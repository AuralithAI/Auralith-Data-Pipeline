"""Tests for the distributed processing module.

All tests use the InMemoryStateStore so they run without Redis.
"""

import threading
import time
from pathlib import Path

import pytest

from auralith_pipeline.distributed.config import (
    CoordinatorConfig,
    DistributedConfig,
    JobConfig,
    WorkerPoolConfig,
)
from auralith_pipeline.distributed.coordinator import Coordinator, JobManager
from auralith_pipeline.distributed.state import InMemoryStateStore
from auralith_pipeline.distributed.strategies import (
    DynamicDistribution,
    LeastBusyDistribution,
    RoundRobinDistribution,
)
from auralith_pipeline.distributed.worker import Worker, WorkerPool
from auralith_pipeline.utils.file_types import classify_file as _classify_file

# =====================================================================
# InMemoryStateStore
# =====================================================================


class TestInMemoryStateStore:
    """Tests for InMemoryStateStore."""

    def test_connect_disconnect_noop(self):
        store = InMemoryStateStore()
        store.connect()
        store.disconnect()

    def test_set_get(self):
        store = InMemoryStateStore()
        store.set("key1", {"hello": "world"})
        assert store.get("key1") == {"hello": "world"}

    def test_get_missing_returns_none(self):
        store = InMemoryStateStore()
        assert store.get("missing") is None

    def test_delete(self):
        store = InMemoryStateStore()
        store.set("k", 42)
        store.delete("k")
        assert store.get("k") is None

    def test_exists(self):
        store = InMemoryStateStore()
        assert not store.exists("k")
        store.set("k", 1)
        assert store.exists("k")

    def test_queue_push_pop(self):
        store = InMemoryStateStore()
        store.push_queue("q", {"id": 1})
        store.push_queue("q", {"id": 2})
        assert store.queue_length("q") == 2
        assert store.pop_queue("q") == {"id": 1}
        assert store.pop_queue("q") == {"id": 2}
        assert store.pop_queue("q") is None

    def test_pop_empty_queue_returns_none(self):
        store = InMemoryStateStore()
        assert store.pop_queue("nonexistent") is None

    def test_set_heartbeat_and_check(self):
        store = InMemoryStateStore()
        store.set_heartbeat("worker-1", ttl=30)
        assert store.check_heartbeat("worker-1")
        assert not store.check_heartbeat("worker-999")

    def test_get_all_workers(self):
        store = InMemoryStateStore()
        assert store.get_all_workers() == []
        store.set_heartbeat("w1")
        store.set_heartbeat("w2")
        workers = sorted(store.get_all_workers())
        assert workers == ["w1", "w2"]

    def test_get_all_workers_after_delete(self):
        store = InMemoryStateStore()
        store.set_heartbeat("w1")
        store.delete("heartbeat:w1")
        assert store.get_all_workers() == []

    def test_atomic_update_basic(self):
        store = InMemoryStateStore()
        store.set("counter", {"value": 0})

        def inc(current):
            current["value"] += 1
            return current

        result = store.atomic_update("counter", inc)
        assert result == {"value": 1}
        assert store.get("counter") == {"value": 1}

    def test_atomic_update_missing_key(self):
        store = InMemoryStateStore()

        def init(current):
            if current is None:
                return {"value": 42}
            return current

        result = store.atomic_update("new_key", init)
        assert result == {"value": 42}

    def test_atomic_update_thread_safety(self):
        """Concurrent atomic_update calls must not lose increments."""
        store = InMemoryStateStore()
        store.set("job:stress", {"completed_tasks": 0})
        n_threads = 20
        increments_per_thread = 50
        barrier = threading.Barrier(n_threads)

        def worker():
            barrier.wait()
            for _ in range(increments_per_thread):
                def inc(job):
                    job["completed_tasks"] = job.get("completed_tasks", 0) + 1
                    return job
                store.atomic_update("job:stress", inc)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        job = store.get("job:stress")
        assert job["completed_tasks"] == n_threads * increments_per_thread


# =====================================================================
# CoordinatorConfig
# =====================================================================


class TestCoordinatorConfig:
    """Tests for CoordinatorConfig defaults and fields."""

    def test_defaults(self):
        cfg = CoordinatorConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 8080
        assert cfg.state_store_type == "redis"
        assert cfg.state_store_password is None
        assert cfg.max_retries == 3

    def test_memory_state_store(self):
        cfg = CoordinatorConfig(state_store_type="memory")
        assert cfg.state_store_type == "memory"


# =====================================================================
# JobManager
# =====================================================================


class TestJobManager:
    """Tests for JobManager task lifecycle."""

    @pytest.fixture()
    def manager(self):
        store = InMemoryStateStore()
        store.connect()
        cfg = CoordinatorConfig(state_store_type="memory", max_retries=2)
        return JobManager(cfg, store)

    def test_create_job(self, manager: JobManager):
        job = manager.create_job("job-1", {"input": "/data"})
        assert job["id"] == "job-1"
        assert job["status"] == "pending"
        assert job["total_tasks"] == 0
        # Also persisted in state store
        stored = manager.state_store.get("job:job-1")
        assert stored["id"] == "job-1"

    def test_update_job_status(self, manager: JobManager):
        manager.create_job("j", {})
        manager.update_job_status("j", "running")
        assert manager.jobs["j"]["status"] == "running"

    def test_submit_task_no_workers_queues_pending(self, manager: JobManager):
        manager.create_job("j", {})
        manager.submit_task("j", {"id": "t1"})
        # Should land in pending_tasks since no workers have heartbeats
        assert manager.state_store.queue_length("pending_tasks") == 1

    def test_submit_task_with_worker(self, manager: JobManager):
        manager.create_job("j", {})
        # Register a worker heartbeat
        manager.state_store.set_heartbeat("worker-a")
        manager.submit_task("j", {"id": "t1"})
        # Task should be in worker-a's queue
        assert manager.state_store.queue_length("tasks:worker-a") == 1
        task = manager.state_store.pop_queue("tasks:worker-a")
        assert task["id"] == "t1"
        assert task["worker_id"] == "worker-a"

    def test_submit_tasks_batch(self, manager: JobManager):
        manager.create_job("j", {})
        manager.state_store.set_heartbeat("w1")
        manager.state_store.set_heartbeat("w2")

        tasks = [{"id": f"t{i}"} for i in range(4)]
        manager.submit_tasks("j", tasks)

        job = manager.jobs["j"]
        assert job["total_tasks"] == 4
        assert job["status"] == "running"

        # Tasks should be distributed across w1 and w2 (round-robin)
        q1 = manager.state_store.queue_length("tasks:w1")
        q2 = manager.state_store.queue_length("tasks:w2")
        assert q1 + q2 == 4

    def test_record_task_result_success(self, manager: JobManager):
        manager.create_job("j", {})
        manager.record_task_result("j", "t1", success=True)
        assert manager.jobs["j"]["completed_tasks"] == 1

    def test_record_task_result_failure(self, manager: JobManager):
        manager.create_job("j", {})
        manager.record_task_result("j", "t1", success=False)
        assert manager.jobs["j"]["failed_tasks"] == 1

    def test_handle_worker_failure_requeues_tasks(self, manager: JobManager):
        manager.create_job("j", {})
        manager.state_store.set_heartbeat("dead-worker")
        # Put tasks in dead worker's queue
        for i in range(3):
            manager.state_store.push_queue(
                "tasks:dead-worker",
                {"id": f"t{i}", "job_id": "j", "retries": 0},
            )
        manager._handle_worker_failure("dead-worker")

        # Heartbeat should be removed
        assert not manager.state_store.check_heartbeat("dead-worker")
        # Tasks should be in pending queue
        assert manager.state_store.queue_length("pending_tasks") == 3

    def test_handle_worker_failure_exceeds_max_retries(self, manager: JobManager):
        """Tasks that already hit max_retries should be marked failed."""
        manager.create_job("j", {})
        manager.state_store.set_heartbeat("dead")
        # Task already at max retries (2) — next bump makes it 3 > max_retries=2
        manager.state_store.push_queue(
            "tasks:dead",
            {"id": "t-old", "job_id": "j", "retries": 2},
        )
        manager._handle_worker_failure("dead")

        # Should NOT be in pending queue
        assert manager.state_store.queue_length("pending_tasks") == 0
        # Should be marked failed in state store
        task = manager.state_store.get("task:t-old")
        assert task["status"] == "failed"
        # Job should have 1 failed task
        assert manager.jobs["j"]["failed_tasks"] == 1

    def test_set_strategy_round_robin(self, manager: JobManager):
        manager.set_strategy("round_robin")
        assert isinstance(manager.distribution_strategy, RoundRobinDistribution)

    def test_set_strategy_least_busy(self, manager: JobManager):
        manager.set_strategy("least_busy")
        assert isinstance(manager.distribution_strategy, LeastBusyDistribution)

    def test_set_strategy_unknown_keeps_current(self, manager: JobManager):
        original = manager.distribution_strategy
        manager.set_strategy("nonexistent_strategy")
        assert manager.distribution_strategy is original


# =====================================================================
# Coordinator
# =====================================================================


class TestCoordinator:
    """Tests for Coordinator lifecycle."""

    def test_start_stop_memory(self):
        cfg = CoordinatorConfig(state_store_type="memory")
        coord = Coordinator(cfg)
        coord.start()
        assert coord.is_running()
        coord.stop()
        assert not coord.is_running()

    def test_create_state_store_redis_import(self):
        """RedisStateStore is created (but not connected) for type='redis'."""
        cfg = CoordinatorConfig(state_store_type="redis")
        coord = Coordinator(cfg)
        from auralith_pipeline.distributed.state import RedisStateStore

        assert isinstance(coord.state_store, RedisStateStore)

    def test_create_state_store_invalid_raises(self):
        cfg = CoordinatorConfig(state_store_type="cassandra")
        with pytest.raises(ValueError, match="Unsupported"):
            Coordinator(cfg)


# =====================================================================
# Worker helpers
# =====================================================================


class TestClassifyFile:
    """Tests for the classify_file helper (shared via utils.file_types)."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("file.txt", "text"),
            ("file.md", "text"),
            ("file.json", "text"),
            ("file.png", "image"),
            ("file.jpg", "image"),
            ("file.wav", "audio"),
            ("file.mp3", "audio"),
            ("file.mp4", "video"),
            ("file.mkv", "video"),
            ("file.xyz", None),
        ],
    )
    def test_classify(self, name: str, expected: str | None):
        assert _classify_file(Path(name)) == expected

    def test_npy_in_image_dir(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        assert _classify_file(img_dir / "data.npy") == "image"

    def test_npy_in_audio_dir(self, tmp_path):
        aud_dir = tmp_path / "audio"
        aud_dir.mkdir()
        assert _classify_file(aud_dir / "data.npy") == "audio"

    def test_npy_ambiguous(self, tmp_path):
        misc_dir = tmp_path / "misc"
        misc_dir.mkdir()
        assert _classify_file(misc_dir / "data.npy") is None


# =====================================================================
# Worker lifecycle
# =====================================================================


class TestWorker:
    """Tests for Worker start / stop / heartbeat."""

    def test_start_without_connect_raises(self):
        w = Worker("w1", "localhost", 8080)
        with pytest.raises(RuntimeError, match="not connected"):
            w.start()

    def test_start_stop(self):
        store = InMemoryStateStore()
        store.connect()

        w = Worker("w1", "localhost", 8080)
        w.connect(store)
        w.start()
        # Give heartbeat thread time to write
        time.sleep(0.5)
        assert store.check_heartbeat("w1")
        w.stop()
        # Heartbeat should be cleaned up
        assert not store.check_heartbeat("w1")

    def test_worker_processes_task(self, tmp_path: Path):
        """Worker picks up a task and marks it completed."""
        store = InMemoryStateStore()
        store.connect()

        # Create a simple text file
        txt = tmp_path / "hello.txt"
        txt.write_text("Hello distributed world!")

        # Create a minimal tokenizers dir with a dummy text tokenizer
        tok_dir = tmp_path / "tokenizers" / "text"
        tok_dir.mkdir(parents=True)

        w = Worker("w1", "localhost", 8080)
        w.connect(store)
        w.start()
        time.sleep(0.3)  # let heartbeat register

        # Push a task — but since we don't have a real tokenizer,
        # we expect the task to fail (which proves the worker ran it)
        task = {
            "id": "task-1",
            "job_id": "job-1",
            "status": "pending",
            "retries": 0,
            "files": [str(txt)],
            "tokenizers_dir": str(tmp_path / "tokenizers"),
            "output_dir": str(tmp_path / "output" / "task_task-1"),
            "max_seq_len": 128,
            "shard_size": 100,
        }
        # Create the job in state store so result recording works
        store.set(
            "job:job-1",
            {
                "id": "job-1",
                "status": "running",
                "total_tasks": 1,
                "completed_tasks": 0,
                "failed_tasks": 0,
            },
        )
        store.push_queue("tasks:w1", task)

        # Wait for worker to pick it up
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            t = store.get("task:task-1")
            if t and t.get("status") in ("completed", "failed"):
                break
            time.sleep(0.2)

        w.stop()

        result_task = store.get("task:task-1")
        assert result_task is not None
        # Task ran — either completed (no tokenizer → no samples → completed
        # with 0 samples) or failed (missing tokenizer files).
        assert result_task["status"] in ("completed", "failed")


# =====================================================================
# WorkerPool
# =====================================================================


class TestWorkerPool:
    """Tests for WorkerPool."""

    def test_start_stop_with_shared_store(self):
        store = InMemoryStateStore()
        store.connect()

        cfg = WorkerPoolConfig(
            name="test-pool",
            worker_ids=["wp-1", "wp-2"],
        )
        pool = WorkerPool(cfg)
        pool.start_workers("localhost", 8080, state_store=store)
        assert len(pool.workers) == 2

        time.sleep(0.3)
        # Both should have heartbeats
        assert store.check_heartbeat("wp-1")
        assert store.check_heartbeat("wp-2")

        pool.stop_workers()


# =====================================================================
# Strategies
# =====================================================================


class TestDistributionStrategies:
    """Tests for task distribution strategies."""

    def test_round_robin(self):
        rr = RoundRobinDistribution()
        workers = ["a", "b", "c"]
        assignments = [rr.assign_task({}, workers) for _ in range(6)]
        assert assignments == ["a", "b", "c", "a", "b", "c"]

    def test_round_robin_no_workers_raises(self):
        rr = RoundRobinDistribution()
        with pytest.raises(ValueError, match="No available workers"):
            rr.assign_task({}, [])

    def test_least_busy(self):
        store = InMemoryStateStore()
        store.connect()
        # w1 has 5 tasks, w2 has 1
        for _ in range(5):
            store.push_queue("tasks:w1", {})
        store.push_queue("tasks:w2", {})

        lb = LeastBusyDistribution(store)
        assert lb.assign_task({}, ["w1", "w2"]) == "w2"

    def test_dynamic_distribution(self):
        store = InMemoryStateStore()
        store.connect()
        # w1 has high CPU, w2 is idle
        store.set("metrics:w1", {"cpu_usage": 90, "memory_usage": 80})
        store.set("metrics:w2", {"cpu_usage": 10, "memory_usage": 20})

        dd = DynamicDistribution(store)
        assert dd.assign_task({}, ["w1", "w2"]) == "w2"


# =====================================================================
# DistributedPipeline — task creation
# =====================================================================


class TestDistributedPipelineTaskCreation:
    """Test task splitting logic (no workers needed)."""

    def test_create_tasks_splits_evenly(self):
        from auralith_pipeline.distributed.pipeline import DistributedPipeline

        files = [f"/data/file_{i}.txt" for i in range(10)]
        tasks = DistributedPipeline._create_tasks(
            job_id="j1",
            files=files,
            tokenizers_dir="/tok",
            output_dir="/out",
            max_seq_len=256,
            shard_size=100,
            files_per_task=3,
        )
        # 10 files / 3 per task = ceil(3.33) = 4 tasks
        assert len(tasks) == 4
        # First task has 3 files, last has 1
        assert len(tasks[0]["files"]) == 3
        assert len(tasks[3]["files"]) == 1
        # All tasks reference the correct job
        assert all(t["job_id"] == "j1" for t in tasks)
        # Each task has its own output subdirectory (no shard_offset)
        output_dirs = [t["output_dir"] for t in tasks]
        assert len(set(output_dirs)) == 4  # all unique
        assert all("task_" in d for d in output_dirs)
        assert all("shard_offset" not in t for t in tasks)

    def test_create_tasks_single_chunk(self):
        from auralith_pipeline.distributed.pipeline import DistributedPipeline

        files = ["/data/a.txt", "/data/b.txt"]
        tasks = DistributedPipeline._create_tasks(
            job_id="j",
            files=files,
            tokenizers_dir="/tok",
            output_dir="/out",
            max_seq_len=128,
            shard_size=50,
            files_per_task=100,
        )
        assert len(tasks) == 1
        assert len(tasks[0]["files"]) == 2
        assert "task_" in tasks[0]["output_dir"]
        assert "shard_offset" not in tasks[0]


# =====================================================================
# DistributedPipeline — embedded end-to-end
# =====================================================================


class TestDistributedPipelineEmbedded:
    """End-to-end test with embedded coordinator + workers (in-memory)."""

    def test_run_embedded_no_files_raises(self, tmp_path: Path):
        from auralith_pipeline.distributed.pipeline import DistributedPipeline

        job_cfg = JobConfig(name="test", num_workers=1)
        dp = DistributedPipeline(job_cfg, embedded=True, num_workers=1)

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No files found"):
            dp.run(
                input_dir=str(empty_dir),
                output_dir=str(tmp_path / "out"),
                tokenizers_dir=str(tmp_path / "tok"),
            )

    def test_run_embedded_completes(self, tmp_path: Path):
        """Full embedded run — files exist but no real tokenizer, so tasks
        complete with 0 samples (worker handles gracefully)."""
        from auralith_pipeline.distributed.pipeline import DistributedPipeline

        # Create input files
        in_dir = tmp_path / "input"
        in_dir.mkdir()
        for i in range(5):
            (in_dir / f"doc_{i}.txt").write_text(f"Document {i} content")

        # Create tokenizers dir (no actual tokenizer — worker produces 0 samples)
        tok_dir = tmp_path / "tokenizers"
        tok_dir.mkdir()

        out_dir = tmp_path / "output"

        job_cfg = JobConfig(name="e2e-test", num_workers=2, timeout=30)
        dp = DistributedPipeline(job_cfg, embedded=True, num_workers=2)

        result = dp.run(
            input_dir=str(in_dir),
            output_dir=str(out_dir),
            tokenizers_dir=str(tok_dir),
            max_seq_len=128,
            shard_size=100,
            files_per_task=3,
            poll_interval=0.5,
            timeout=30,
        )

        assert result["status"] in ("completed", "completed_with_errors")
        assert result["total_tasks"] == 2  # ceil(5/3)


# =====================================================================
# DistributedConfig YAML round-trip
# =====================================================================


class TestDistributedConfigYaml:
    """Test YAML serialisation."""

    def test_to_yaml_from_yaml(self, tmp_path: Path):
        cfg = DistributedConfig(
            coordinator=CoordinatorConfig(host="10.0.0.1", port=9090),
            workers=[
                WorkerPoolConfig(name="pool-a", worker_ids=["w1", "w2"]),
            ],
            job=JobConfig(name="roundtrip-test", num_workers=4),
        )
        path = tmp_path / "dist.yaml"
        cfg.to_yaml(str(path))
        assert path.exists()

        loaded = DistributedConfig.from_yaml(str(path))
        assert loaded.coordinator.host == "10.0.0.1"
        assert loaded.coordinator.port == 9090
        assert loaded.job.name == "roundtrip-test"
        assert len(loaded.workers) == 1
        assert loaded.workers[0].name == "pool-a"
