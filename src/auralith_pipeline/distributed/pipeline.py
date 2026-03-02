"""Distributed pipeline implementation.

``DistributedPipeline`` is the top-level entry-point for distributed
processing.  It scans an input directory for raw files, splits them into
chunks (one chunk per task), submits the tasks to a running
``Coordinator`` via the shared state-store, and polls until all tasks
are finished (or failed / timed-out).
"""

import logging
import math
import time
import uuid
from pathlib import Path
from typing import Any

from auralith_pipeline.distributed.config import DistributedConfig, JobConfig
from auralith_pipeline.distributed.coordinator import Coordinator
from auralith_pipeline.distributed.state import StateStore
from auralith_pipeline.distributed.worker import Worker

logger = logging.getLogger(__name__)


class DistributedPipeline:
    """Distribute file-level tokenization + sharding across workers.

    Modes of operation
    ------------------
    1. **External coordinator** – you start a ``Coordinator`` and N ``Worker``
       processes separately (possibly on different machines) and point this
       pipeline at the same state-store so it only *submits* tasks.
    2. **Embedded (single-machine)** – pass ``embedded=True`` and the pipeline
       will spin up a ``Coordinator`` + *N* ``Worker`` threads in-process,
       using the in-memory state-store (no Redis needed).
    """

    def __init__(
        self,
        job_config: JobConfig,
        distributed_config: DistributedConfig | None = None,
        *,
        embedded: bool = False,
        num_workers: int | None = None,
    ):
        self.job_config = job_config
        self.dist_config = distributed_config or DistributedConfig()
        self.embedded = embedded
        self.num_workers = num_workers or self.job_config.num_workers or 2
        self.state_store: StateStore | None = None

        # Embedded-mode handles
        self._coordinator: Coordinator | None = None
        self._workers: list[Worker] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        input_dir: str,
        output_dir: str,
        tokenizers_dir: str,
        *,
        max_seq_len: int = 4096,
        shard_size: int = 10_000,
        files_per_task: int = 500,
        poll_interval: float = 2.0,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Run the full distributed pipeline.

        Parameters
        ----------
        input_dir
            Directory containing raw files (text, images, audio, video).
        output_dir
            Where to write SafeTensors shards.
        tokenizers_dir
            Directory containing trained tokenizer subdirectories.
        max_seq_len
            Maximum sequence length for tokenization.
        shard_size
            Samples per shard.
        files_per_task
            How many files to pack into each distributed task.
        poll_interval
            Seconds between progress polls.
        timeout
            Overall job timeout in seconds (``None`` = use ``job_config.timeout``).

        Returns
        -------
        dict
            Final job summary with completion stats.
        """
        timeout = timeout if timeout is not None else self.job_config.timeout
        in_path = Path(input_dir)
        out_path = Path(output_dir)

        # 1. Discover raw files
        raw_files = sorted(str(p) for p in in_path.rglob("*") if p.is_file())
        if not raw_files:
            raise FileNotFoundError(f"No files found in {input_dir}")

        logger.info("Discovered %d raw files in %s", len(raw_files), input_dir)

        # 2. Start embedded infra if requested
        if self.embedded:
            self._start_embedded()

        # 3. Connect to state store
        state_store = self._get_state_store()

        # 4. Create job
        job_id = str(uuid.uuid4())
        job: dict[str, Any] = {
            "id": job_id,
            "status": "pending",
            "config": {
                "input_dir": input_dir,
                "output_dir": output_dir,
                "tokenizers_dir": tokenizers_dir,
                "max_seq_len": max_seq_len,
                "shard_size": shard_size,
            },
            "tasks": [],
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
        }
        state_store.set(f"job:{job_id}", job)
        logger.info("Created job %s", job_id)

        # 5. Split files into tasks
        tasks = self._create_tasks(
            job_id=job_id,
            files=raw_files,
            tokenizers_dir=str(tokenizers_dir),
            output_dir=str(out_path),
            max_seq_len=max_seq_len,
            shard_size=shard_size,
            files_per_task=files_per_task,
        )
        logger.info("Created %d tasks (%d files/task)", len(tasks), files_per_task)

        # 6. Submit tasks
        if self._coordinator is not None:
            # Embedded — submit via coordinator's JobManager directly
            self._coordinator.job_manager.create_job(job_id, job.get("config", {}))
            self._coordinator.job_manager.submit_tasks(job_id, tasks)
        else:
            # External — push tasks + update job in state store
            job["total_tasks"] = len(tasks)
            job["status"] = "running"
            state_store.set(f"job:{job_id}", job)
            for task in tasks:
                state_store.push_queue("pending_tasks", task)

        # 7. Poll until done
        result = self._poll_job(state_store, job_id, timeout, poll_interval)

        # 8. Tear down embedded infra
        if self.embedded:
            self._stop_embedded()

        return result

    # ------------------------------------------------------------------
    # Task creation
    # ------------------------------------------------------------------

    @staticmethod
    def _create_tasks(
        *,
        job_id: str,
        files: list[str],
        tokenizers_dir: str,
        output_dir: str,
        max_seq_len: int,
        shard_size: int,
        files_per_task: int,
    ) -> list[dict[str, Any]]:
        """Split file list into task dicts."""
        num_tasks = max(1, math.ceil(len(files) / files_per_task))
        tasks: list[dict[str, Any]] = []

        for i in range(num_tasks):
            chunk = files[i * files_per_task : (i + 1) * files_per_task]
            tasks.append(
                {
                    "id": str(uuid.uuid4()),
                    "job_id": job_id,
                    "status": "pending",
                    "retries": 0,
                    "files": chunk,
                    "tokenizers_dir": tokenizers_dir,
                    "output_dir": output_dir,
                    "max_seq_len": max_seq_len,
                    "shard_size": shard_size,
                    "shard_offset": i * shard_size,
                }
            )

        return tasks

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    @staticmethod
    def _poll_job(
        state_store: StateStore,
        job_id: str,
        timeout: int,
        poll_interval: float,
    ) -> dict[str, Any]:
        """Block until the job completes or times out."""
        start = time.monotonic()
        last_log = 0.0

        while True:
            elapsed = time.monotonic() - start
            if elapsed > timeout:
                logger.error("Job %s timed out after %ds", job_id, timeout)
                return {"job_id": job_id, "status": "timeout", "elapsed": elapsed}

            job = state_store.get(f"job:{job_id}")
            if not job:
                time.sleep(poll_interval)
                continue

            status = job.get("status", "unknown")
            completed = job.get("completed_tasks", 0)
            failed = job.get("failed_tasks", 0)
            total = job.get("total_tasks", 0)

            # Log progress every 10 s
            if elapsed - last_log >= 10:
                logger.info(
                    "Job %s — %d/%d completed, %d failed (%.0fs)",
                    job_id,
                    completed,
                    total,
                    failed,
                    elapsed,
                )
                last_log = elapsed

            if status in ("completed", "failed", "cancelled"):
                return {
                    "job_id": job_id,
                    "status": status,
                    "total_tasks": total,
                    "completed_tasks": completed,
                    "failed_tasks": failed,
                    "elapsed_seconds": elapsed,
                }

            # Also check if all tasks are done even if status wasn't flipped
            if total > 0 and completed + failed >= total:
                final_status = "completed" if failed == 0 else "completed_with_errors"
                job["status"] = final_status
                state_store.set(f"job:{job_id}", job)
                return {
                    "job_id": job_id,
                    "status": final_status,
                    "total_tasks": total,
                    "completed_tasks": completed,
                    "failed_tasks": failed,
                    "elapsed_seconds": elapsed,
                }

            time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Embedded mode helpers
    # ------------------------------------------------------------------

    def _start_embedded(self) -> None:
        """Spin up coordinator + N workers in-process (single-machine)."""
        from auralith_pipeline.distributed.config import CoordinatorConfig

        cfg = CoordinatorConfig(
            state_store_type="memory",
            heartbeat_interval=self.dist_config.coordinator.heartbeat_interval,
            heartbeat_timeout=self.dist_config.coordinator.heartbeat_timeout,
            max_retries=self.dist_config.coordinator.max_retries,
            retry_backoff=self.dist_config.coordinator.retry_backoff,
        )
        self._coordinator = Coordinator(cfg)
        self._coordinator.start()

        store = self._coordinator.state_store
        for i in range(self.num_workers):
            wid = f"embedded-worker-{i}"
            w = Worker(wid, "localhost", cfg.port)
            w.connect(store)
            w.start()
            self._workers.append(w)

        logger.info("Embedded mode: coordinator + %d workers started", self.num_workers)

    def _stop_embedded(self) -> None:
        """Stop embedded workers and coordinator."""
        for w in self._workers:
            w.stop()
        self._workers.clear()
        if self._coordinator:
            self._coordinator.stop()
            self._coordinator = None
        logger.info("Embedded infrastructure stopped")

    def _get_state_store(self) -> StateStore:
        """Return the state store, connecting if needed."""
        if self._coordinator is not None:
            return self._coordinator.state_store

        # External coordinator — connect via config
        cfg = self.dist_config.coordinator
        if cfg.state_store_type == "redis":
            from auralith_pipeline.distributed.state import RedisStateStore

            store = RedisStateStore(
                host=cfg.state_store_host,
                port=cfg.state_store_port,
                db=cfg.state_store_db,
                password=cfg.state_store_password,
            )
            store.connect()
            self.state_store = store
            return store

        if cfg.state_store_type == "memory":
            from auralith_pipeline.distributed.state import InMemoryStateStore

            store = InMemoryStateStore()
            store.connect()
            self.state_store = store
            return store

        raise ValueError(f"Unsupported state store type: {cfg.state_store_type}")
