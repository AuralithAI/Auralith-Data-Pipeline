"""Worker node for distributed processing.

A Worker connects to the shared state-store (Redis or in-memory),
registers a heartbeat, and continuously polls its personal task queue
(``tasks:<worker_id>``).  Each task carries a list of file paths plus
tokenizer / output configuration so the worker can independently
tokenize files and write SafeTensors shards.
"""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from auralith_pipeline.distributed.config import WorkerPoolConfig
from auralith_pipeline.distributed.state import StateStore
from auralith_pipeline.utils.file_types import (
    AUDIO_TOKEN_OFFSET as _AUDIO_TOKEN_OFFSET,
)
from auralith_pipeline.utils.file_types import (
    IMAGE_TOKEN_OFFSET as _IMAGE_TOKEN_OFFSET,
)
from auralith_pipeline.utils.file_types import (
    MODALITY_ID as _MODALITY_ID,
)
from auralith_pipeline.utils.file_types import (
    VIDEO_TOKEN_OFFSET as _VIDEO_TOKEN_OFFSET,
)
from auralith_pipeline.utils.file_types import (
    classify_file as _classify_file,
)

logger = logging.getLogger(__name__)

try:
    import psutil  # type: ignore[import-untyped]

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ── Worker ─────────────────────────────────────────────────────────────


class Worker:
    """Worker node that processes pipeline tasks.

    Each worker runs two daemon threads:
    * **heartbeat** – periodically writes ``heartbeat:<worker_id>`` to the
      state store so the coordinator knows it's alive.
    * **processing** – pops tasks from ``tasks:<worker_id>`` and executes
      them (tokenize files → write shards).
    """

    def __init__(self, worker_id: str, coordinator_host: str, coordinator_port: int):
        self.worker_id = worker_id
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.state_store: StateStore | None = None
        self._running = False
        self._processing_thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None
        # Metrics – updated after every task
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._samples_processed = 0

    # ── lifecycle ──────────────────────────────────────────────────────

    def connect(self, state_store: StateStore) -> None:
        """Connect to the shared state store."""
        self.state_store = state_store
        self.state_store.connect()
        logger.info(
            "Worker %s connected to coordinator at %s:%s",
            self.worker_id,
            self.coordinator_host,
            self.coordinator_port,
        )

    def start(self) -> None:
        """Start heartbeat and task-processing threads."""
        if not self.state_store:
            raise RuntimeError("Worker not connected to state store")

        self._running = True

        self._heartbeat_thread = threading.Thread(target=self._send_heartbeats, daemon=True)
        self._heartbeat_thread.start()

        self._processing_thread = threading.Thread(target=self._process_tasks, daemon=True)
        self._processing_thread.start()

        logger.info("Worker %s started", self.worker_id)

    def stop(self) -> None:
        """Stop the worker gracefully."""
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
        if self._processing_thread:
            self._processing_thread.join(timeout=5)
        # Remove heartbeat so coordinator doesn't keep pinging
        if self.state_store:
            self.state_store.delete(f"heartbeat:{self.worker_id}")
        logger.info("Worker %s stopped", self.worker_id)

    # ── heartbeat ─────────────────────────────────────────────────────

    def _send_heartbeats(self) -> None:
        """Periodically write heartbeat key with TTL."""
        while self._running:
            try:
                if self.state_store is None:
                    break
                heartbeat_data = {
                    "worker_id": self.worker_id,
                    "timestamp": datetime.now().isoformat(),
                    "status": "alive",
                    "tasks_completed": self._tasks_completed,
                    "tasks_failed": self._tasks_failed,
                    "samples_processed": self._samples_processed,
                }
                self.state_store.set(f"heartbeat:{self.worker_id}", heartbeat_data, ttl=30)
                # Expose detailed metrics for LeastBusy / Dynamic strategies
                self.state_store.set(
                    f"metrics:{self.worker_id}",
                    {
                        "cpu_usage": psutil.cpu_percent() if _HAS_PSUTIL else 0,
                        "memory_usage": (psutil.virtual_memory().percent if _HAS_PSUTIL else 0),
                        "tasks_completed": self._tasks_completed,
                        "samples_processed": self._samples_processed,
                    },
                )
                time.sleep(10)
            except Exception:
                logger.exception("Error sending heartbeat")

    # ── task loop ─────────────────────────────────────────────────────

    def _process_tasks(self) -> None:
        """Poll personal queue and execute tasks."""
        queue_name = f"tasks:{self.worker_id}"

        while self._running:
            try:
                if self.state_store is None:
                    break
                task = self.state_store.pop_queue(queue_name, timeout=5)
                if task:
                    self._execute_task(task)
            except Exception:
                logger.exception("Error in task loop")

    def _execute_task(self, task: dict[str, Any]) -> None:
        """Execute a single task and write results back to state store."""
        if self.state_store is None:
            return
        task_id = task.get("id", "unknown")
        logger.info("Executing task %s", task_id)

        try:
            task["status"] = "running"
            task["started_at"] = datetime.now().isoformat()
            self.state_store.set(f"task:{task_id}", task)

            result = self._run_pipeline_task(task)

            task["status"] = "completed"
            task["completed_at"] = datetime.now().isoformat()
            task["result"] = result
            self.state_store.set(f"task:{task_id}", task)

            self._tasks_completed += 1
            self._samples_processed += result.get("samples_processed", 0)

            # Notify coordinator
            job_id = task.get("job_id")
            if job_id:
                self._record_result(job_id, task_id, success=True)

            logger.info("Task %s completed — %s", task_id, result)

        except Exception as exc:
            logger.error("Task %s failed: %s", task_id, exc)
            task["status"] = "failed"
            task["error"] = str(exc)
            task["failed_at"] = datetime.now().isoformat()
            self.state_store.set(f"task:{task_id}", task)

            self._tasks_failed += 1
            job_id = task.get("job_id")
            if job_id:
                self._record_result(job_id, task_id, success=False)

    def _record_result(self, job_id: str, task_id: str, *, success: bool) -> None:
        """Update the job counters in the state store (coordinator-side)."""
        if self.state_store is None:
            return
        job = self.state_store.get(f"job:{job_id}")
        if not job:
            return
        if success:
            job["completed_tasks"] = job.get("completed_tasks", 0) + 1
        else:
            job["failed_tasks"] = job.get("failed_tasks", 0) + 1

        total = job.get("total_tasks", 0)
        done = job.get("completed_tasks", 0) + job.get("failed_tasks", 0)
        if total > 0 and done >= total:
            job["status"] = "completed"
            job["completed_at"] = datetime.now().isoformat()

        self.state_store.set(f"job:{job_id}", job)

    # ── real pipeline task execution ──────────────────────────────────

    def _run_pipeline_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Tokenize files and write SafeTensors shards.

        Expected task keys
        ------------------
        files : list[str]          – absolute paths to raw files
        tokenizers_dir : str       – path to tokenizers/ directory
        output_dir : str           – shard output directory
        max_seq_len : int          – maximum sequence length (default 4096)
        shard_size : int           – samples per shard (default 10000)
        shard_offset : int         – starting shard index (default 0)
        """
        files = task.get("files", [])
        tokenizers_dir = Path(task["tokenizers_dir"])
        output_dir = Path(task["output_dir"])
        max_seq_len: int = task.get("max_seq_len", 4096)
        shard_size: int = task.get("shard_size", 10_000)
        shard_offset: int = task.get("shard_offset", 0)

        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load tokenizers (lazy import to keep worker module lightweight)
        tokenizers = self._load_tokenizers(tokenizers_dir)
        loaded = [name for name, tok in tokenizers.items() if tok is not None]
        if not loaded:
            raise RuntimeError(f"No tokenizers found in {tokenizers_dir}")
        logger.info("Worker %s loaded tokenizers: %s", self.worker_id, loaded)

        # 2. Tokenize every file
        samples: list[dict[str, Any]] = []
        skipped = 0
        for file_str in files:
            file_path = Path(file_str)
            result = self._tokenize_file(file_path, tokenizers, max_seq_len)
            if result is not None:
                samples.append(result)
            else:
                skipped += 1

        logger.info(
            "Worker %s tokenized %d files (%d skipped)",
            self.worker_id,
            len(samples),
            skipped,
        )

        if not samples:
            return {
                "samples_processed": 0,
                "shards_written": 0,
                "skipped": skipped,
                "status": "no_samples",
            }

        # 3. Write shards
        num_shards = self._write_shards(samples, output_dir, max_seq_len, shard_size, shard_offset)

        return {
            "samples_processed": len(samples),
            "shards_written": num_shards,
            "skipped": skipped,
            "status": "success",
        }

    # ── tokenizer loading ─────────────────────────────────────────────

    @staticmethod
    def _load_tokenizers(tokenizers_dir: Path) -> dict[str, Any]:
        """Load all available tokenizers from a directory."""
        result: dict[str, Any] = {
            "text": None,
            "image": None,
            "audio": None,
            "video": None,
        }

        text_dir = tokenizers_dir / "text"
        if text_dir.is_dir() and (text_dir / "vocab.json").exists():
            from auralith_pipeline.tokenization import BPETokenizer

            result["text"] = BPETokenizer.load(text_dir)

        image_dir = tokenizers_dir / "image"
        if image_dir.is_dir() and (image_dir / "config.json").exists():
            from auralith_pipeline.tokenization import ImageTokenizer

            result["image"] = ImageTokenizer.load(image_dir)

        audio_dir = tokenizers_dir / "audio"
        if audio_dir.is_dir() and (audio_dir / "config.json").exists():
            from auralith_pipeline.tokenization import AudioTokenizer

            result["audio"] = AudioTokenizer.load(audio_dir)

        video_dir = tokenizers_dir / "video"
        if video_dir.is_dir() and (video_dir / "config.json").exists():
            from auralith_pipeline.tokenization import VideoTokenizer

            result["video"] = VideoTokenizer.load(video_dir)

        return result

    # ── per-file tokenization ─────────────────────────────────────────

    @staticmethod
    def _tokenize_file(
        file_path: Path,
        tokenizers: dict[str, Any],
        max_seq_len: int,
    ) -> dict[str, Any] | None:
        """Tokenize a single raw file and return input_ids + modality_mask."""
        modality = _classify_file(file_path)
        if modality is None:
            return None

        tokenizer = tokenizers.get(modality)
        if tokenizer is None:
            return None

        try:
            if modality == "text":
                text = file_path.read_text(encoding="utf-8", errors="replace")
                if not text.strip():
                    return None
                input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=max_seq_len)
                modality_mask = [_MODALITY_ID["text"]] * len(input_ids)

            elif modality == "image":
                codes = tokenizer.encode(file_path)
                from auralith_pipeline.tokenization import BPETokenizer

                img_s = BPETokenizer.SPECIAL_TOKENS["<IMG>"]
                img_e = BPETokenizer.SPECIAL_TOKENS["<IMG_END>"]
                input_ids = [img_s] + [c + _IMAGE_TOKEN_OFFSET for c in codes] + [img_e]
                modality_mask = [_MODALITY_ID["image"]] * len(input_ids)

            elif modality == "audio":
                codes = tokenizer.encode(file_path)
                from auralith_pipeline.tokenization import BPETokenizer

                aud_s = BPETokenizer.SPECIAL_TOKENS["<AUDIO>"]
                aud_e = BPETokenizer.SPECIAL_TOKENS["<AUDIO_END>"]
                input_ids = [aud_s] + [c + _AUDIO_TOKEN_OFFSET for c in codes] + [aud_e]
                modality_mask = [_MODALITY_ID["audio"]] * len(input_ids)

            elif modality == "video":
                codes = tokenizer.encode(file_path)
                from auralith_pipeline.tokenization import BPETokenizer

                vid_s = BPETokenizer.SPECIAL_TOKENS["<VIDEO>"]
                vid_e = BPETokenizer.SPECIAL_TOKENS["<VIDEO_END>"]
                input_ids = [vid_s] + [c + _VIDEO_TOKEN_OFFSET for c in codes] + [vid_e]
                modality_mask = [_MODALITY_ID["video"]] * len(input_ids)

            else:
                return None

            input_ids = input_ids[:max_seq_len]
            modality_mask = modality_mask[:max_seq_len]

            return {
                "input_ids": input_ids,
                "modality_mask": modality_mask,
                "source": str(file_path),
            }

        except Exception as exc:
            logger.warning("Failed to tokenize %s: %s", file_path, exc)
            return None

    # ── shard writing ─────────────────────────────────────────────────

    @staticmethod
    def _write_shards(
        samples: list[dict[str, Any]],
        output_dir: Path,
        max_seq_len: int,
        shard_size: int,
        shard_offset: int = 0,
    ) -> int:
        """Pack tokenized samples into SafeTensors shards."""
        import numpy as np

        try:
            from safetensors.numpy import save_file
        except ImportError:
            raise ImportError(
                "safetensors is required for shard writing. "
                "Install with: pip install safetensors"
            ) from None

        shard_idx = shard_offset
        for start in range(0, len(samples), shard_size):
            batch = samples[start : start + shard_size]

            input_ids_arr = np.zeros((len(batch), max_seq_len), dtype=np.int32)
            attention_mask_arr = np.zeros((len(batch), max_seq_len), dtype=np.uint8)
            modality_mask_arr = np.zeros((len(batch), max_seq_len), dtype=np.uint8)

            for i, sample in enumerate(batch):
                ids = sample["input_ids"]
                mask = sample["modality_mask"]
                seq_len = min(len(ids), max_seq_len)
                input_ids_arr[i, :seq_len] = ids[:seq_len]
                attention_mask_arr[i, :seq_len] = 1
                modality_mask_arr[i, :seq_len] = mask[:seq_len]

            targets_arr = np.zeros_like(input_ids_arr)
            targets_arr[:, :-1] = input_ids_arr[:, 1:]

            shard_path = output_dir / f"shard_{shard_idx:06d}.safetensors"
            metadata = {
                "pipeline_version": "2.0",
                "schema_version": "2",
                "shard_id": str(shard_idx),
                "num_samples": str(len(batch)),
                "seq_length": str(max_seq_len),
            }
            save_file(
                {
                    "input_ids": input_ids_arr,
                    "attention_mask": attention_mask_arr,
                    "modality_mask": modality_mask_arr,
                    "targets": targets_arr,
                },
                str(shard_path),
                metadata=metadata,
            )
            logger.info("Wrote %s (%d samples)", shard_path.name, len(batch))
            shard_idx += 1

        return shard_idx - shard_offset


# ── WorkerPool ─────────────────────────────────────────────────────────


class WorkerPool:
    """Pool of workers for a specific processing stage."""

    def __init__(self, config: WorkerPoolConfig):
        self.config = config
        self.workers: list[Worker] = []

    def start_workers(
        self,
        coordinator_host: str,
        coordinator_port: int,
        state_store: StateStore | None = None,
    ) -> None:
        """Start all workers in the pool.

        Parameters
        ----------
        state_store
            An already-constructed state store to share.  When ``None``
            a new ``RedisStateStore`` is created per worker (legacy
            behaviour).
        """
        for worker_id in self.config.worker_ids:
            w = Worker(worker_id, coordinator_host, coordinator_port)

            if state_store is not None:
                w.connect(state_store)
            else:
                from auralith_pipeline.distributed.state import RedisStateStore

                w.connect(RedisStateStore(host=coordinator_host, port=6379))
            w.start()
            self.workers.append(w)

        logger.info(
            "Started %d workers in pool: %s",
            len(self.workers),
            self.config.name,
        )

    def stop_workers(self) -> None:
        """Stop all workers in the pool."""
        for w in self.workers:
            w.stop()
        logger.info("Stopped workers in pool: %s", self.config.name)
