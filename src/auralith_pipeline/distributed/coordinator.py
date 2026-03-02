"""Coordinator node for distributed processing."""

import logging
import threading
import time
import uuid
from datetime import datetime
from typing import Any

from auralith_pipeline.distributed.config import CoordinatorConfig
from auralith_pipeline.distributed.state import StateStore
from auralith_pipeline.distributed.strategies import (
    LeastBusyDistribution,
    RoundRobinDistribution,
    TaskDistributionStrategy,
)

logger = logging.getLogger(__name__)


class JobManager:
    """Manages distributed jobs and task scheduling."""

    def __init__(self, config: CoordinatorConfig, state_store: StateStore):
        self.config = config
        self.state_store = state_store
        self.distribution_strategy: TaskDistributionStrategy = RoundRobinDistribution()
        self.jobs: dict[str, dict[str, Any]] = {}
        self._running = False
        self._monitor_thread: threading.Thread | None = None
        self._requeue_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    def create_job(self, job_id: str, job_config: dict[str, Any]) -> dict[str, Any]:
        """Create a new distributed job."""
        job: dict[str, Any] = {
            "id": job_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "config": job_config,
            "tasks": [],
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "retried_tasks": 0,
        }
        self.jobs[job_id] = job
        self.state_store.set(f"job:{job_id}", job)
        logger.info("Created job: %s", job_id)
        return job

    def update_job_status(self, job_id: str, status: str) -> None:
        """Update the top-level status of a job."""
        job = self.jobs.get(job_id) or self.state_store.get(f"job:{job_id}") or {}
        job["status"] = status
        self.jobs[job_id] = job
        self.state_store.set(f"job:{job_id}", job)

    def record_task_result(self, job_id: str, task_id: str, *, success: bool) -> None:
        """Increment completed/failed counter for a job."""
        job = self.jobs.get(job_id) or self.state_store.get(f"job:{job_id}")
        if not job:
            return
        if success:
            job["completed_tasks"] = job.get("completed_tasks", 0) + 1
        else:
            job["failed_tasks"] = job.get("failed_tasks", 0) + 1
        self.jobs[job_id] = job
        self.state_store.set(f"job:{job_id}", job)

    # ------------------------------------------------------------------
    # Task submission
    # ------------------------------------------------------------------

    def submit_task(self, job_id: str, task: dict[str, Any]) -> None:
        """Submit a task to an available worker (or pending queue)."""
        task.setdefault("id", str(uuid.uuid4()))
        task["job_id"] = job_id
        task.setdefault("retries", 0)

        available_workers = self.state_store.get_all_workers()
        if not available_workers:
            logger.warning("No workers available — queuing task %s", task["id"])
            self.state_store.push_queue("pending_tasks", task)
            return

        worker_id = self.distribution_strategy.assign_task(task, available_workers)
        task["worker_id"] = worker_id
        task["status"] = "assigned"
        task["assigned_at"] = datetime.now().isoformat()

        self.state_store.push_queue(f"tasks:{worker_id}", task)
        self.state_store.set(f"task:{task['id']}", task)
        logger.debug("Assigned task %s → worker %s", task["id"], worker_id)

    def submit_tasks(self, job_id: str, tasks: list[dict[str, Any]]) -> None:
        """Submit a batch of tasks, updating the job's total counter."""
        job = self.jobs.get(job_id) or self.state_store.get(f"job:{job_id}")
        if job:
            job["total_tasks"] = job.get("total_tasks", 0) + len(tasks)
            job["status"] = "running"
            self.jobs[job_id] = job
            self.state_store.set(f"job:{job_id}", job)

        for task in tasks:
            self.submit_task(job_id, task)

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background monitor and requeue threads."""
        self._running = True

        self._monitor_thread = threading.Thread(target=self._monitor_workers, daemon=True)
        self._monitor_thread.start()

        self._requeue_thread = threading.Thread(target=self._requeue_pending_tasks, daemon=True)
        self._requeue_thread.start()

        logger.info("Job manager started")

    def stop(self) -> None:
        """Stop the job manager."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        if self._requeue_thread:
            self._requeue_thread.join(timeout=5)
        logger.info("Job manager stopped")

    # ------------------------------------------------------------------
    # Worker monitoring
    # ------------------------------------------------------------------

    def _monitor_workers(self) -> None:
        """Monitor heartbeats and handle worker failures."""
        while self._running:
            try:
                workers = self.state_store.get_all_workers()
                for worker_id in workers:
                    if not self.state_store.check_heartbeat(worker_id):
                        logger.warning("Worker %s heartbeat lost", worker_id)
                        self._handle_worker_failure(worker_id)
            except Exception:
                logger.exception("Error in worker monitor loop")
            time.sleep(self.config.heartbeat_interval)

    def _handle_worker_failure(self, worker_id: str) -> None:
        """Drain a dead worker's task queue back to ``pending_tasks``."""
        logger.error("Handling failure for worker: %s", worker_id)
        queue_name = f"tasks:{worker_id}"
        requeued = 0
        while True:
            task = self.state_store.pop_queue(queue_name, timeout=0)
            if task is None:
                break
            task["status"] = "pending"
            task["worker_id"] = None
            retries = task.get("retries", 0) + 1
            task["retries"] = retries

            if retries > self.config.max_retries:
                logger.error(
                    "Task %s exceeded max retries (%d) — marking failed",
                    task.get("id"),
                    self.config.max_retries,
                )
                task["status"] = "failed"
                self.state_store.set(f"task:{task['id']}", task)
                job_id = task.get("job_id")
                if job_id:
                    self.record_task_result(job_id, task["id"], success=False)
                continue

            self.state_store.push_queue("pending_tasks", task)
            requeued += 1

        # Remove the dead worker's heartbeat
        self.state_store.delete(f"heartbeat:{worker_id}")
        logger.info("Requeued %d tasks from dead worker %s", requeued, worker_id)

    def _requeue_pending_tasks(self) -> None:
        """Assign pending tasks to newly available workers."""
        while self._running:
            try:
                workers = self.state_store.get_all_workers()
                if workers:
                    task = self.state_store.pop_queue("pending_tasks", timeout=0)
                    if task:
                        job_id = task.get("job_id", "unknown")
                        self.submit_task(job_id, task)
            except Exception:
                logger.exception("Error in requeue loop")
            time.sleep(self.config.retry_backoff)

    # ------------------------------------------------------------------
    # Strategy helpers
    # ------------------------------------------------------------------

    def set_strategy(self, strategy: str) -> None:
        """Switch the task distribution strategy at runtime."""
        if strategy == "round_robin":
            self.distribution_strategy = RoundRobinDistribution()
        elif strategy == "least_busy":
            self.distribution_strategy = LeastBusyDistribution(self.state_store)
        else:
            logger.warning("Unknown strategy %s — keeping current", strategy)


class Coordinator:
    """Coordinator node that manages distributed processing."""

    def __init__(self, config: CoordinatorConfig):
        self.config = config
        self.state_store = self._create_state_store()
        self.job_manager = JobManager(config, self.state_store)
        self._running = False

    def _create_state_store(self) -> StateStore:
        """Create state store from configuration."""
        if self.config.state_store_type == "redis":
            from auralith_pipeline.distributed.state import RedisStateStore

            return RedisStateStore(
                host=self.config.state_store_host,
                port=self.config.state_store_port,
                db=self.config.state_store_db,
                password=self.config.state_store_password,
            )
        if self.config.state_store_type == "memory":
            from auralith_pipeline.distributed.state import InMemoryStateStore

            return InMemoryStateStore()
        raise ValueError(f"Unsupported state store type: {self.config.state_store_type}")

    def start(self) -> None:
        """Start the coordinator."""
        logger.info("Starting coordinator …")
        self.state_store.connect()
        self.job_manager.start()
        self._running = True
        logger.info("Coordinator running on %s:%s", self.config.host, self.config.port)

    def stop(self) -> None:
        """Stop the coordinator."""
        logger.info("Stopping coordinator …")
        self._running = False
        self.job_manager.stop()
        self.state_store.disconnect()
        logger.info("Coordinator stopped")

    def is_running(self) -> bool:
        """Check if the coordinator is running."""
        return self._running
