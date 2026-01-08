"""Coordinator node for distributed processing."""

import logging
import threading
import time
from datetime import datetime
from typing import Any

from auralith_pipeline.distributed.config import CoordinatorConfig
from auralith_pipeline.distributed.state import RedisStateStore, StateStore
from auralith_pipeline.distributed.strategies import (
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
        self.jobs = {}
        self._running = False
        self._monitor_thread = None

    def create_job(self, job_id: str, job_config: dict[str, Any]) -> dict:
        """Create a new distributed job."""
        job = {
            "id": job_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "config": job_config,
            "tasks": [],
            "completed_tasks": 0,
            "failed_tasks": 0,
        }
        self.jobs[job_id] = job
        self.state_store.set(f"job:{job_id}", job)
        logger.info(f"Created job: {job_id}")
        return job

    def submit_task(self, job_id: str, task: dict[str, Any]):
        """Submit a task for a job."""
        available_workers = self.state_store.get_all_workers()
        if not available_workers:
            logger.warning("No available workers, queuing task")
            self.state_store.push_queue("pending_tasks", task)
            return

        # Assign task to worker
        worker_id = self.distribution_strategy.assign_task(task, available_workers)
        task["worker_id"] = worker_id
        task["status"] = "assigned"

        # Push to worker queue
        self.state_store.push_queue(f"tasks:{worker_id}", task)
        logger.debug(f"Assigned task to worker: {worker_id}")

    def start(self):
        """Start the job manager."""
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_workers)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        logger.info("Job manager started")

    def stop(self):
        """Stop the job manager."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("Job manager stopped")

    def _monitor_workers(self):
        """Monitor worker heartbeats."""
        while self._running:
            workers = self.state_store.get_all_workers()
            for worker_id in workers:
                if not self.state_store.check_heartbeat(worker_id):
                    logger.warning(f"Worker {worker_id} heartbeat failed")
                    self._handle_worker_failure(worker_id)
            time.sleep(self.config.heartbeat_interval)

    def _handle_worker_failure(self, worker_id: str):
        """Handle worker failure by reassigning tasks."""
        logger.error(f"Handling failure for worker: {worker_id}")
        # Reassign pending tasks from failed worker
        # Implementation would move tasks back to pending queue


class Coordinator:
    """Coordinator node that manages distributed processing."""

    def __init__(self, config: CoordinatorConfig):
        self.config = config
        self.state_store = self._create_state_store()
        self.job_manager = JobManager(config, self.state_store)
        self._running = False

    def _create_state_store(self) -> StateStore:
        """Create state store based on configuration."""
        if self.config.state_store_type == "redis":
            return RedisStateStore(
                host=self.config.state_store_host,
                port=self.config.state_store_port,
                db=self.config.state_store_db,
            )
        else:
            raise ValueError(f"Unsupported state store type: {self.config.state_store_type}")

    def start(self):
        """Start the coordinator."""
        logger.info("Starting coordinator...")
        self.state_store.connect()
        self.job_manager.start()
        self._running = True
        logger.info(f"Coordinator running at {self.config.host}:{self.config.port}")

    def stop(self):
        """Stop the coordinator."""
        logger.info("Stopping coordinator...")
        self._running = False
        self.job_manager.stop()
        self.state_store.disconnect()
        logger.info("Coordinator stopped")

    def is_running(self) -> bool:
        """Check if coordinator is running."""
        return self._running
