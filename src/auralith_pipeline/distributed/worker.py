"""Worker node for distributed processing."""

import logging
import threading
import time
from datetime import datetime
from typing import Any

from auralith_pipeline.distributed.config import WorkerPoolConfig
from auralith_pipeline.distributed.state import RedisStateStore, StateStore

logger = logging.getLogger(__name__)


class Worker:
    """Worker node that processes tasks."""

    def __init__(self, worker_id: str, coordinator_host: str, coordinator_port: int):
        self.worker_id = worker_id
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.state_store: StateStore | None = None
        self._running = False
        self._processing_thread = None
        self._heartbeat_thread = None

    def connect(self, state_store: StateStore):
        """Connect to state store."""
        self.state_store = state_store
        self.state_store.connect()
        logger.info(
            f"Worker {self.worker_id} connected to coordinator at "
            f"{self.coordinator_host}:{self.coordinator_port}"
        )

    def start(self):
        """Start processing tasks."""
        if not self.state_store:
            raise RuntimeError("Worker not connected to state store")

        self._running = True

        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(target=self._send_heartbeats)
        self._heartbeat_thread.daemon = True
        self._heartbeat_thread.start()

        # Start processing thread
        self._processing_thread = threading.Thread(target=self._process_tasks)
        self._processing_thread.daemon = True
        self._processing_thread.start()

        logger.info(f"Worker {self.worker_id} started")

    def stop(self):
        """Stop the worker."""
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
        if self._processing_thread:
            self._processing_thread.join(timeout=5)
        logger.info(f"Worker {self.worker_id} stopped")

    def _send_heartbeats(self):
        """Send periodic heartbeats to coordinator."""
        while self._running:
            try:
                heartbeat_data = {
                    "worker_id": self.worker_id,
                    "timestamp": datetime.now().isoformat(),
                    "status": "alive",
                }
                self.state_store.set(f"heartbeat:{self.worker_id}", heartbeat_data, ttl=30)
                time.sleep(10)  # Send heartbeat every 10 seconds
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")

    def _process_tasks(self):
        """Process tasks from the queue."""
        queue_name = f"tasks:{self.worker_id}"

        while self._running:
            try:
                # Get task from queue (with timeout)
                task = self.state_store.pop_queue(queue_name, timeout=5)
                if task:
                    self._execute_task(task)
            except Exception as e:
                logger.error(f"Error processing task: {e}")

    def _execute_task(self, task: dict[str, Any]):
        """Execute a single task."""
        task_id = task.get("id", "unknown")
        logger.info(f"Executing task {task_id}")

        try:
            # Update task status
            task["status"] = "running"
            task["started_at"] = datetime.now().isoformat()
            self.state_store.set(f"task:{task_id}", task)

            # Execute task logic
            result = self._run_pipeline_task(task)

            # Mark task as completed
            task["status"] = "completed"
            task["completed_at"] = datetime.now().isoformat()
            task["result"] = result
            self.state_store.set(f"task:{task_id}", task)

            logger.info(f"Task {task_id} completed successfully")

        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            task["status"] = "failed"
            task["error"] = str(e)
            task["failed_at"] = datetime.now().isoformat()
            self.state_store.set(f"task:{task_id}", task)

    def _run_pipeline_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Run a pipeline task."""
        # This is a simplified implementation
        # Real implementation would deserialize pipeline config and run it
        logger.debug(f"Running pipeline task: {task}")
        time.sleep(1)  # Simulate work
        return {"samples_processed": 100, "status": "success"}


class WorkerPool:
    """Pool of workers for a specific processing stage."""

    def __init__(self, config: WorkerPoolConfig):
        self.config = config
        self.workers: list[Worker] = []

    def start_workers(self, coordinator_host: str, coordinator_port: int):
        """Start all workers in the pool."""
        for worker_id in self.config.worker_ids:
            worker = Worker(worker_id, coordinator_host, coordinator_port)
            # Create state store for worker
            state_store = RedisStateStore(host=coordinator_host, port=6379)  # Simplified
            worker.connect(state_store)
            worker.start()
            self.workers.append(worker)

        logger.info(f"Started {len(self.workers)} workers in pool: {self.config.name}")

    def stop_workers(self):
        """Stop all workers in the pool."""
        for worker in self.workers:
            worker.stop()
        logger.info(f"Stopped workers in pool: {self.config.name}")
