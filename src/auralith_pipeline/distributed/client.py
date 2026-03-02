"""Client for interacting with distributed pipeline."""

import logging
from typing import Any

from auralith_pipeline.distributed.state import RedisStateStore

logger = logging.getLogger(__name__)


class DistributedClient:
    """Client for monitoring and controlling distributed pipelines."""

    def __init__(self, coordinator_address: str):
        """Initialize client.

        Args:
            coordinator_address: Format "host:port"
        """
        host, port = coordinator_address.split(":")
        self.coordinator_host = host
        self.coordinator_port = int(port)
        self.state_store = RedisStateStore(host=host, port=6379)
        self.state_store.connect()

    def get_job(self, job_id: str) -> dict[str, Any]:
        """Get job status."""
        job = self.state_store.get(f"job:{job_id}")
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        return job

    def list_workers(self) -> list[dict[str, Any]]:
        """List all active workers."""
        worker_ids = self.state_store.get_all_workers()
        workers = []
        for worker_id in worker_ids:
            metrics = self.state_store.get(f"metrics:{worker_id}") or {}
            workers.append(
                {
                    "id": worker_id,
                    "status": "active",
                    "cpu_usage": metrics.get("cpu_usage", 0),
                    "memory_usage": metrics.get("memory_usage", 0),
                }
            )
        return workers

    def get_metrics(self, job_id: str) -> dict[str, Any]:
        """Get job metrics."""
        job = self.get_job(job_id)
        total_tasks = job.get("total_tasks", 0)
        completed = job.get("completed_tasks", 0)
        failed = job.get("failed_tasks", 0)

        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed,
            "failed_tasks": failed,
            "throughput": completed / max(1, total_tasks) * 100,
            "error_rate": failed / max(1, total_tasks),
        }

    def cancel_job(self, job_id: str):
        """Cancel a running job."""
        job = self.get_job(job_id)
        job["status"] = "cancelled"
        self.state_store.set(f"job:{job_id}", job)
        logger.info(f"Cancelled job: {job_id}")

    def close(self):
        """Close client connection."""
        self.state_store.disconnect()
