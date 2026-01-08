"""Load balancing and task distribution strategies."""

from enum import Enum
from abc import ABC, abstractmethod
from typing import Any
import logging

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for task distribution."""

    ROUND_ROBIN = "round_robin"
    LEAST_BUSY = "least_busy"
    DYNAMIC = "dynamic"
    RANDOM = "random"


class TaskDistributionStrategy(ABC):
    """Abstract base class for task distribution strategies."""

    @abstractmethod
    def assign_task(self, task: Any, available_workers: list[str]) -> str:
        """Assign a task to a worker."""
        pass


class RoundRobinDistribution(TaskDistributionStrategy):
    """Round-robin task distribution."""

    def __init__(self):
        self.current_index = 0

    def assign_task(self, task: Any, available_workers: list[str]) -> str:
        """Assign task using round-robin."""
        if not available_workers:
            raise ValueError("No available workers")

        worker = available_workers[self.current_index % len(available_workers)]
        self.current_index += 1
        return worker


class LeastBusyDistribution(TaskDistributionStrategy):
    """Assign tasks to least busy worker."""

    def __init__(self, state_store):
        self.state_store = state_store

    def assign_task(self, task: Any, available_workers: list[str]) -> str:
        """Assign task to least busy worker."""
        if not available_workers:
            raise ValueError("No available workers")

        # Get queue lengths for each worker
        worker_loads = {}
        for worker in available_workers:
            queue_name = f"tasks:{worker}"
            worker_loads[worker] = self.state_store.queue_length(queue_name)

        # Return worker with minimum load
        return min(worker_loads, key=worker_loads.get)


class DynamicDistribution(TaskDistributionStrategy):
    """Dynamic task distribution based on worker metrics."""

    def __init__(self, state_store):
        self.state_store = state_store

    def assign_task(self, task: Any, available_workers: list[str]) -> str:
        """Assign task dynamically based on worker state."""
        if not available_workers:
            raise ValueError("No available workers")

        # Get worker metrics
        worker_scores = {}
        for worker in available_workers:
            metrics_key = f"metrics:{worker}"
            metrics = self.state_store.get(metrics_key) or {}

            # Calculate score (lower is better)
            queue_length = self.state_store.queue_length(f"tasks:{worker}")
            cpu_usage = metrics.get("cpu_usage", 50)
            memory_usage = metrics.get("memory_usage", 50)

            # Weighted score
            score = (queue_length * 0.4) + (cpu_usage * 0.3) + (memory_usage * 0.3)
            worker_scores[worker] = score

        # Return worker with minimum score
        return min(worker_scores, key=worker_scores.get)
