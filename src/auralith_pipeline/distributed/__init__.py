"""Distributed processing module."""

from auralith_pipeline.distributed.coordinator import Coordinator, JobManager
from auralith_pipeline.distributed.worker import Worker, WorkerPool
from auralith_pipeline.distributed.pipeline import DistributedPipeline
from auralith_pipeline.distributed.config import (
    JobConfig,
    WorkerPoolConfig,
    CoordinatorConfig,
)
from auralith_pipeline.distributed.client import DistributedClient
from auralith_pipeline.distributed.state import StateStore, RedisStateStore
from auralith_pipeline.distributed.strategies import (
    LoadBalancingStrategy,
    TaskDistributionStrategy,
)

__all__ = [
    "Coordinator",
    "JobManager",
    "Worker",
    "WorkerPool",
    "DistributedPipeline",
    "JobConfig",
    "WorkerPoolConfig",
    "CoordinatorConfig",
    "DistributedClient",
    "StateStore",
    "RedisStateStore",
    "LoadBalancingStrategy",
    "TaskDistributionStrategy",
]
