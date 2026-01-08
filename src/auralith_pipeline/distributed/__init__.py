"""Distributed processing module."""

from auralith_pipeline.distributed.client import DistributedClient
from auralith_pipeline.distributed.config import (
    CoordinatorConfig,
    DistributedConfig,
    JobConfig,
    WorkerPoolConfig,
)
from auralith_pipeline.distributed.coordinator import Coordinator, JobManager
from auralith_pipeline.distributed.pipeline import DistributedPipeline
from auralith_pipeline.distributed.state import RedisStateStore, StateStore
from auralith_pipeline.distributed.strategies import (
    LoadBalancingStrategy,
    TaskDistributionStrategy,
)
from auralith_pipeline.distributed.worker import Worker, WorkerPool

__all__ = [
    "Coordinator",
    "JobManager",
    "Worker",
    "WorkerPool",
    "DistributedPipeline",
    "JobConfig",
    "WorkerPoolConfig",
    "CoordinatorConfig",
    "DistributedConfig",
    "DistributedClient",
    "StateStore",
    "RedisStateStore",
    "LoadBalancingStrategy",
    "TaskDistributionStrategy",
]
