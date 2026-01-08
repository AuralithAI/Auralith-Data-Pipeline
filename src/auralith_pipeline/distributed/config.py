"""Configuration for distributed processing."""

from dataclasses import dataclass, field


@dataclass
class CoordinatorConfig:
    """Configuration for coordinator node."""

    host: str = "localhost"
    port: int = 8080

    # State store configuration
    state_store_type: str = "redis"  # or "etcd"
    state_store_host: str = "localhost"
    state_store_port: int = 6379
    state_store_db: int = 0

    # Heartbeat settings
    heartbeat_interval: int = 10  # seconds
    heartbeat_timeout: int = 30  # seconds

    # Task configuration
    max_retries: int = 3
    retry_backoff: int = 5  # seconds
    task_timeout: int = 3600  # seconds

    # Checkpointing
    enable_checkpointing: bool = True
    checkpoint_interval: int = 300  # seconds
    checkpoint_dir: str = "./checkpoints"
    keep_last_n_checkpoints: int = 5


@dataclass
class WorkerPoolConfig:
    """Configuration for worker pool."""

    name: str
    worker_ids: list[str] = field(default_factory=list)
    batch_size: int = 100
    num_processes: int = 4

    # Resource limits
    cpu_cores: int | None = None
    memory_gb: int | None = None
    gpu_count: int = 0

    # Cache configuration
    cache_dir: str = "/tmp/auralith_cache"
    max_cache_size_gb: int = 50
    cleanup_on_exit: bool = True


@dataclass
class JobConfig:
    """Configuration for distributed job."""

    name: str
    coordinator_host: str = "localhost"
    coordinator_port: int = 8080

    # Job parameters
    num_workers: int = 1
    timeout: int = 86400  # 24 hours

    # Monitoring
    enable_monitoring: bool = True
    monitoring_port: int = 9090

    # Retry configuration
    max_task_retries: int = 3
    retry_backoff_multiplier: int = 2
    max_retry_backoff: int = 300  # seconds

    # Output configuration
    output_dir: str = "./output"
    checkpoint_enabled: bool = True


@dataclass
class DistributedConfig:
    """Complete distributed processing configuration."""

    coordinator: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    workers: list[WorkerPoolConfig] = field(default_factory=list)
    job: JobConfig = field(default_factory=JobConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "DistributedConfig":
        """Load configuration from YAML file."""
        from pathlib import Path

        import yaml

        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        coordinator = CoordinatorConfig(**data.get("coordinator", {}))
        workers = [WorkerPoolConfig(**w) for w in data.get("workers", [])]
        job = JobConfig(**data.get("job", {}))

        return cls(coordinator=coordinator, workers=workers, job=job)

    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        from pathlib import Path

        import yaml

        data = {
            "coordinator": {
                "host": self.coordinator.host,
                "port": self.coordinator.port,
                "state_store_type": self.coordinator.state_store_type,
                "state_store_host": self.coordinator.state_store_host,
                "state_store_port": self.coordinator.state_store_port,
                "heartbeat_interval": self.coordinator.heartbeat_interval,
                "heartbeat_timeout": self.coordinator.heartbeat_timeout,
            },
            "workers": [
                {
                    "name": w.name,
                    "worker_ids": w.worker_ids,
                    "batch_size": w.batch_size,
                    "num_processes": w.num_processes,
                }
                for w in self.workers
            ],
            "job": {
                "name": self.job.name,
                "num_workers": self.job.num_workers,
                "timeout": self.job.timeout,
            },
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
