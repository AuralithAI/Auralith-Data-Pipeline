"""Ray-based distributed pipeline runner.

Replaces Redis orchestration with Ray for true horizontal scaling
on K8s / DGX Cloud.
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def _check_ray() -> bool:
    """Check if Ray is available."""
    try:
        import ray  # noqa: F401

        return True
    except ImportError:
        return False


class RayPipelineRunner:
    """Distribute pipeline workloads across a Ray cluster.

    Usage:
        runner = RayPipelineRunner(address="auto")
        runner.connect()
        stats = runner.run_distributed(
            datasets=["wikipedia", "c4"],
            config=pipeline_config,
        )
    """

    def __init__(
        self,
        address: str = "auto",
        num_cpus: int | None = None,
        num_gpus: int | None = None,
        runtime_env: dict[str, Any] | None = None,
    ):
        """Initialize Ray runner.

        Args:
            address: Ray cluster address ('auto', 'local', or 'ray://...')
            num_cpus: CPUs per worker
            num_gpus: GPUs per worker (for VQ encoding)
            runtime_env: Ray runtime environment config
        """
        self.address = address
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.runtime_env = runtime_env or {}
        self._connected = False

    def connect(self) -> None:
        """Connect to (or start) the Ray cluster."""
        if not _check_ray():
            raise ImportError("Ray is not installed. Install with: pip install 'ray[default]'")

        import ray

        if not ray.is_initialized():
            ray.init(
                address=self.address,
                num_cpus=self.num_cpus,
                num_gpus=self.num_gpus,
                runtime_env=self.runtime_env,
                ignore_reinit_error=True,
            )
            logger.info(f"Ray connected: {ray.cluster_resources()}")

        self._connected = True

    def run_distributed(
        self,
        datasets: list[str],
        config: Any,  # PipelineConfig
        max_samples_per_dataset: int | None = None,
    ) -> dict[str, Any]:
        """Run pipeline across datasets in parallel using Ray tasks.

        Each dataset is processed as an independent Ray remote task.

        Returns:
            Aggregated stats dict.
        """
        if not self._connected:
            self.connect()

        import ray

        @ray.remote
        def process_dataset(dataset_name: str, config_dict: dict, max_samples: int | None) -> dict:
            """Remote task: process one dataset."""
            from auralith_pipeline.config.pipeline_config import PipelineConfig
            from auralith_pipeline.pipeline import Pipeline
            from auralith_pipeline.sources.data_sources import create_source

            cfg = PipelineConfig.from_dict(config_dict)
            cfg.output_dir = f"{cfg.output_dir}/{dataset_name}"

            pipeline = Pipeline(cfg)
            source = create_source(dataset_name, streaming=True, max_samples=max_samples)
            pipeline.add_source(source)

            stats = pipeline.run(max_samples=max_samples)
            return {
                "dataset": dataset_name,
                "total_samples": stats.total_samples,
                "samples_tokenized": stats.samples_tokenized,
                "total_tokens": stats.total_tokens,
                "num_shards": stats.num_shards,
                "total_size_bytes": stats.total_size_bytes,
                "elapsed": stats.elapsed_time_seconds,
            }

        # Submit all dataset tasks in parallel
        start = time.time()
        config_dict = config.to_dict()
        futures = [
            process_dataset.remote(ds, config_dict, max_samples_per_dataset) for ds in datasets
        ]

        # Gather results
        results = ray.get(futures)

        total_stats = {
            "datasets": len(results),
            "total_samples": sum(r["total_samples"] for r in results),
            "total_tokens": sum(r["total_tokens"] for r in results),
            "total_shards": sum(r["num_shards"] for r in results),
            "total_size_bytes": sum(r["total_size_bytes"] for r in results),
            "elapsed_seconds": time.time() - start,
            "per_dataset": results,
        }

        logger.info(
            f"Ray distributed run complete: {total_stats['total_samples']:,} samples, "
            f"{total_stats['total_shards']} shards in {total_stats['elapsed_seconds']:.1f}s"
        )

        return total_stats

    def shutdown(self) -> None:
        """Shutdown Ray connection."""
        if _check_ray():
            import ray

            if ray.is_initialized():
                ray.shutdown()
                logger.info("Ray shutdown complete")
        self._connected = False
