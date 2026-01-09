"""Command-line interface for Auralith Data Pipeline."""

import logging

import click

from auralith_pipeline.config.pipeline_config import PipelineConfig
from auralith_pipeline.pipeline import Pipeline
from auralith_pipeline.sources.data_sources import DATASET_REGISTRY, create_source
from auralith_pipeline.storage.backends import create_storage_backend
from auralith_pipeline.utils.helpers import format_size, setup_logging

logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def main(verbose: bool):
    """Auralith Data Pipeline - Production-grade data processing for LLMs."""
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level)


@main.command()
def list_datasets():
    """List available datasets."""
    click.echo("\nAvailable Datasets:")
    click.echo("-" * 60)

    for name, info in DATASET_REGISTRY.items():
        desc = info.get("description", "No description")
        click.echo(f"  {name:20} {desc}")

    click.echo()


@main.command()
@click.option("--dataset", "-d", required=True, help="Dataset name from registry")
@click.option("--output", "-o", default="./data/shards", help="Output directory")
@click.option("--max-samples", "-n", type=int, default=None, help="Maximum samples to process")
@click.option("--streaming/--no-streaming", default=True, help="Use streaming mode")
@click.option("--deduplicate/--no-deduplicate", default=True, help="Enable deduplication")
@click.option("--quality-filter/--no-quality-filter", default=True, help="Enable quality filtering")
@click.option("--preset", type=click.Choice(["small", "medium", "production"]), default="medium")
def collect(
    dataset: str,
    output: str,
    max_samples: int,
    streaming: bool,
    deduplicate: bool,
    quality_filter: bool,
    preset: str,
):
    """Collect and process a dataset."""
    click.echo(f"Collecting dataset: {dataset}")

    # Create config from preset
    config = PipelineConfig.from_preset(preset)
    config.output_dir = output
    config.deduplicate = deduplicate
    config.quality_filter = quality_filter
    config.streaming = streaming

    # Create pipeline
    pipeline = Pipeline(config)

    # Add source
    source = create_source(dataset, streaming=streaming, max_samples=max_samples)
    pipeline.add_source(source)

    # Run with progress bar
    with click.progressbar(length=max_samples or 100000, label="Processing") as bar:

        def progress(count):
            bar.update(1000)

        stats = pipeline.run(max_samples=max_samples, progress_callback=progress)

    click.echo(stats.summary())


@main.command()
@click.option("--source", "-s", required=True, help="Local source directory")
@click.option(
    "--dest", "-d", required=True, help="Destination (e.g., hf://org/repo, s3://bucket/prefix)"
)
@click.option("--token", envvar="HF_TOKEN", help="HuggingFace token")
def upload(source: str, dest: str, token: str):
    """Upload shards to storage."""
    click.echo(f"Uploading {source} to {dest}")

    # Parse destination
    if dest.startswith("hf://") or dest.startswith("huggingface://"):
        repo_id = dest.split("://")[1]
        backend = create_storage_backend("huggingface", repo_id=repo_id, token=token)
    elif dest.startswith("s3://"):
        parts = dest[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        backend = create_storage_backend("s3", bucket=bucket, prefix=prefix)
    elif dest.startswith("gs://"):
        parts = dest[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        backend = create_storage_backend("gcs", bucket=bucket, prefix=prefix)
    else:
        click.echo("Error: Unknown destination format. Use hf://, s3://, or gs://")
        return

    result = backend.upload(source, "")

    if result.success:
        click.echo(
            f"Success! Uploaded {result.files_uploaded} files ({format_size(result.bytes_uploaded)})"
        )
        click.echo(f"URL: {result.url}")
    else:
        click.echo(f"Error: {result.error}")


@main.command()
@click.option("--source", "-s", required=True, help="Remote source (e.g., hf://org/repo)")
@click.option("--dest", "-d", required=True, help="Local destination directory")
@click.option("--token", envvar="HF_TOKEN", help="HuggingFace token")
def download(source: str, dest: str, token: str):
    """Download shards from storage."""
    click.echo(f"Downloading {source} to {dest}")

    # Parse source
    if source.startswith("hf://") or source.startswith("huggingface://"):
        repo_id = source.split("://")[1]
        backend = create_storage_backend("huggingface", repo_id=repo_id, token=token)
    elif source.startswith("s3://"):
        parts = source[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        backend = create_storage_backend("s3", bucket=bucket, prefix=prefix)
    elif source.startswith("gs://"):
        parts = source[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        backend = create_storage_backend("gcs", bucket=bucket, prefix=prefix)
    else:
        click.echo("Error: Unknown source format. Use hf://, s3://, or gs://")
        return

    success = backend.download("", dest)

    if success:
        click.echo(f"Success! Downloaded to {dest}")
    else:
        click.echo("Download failed")


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
def run(config_path: str):
    """Run pipeline from a config file."""
    click.echo(f"Loading config from {config_path}")

    config = PipelineConfig.from_yaml(config_path)
    pipeline = Pipeline(config)

    # Sources would need to be configured in the YAML
    # For now, this is a placeholder
    click.echo("Running pipeline...")
    stats = pipeline.run()
    click.echo(stats.summary())


# ============================================================================
# Distributed Processing Commands
# ============================================================================


@main.command()
@click.option("--config", "-c", required=True, help="Configuration file path")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8080, type=int, help="Port to bind to")
def coordinator(config: str, host: str, port: int):
    """Start a coordinator node for distributed processing."""
    from auralith_pipeline.distributed import Coordinator, DistributedConfig

    click.echo("=" * 60)
    click.echo("Starting Auralith Coordinator Node")
    click.echo("=" * 60)

    # Load configuration
    dist_config = DistributedConfig.from_yaml(config)
    dist_config.coordinator.host = host
    dist_config.coordinator.port = port

    click.echo(f"\nCoordinator: {host}:{port}")
    click.echo(
        f"State Store: {dist_config.coordinator.state_store_type} "
        f"at {dist_config.coordinator.state_store_host}:{dist_config.coordinator.state_store_port}"
    )

    # Create and start coordinator
    coord = Coordinator(dist_config.coordinator)

    try:
        coord.start()
        click.echo("\n[OK] Coordinator started successfully")
        click.echo("\nPress Ctrl+C to stop...")

        # Keep running
        import time

        while coord.is_running():
            time.sleep(1)

    except KeyboardInterrupt:
        click.echo("\n\nShutting down coordinator...")
        coord.stop()
        click.echo("[OK] Coordinator stopped")
    except Exception as e:
        click.echo(f"\n[ERROR] {e}", err=True)
        coord.stop()
        raise


@main.command()
@click.option("--config", "-c", required=True, help="Configuration file path")
@click.option("--coordinator", required=True, help="Coordinator address (host:port)")
@click.option("--worker-id", required=True, help="Unique worker ID")
def worker(config: str, coordinator: str, worker_id: str):
    """Start a worker node for distributed processing."""
    from auralith_pipeline.distributed import Worker
    from auralith_pipeline.distributed.state import RedisStateStore

    click.echo("=" * 60)
    click.echo("Starting Auralith Worker Node")
    click.echo("=" * 60)

    # Parse coordinator address
    host, port = coordinator.split(":")

    click.echo(f"\nWorker ID: {worker_id}")
    click.echo(f"Coordinator: {host}:{port}")

    # Create worker
    w = Worker(worker_id, host, int(port))

    # Connect to state store (use Redis by default)
    state_store = RedisStateStore(host=host, port=6379)

    try:
        w.connect(state_store)
        w.start()

        click.echo("\n[OK] Worker started successfully")
        click.echo("[OK] Heartbeat active")
        click.echo("\nWaiting for tasks... (Press Ctrl+C to stop)")

        # Keep running
        import time

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        click.echo("\n\nShutting down worker...")
        w.stop()
        click.echo("[OK] Worker stopped")
    except Exception as e:
        click.echo(f"\n[ERROR] {e}", err=True)
        w.stop()
        raise


@main.command("submit-job")
@click.option("--config", "-c", required=True, help="Configuration file path")
@click.option("--coordinator", required=True, help="Coordinator address (host:port)")
@click.option("--job-name", required=True, help="Job name")
@click.option("--dataset", "-d", required=True, help="Dataset to process")
@click.option("--output-dir", "-o", required=True, help="Output directory")
@click.option("--max-samples", "-n", type=int, help="Maximum samples")
def submit_job(
    config: str,
    coordinator: str,
    job_name: str,
    dataset: str,
    output_dir: str,
    max_samples: int,
):
    """Submit a distributed processing job."""
    from auralith_pipeline.distributed import DistributedPipeline, JobConfig
    from auralith_pipeline.sources.data_sources import create_source

    click.echo("=" * 60)
    click.echo("Submitting Distributed Job")
    click.echo("=" * 60)

    # Parse coordinator address
    host, port = coordinator.split(":")

    click.echo(f"\nJob Name: {job_name}")
    click.echo(f"Dataset: {dataset}")
    click.echo(f"Output: {output_dir}")
    click.echo(f"Coordinator: {host}:{port}")

    # Create job config
    job_config = JobConfig(
        name=job_name,
        coordinator_host=host,
        coordinator_port=int(port),
        output_dir=output_dir,
    )

    # Load pipeline config
    pipeline_config = PipelineConfig.from_yaml(config)

    # Create distributed pipeline
    pipeline = DistributedPipeline(job_config, pipeline_config=pipeline_config)

    # Add source
    source = create_source(dataset, streaming=True, max_samples=max_samples)
    pipeline.add_source(source)

    click.echo("\n[OK] Job configured")
    click.echo("Starting job execution...")

    try:
        # Run the job
        click.echo("\nStarting job execution...")
        stats = pipeline.run(output_path=output_dir, monitor=True)

        click.echo("\n" + "=" * 60)
        click.echo("Job Completed Successfully")
        click.echo("=" * 60)
        click.echo(f"\nSamples: {stats['total_samples']:,}")
        click.echo(f"Shards: {stats.get('shard_count', 'N/A')}")
        click.echo(f"Time: {stats.get('processing_time_seconds', 0):.1f}s")
        click.echo(f"Output: {output_dir}")

    except Exception as e:
        click.echo(f"\n[ERROR] Job failed: {e}", err=True)
        raise


@main.command("job-status")
@click.option("--coordinator", required=True, help="Coordinator address (host:port)")
@click.option("--job-id", required=True, help="Job ID")
def job_status(coordinator: str, job_id: str):
    """Check status of a distributed job."""
    from auralith_pipeline.distributed import DistributedClient

    click.echo(f"Checking status for job: {job_id}")

    try:
        client = DistributedClient(coordinator)

        # Get job info
        job = client.get_job(job_id)

        click.echo("\n" + "=" * 60)
        click.echo(f"Job: {job_id}")
        click.echo("=" * 60)
        click.echo(f"Status: {job['status']}")
        click.echo(f"Created: {job.get('created_at', 'N/A')}")

        # Get metrics
        metrics = client.get_metrics(job_id)
        click.echo(f"\nProgress: {metrics['completed_tasks']}/{metrics['total_tasks']}")
        click.echo(f"Throughput: {metrics['throughput']:.1f}%")
        click.echo(f"Errors: {metrics['failed_tasks']}")

        client.close()

    except Exception as e:
        click.echo(f"\n[ERROR] {e}", err=True)
        raise


@main.command()
@click.option("--coordinator", required=True, help="Coordinator address (host:port)")
def status(coordinator: str):
    """Check status of distributed system."""
    from auralith_pipeline.distributed import DistributedClient

    click.echo("=" * 60)
    click.echo("Distributed System Status")
    click.echo("=" * 60)

    try:
        client = DistributedClient(coordinator)

        # List workers
        workers = client.list_workers()

        click.echo(f"\nCoordinator: {coordinator}")
        click.echo(f"Active Workers: {len(workers)}")

        if workers:
            click.echo("\nWorkers:")
            for worker in workers:
                click.echo(
                    f"  • {worker['id']}: "
                    f"CPU={worker.get('cpu_usage', 0)}% "
                    f"Memory={worker.get('memory_usage', 0)}%"
                )
        else:
            click.echo("\n[WARNING] No active workers")

        client.close()

    except Exception as e:
        click.echo(f"\n[ERROR] {e}", err=True)
        click.echo("\nMake sure the coordinator is running and Redis is accessible.")


@main.command("spark-submit")
@click.option("--input", "-i", required=True, help="Input data path")
@click.option("--output", "-o", required=True, help="Output path for processed data")
@click.option("--dataset-name", "-d", required=True, help="Dataset name")
@click.option("--master", "-m", default="local[*]", help="Spark master URL")
@click.option("--executor-memory", default="4g", help="Executor memory")
@click.option("--driver-memory", default="2g", help="Driver memory")
@click.option("--num-executors", type=int, default=2, help="Number of executors")
@click.option("--tokenizer", default="gpt2", help="Tokenizer name")
@click.option("--max-length", type=int, default=2048, help="Maximum sequence length")
@click.option("--deduplicate/--no-deduplicate", default=True, help="Enable deduplication")
@click.option("--quality-filter/--no-quality-filter", default=True, help="Enable quality filtering")
@click.option("--remove-pii/--no-remove-pii", default=True, help="Remove PII")
@click.option("--num-partitions", type=int, default=100, help="Number of output partitions")
def spark_submit(
    input: str,
    output: str,
    dataset_name: str,
    master: str,
    executor_memory: str,
    driver_memory: str,
    num_executors: int,
    tokenizer: str,
    max_length: int,
    deduplicate: bool,
    quality_filter: bool,
    remove_pii: bool,
    num_partitions: int,
):
    """Submit a large-scale processing job to Apache Spark."""
    from auralith_pipeline.spark import SparkConfig, SparkJobConfig, SparkPipelineRunner

    click.echo("=" * 60)
    click.echo("Spark Pipeline Job")
    click.echo("=" * 60)
    click.echo(f"Input: {input}")
    click.echo(f"Output: {output}")
    click.echo(f"Dataset: {dataset_name}")
    click.echo(f"Master: {master}")
    click.echo("-" * 60)

    # Create Spark configuration
    spark_config = SparkConfig(
        app_name=f"Auralith-{dataset_name}",
        master=master,
        executor_memory=executor_memory,
        driver_memory=driver_memory,
        num_executors=num_executors,
    )

    # Create job configuration
    job_config = SparkJobConfig(
        input_path=input,
        output_path=output,
        dataset_name=dataset_name,
        tokenizer_name=tokenizer,
        max_length=max_length,
        deduplicate=deduplicate,
        quality_filter=quality_filter,
        remove_pii=remove_pii,
        num_partitions=num_partitions,
    )

    # Run the job
    runner = SparkPipelineRunner(spark_config)

    try:
        click.echo("\n[OK] Starting Spark job...")
        stats = runner.run(job_config)

        click.echo("\n[OK] Job completed successfully!")
        click.echo("-" * 60)
        click.echo(f"Initial samples: {stats['initial_samples']:,}")
        click.echo(f"Final samples: {stats['final_samples']:,}")
        click.echo(f"Filtered: {stats['filtered_samples']:,}")
        click.echo(f"Output: {stats['output_path']}")

    except Exception as e:
        click.echo(f"\n[ERROR] Job failed: {e}", err=True)
        raise
    finally:
        runner.stop()


if __name__ == "__main__":
    main()
