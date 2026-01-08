"""
Example of using distributed processing.

This example shows how to set up and run a distributed pipeline.
"""

from auralith_pipeline.distributed import (
    DistributedPipeline,
    JobConfig,
    WorkerPoolConfig,
    Coordinator,
    CoordinatorConfig,
)
from auralith_pipeline.sources import HuggingFaceSource
from auralith_pipeline.config import PipelineConfig


def example_distributed_setup():
    """Example of setting up distributed processing."""

    # 1. Start coordinator (run this on coordinator node)
    print("=== Starting Coordinator ===")
    coordinator_config = CoordinatorConfig(
        host="localhost",
        port=8080,
        state_store_host="localhost",
        state_store_port=6379,
    )

    coordinator = Coordinator(coordinator_config)
    # coordinator.start()  # Uncomment to actually start

    # 2. Configure job
    print("\n=== Configuring Distributed Job ===")
    job_config = JobConfig(
        name="wikipedia-processing",
        coordinator_host="localhost",
        coordinator_port=8080,
        num_workers=3,
        timeout=86400,
    )

    # 3. Configure worker pools
    print("\n=== Configuring Worker Pools ===")
    collection_pool = WorkerPoolConfig(
        name="collection",
        worker_ids=["worker-1", "worker-2"],
        batch_size=1000,
    )

    processing_pool = WorkerPoolConfig(
        name="processing",
        worker_ids=["worker-3", "worker-4"],
        batch_size=500,
    )

    # 4. Create distributed pipeline
    print("\n=== Creating Distributed Pipeline ===")
    pipeline_config = PipelineConfig.from_preset("production")
    pipeline = DistributedPipeline(
        job_config,
        pipeline_config=pipeline_config,
        worker_pools=[collection_pool, processing_pool],
    )

    # 5. Add sources
    print("\n=== Adding Data Sources ===")
    pipeline.add_source(HuggingFaceSource("wikipedia", split="train"))

    # 6. Configure pipeline
    print("\n=== Configuring Pipeline Settings ===")
    pipeline.configure_batching(
        collection_batch_size=1000,
        preprocessing_batch_size=500,
        tokenization_batch_size=100,
    )

    # 7. Run pipeline
    print("\n=== Running Distributed Pipeline ===")
    print("Note: This requires Redis and workers to be running")
    print("To actually run, uncomment the following line:")
    # stats = pipeline.run(
    #     output_path="s3://auralith-datasets/wikipedia/shards",
    #     monitor=True,
    #     enable_checkpointing=True,
    # )
    #
    # print(f"\n=== Results ===")
    # print(f"Samples processed: {stats['total_samples']:,}")
    # print(f"Shards created: {stats['shard_count']}")
    # print(f"Processing time: {stats['processing_time_seconds']:.1f}s")

    return pipeline


def example_monitoring():
    """Example of monitoring a distributed job."""
    from auralith_pipeline.distributed import DistributedClient

    print("\n=== Monitoring Distributed Job ===")

    # Connect to coordinator
    client = DistributedClient("localhost:8080")

    # Get job status
    job = client.get_job("wikipedia-processing")
    print(f"Job Status: {job['status']}")

    # List workers
    workers = client.list_workers()
    print(f"\nActive Workers: {len(workers)}")
    for worker in workers:
        print(f"  - {worker['id']}: CPU={worker['cpu_usage']}%")

    # Get metrics
    metrics = client.get_metrics("wikipedia-processing")
    print(f"\nMetrics:")
    print(f"  Completed: {metrics['completed_tasks']}/{metrics['total_tasks']}")
    print(f"  Throughput: {metrics['throughput']:.1f}%")

    client.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Auralith Data Pipeline - Distributed Processing Example")
    print("=" * 60)

    # Run setup example
    pipeline = example_distributed_setup()

    print("\n" + "=" * 60)
    print("\nTo actually run distributed processing:")
    print("1. Start Redis: docker run -p 6379:6379 redis")
    print("2. Start coordinator: auralith-pipeline coordinator")
    print("3. Start workers: auralith-pipeline worker --worker-id worker-1")
    print("4. Submit job using the pipeline.run() method")
    print("\nFor monitoring, run:")
    print("  example_monitoring()")
    print("=" * 60)
