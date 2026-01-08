"""Command-line interface for Auralith Data Pipeline."""

import click
import logging
from pathlib import Path

from auralith_pipeline.config.pipeline_config import PipelineConfig
from auralith_pipeline.sources.data_sources import DATASET_REGISTRY, create_source
from auralith_pipeline.pipeline import Pipeline
from auralith_pipeline.storage.backends import create_storage_backend
from auralith_pipeline.utils.helpers import setup_logging, format_size

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
@click.option("--dest", "-d", required=True, help="Destination (e.g., hf://org/repo, s3://bucket/prefix)")
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
        click.echo(f"Success! Uploaded {result.files_uploaded} files ({format_size(result.bytes_uploaded)})")
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


if __name__ == "__main__":
    main()
