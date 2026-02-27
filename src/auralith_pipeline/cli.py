"""Command-line interface for Auralith Data Pipeline."""

import logging
from pathlib import Path

import click

from auralith_pipeline.config.pipeline_config import PipelineConfig
from auralith_pipeline.pipeline import Pipeline
from auralith_pipeline.sources.data_sources import DATASET_REGISTRY, create_source
from auralith_pipeline.storage.backends import create_storage_backend
from auralith_pipeline.utils.helpers import format_size, setup_logging

logger = logging.getLogger(__name__)

_NPY_GLOB = "*.npy"


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

    # Sources are configured in the YAML file
    click.echo("Running pipeline...")
    stats = pipeline.run()
    click.echo(stats.summary())


# ============================================================================
# Tokenizer Training Commands
# ============================================================================


@main.group("train-tokenizer")
def train_tokenizer():
    """Train tokenizers for text and multimodal data.

    \b
    Examples:
      auralith-pipeline train-tokenizer text  --corpus data/corpus.txt --output tokenizers/bpe_32k
      auralith-pipeline train-tokenizer image --images data/images/    --output tokenizers/image_vq
      auralith-pipeline train-tokenizer audio --audio  data/audio/     --output tokenizers/audio_vq
      auralith-pipeline train-tokenizer video --videos data/videos/    --output tokenizers/video_vq
      auralith-pipeline train-tokenizer all   --corpus data/corpus.txt --output tokenizers/
    """
    pass


@train_tokenizer.command()
@click.option(
    "--corpus",
    type=click.Path(exists=True),
    required=True,
    help="Path to text file or directory containing .txt training files",
)
@click.option("--output", "-o", type=click.Path(), required=True, help="Output directory")
@click.option(
    "--vocab-size", type=int, default=32000, help="Target vocabulary size (default: 32000)"
)
@click.option("--min-frequency", type=int, default=2, help="Min pair frequency for merging")
@click.option("--lowercase", is_flag=True, help="Lowercase all text during training")
@click.option(
    "--max-corpus-size",
    type=int,
    default=None,
    help="Max corpus size in characters (truncates single files, caps multi-file reads)",
)
def text(corpus, output, vocab_size, min_frequency, lowercase, max_corpus_size):
    """Train BPE tokenizer on a text corpus."""
    from auralith_pipeline.tokenization import BPETokenizer

    corpus_path = Path(corpus)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    click.echo("=" * 60)
    click.echo("Training BPE Text Tokenizer")
    click.echo("=" * 60)
    click.echo(f"  Corpus:     {corpus_path}")
    click.echo(f"  Vocab size: {vocab_size}")
    click.echo(f"  Output:     {output_path}")

    tokenizer = BPETokenizer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        lowercase=lowercase,
    )

    if corpus_path.is_file():
        click.echo(f"\nTraining from file: {corpus_path}")
        if max_corpus_size:
            raw = corpus_path.read_text(encoding="utf-8")
            if len(raw) > max_corpus_size:
                click.echo(
                    f"  Truncating corpus from {len(raw):,} to {max_corpus_size:,} characters"
                )
                raw = raw[:max_corpus_size]
            tokenizer.train(raw)
        else:
            tokenizer.train_from_file(corpus_path)
    elif corpus_path.is_dir():
        text_files: list[str | Path] = [str(p) for p in sorted(corpus_path.rglob("*.txt"))]
        if not text_files:
            click.echo("Error: No .txt files found in directory!", err=True)
            raise SystemExit(1)
        click.echo(f"\nTraining from {len(text_files)} files in {corpus_path}")
        tokenizer.train_from_files(text_files, max_size=max_corpus_size)
    else:
        click.echo(f"Error: Corpus path not found: {corpus_path}", err=True)
        raise SystemExit(1)

    tokenizer.save(output_path)

    click.echo(f"\n[OK] BPE tokenizer saved to {output_path}")
    click.echo(f"  Vocabulary size: {tokenizer.get_vocab_size()}")
    click.echo(f"  Merge rules:     {len(tokenizer.merge_rules)}")


@train_tokenizer.command()
@click.option(
    "--images",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing .npy image files",
)
@click.option("--output", "-o", type=click.Path(), required=True, help="Output directory")
@click.option("--codebook-size", type=int, default=1024, help="VQ codebook size (default: 1024)")
@click.option("--image-size", type=int, default=224, help="Target image size (default: 224)")
@click.option("--patch-size", type=int, default=16, help="Patch size (default: 16)")
@click.option("--sample-size", type=int, default=None, help="Max images to sample for training")
def image(images, output, codebook_size, image_size, patch_size, sample_size):
    """Train image VQ tokenizer on an image dataset (.npy format)."""
    from auralith_pipeline.tokenization import ImageTokenizer

    images_path = Path(images)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = [str(p) for p in sorted(images_path.rglob(_NPY_GLOB))]
    if not image_files:
        click.echo("Error: No .npy image files found!", err=True)
        raise SystemExit(1)

    click.echo("=" * 60)
    click.echo("Training Image VQ Tokenizer")
    click.echo("=" * 60)
    click.echo(f"  Images:        {len(image_files)} files in {images_path}")
    click.echo(f"  Image size:    {image_size}x{image_size}")
    click.echo(f"  Patch size:    {patch_size}x{patch_size}")
    click.echo(f"  Codebook size: {codebook_size}")
    click.echo(f"  Output:        {output_path}")

    tokenizer = ImageTokenizer(
        image_size=image_size,
        patch_size=patch_size,
        codebook_size=codebook_size,
    )
    tokenizer.train(image_files, sample_size=sample_size)  # type: ignore[arg-type]
    tokenizer.save(output_path)

    click.echo(f"\n[OK] Image tokenizer saved to {output_path}")
    click.echo(f"  Tokens per image: {tokenizer.num_patches}")


@train_tokenizer.command()
@click.option(
    "--audio",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing .npy audio files",
)
@click.option("--output", "-o", type=click.Path(), required=True, help="Output directory")
@click.option("--codebook-size", type=int, default=512, help="VQ codebook size (default: 512)")
@click.option("--sample-rate", type=int, default=16000, help="Sample rate in Hz (default: 16000)")
@click.option("--sample-size", type=int, default=None, help="Max audio files to sample")
def audio(audio, output, codebook_size, sample_rate, sample_size):
    """Train audio VQ tokenizer on an audio dataset (.npy format)."""
    from auralith_pipeline.tokenization import AudioTokenizer

    audio_path = Path(audio)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    audio_files = [str(p) for p in sorted(audio_path.rglob(_NPY_GLOB))]
    if not audio_files:
        click.echo("Error: No .npy audio files found!", err=True)
        raise SystemExit(1)

    click.echo("=" * 60)
    click.echo("Training Audio VQ Tokenizer")
    click.echo("=" * 60)
    click.echo(f"  Audio files:   {len(audio_files)} files in {audio_path}")
    click.echo(f"  Sample rate:   {sample_rate} Hz")
    click.echo(f"  Codebook size: {codebook_size}")
    click.echo(f"  Output:        {output_path}")

    tokenizer = AudioTokenizer(
        sample_rate=sample_rate,
        codebook_size=codebook_size,
    )
    tokenizer.train(audio_files, sample_size=sample_size)
    tokenizer.save(output_path)

    click.echo(f"\n[OK] Audio tokenizer saved to {output_path}")


@train_tokenizer.command()
@click.option(
    "--videos",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing video files (.mp4, .avi, .mov, .mkv)",
)
@click.option("--output", "-o", type=click.Path(), required=True, help="Output directory")
@click.option("--codebook-size", type=int, default=1024, help="VQ codebook size (default: 1024)")
@click.option("--image-size", type=int, default=224, help="Frame resize (default: 224)")
@click.option("--patch-size", type=int, default=16, help="Patch size (default: 16)")
@click.option("--max-frames", type=int, default=32, help="Max frames per video (default: 32)")
@click.option("--sample-size", type=int, default=None, help="Max videos to use for training")
def video(videos, output, codebook_size, image_size, patch_size, max_frames, sample_size):
    """Train video VQ tokenizer on a video dataset."""
    from auralith_pipeline.tokenization import VideoTokenizer

    videos_path = Path(videos)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find video files
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_files: list[str | Path] = sorted(
        str(p) for p in videos_path.rglob("*") if p.suffix.lower() in video_exts
    )
    if not video_files:
        click.echo("Error: No video files found (.mp4/.avi/.mov/.mkv/.webm)!", err=True)
        raise SystemExit(1)

    if sample_size and len(video_files) > sample_size:
        video_files = video_files[:sample_size]

    click.echo("=" * 60)
    click.echo("Training Video VQ Tokenizer")
    click.echo("=" * 60)
    click.echo(f"  Videos:        {len(video_files)} files in {videos_path}")
    click.echo(f"  Frame size:    {image_size}x{image_size}")
    click.echo(f"  Patch size:    {patch_size}x{patch_size}")
    click.echo(f"  Codebook size: {codebook_size}")
    click.echo(f"  Max frames:    {max_frames}")
    click.echo(f"  Output:        {output_path}")

    tokenizer = VideoTokenizer(
        image_size=image_size,
        patch_size=patch_size,
        codebook_size=codebook_size,
        max_frames=max_frames,
    )
    tokenizer.train_from_video_files(
        video_files,
        max_frames_per_video=max_frames,
    )
    tokenizer.save(output_path)

    click.echo(f"\n[OK] Video tokenizer saved to {output_path}")
    click.echo(f"  Patches per frame: {tokenizer.patches_per_frame}")
    click.echo(f"  Max tokens/video:  {tokenizer.patches_per_frame * max_frames}")


@train_tokenizer.command("all")
@click.option(
    "--corpus",
    type=click.Path(exists=True),
    default=None,
    help="Text corpus file or directory (required for text modality)",
)
@click.option("--images", type=click.Path(exists=True), default=None, help="Image .npy directory")
@click.option("--audio", type=click.Path(exists=True), default=None, help="Audio .npy directory")
@click.option("--videos", type=click.Path(exists=True), default=None, help="Video files directory")
@click.option("--output", "-o", type=click.Path(), required=True, help="Root output directory")
@click.option("--vocab-size", type=int, default=32000, help="BPE vocab size")
@click.option("--codebook-size", type=int, default=1024, help="VQ codebook size (image/video)")
@click.option("--audio-codebook-size", type=int, default=512, help="Audio VQ codebook size")
def train_all(
    corpus, images, audio, videos, output, vocab_size, codebook_size, audio_codebook_size
):
    """Train ALL tokenizers in one command.

    \b
    Trains whichever modalities have data provided:
      --corpus  → BPE text tokenizer   → <output>/text/
      --images  → Image VQ tokenizer   → <output>/image/
      --audio   → Audio VQ tokenizer   → <output>/audio/
      --videos  → Video VQ tokenizer   → <output>/video/

    \b
    Example:
      auralith-pipeline train-tokenizer all \\
        --corpus data/corpus.txt \\
        --images data/images/ \\
        --audio  data/audio/ \\
        --videos data/videos/ \\
        --output tokenizers/ \\
        --vocab-size 32000
    """
    output_root = Path(output)

    if not any([corpus, images, audio, videos]):
        click.echo("Error: provide at least one of --corpus, --images, --audio, --videos", err=True)
        raise SystemExit(1)

    click.echo("=" * 60)
    click.echo("Training All Tokenizers")
    click.echo("=" * 60)

    trained = _train_all_modalities(
        corpus=corpus,
        images=images,
        audio=audio,
        videos=videos,
        output_root=output_root,
        vocab_size=vocab_size,
        codebook_size=codebook_size,
        audio_codebook_size=audio_codebook_size,
    )

    click.echo("\n" + "=" * 60)
    if trained:
        click.echo(f"[OK] Trained {len(trained)} tokenizer(s): {', '.join(trained)}")
        click.echo(f"Output: {output_root}")
    else:
        click.echo("[WARNING] No tokenizers were trained")
    click.echo("=" * 60)


def _train_all_modalities(
    *,
    corpus: str | None,
    images: str | None,
    audio: str | None,
    videos: str | None,
    output_root: Path,
    vocab_size: int,
    codebook_size: int,
    audio_codebook_size: int,
) -> list[str]:
    """Train each requested modality tokenizer and return list of trained names."""
    trained: list[str] = []

    # --- 1. Text BPE ---
    if corpus:
        trained += _train_text_bpe(corpus, output_root / "text", vocab_size)
    else:
        click.echo("\n[1/4] Text BPE  →  skipped (no --corpus)")

    # --- 2. Image VQ ---
    if images:
        trained += _train_image_vq(images, output_root / "image", codebook_size)
    else:
        click.echo("\n[2/4] Image VQ  →  skipped (no --images)")

    # --- 3. Audio VQ ---
    if audio:
        trained += _train_audio_vq(audio, output_root / "audio", audio_codebook_size)
    else:
        click.echo("\n[3/4] Audio VQ  →  skipped (no --audio)")

    # --- 4. Video VQ ---
    if videos:
        trained += _train_video_vq(videos, output_root / "video", codebook_size)
    else:
        click.echo("\n[4/4] Video VQ  →  skipped (no --videos)")

    return trained


def _train_text_bpe(corpus: str, out_dir: Path, vocab_size: int) -> list[str]:
    """Train BPE text tokenizer; returns ['text'] on success, else []."""
    from auralith_pipeline.tokenization import BPETokenizer

    corpus_path = Path(corpus)
    out_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"\n[1/4] Text BPE  →  {out_dir}")

    tokenizer = BPETokenizer(vocab_size=vocab_size)
    if corpus_path.is_file():
        tokenizer.train_from_file(corpus_path)
    else:
        txt_files: list[str | Path] = [str(p) for p in sorted(corpus_path.rglob("*.txt"))]
        if txt_files:
            tokenizer.train_from_files(txt_files)
        else:
            click.echo("  Warning: No .txt files found, skipping text")

    if tokenizer.merge_rules:
        tokenizer.save(out_dir)
        click.echo(
            f"  [OK] vocab={tokenizer.get_vocab_size()}, merges={len(tokenizer.merge_rules)}"
        )
        return ["text"]
    click.echo("  [SKIP] No merge rules learned")
    return []


def _train_image_vq(images: str, out_dir: Path, codebook_size: int) -> list[str]:
    """Train image VQ tokenizer; returns ['image'] on success, else []."""
    from auralith_pipeline.tokenization import ImageTokenizer

    images_path = Path(images)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_files: list[str | Path] = [str(p) for p in sorted(images_path.rglob(_NPY_GLOB))]
    click.echo(f"\n[2/4] Image VQ  →  {out_dir}  ({len(img_files)} files)")

    if img_files:
        tok = ImageTokenizer(codebook_size=codebook_size)
        tok.train(img_files)
        tok.save(out_dir)
        click.echo(f"  [OK] codebook={codebook_size}, patches/img={tok.num_patches}")
        return ["image"]
    click.echo("  [SKIP] No .npy files found")
    return []


def _train_audio_vq(audio: str, out_dir: Path, codebook_size: int) -> list[str]:
    """Train audio VQ tokenizer; returns ['audio'] on success, else []."""
    from auralith_pipeline.tokenization import AudioTokenizer

    audio_path = Path(audio)
    out_dir.mkdir(parents=True, exist_ok=True)

    aud_files: list[str | Path] = [str(p) for p in sorted(audio_path.rglob(_NPY_GLOB))]
    click.echo(f"\n[3/4] Audio VQ  →  {out_dir}  ({len(aud_files)} files)")

    if aud_files:
        tok = AudioTokenizer(codebook_size=codebook_size)
        tok.train(aud_files)
        tok.save(out_dir)
        click.echo(f"  [OK] codebook={codebook_size}")
        return ["audio"]
    click.echo("  [SKIP] No .npy files found")
    return []


def _train_video_vq(videos: str, out_dir: Path, codebook_size: int) -> list[str]:
    """Train video VQ tokenizer; returns ['video'] on success, else []."""
    from auralith_pipeline.tokenization import VideoTokenizer

    videos_path = Path(videos)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    vid_files: list[str | Path] = [
        str(p) for p in sorted(videos_path.rglob("*")) if p.suffix.lower() in video_exts
    ]
    click.echo(f"\n[4/4] Video VQ  →  {out_dir}  ({len(vid_files)} files)")

    if vid_files:
        tok = VideoTokenizer(codebook_size=codebook_size)
        tok.train_from_video_files(vid_files)
        tok.save(out_dir)
        click.echo(f"  [OK] codebook={codebook_size}, patches/frame={tok.patches_per_frame}")
        return ["video"]
    click.echo("  [SKIP] No video files found")
    return []


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
