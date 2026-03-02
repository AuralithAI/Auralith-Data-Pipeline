"""Command-line interface for Auralith Data Pipeline."""

import logging
import platform
import sys
from pathlib import Path
from typing import Any

import click

from auralith_pipeline import __version__
from auralith_pipeline.config.pipeline_config import PipelineConfig
from auralith_pipeline.pipeline import Pipeline
from auralith_pipeline.sources.data_sources import DATASET_REGISTRY, create_source
from auralith_pipeline.storage.backends import create_storage_backend
from auralith_pipeline.utils.file_types import (
    AUDIO_EXTS as _AUDIO_EXTS,  # noqa: F401  (re-exported for tests)
    AUDIO_TOKEN_OFFSET as _AUDIO_TOKEN_OFFSET,
    IMAGE_EXTS as _IMAGE_EXTS,  # noqa: F401  (re-exported for tests)
    IMAGE_TOKEN_OFFSET as _IMAGE_TOKEN_OFFSET,
    MODALITY_ID as _MODALITY_ID,
    TEXT_EXTS as _TEXT_EXTS,  # noqa: F401  (re-exported for tests)
    VIDEO_EXTS as _VIDEO_EXTS,
    VIDEO_TOKEN_OFFSET as _VIDEO_TOKEN_OFFSET,
    classify_file as _classify_file,
)
from auralith_pipeline.utils.helpers import format_size, setup_logging

logger = logging.getLogger(__name__)

_NPY_GLOB = "*.npy"
_CONFIG_JSON = "config.json"

# ── ASCII banner ──────────────────────────────────────────────────────────
# Displayed once when the CLI starts.  Uses Rich markup for gradient color.
_BANNER_ART = r"""
[bold bright_magenta]       █████╗  ██╗   ██╗ ██████╗   █████╗  ██╗      ██╗ ████████╗ ██╗  ██╗[/]
[bold magenta]      ██╔══██╗ ██║   ██║ ██╔══██╗ ██╔══██╗ ██║      ██║ ╚══██╔══╝ ██║  ██║[/]
[bold bright_cyan]      ███████║ ██║   ██║ ██████╔╝ ███████║ ██║      ██║    ██║    ███████║[/]
[bold cyan]      ██╔══██║ ██║   ██║ ██╔══██╗ ██╔══██║ ██║      ██║    ██║    ██╔══██║[/]
[bold bright_blue]      ██║  ██║ ╚██████╔╝ ██║  ██║ ██║  ██║ ███████╗ ██║    ██║    ██║  ██║[/]
[bold blue]      ╚═╝  ╚═╝  ╚═════╝  ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚══════╝ ╚═╝    ╚═╝    ╚═╝  ╚═╝[/]
"""

_BANNER_TAGLINE = "[bold bright_white]Multimodal Data Pipeline for RT-DLM[/bold bright_white]"


def _print_banner() -> None:
    """Print the vibrant startup banner using Rich."""
    try:
        from rich.console import Console
        from rich.panel import Panel

        console = Console(stderr=True)

        # Build info line
        py_ver = platform.python_version()
        os_name = platform.system()
        info_line = (
            f"[bold green]v{__version__}[/]  "
            f"[dim]•[/dim]  [white]Python {py_ver}[/]  "
            f"[dim]•[/dim]  [white]{os_name}[/]"
        )

        body = _BANNER_ART + "\n" + _BANNER_TAGLINE + "\n" + info_line

        console.print(
            Panel(
                body,
                border_style="bright_magenta",
                padding=(0, 2),
                subtitle="[dim italic]github.com/AuralithAI[/]",
                subtitle_align="right",
            )
        )

        # Tips
        console.print(
            "[dim]Getting started:[/dim]\n"
            "  [bold cyan]1.[/] Run [bold cyan]auralith-pipeline --help[/] "
            "to see all commands.\n"
            "  [bold cyan]2.[/] Use [bold cyan]-v[/] for verbose / debug output.\n"
            "  [bold cyan]3.[/] Install all extras: "
            '[bold yellow]pip install "auralith-data-pipeline\\[all]"[/]\n',
            highlight=False,
        )
    except Exception:  # noqa: BLE001 – never crash on a cosmetic banner
        # Graceful fallback – plain text
        click.echo(f"Auralith Data Pipeline  v{__version__}")


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--no-banner",
    is_flag=True,
    default=False,
    envvar="AURALITH_NO_BANNER",
    help="Suppress the startup banner",
)
def main(verbose: bool, no_banner: bool):
    """Auralith Data Pipeline - Production-grade data processing for LLMs."""
    if not no_banner and sys.stderr.isatty():
        _print_banner()
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

    click.echo(f"\n[OK] Text BPE tokenizer saved to {output_path}")
    click.echo(f"  vocab={tokenizer.get_vocab_size()}, merges={len(tokenizer.merge_rules)}")


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

    image_files: list[str | Path] = [str(p) for p in sorted(images_path.rglob(_NPY_GLOB))]
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
    tokenizer.train(image_files, sample_size=sample_size)
    tokenizer.save(output_path)

    click.echo(f"\n[OK] Image VQ tokenizer saved to {output_path}")
    click.echo(f"  codebook={codebook_size}, patches/img={tokenizer.num_patches}")


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

    audio_files: list[str | Path] = [str(p) for p in sorted(audio_path.rglob(_NPY_GLOB))]
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

    click.echo(f"\n[OK] Audio VQ tokenizer saved to {output_path}")
    click.echo(f"  codebook={codebook_size}")


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
    video_files: list[str | Path] = sorted(
        str(p) for p in videos_path.rglob("*") if p.suffix.lower() in _VIDEO_EXTS
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
    # max_frames is intentionally used for both the constructor (encoding cap)
    # and training (frames sampled per video) to keep the distribution aligned.
    tokenizer.train_from_video_files(
        video_files,
        max_frames_per_video=max_frames,
    )
    tokenizer.save(output_path)

    click.echo(f"\n[OK] Video VQ tokenizer saved to {output_path}")
    click.echo(
        f"  codebook={codebook_size}, patches/frame={tokenizer.patches_per_frame}, "
        f"max_tokens/video={tokenizer.patches_per_frame * max_frames}"
    )


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
@click.option("--image-size", type=int, default=224, help="Image/frame resize (default: 224)")
@click.option("--patch-size", type=int, default=16, help="Patch size for image/video (default: 16)")
@click.option(
    "--sample-rate", type=int, default=16000, help="Audio sample rate Hz (default: 16000)"
)
@click.option("--max-frames", type=int, default=32, help="Max frames per video (default: 32)")
def train_all(
    corpus,
    images,
    audio,
    videos,
    output,
    vocab_size,
    codebook_size,
    audio_codebook_size,
    image_size,
    patch_size,
    sample_rate,
    max_frames,
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
        --vocab-size 32000 \\
        --codebook-size 1024 \\
        --audio-codebook-size 512
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
        image_size=image_size,
        patch_size=patch_size,
        sample_rate=sample_rate,
        max_frames=max_frames,
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
    image_size: int,
    patch_size: int,
    sample_rate: int,
    max_frames: int,
) -> list[str]:
    """Train each requested modality tokenizer and return list of trained names."""
    trained: list[str] = []
    modalities = [
        ("Text BPE", corpus),
        ("Image VQ", images),
        ("Audio VQ", audio),
        ("Video VQ", videos),
    ]
    active = [(name, val) for name, val in modalities if val]
    total = len(active)
    step = 0

    # --- Text BPE ---
    if corpus:
        step += 1
        trained += _train_text_bpe(corpus, output_root / "text", vocab_size, step, total)
    else:
        click.echo("\n  Text BPE  →  skipped (no --corpus)")

    # --- Image VQ ---
    if images:
        step += 1
        trained += _train_image_vq(
            images, output_root / "image", codebook_size, image_size, patch_size, step, total
        )
    else:
        click.echo("\n  Image VQ  →  skipped (no --images)")

    # --- Audio VQ ---
    if audio:
        step += 1
        trained += _train_audio_vq(
            audio, output_root / "audio", audio_codebook_size, sample_rate, step, total
        )
    else:
        click.echo("\n  Audio VQ  →  skipped (no --audio)")

    # --- Video VQ ---
    if videos:
        step += 1
        trained += _train_video_vq(
            videos,
            output_root / "video",
            codebook_size,
            image_size,
            patch_size,
            max_frames,
            step,
            total,
        )
    else:
        click.echo("\n  Video VQ  →  skipped (no --videos)")

    return trained


def _train_text_bpe(
    corpus: str, out_dir: Path, vocab_size: int, step: int, total: int
) -> list[str]:
    """Train BPE text tokenizer; returns ['text'] on success, else []."""
    from auralith_pipeline.tokenization import BPETokenizer

    corpus_path = Path(corpus)
    out_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"\n[{step}/{total}] Text BPE  →  {out_dir}")

    tokenizer = BPETokenizer(vocab_size=vocab_size)
    if corpus_path.is_file():
        tokenizer.train_from_file(corpus_path)
    else:
        txt_files: list[str | Path] = [str(p) for p in sorted(corpus_path.rglob("*.txt"))]
        if txt_files:
            tokenizer.train_from_files(txt_files)
        else:
            click.echo("  [SKIP] No .txt files found")
            return []

    if tokenizer.merge_rules:
        tokenizer.save(out_dir)
        click.echo(
            f"  [OK] Text BPE tokenizer saved — "
            f"vocab={tokenizer.get_vocab_size()}, merges={len(tokenizer.merge_rules)}"
        )
        return ["text"]
    click.echo("  [SKIP] No merge rules learned")
    return []


def _train_image_vq(
    images: str,
    out_dir: Path,
    codebook_size: int,
    image_size: int,
    patch_size: int,
    step: int,
    total: int,
) -> list[str]:
    """Train image VQ tokenizer; returns ['image'] on success, else []."""
    from auralith_pipeline.tokenization import ImageTokenizer

    images_path = Path(images)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_files: list[str | Path] = [str(p) for p in sorted(images_path.rglob(_NPY_GLOB))]
    click.echo(f"\n[{step}/{total}] Image VQ  →  {out_dir}  ({len(img_files)} files)")

    if img_files:
        tok = ImageTokenizer(
            image_size=image_size, patch_size=patch_size, codebook_size=codebook_size
        )
        tok.train(img_files)
        tok.save(out_dir)
        click.echo(
            f"  [OK] Image VQ tokenizer saved — "
            f"codebook={codebook_size}, patches/img={tok.num_patches}"
        )
        return ["image"]
    click.echo("  [SKIP] No .npy files found")
    return []


def _train_audio_vq(
    audio: str,
    out_dir: Path,
    codebook_size: int,
    sample_rate: int,
    step: int,
    total: int,
) -> list[str]:
    """Train audio VQ tokenizer; returns ['audio'] on success, else []."""
    from auralith_pipeline.tokenization import AudioTokenizer

    audio_path = Path(audio)
    out_dir.mkdir(parents=True, exist_ok=True)

    aud_files: list[str | Path] = [str(p) for p in sorted(audio_path.rglob(_NPY_GLOB))]
    click.echo(f"\n[{step}/{total}] Audio VQ  →  {out_dir}  ({len(aud_files)} files)")

    if aud_files:
        tok = AudioTokenizer(sample_rate=sample_rate, codebook_size=codebook_size)
        tok.train(aud_files)
        tok.save(out_dir)
        click.echo(f"  [OK] Audio VQ tokenizer saved — codebook={codebook_size}")
        return ["audio"]
    click.echo("  [SKIP] No .npy files found")
    return []


def _train_video_vq(
    videos: str,
    out_dir: Path,
    codebook_size: int,
    image_size: int,
    patch_size: int,
    max_frames: int,
    step: int,
    total: int,
) -> list[str]:
    """Train video VQ tokenizer; returns ['video'] on success, else []."""
    from auralith_pipeline.tokenization import VideoTokenizer

    videos_path = Path(videos)
    out_dir.mkdir(parents=True, exist_ok=True)

    vid_files: list[str | Path] = [
        str(p) for p in sorted(videos_path.rglob("*")) if p.suffix.lower() in _VIDEO_EXTS
    ]
    click.echo(f"\n[{step}/{total}] Video VQ  →  {out_dir}  ({len(vid_files)} files)")

    if vid_files:
        tok = VideoTokenizer(
            image_size=image_size,
            patch_size=patch_size,
            codebook_size=codebook_size,
            max_frames=max_frames,
        )
        tok.train_from_video_files(vid_files, max_frames_per_video=max_frames)
        tok.save(out_dir)
        click.echo(
            f"  [OK] Video VQ tokenizer saved — "
            f"codebook={codebook_size}, patches/frame={tok.patches_per_frame}"
        )
        return ["video"]
    click.echo("  [SKIP] No video files found")
    return []


# ============================================================================
# Process Command — Raw Data → Production SafeTensors Shards
# ============================================================================


@main.command()
@click.option(
    "--input",
    "-i",
    "input_dir",
    type=click.Path(exists=True),
    required=True,
    help="Folder with raw files (.txt, .jpg, .png, .wav, .mp4, .npy, etc.)",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(),
    required=True,
    help="Where .safetensors shards are written",
)
@click.option(
    "--tokenizers",
    "tokenizers_dir",
    type=click.Path(exists=True),
    required=True,
    help="Root folder of trained tokenizers (text/, image/, audio/, video/ subdirs)",
)
@click.option(
    "--max-seq-len",
    type=int,
    default=4096,
    help="Maximum sequence length per sample (default: 4096)",
)
@click.option(
    "--shard-size",
    type=int,
    default=10000,
    help="Max samples per shard (default: 10000)",
)
def process(
    input_dir: str, output_dir: str, tokenizers_dir: str, max_seq_len: int, shard_size: int
):
    """Process raw files → production-ready .safetensors shards.

    \b
    Takes a folder of mixed raw data (text, images, audio, video) and
    uses frozen tokenizers to produce sharded SafeTensors files with
    input_ids, attention_mask, modality_mask, and targets.

    \b
    The tokenizers directory should contain subdirectories created by
    `train-tokenizer all`:
        tokenizers/
        ├── text/     (vocab.json, merges.txt, config.json)
        ├── image/    (config.json, vq_codebook.json)
        ├── audio/    (config.json, vq_codebook.json)
        └── video/    (config.json, vq_codebook.json)

    \b
    Example:
        auralith-pipeline process \\
            --input  data/raw/ \\
            --output shards/ \\
            --tokenizers tokenizers/ \\
            --max-seq-len 4096 \\
            --shard-size 10000
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    tok_path = Path(tokenizers_dir)

    click.echo("=" * 60)
    click.echo("Processing Raw Data → SafeTensors Shards")
    click.echo("=" * 60)
    click.echo(f"  Input:       {in_path}")
    click.echo(f"  Output:      {out_path}")
    click.echo(f"  Tokenizers:  {tok_path}")
    click.echo(f"  Max seq len: {max_seq_len}")
    click.echo(f"  Shard size:  {shard_size}")

    # ------------------------------------------------------------------
    # 1. Load tokenizers
    # ------------------------------------------------------------------
    click.echo("\n[1/3] Loading tokenizers ...")
    tokenizers = _load_all_tokenizers(tok_path)
    loaded = [name for name, tok in tokenizers.items() if tok is not None]
    if not loaded:
        click.echo("[ERROR] No tokenizers found in the tokenizers directory!", err=True)
        raise SystemExit(1)
    click.echo(f"  Loaded: {', '.join(loaded)}")

    # ------------------------------------------------------------------
    # 2. Discover & tokenize raw files
    # ------------------------------------------------------------------
    click.echo("\n[2/3] Tokenizing raw files ...")
    out_path.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(p for p in in_path.rglob("*") if p.is_file())
    if not raw_files:
        click.echo("[ERROR] No files found in input directory!", err=True)
        raise SystemExit(1)

    click.echo(f"  Found {len(raw_files)} files")

    samples: list[dict[str, Any]] = []
    skipped = 0

    for idx, file_path in enumerate(raw_files):
        if (idx + 1) % 500 == 0:
            click.echo(f"  ... processed {idx + 1}/{len(raw_files)}")

        result = _tokenize_file(file_path, tokenizers, max_seq_len)
        if result is not None:
            samples.append(result)
        else:
            skipped += 1

    click.echo(f"  Tokenized {len(samples)} files ({skipped} skipped)")

    if not samples:
        click.echo("[ERROR] No files could be tokenized!", err=True)
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # 3. Pack into shards and write SafeTensors
    # ------------------------------------------------------------------
    click.echo("\n[3/3] Writing shards ...")
    num_shards = _write_shards(samples, out_path, max_seq_len, shard_size)

    click.echo("\n" + "=" * 60)
    click.echo(
        f"[OK] Created {num_shards} shard(s) in {out_path}  "
        f"({len(samples)} samples, max_seq_len={max_seq_len})"
    )
    click.echo("=" * 60)


# ---- process helpers -----------------------------------------------------


def _load_all_tokenizers(tokenizers_dir: Path) -> dict[str, Any]:
    """Load all available tokenizers from a directory.

    Looks for subdirectories named text/, image/, audio/, video/ and
    loads the corresponding tokenizer class from each.

    Returns:
        Dict mapping modality name → tokenizer instance (or None if absent).
    """
    result: dict[str, Any] = {"text": None, "image": None, "audio": None, "video": None}

    text_dir = tokenizers_dir / "text"
    if text_dir.is_dir() and (text_dir / "vocab.json").exists():
        from auralith_pipeline.tokenization import BPETokenizer

        result["text"] = BPETokenizer.load(text_dir)
        logger.info("Loaded text BPE tokenizer from %s", text_dir)

    image_dir = tokenizers_dir / "image"
    if image_dir.is_dir() and (image_dir / _CONFIG_JSON).exists():
        from auralith_pipeline.tokenization import ImageTokenizer

        result["image"] = ImageTokenizer.load(image_dir)
        logger.info("Loaded image VQ tokenizer from %s", image_dir)

    audio_dir = tokenizers_dir / "audio"
    if audio_dir.is_dir() and (audio_dir / _CONFIG_JSON).exists():
        from auralith_pipeline.tokenization import AudioTokenizer

        result["audio"] = AudioTokenizer.load(audio_dir)
        logger.info("Loaded audio VQ tokenizer from %s", audio_dir)

    video_dir = tokenizers_dir / "video"
    if video_dir.is_dir() and (video_dir / _CONFIG_JSON).exists():
        from auralith_pipeline.tokenization import VideoTokenizer

        result["video"] = VideoTokenizer.load(video_dir)
        logger.info("Loaded video VQ tokenizer from %s", video_dir)

    return result


def _tokenize_file(
    file_path: Path,
    tokenizers: dict[str, Any],
    max_seq_len: int,
) -> dict[str, Any] | None:
    """Tokenize a single raw file and return input_ids + modality_mask.

    Returns None if the file type is unsupported or the relevant tokenizer
    is not available.
    """
    modality = _classify_file(file_path)
    if modality is None:
        return None

    tokenizer = tokenizers.get(modality)
    if tokenizer is None:
        return None

    try:
        if modality == "text":
            text = file_path.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                return None
            input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=max_seq_len)
            modality_mask = [_MODALITY_ID["text"]] * len(input_ids)

        elif modality == "image":
            codes = tokenizer.encode(file_path)
            # Wrap with special tokens: <IMG> [codes] <IMG_END>
            from auralith_pipeline.tokenization import BPETokenizer

            img_start = BPETokenizer.SPECIAL_TOKENS["<IMG>"]
            img_end = BPETokenizer.SPECIAL_TOKENS["<IMG_END>"]
            input_ids = [img_start] + [c + _IMAGE_TOKEN_OFFSET for c in codes] + [img_end]
            modality_mask = [_MODALITY_ID["image"]] * len(input_ids)

        elif modality == "audio":
            codes = tokenizer.encode(file_path)
            from auralith_pipeline.tokenization import BPETokenizer

            aud_start = BPETokenizer.SPECIAL_TOKENS["<AUDIO>"]
            aud_end = BPETokenizer.SPECIAL_TOKENS["<AUDIO_END>"]
            input_ids = [aud_start] + [c + _AUDIO_TOKEN_OFFSET for c in codes] + [aud_end]
            modality_mask = [_MODALITY_ID["audio"]] * len(input_ids)

        elif modality == "video":
            codes = tokenizer.encode(file_path)
            from auralith_pipeline.tokenization import BPETokenizer

            vid_start = BPETokenizer.SPECIAL_TOKENS["<VIDEO>"]
            vid_end = BPETokenizer.SPECIAL_TOKENS["<VIDEO_END>"]
            input_ids = [vid_start] + [c + _VIDEO_TOKEN_OFFSET for c in codes] + [vid_end]
            modality_mask = [_MODALITY_ID["video"]] * len(input_ids)

        else:
            return None

        # Truncate to max_seq_len
        input_ids = input_ids[:max_seq_len]
        modality_mask = modality_mask[:max_seq_len]

        return {
            "input_ids": input_ids,
            "modality_mask": modality_mask,
            "source": str(file_path),
        }

    except Exception as e:
        logger.warning("Failed to tokenize %s: %s", file_path, e)
        return None


def _write_shards(
    samples: list[dict[str, Any]],
    output_dir: Path,
    max_seq_len: int,
    shard_size: int,
) -> int:
    """Pack tokenized samples into fixed-length SafeTensors shards.

    Each shard contains up to ``shard_size`` samples. Every sample is
    padded/truncated to ``max_seq_len``.

    Returns the number of shards written.
    """
    import numpy as np

    try:
        from safetensors.numpy import save_file
    except ImportError:
        click.echo(
            "[ERROR] safetensors is not installed. Install with: pip install safetensors",
            err=True,
        )
        raise SystemExit(1) from None

    shard_idx = 0
    for start in range(0, len(samples), shard_size):
        batch = samples[start : start + shard_size]

        input_ids_arr = np.zeros((len(batch), max_seq_len), dtype=np.int32)
        attention_mask_arr = np.zeros((len(batch), max_seq_len), dtype=np.uint8)
        modality_mask_arr = np.zeros((len(batch), max_seq_len), dtype=np.uint8)

        for i, sample in enumerate(batch):
            ids = sample["input_ids"]
            mask = sample["modality_mask"]
            seq_len = min(len(ids), max_seq_len)

            input_ids_arr[i, :seq_len] = ids[:seq_len]
            attention_mask_arr[i, :seq_len] = 1
            modality_mask_arr[i, :seq_len] = mask[:seq_len]

        # targets = right-shifted input_ids (causal LM next-token prediction)
        targets_arr = np.zeros_like(input_ids_arr)
        targets_arr[:, :-1] = input_ids_arr[:, 1:]

        shard_path = output_dir / f"shard_{shard_idx:06d}.safetensors"

        metadata = {
            "pipeline_version": "2.0",
            "schema_version": "2",
            "shard_id": str(shard_idx),
            "num_samples": str(len(batch)),
            "seq_length": str(max_seq_len),
        }

        save_file(
            {
                "input_ids": input_ids_arr,
                "attention_mask": attention_mask_arr,
                "modality_mask": modality_mask_arr,
                "targets": targets_arr,
            },
            str(shard_path),
            metadata=metadata,
        )

        click.echo(f"  Wrote {shard_path.name} ({len(batch)} samples)")
        shard_idx += 1

    return shard_idx


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
@click.option(
    "--state-store",
    type=click.Choice(["redis", "memory"]),
    default="redis",
    help="State store backend (default: redis)",
)
def worker(config: str, coordinator: str, worker_id: str, state_store: str):
    """Start a worker node for distributed processing."""
    from auralith_pipeline.distributed import DistributedConfig, Worker

    click.echo("=" * 60)
    click.echo("Starting Auralith Worker Node")
    click.echo("=" * 60)

    # Parse coordinator address
    host, port = coordinator.split(":")
    dist_config = DistributedConfig.from_yaml(config)

    click.echo(f"\nWorker ID: {worker_id}")
    click.echo(f"Coordinator: {host}:{port}")
    click.echo(f"State store: {state_store}")

    # Create worker
    w = Worker(worker_id, host, int(port))

    # Connect to state store
    if state_store == "memory":
        from auralith_pipeline.distributed.state import InMemoryStateStore

        store = InMemoryStateStore()
    else:
        from auralith_pipeline.distributed.state import RedisStateStore

        store = RedisStateStore(
            host=dist_config.coordinator.state_store_host,
            port=dist_config.coordinator.state_store_port,
            db=dist_config.coordinator.state_store_db,
            password=dist_config.coordinator.state_store_password,
        )

    try:
        w.connect(store)
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
@click.option("--input-dir", "-i", required=True, help="Input directory of raw files")
@click.option("--output-dir", "-o", required=True, help="Output directory for shards")
@click.option("--tokenizers-dir", "-t", required=True, help="Tokenizers directory")
@click.option("--job-name", default="auralith-job", help="Job name")
@click.option("--max-seq-len", default=4096, type=int, help="Max sequence length")
@click.option("--shard-size", default=10000, type=int, help="Samples per shard")
@click.option("--files-per-task", default=500, type=int, help="Files per distributed task")
@click.option("--num-workers", "-w", default=2, type=int, help="Workers (embedded mode)")
@click.option(
    "--embedded/--external",
    default=False,
    help="Run coordinator + workers in-process (no Redis needed)",
)
def submit_job(
    config: str,
    input_dir: str,
    output_dir: str,
    tokenizers_dir: str,
    job_name: str,
    max_seq_len: int,
    shard_size: int,
    files_per_task: int,
    num_workers: int,
    embedded: bool,
):
    """Submit a distributed processing job.

    \b
    Two modes:
      --embedded   Spin up coordinator + N workers in-process (no Redis).
      --external   Connect to an already-running coordinator via Redis.

    \b
    Embedded example (single machine, no Redis):
        auralith-pipeline submit-job \\
            -c configs/distributed.yaml \\
            -i data/raw/ -o shards/ -t tokenizers/ \\
            --embedded -w 4

    \b
    External example (coordinator + workers already running):
        auralith-pipeline submit-job \\
            -c configs/distributed.yaml \\
            -i data/raw/ -o shards/ -t tokenizers/ \\
            --external
    """
    from auralith_pipeline.distributed import DistributedConfig, DistributedPipeline, JobConfig

    click.echo("=" * 60)
    click.echo("Submitting Distributed Job")
    click.echo("=" * 60)

    dist_config = DistributedConfig.from_yaml(config)

    click.echo(f"\n  Job Name:       {job_name}")
    click.echo(f"  Input:          {input_dir}")
    click.echo(f"  Output:         {output_dir}")
    click.echo(f"  Tokenizers:     {tokenizers_dir}")
    click.echo(f"  Max seq len:    {max_seq_len}")
    click.echo(f"  Shard size:     {shard_size}")
    click.echo(f"  Files/task:     {files_per_task}")
    click.echo(f"  Mode:           {'embedded' if embedded else 'external'}")
    if embedded:
        click.echo(f"  Workers:        {num_workers}")

    job_config = JobConfig(
        name=job_name,
        coordinator_host=dist_config.coordinator.host,
        coordinator_port=dist_config.coordinator.port,
        num_workers=num_workers,
        output_dir=output_dir,
    )

    pipeline = DistributedPipeline(
        job_config,
        distributed_config=dist_config,
        embedded=embedded,
        num_workers=num_workers,
    )

    click.echo("\n[OK] Job configured — starting execution...")

    try:
        result = pipeline.run(
            input_dir=input_dir,
            output_dir=output_dir,
            tokenizers_dir=tokenizers_dir,
            max_seq_len=max_seq_len,
            shard_size=shard_size,
            files_per_task=files_per_task,
        )

        click.echo("\n" + "=" * 60)
        click.echo("Job Completed")
        click.echo("=" * 60)
        click.echo(f"\n  Status:     {result.get('status', 'unknown')}")
        click.echo(
            f"  Tasks:      {result.get('completed_tasks', 0)}/{result.get('total_tasks', 0)}"
        )
        click.echo(f"  Failed:     {result.get('failed_tasks', 0)}")
        click.echo(f"  Time:       {result.get('elapsed_seconds', 0):.1f}s")
        click.echo(f"  Output:     {output_dir}")

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
