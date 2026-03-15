#!/usr/bin/env python3
"""Download sample training data for tokenizer training.

This script downloads small datasets for each modality and saves them
in the format expected by `auralith-pipeline train-tokenizer`.

Usage:
    python scripts/download_training_data.py                # Download all modalities
    python scripts/download_training_data.py --text-only     # Text only (fastest, ~2 min)
    python scripts/download_training_data.py --no-video       # Skip video (saves time)
    python scripts/download_training_data.py --output data/raw  # Custom output dir

Output structure:
    data/raw/
    ├── corpus/          # .txt files for BPE training
    ├── images/          # .npy arrays (H, W, 3) uint8
    ├── audio/           # .npy arrays (1D float32 waveforms)
    └── videos/          # .mp4 files
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def download_text(output_dir: Path, max_samples: int = 50_000) -> None:
    """Download text corpus from Wikitext-103 (small, ~180MB raw text).

    This is the easiest text dataset -- no auth needed, fast download.
    """
    from datasets import load_dataset

    text_dir = output_dir / "corpus"
    text_dir.mkdir(parents=True, exist_ok=True)

    output_file = text_dir / "wikitext103.txt"
    if output_file.exists():
        logger.info(f"Text corpus already exists: {output_file}")
        return

    logger.info("Downloading Wikitext-103 from HuggingFace...")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")

    logger.info(f"Dataset has {len(ds)} rows, taking up to {max_samples}...")

    lines = []
    count = 0
    for row in ds:
        text = row["text"].strip()
        if len(text) > 50:  # Skip empty/short lines
            lines.append(text)
            count += 1
            if count >= max_samples:
                break

    corpus = "\n".join(lines)
    output_file.write_text(corpus, encoding="utf-8")
    logger.info(
        f"Saved {count:,} text samples to {output_file} " f"({len(corpus) / 1_000_000:.1f} MB)"
    )


def download_images(output_dir: Path, max_samples: int = 2_000) -> None:
    """Download images from CIFAR-10 and save as .npy arrays.

    CIFAR-10 images are 32x32x3 uint8 -- small but sufficient for
    VQ codebook training. No auth needed.
    """
    from datasets import load_dataset

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing = list(image_dir.glob("*.npy"))
    if len(existing) >= max_samples:
        logger.info(f"Images already downloaded: {len(existing)} files in {image_dir}")
        return

    logger.info("Downloading CIFAR-10 images from HuggingFace...")
    ds = load_dataset("uoft-cs/cifar10", split="train")

    count = 0
    for i, row in enumerate(ds):
        if count >= max_samples:
            break

        img = row["img"]  # PIL Image

        # Resize to 224x224 for realistic patch tokenization
        from PIL import Image

        img_resized = img.resize((224, 224), Image.LANCZOS)
        arr_resized = np.array(img_resized, dtype=np.uint8)  # (224, 224, 3)

        out_path = image_dir / f"img_{i:06d}.npy"
        np.save(out_path, arr_resized)
        count += 1

        if count % 500 == 0:
            logger.info(f"  Saved {count}/{max_samples} images...")

    logger.info(f"Saved {count:,} images to {image_dir}")


def download_audio(output_dir: Path, max_samples: int = 1_000) -> None:
    """Download audio from LibriSpeech and save as .npy waveform arrays.

    Each .npy file is a 1D float32 array (raw waveform at 16kHz).
    """
    from datasets import load_dataset

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    existing = list(audio_dir.glob("*.npy"))
    if len(existing) >= max_samples:
        logger.info(f"Audio already downloaded: {len(existing)} files in {audio_dir}")
        return

    logger.info("Downloading LibriSpeech (clean-100) from HuggingFace...")
    ds = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split="train.100",
        streaming=True,
        trust_remote_code=True,
    )

    count = 0
    for row in ds:
        if count >= max_samples:
            break

        audio = row["audio"]
        waveform = np.array(audio["array"], dtype=np.float32)

        # Resample to 16kHz if needed
        sr = audio["sampling_rate"]
        if sr != 16000:
            from scipy.signal import resample

            num_samples = int(len(waveform) * 16000 / sr)
            waveform = resample(waveform, num_samples).astype(np.float32)

        out_path = audio_dir / f"audio_{count:06d}.npy"
        np.save(out_path, waveform)
        count += 1

        if count % 200 == 0:
            logger.info(f"  Saved {count}/{max_samples} audio files...")

    logger.info(f"Saved {count:,} audio waveforms to {audio_dir}")


def _save_video_row(video_data: object, out_path: Path) -> bool:
    """Save a single video row to disk. Returns True on success."""
    if isinstance(video_data, bytes):
        out_path.write_bytes(video_data)
        return True
    if isinstance(video_data, dict) and "bytes" in video_data:
        out_path.write_bytes(video_data["bytes"])
        return True
    if isinstance(video_data, dict) and "path" in video_data:
        import shutil

        shutil.copy2(video_data["path"], out_path)
        return True
    return False


def download_videos(output_dir: Path, max_samples: int = 50) -> None:
    """Download sample videos from HuggingFace.

    Downloads small .mp4 clips. Video is the heaviest modality,
    so we default to just 50 clips.
    """
    from datasets import load_dataset

    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    existing = list(video_dir.glob("*.mp4"))
    if len(existing) >= max_samples:
        logger.info(f"Videos already downloaded: {len(existing)} files in {video_dir}")
        return

    logger.info("Downloading sample videos from HuggingFace...")
    logger.info("(This may take a while depending on dataset availability)")

    try:
        # Use a small video dataset
        ds = load_dataset(
            "nateraw/kinetics",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        count = 0
        for row in ds:
            if count >= max_samples:
                break

            video_data = row.get("video")
            if video_data is None:
                continue

            out_path = video_dir / f"video_{count:06d}.mp4"
            if not _save_video_row(video_data, out_path):
                logger.warning(f"  Skipping row {count}: unexpected video format")
                continue

            count += 1
            if count % 10 == 0:
                logger.info(f"  Saved {count}/{max_samples} videos...")

        logger.info(f"Saved {count} videos to {video_dir}")

    except Exception as e:
        logger.warning(f"Video download failed: {e}")
        logger.info(
            "Video datasets often require special access. You can:\n"
            "  1. Manually place .mp4 files in data/raw/videos/\n"
            "  2. Skip video and train other modalities first:\n"
            "     auralith-pipeline train-tokenizer all --corpus data/raw/corpus/ "
            "--images data/raw/images/ --audio data/raw/audio/ --output tokenizers/"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download training data for tokenizer training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "After downloading, train tokenizers with:\n\n"
            "  auralith-pipeline train-tokenizer all \\\n"
            "    --corpus  data/raw/corpus/ \\\n"
            "    --images  data/raw/images/ \\\n"
            "    --audio   data/raw/audio/ \\\n"
            "    --videos  data/raw/videos/ \\\n"
            "    --output  tokenizers/ \\\n"
            "    --vocab-size 32000 \\\n"
            "    --codebook-size 1024 \\\n"
            "    --audio-codebook-size 512\n"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/raw",
        help="Output directory (default: data/raw)",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Download text corpus only (fastest, ~2 min)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video download (video datasets can be slow)",
    )
    parser.add_argument(
        "--text-samples",
        type=int,
        default=50_000,
        help="Max text samples (default: 50000)",
    )
    parser.add_argument(
        "--image-samples",
        type=int,
        default=2_000,
        help="Max image samples (default: 2000)",
    )
    parser.add_argument(
        "--audio-samples",
        type=int,
        default=1_000,
        help="Max audio samples (default: 1000)",
    )
    parser.add_argument(
        "--video-samples",
        type=int,
        default=50,
        help="Max video samples (default: 50)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Downloading Training Data for Tokenizer Training")
    logger.info(f"Output directory: {output_dir.resolve()}")
    logger.info("=" * 60)

    # Always download text
    download_text(output_dir, max_samples=args.text_samples)

    if not args.text_only:
        download_images(output_dir, max_samples=args.image_samples)
        download_audio(output_dir, max_samples=args.audio_samples)

        if not args.no_video:
            download_videos(output_dir, max_samples=args.video_samples)
        else:
            logger.info("Skipping video download (--no-video)")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Download complete!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next step -- train tokenizers:")
    logger.info("")
    if args.text_only:
        logger.info(
            "  auralith-pipeline train-tokenizer text \\\n"
            f"    --corpus {output_dir}/corpus/ \\\n"
            "    --output tokenizers/text \\\n"
            "    --vocab-size 32000"
        )
    else:
        cmd_parts = [
            "  auralith-pipeline train-tokenizer all \\",
            f"    --corpus  {output_dir}/corpus/ \\",
            f"    --images  {output_dir}/images/ \\",
            f"    --audio   {output_dir}/audio/ \\",
        ]
        if not args.no_video:
            cmd_parts.append(f"    --videos  {output_dir}/videos/ \\")
        cmd_parts += [
            "    --output  tokenizers/ \\",
            "    --vocab-size 32000 \\",
            "    --codebook-size 1024 \\",
            "    --audio-codebook-size 512",
        ]
        logger.info("\n".join(cmd_parts))

    logger.info("")
    logger.info("Then process into shards:")
    logger.info("")
    logger.info(
        "  auralith-pipeline process \\\n"
        f"    --input {output_dir}/ \\\n"
        "    --output shards/ \\\n"
        "    --tokenizers tokenizers/ \\\n"
        "    --max-seq-len 4096"
    )


if __name__ == "__main__":
    main()
