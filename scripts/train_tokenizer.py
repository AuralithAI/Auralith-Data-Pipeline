"""CLI commands for training tokenizers."""

import logging
from pathlib import Path

import click

from auralith_pipeline.tokenization import (
    AudioTokenizer,
    BPETokenizer,
    ImageTokenizer,
)

logger = logging.getLogger(__name__)


@click.group()
def train_tokenizer():
    """Train custom tokenizers for text and multimodal data."""
    pass


@train_tokenizer.command()
@click.option(
    "--corpus",
    type=click.Path(exists=True),
    required=True,
    help="Path to text file or directory containing training corpus",
)
@click.option(
    "--output",
    type=click.Path(),
    required=True,
    help="Output directory to save trained tokenizer",
)
@click.option(
    "--vocab-size",
    type=int,
    default=32000,
    help="Target vocabulary size (default: 32,000)",
)
@click.option(
    "--min-frequency",
    type=int,
    default=2,
    help="Minimum pair frequency for merging (default: 2)",
)
@click.option(
    "--lowercase",
    is_flag=True,
    help="Lowercase all text during training",
)
@click.option(
    "--max-corpus-size",
    type=int,
    default=None,
    help="Maximum corpus size in characters (for sampling large corpora)",
)
def text(corpus, output, vocab_size, min_frequency, lowercase, max_corpus_size):
    """Train BPE tokenizer on text corpus.

    Example:

        auralith-pipeline train-tokenizer text --corpus data/train.txt --output tokenizer/ --vocab-size 32000
    """
    logger.info("=" * 80)
    logger.info("Training BPE Text Tokenizer")
    logger.info("=" * 80)

    corpus_path = Path(corpus)
    output_path = Path(output)

    # Initialize tokenizer
    tokenizer = BPETokenizer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        lowercase=lowercase,
    )

    # Load corpus
    if corpus_path.is_file():
        logger.info(f"Training from single file: {corpus_path}")
        tokenizer.train_from_file(corpus_path)
    elif corpus_path.is_dir():
        # Find all text files
        text_files = list(corpus_path.rglob("*.txt"))
        logger.info(f"Found {len(text_files)} text files in {corpus_path}")

        if not text_files:
            logger.error("No .txt files found in directory!")
            return

        tokenizer.train_from_files(text_files, max_size=max_corpus_size)
    else:
        logger.error(f"Corpus path not found: {corpus_path}")
        return

    # Save tokenizer
    tokenizer.save(output_path)

    logger.info("=" * 80)
    logger.info(" Tokenizer training complete!")
    logger.info(f"  Vocabulary size: {tokenizer.get_vocab_size()}")
    logger.info(f"  Merge rules: {len(tokenizer.merge_rules)}")
    logger.info(f"  Saved to: {output_path}")
    logger.info("=" * 80)


@train_tokenizer.command()
@click.option(
    "--images",
    type=click.Path(exists=True),
    required=True,
    help="Path to directory containing training images (.npy format)",
)
@click.option(
    "--output",
    type=click.Path(),
    required=True,
    help="Output directory to save trained image tokenizer",
)
@click.option(
    "--codebook-size",
    type=int,
    default=1024,
    help="VQ codebook size (default: 1024)",
)
@click.option(
    "--image-size",
    type=int,
    default=224,
    help="Target image size (default: 224)",
)
@click.option(
    "--patch-size",
    type=int,
    default=16,
    help="Patch size (default: 16)",
)
@click.option(
    "--sample-size",
    type=int,
    default=None,
    help="Number of images to sample for training",
)
def image(images, output, codebook_size, image_size, patch_size, sample_size):
    """Train image tokenizer on image dataset.

    Example:

        auralith-pipeline train-tokenizer image --images data/images/ --output image_tokenizer/ --codebook-size 1024
    """
    logger.info("=" * 80)
    logger.info("Training Image Tokenizer")
    logger.info("=" * 80)

    images_path = Path(images)
    output_path = Path(output)

    # Find image files
    image_files = list(images_path.rglob("*.npy"))
    logger.info(f"Found {len(image_files)} image files")

    if not image_files:
        logger.error("No .npy image files found!")
        logger.info("Note: Images must be preprocessed to .npy format")
        return

    # Initialize tokenizer
    tokenizer = ImageTokenizer(
        image_size=image_size,
        patch_size=patch_size,
        codebook_size=codebook_size,
    )

    # Train
    tokenizer.train(image_files, sample_size=sample_size)

    # Save
    tokenizer.save(output_path)

    logger.info("=" * 80)
    logger.info(" Image tokenizer training complete!")
    logger.info(f"  Codebook size: {codebook_size}")
    logger.info(f"  Image size: {image_size}x{image_size}")
    logger.info(f"  Patch size: {patch_size}x{patch_size}")
    logger.info(f"  Tokens per image: {tokenizer.num_patches}")
    logger.info(f"  Saved to: {output_path}")
    logger.info("=" * 80)


@train_tokenizer.command()
@click.option(
    "--audio",
    type=click.Path(exists=True),
    required=True,
    help="Path to directory containing training audio (.npy format)",
)
@click.option(
    "--output",
    type=click.Path(),
    required=True,
    help="Output directory to save trained audio tokenizer",
)
@click.option(
    "--codebook-size",
    type=int,
    default=512,
    help="VQ codebook size (default: 512)",
)
@click.option(
    "--sample-rate",
    type=int,
    default=16000,
    help="Sample rate (default: 16000 Hz)",
)
@click.option(
    "--sample-size",
    type=int,
    default=None,
    help="Number of audio files to sample for training",
)
def audio(audio, output, codebook_size, sample_rate, sample_size):
    """Train audio tokenizer on audio dataset.

    Example:

        auralith-pipeline train-tokenizer audio --audio data/audio/ --output audio_tokenizer/ --codebook-size 512
    """
    logger.info("=" * 80)
    logger.info("Training Audio Tokenizer")
    logger.info("=" * 80)

    audio_path = Path(audio)
    output_path = Path(output)

    # Find audio files
    audio_files = list(audio_path.rglob("*.npy"))
    logger.info(f"Found {len(audio_files)} audio files")

    if not audio_files:
        logger.error("No .npy audio files found!")
        logger.info("Note: Audio must be preprocessed to .npy format")
        return

    # Initialize tokenizer
    tokenizer = AudioTokenizer(
        sample_rate=sample_rate,
        codebook_size=codebook_size,
    )

    # Train
    tokenizer.train(audio_files, sample_size=sample_size)

    # Save
    tokenizer.save(output_path)

    logger.info("=" * 80)
    logger.info(" Audio tokenizer training complete!")
    logger.info(f"  Codebook size: {codebook_size}")
    logger.info(f"  Sample rate: {sample_rate} Hz")
    logger.info(f"  Saved to: {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    train_tokenizer()
