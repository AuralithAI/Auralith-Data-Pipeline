"""Tests for the CLI tokenizer-training commands.

Covers argument parsing, file-discovery logic, error handling,
success-message formatting, and integration with the tokenization classes.
"""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from auralith_pipeline.cli import (
    _VIDEO_EXTS,
    _train_all_modalities,
    _train_audio_vq,
    _train_image_vq,
    _train_text_bpe,
    _train_video_vq,
    main,
)


@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture()
def tmp_dir(tmp_path):
    """Provide a temporary directory (pytest's built-in tmp_path)."""
    return tmp_path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify module-level constants."""

    def test_video_exts_contains_all(self):
        expected = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        assert _VIDEO_EXTS == expected

    def test_video_exts_is_frozenset(self):
        assert isinstance(_VIDEO_EXTS, frozenset)


# ---------------------------------------------------------------------------
# train-tokenizer group
# ---------------------------------------------------------------------------


class TestTrainTokenizerGroup:
    """Test the train-tokenizer CLI group itself."""

    def test_group_help(self, runner):
        result = runner.invoke(main, ["train-tokenizer", "--help"])
        assert result.exit_code == 0
        assert "text" in result.output
        assert "image" in result.output
        assert "audio" in result.output
        assert "video" in result.output
        assert "all" in result.output

    def test_group_no_subcommand(self, runner):
        result = runner.invoke(main, ["train-tokenizer"])
        # Click shows help/usage when no subcommand is given.
        # Exit code varies by Click version (0 or 2), so just check output.
        assert result.exit_code in (0, 2)
        assert "Usage" in result.output or "train-tokenizer" in result.output


# ---------------------------------------------------------------------------
# text subcommand
# ---------------------------------------------------------------------------


class TestTextCommand:
    """Test the `train-tokenizer text` command."""

    def test_text_from_single_file(self, runner, tmp_dir):
        corpus_file = tmp_dir / "corpus.txt"
        corpus_file.write_text("the quick brown fox jumps over the lazy dog " * 100)
        out = tmp_dir / "bpe_out"

        result = runner.invoke(
            main,
            [
                "train-tokenizer",
                "text",
                "--corpus",
                str(corpus_file),
                "--output",
                str(out),
                "--vocab-size",
                "500",
            ],
        )
        assert result.exit_code == 0
        assert "[OK] Text BPE tokenizer saved to" in result.output
        assert "vocab=" in result.output
        assert "merges=" in result.output
        # Verify output files created
        assert out.exists()

    def test_text_from_directory(self, runner, tmp_dir):
        corpus_dir = tmp_dir / "corpus"
        corpus_dir.mkdir()
        for i in range(3):
            (corpus_dir / f"file_{i}.txt").write_text("hello world test data sample " * 50)
        out = tmp_dir / "bpe_out"

        result = runner.invoke(
            main,
            [
                "train-tokenizer",
                "text",
                "--corpus",
                str(corpus_dir),
                "--output",
                str(out),
                "--vocab-size",
                "500",
            ],
        )
        assert result.exit_code == 0
        assert "Training from 3 files" in result.output
        assert "[OK] Text BPE tokenizer saved to" in result.output

    def test_text_empty_directory_error(self, runner, tmp_dir):
        empty_dir = tmp_dir / "empty"
        empty_dir.mkdir()
        out = tmp_dir / "bpe_out"

        result = runner.invoke(
            main,
            [
                "train-tokenizer",
                "text",
                "--corpus",
                str(empty_dir),
                "--output",
                str(out),
            ],
        )
        assert result.exit_code != 0
        assert "No .txt files found" in result.output

    def test_text_max_corpus_size_truncates_single_file(self, runner, tmp_dir):
        corpus_file = tmp_dir / "big.txt"
        content = "abcdefghij" * 1000  # 10,000 chars
        corpus_file.write_text(content)
        out = tmp_dir / "bpe_out"

        result = runner.invoke(
            main,
            [
                "train-tokenizer",
                "text",
                "--corpus",
                str(corpus_file),
                "--output",
                str(out),
                "--vocab-size",
                "500",
                "--max-corpus-size",
                "500",
            ],
        )
        assert result.exit_code == 0
        assert "Truncating corpus from" in result.output

    def test_text_max_corpus_size_no_truncate_when_small(self, runner, tmp_dir):
        corpus_file = tmp_dir / "small.txt"
        corpus_file.write_text("hello world test " * 50)
        out = tmp_dir / "bpe_out"

        result = runner.invoke(
            main,
            [
                "train-tokenizer",
                "text",
                "--corpus",
                str(corpus_file),
                "--output",
                str(out),
                "--vocab-size",
                "500",
                "--max-corpus-size",
                "999999",
            ],
        )
        assert result.exit_code == 0
        assert "Truncating" not in result.output

    def test_text_missing_corpus_option(self, runner):
        result = runner.invoke(main, ["train-tokenizer", "text", "--output", "out"])
        assert result.exit_code != 0
        assert "Missing" in result.output or "required" in result.output.lower()

    def test_text_missing_output_option(self, runner, tmp_dir):
        corpus = tmp_dir / "c.txt"
        corpus.write_text("data")
        result = runner.invoke(main, ["train-tokenizer", "text", "--corpus", str(corpus)])
        assert result.exit_code != 0

    def test_text_lowercase_flag(self, runner, tmp_dir):
        corpus_file = tmp_dir / "corpus.txt"
        corpus_file.write_text("THE QUICK BROWN FOX " * 100)
        out = tmp_dir / "bpe_out"

        result = runner.invoke(
            main,
            [
                "train-tokenizer",
                "text",
                "--corpus",
                str(corpus_file),
                "--output",
                str(out),
                "--vocab-size",
                "500",
                "--lowercase",
            ],
        )
        assert result.exit_code == 0
        assert "[OK]" in result.output


# ---------------------------------------------------------------------------
# image subcommand
# ---------------------------------------------------------------------------


class TestImageCommand:
    """Test the `train-tokenizer image` command."""

    def _create_npy_images(self, directory: Path, count: int = 3):
        directory.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img = np.random.rand(224, 224, 3).astype(np.float32)
            np.save(directory / f"img_{i}.npy", img)

    def test_image_basic(self, runner, tmp_dir):
        img_dir = tmp_dir / "images"
        self._create_npy_images(img_dir)
        out = tmp_dir / "image_out"

        result = runner.invoke(
            main,
            [
                "train-tokenizer",
                "image",
                "--images",
                str(img_dir),
                "--output",
                str(out),
                "--codebook-size",
                "64",
            ],
        )
        assert result.exit_code == 0
        assert "[OK] Image VQ tokenizer saved to" in result.output
        assert "codebook=" in result.output
        assert "patches/img=" in result.output

    def test_image_empty_dir_error(self, runner, tmp_dir):
        empty = tmp_dir / "empty_imgs"
        empty.mkdir()
        out = tmp_dir / "image_out"

        result = runner.invoke(
            main,
            [
                "train-tokenizer",
                "image",
                "--images",
                str(empty),
                "--output",
                str(out),
            ],
        )
        assert result.exit_code != 0
        assert "No .npy image files found" in result.output

    def test_image_custom_sizes(self, runner, tmp_dir):
        img_dir = tmp_dir / "images"
        img_dir.mkdir()
        # Need enough patches (>= 64) to train VQ codebook.
        # 128/32 = 4 patches per side → 16 patches per image → need >= 4 images.
        for i in range(5):
            img = np.random.rand(128, 128, 3).astype(np.float32)
            np.save(img_dir / f"img_{i}.npy", img)
        out = tmp_dir / "image_out"

        result = runner.invoke(
            main,
            [
                "train-tokenizer",
                "image",
                "--images",
                str(img_dir),
                "--output",
                str(out),
                "--image-size",
                "128",
                "--patch-size",
                "32",
                "--codebook-size",
                "64",
            ],
        )
        assert result.exit_code == 0
        assert "128x128" in result.output
        assert "32x32" in result.output


# ---------------------------------------------------------------------------
# audio subcommand
# ---------------------------------------------------------------------------


class TestAudioCommand:
    """Test the `train-tokenizer audio` command."""

    def _create_npy_audio(self, directory: Path, count: int = 25):
        directory.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            # 1 second of random audio at 16kHz — need many files
            # because each produces only a few spectrogram patches.
            audio = np.random.rand(16000).astype(np.float32)
            np.save(directory / f"audio_{i}.npy", audio)

    def test_audio_basic(self, runner, tmp_dir):
        audio_dir = tmp_dir / "audio"
        self._create_npy_audio(audio_dir)
        out = tmp_dir / "audio_out"

        result = runner.invoke(
            main,
            [
                "train-tokenizer",
                "audio",
                "--audio",
                str(audio_dir),
                "--output",
                str(out),
                "--codebook-size",
                "64",
            ],
        )
        assert result.exit_code == 0
        assert "[OK] Audio VQ tokenizer saved to" in result.output
        assert "codebook=" in result.output

    def test_audio_empty_dir_error(self, runner, tmp_dir):
        empty = tmp_dir / "empty_audio"
        empty.mkdir()
        out = tmp_dir / "audio_out"

        result = runner.invoke(
            main,
            [
                "train-tokenizer",
                "audio",
                "--audio",
                str(empty),
                "--output",
                str(out),
            ],
        )
        assert result.exit_code != 0
        assert "No .npy audio files found" in result.output


# ---------------------------------------------------------------------------
# video subcommand
# ---------------------------------------------------------------------------


class TestVideoCommand:
    """Test the `train-tokenizer video` command."""

    def test_video_empty_dir_error(self, runner, tmp_dir):
        empty = tmp_dir / "empty_vids"
        empty.mkdir()
        out = tmp_dir / "video_out"

        result = runner.invoke(
            main,
            [
                "train-tokenizer",
                "video",
                "--videos",
                str(empty),
                "--output",
                str(out),
            ],
        )
        assert result.exit_code != 0
        assert "No video files found" in result.output

    def test_video_discovers_extensions(self, tmp_dir):
        """Verify that video file discovery respects _VIDEO_EXTS."""
        vid_dir = tmp_dir / "vids"
        vid_dir.mkdir()
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm", ".txt", ".npy"]:
            (vid_dir / f"file{ext}").touch()

        found = sorted(str(p) for p in vid_dir.rglob("*") if p.suffix.lower() in _VIDEO_EXTS)
        # Should find exactly 5 video files, ignore .txt and .npy
        assert len(found) == 5


# ---------------------------------------------------------------------------
# all subcommand
# ---------------------------------------------------------------------------


class TestTrainAllCommand:
    """Test the `train-tokenizer all` command."""

    def test_all_no_modalities_error(self, runner, tmp_dir):
        out = tmp_dir / "all_out"
        result = runner.invoke(
            main,
            ["train-tokenizer", "all", "--output", str(out)],
        )
        assert result.exit_code != 0
        assert "provide at least one" in result.output

    def test_all_text_only(self, runner, tmp_dir):
        corpus = tmp_dir / "corpus.txt"
        corpus.write_text("the quick brown fox jumps " * 100)
        out = tmp_dir / "all_out"

        result = runner.invoke(
            main,
            [
                "train-tokenizer",
                "all",
                "--corpus",
                str(corpus),
                "--output",
                str(out),
                "--vocab-size",
                "500",
            ],
        )
        assert result.exit_code == 0
        assert "Trained 1 tokenizer(s): text" in result.output
        # Should NOT show [1/4]; should show [1/1]
        assert "[1/1]" in result.output

    def test_all_skipped_modalities_message(self, runner, tmp_dir):
        corpus = tmp_dir / "corpus.txt"
        corpus.write_text("hello world " * 100)
        out = tmp_dir / "all_out"

        result = runner.invoke(
            main,
            [
                "train-tokenizer",
                "all",
                "--corpus",
                str(corpus),
                "--output",
                str(out),
                "--vocab-size",
                "500",
            ],
        )
        assert result.exit_code == 0
        # Skipped modalities show descriptive message, not numbered
        assert "Image VQ  →  skipped" in result.output
        assert "Audio VQ  →  skipped" in result.output
        assert "Video VQ  →  skipped" in result.output

    def test_all_help_shows_all_params(self, runner):
        result = runner.invoke(main, ["train-tokenizer", "all", "--help"])
        assert result.exit_code == 0
        assert "--vocab-size" in result.output
        assert "--codebook-size" in result.output
        assert "--audio-codebook-size" in result.output
        assert "--image-size" in result.output
        assert "--patch-size" in result.output
        assert "--sample-rate" in result.output
        assert "--max-frames" in result.output


# ---------------------------------------------------------------------------
# _train_all_modalities (dynamic progress)
# ---------------------------------------------------------------------------


class TestTrainAllModalities:
    """Test _train_all_modalities helper for dynamic step numbering."""

    def test_dynamic_step_text_only(self, tmp_dir):
        """When only --corpus is provided, steps should be [1/1]."""
        corpus = tmp_dir / "corpus.txt"
        corpus.write_text("data " * 500)

        result = _train_all_modalities(
            corpus=str(corpus),
            images=None,
            audio=None,
            videos=None,
            output_root=tmp_dir / "out",
            vocab_size=500,
            codebook_size=1024,
            audio_codebook_size=512,
            image_size=224,
            patch_size=16,
            sample_rate=16000,
            max_frames=32,
        )
        assert result == ["text"]


# ---------------------------------------------------------------------------
# _train_text_bpe helper
# ---------------------------------------------------------------------------


class TestTrainTextBpeHelper:
    """Test the _train_text_bpe helper function."""

    def test_single_file(self, tmp_dir):
        f = tmp_dir / "corpus.txt"
        f.write_text("hello world test data " * 100)
        out = tmp_dir / "text_out"

        result = _train_text_bpe(str(f), out, vocab_size=500, step=1, total=1)
        assert result == ["text"]
        assert out.exists()

    def test_directory_with_txt(self, tmp_dir):
        d = tmp_dir / "corpus_dir"
        d.mkdir()
        (d / "a.txt").write_text("data " * 200)
        (d / "b.txt").write_text("more data " * 200)
        out = tmp_dir / "text_out"

        result = _train_text_bpe(str(d), out, vocab_size=500, step=1, total=2)
        assert result == ["text"]

    def test_directory_no_txt_early_return(self, tmp_dir):
        d = tmp_dir / "no_txt"
        d.mkdir()
        (d / "data.csv").write_text("a,b,c")
        out = tmp_dir / "text_out"

        result = _train_text_bpe(str(d), out, vocab_size=500, step=1, total=1)
        assert result == []


# ---------------------------------------------------------------------------
# _train_image_vq helper
# ---------------------------------------------------------------------------


class TestTrainImageVqHelper:
    """Test the _train_image_vq helper function."""

    def test_with_images(self, tmp_dir):
        img_dir = tmp_dir / "images"
        img_dir.mkdir()
        for i in range(3):
            np.save(img_dir / f"img_{i}.npy", np.random.rand(224, 224, 3).astype(np.float32))
        out = tmp_dir / "image_out"

        result = _train_image_vq(
            str(img_dir), out, codebook_size=64, image_size=224, patch_size=16, step=1, total=1
        )
        assert result == ["image"]
        assert out.exists()

    def test_empty_dir(self, tmp_dir):
        empty = tmp_dir / "empty"
        empty.mkdir()
        out = tmp_dir / "image_out"

        result = _train_image_vq(
            str(empty), out, codebook_size=64, image_size=224, patch_size=16, step=1, total=1
        )
        assert result == []


# ---------------------------------------------------------------------------
# _train_audio_vq helper
# ---------------------------------------------------------------------------


class TestTrainAudioVqHelper:
    """Test the _train_audio_vq helper function."""

    def test_with_audio(self, tmp_dir):
        aud_dir = tmp_dir / "audio"
        aud_dir.mkdir()
        # Need enough spectrogram patches (>= 64) for VQ codebook training.
        for i in range(25):
            np.save(aud_dir / f"aud_{i}.npy", np.random.rand(16000).astype(np.float32))
        out = tmp_dir / "audio_out"

        result = _train_audio_vq(
            str(aud_dir), out, codebook_size=64, sample_rate=16000, step=1, total=1
        )
        assert result == ["audio"]
        assert out.exists()

    def test_empty_dir(self, tmp_dir):
        empty = tmp_dir / "empty"
        empty.mkdir()
        out = tmp_dir / "audio_out"

        result = _train_audio_vq(
            str(empty), out, codebook_size=64, sample_rate=16000, step=1, total=1
        )
        assert result == []


# ---------------------------------------------------------------------------
# _train_video_vq helper
# ---------------------------------------------------------------------------


class TestTrainVideoVqHelper:
    """Test the _train_video_vq helper function."""

    def test_empty_dir(self, tmp_dir):
        empty = tmp_dir / "empty"
        empty.mkdir()
        out = tmp_dir / "video_out"

        result = _train_video_vq(
            str(empty),
            out,
            codebook_size=64,
            image_size=224,
            patch_size=16,
            max_frames=32,
            step=1,
            total=1,
        )
        assert result == []


# ---------------------------------------------------------------------------
# Success message format consistency
# ---------------------------------------------------------------------------


class TestSuccessMessageFormat:
    """Verify that success messages use the standardized format."""

    def test_text_success_format(self, runner, tmp_dir):
        f = tmp_dir / "corpus.txt"
        f.write_text("the quick brown fox jumps " * 100)
        out = tmp_dir / "bpe_out"

        result = runner.invoke(
            main,
            [
                "train-tokenizer",
                "text",
                "--corpus",
                str(f),
                "--output",
                str(out),
                "--vocab-size",
                "500",
            ],
        )
        assert result.exit_code == 0
        # Format: [OK] Text BPE tokenizer saved to <path>
        assert "[OK] Text BPE tokenizer saved to" in result.output
        # Summary line: vocab=..., merges=...
        assert "vocab=" in result.output
        assert "merges=" in result.output

    def test_image_success_format(self, runner, tmp_dir):
        img_dir = tmp_dir / "images"
        img_dir.mkdir()
        np.save(img_dir / "img.npy", np.random.rand(224, 224, 3).astype(np.float32))
        out = tmp_dir / "image_out"

        result = runner.invoke(
            main,
            [
                "train-tokenizer",
                "image",
                "--images",
                str(img_dir),
                "--output",
                str(out),
                "--codebook-size",
                "64",
            ],
        )
        assert result.exit_code == 0
        assert "[OK] Image VQ tokenizer saved to" in result.output
        assert "codebook=" in result.output
        assert "patches/img=" in result.output

    def test_audio_success_format(self, runner, tmp_dir):
        aud_dir = tmp_dir / "audio"
        aud_dir.mkdir()
        for i in range(25):
            np.save(aud_dir / f"aud_{i}.npy", np.random.rand(16000).astype(np.float32))
        out = tmp_dir / "audio_out"

        result = runner.invoke(
            main,
            [
                "train-tokenizer",
                "audio",
                "--audio",
                str(aud_dir),
                "--output",
                str(out),
                "--codebook-size",
                "64",
            ],
        )
        assert result.exit_code == 0
        assert "[OK] Audio VQ tokenizer saved to" in result.output
        assert "codebook=" in result.output
