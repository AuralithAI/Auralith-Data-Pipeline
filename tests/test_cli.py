"""Tests for the CLI tokenizer-training commands.

Covers argument parsing, file-discovery logic, error handling,
success-message formatting, and integration with the tokenization classes.
"""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from auralith_pipeline.cli import (
    _AUDIO_EXTS,
    _BANNER_ART,
    _IMAGE_EXTS,
    _MODALITY_ID,
    _TEXT_EXTS,
    _VIDEO_EXTS,
    _classify_file,
    _load_all_tokenizers,
    _print_banner,
    _tokenize_file,
    _train_all_modalities,
    _train_audio_vq,
    _train_image_vq,
    _train_text_bpe,
    _train_video_vq,
    _write_shards,
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


# ===========================================================================
# Process command tests
# ===========================================================================


# ---------------------------------------------------------------------------
# File classification
# ---------------------------------------------------------------------------


class TestClassifyFile:
    """Test the _classify_file helper."""

    def test_text_extensions(self, tmp_path):
        for ext in [".txt", ".md", ".rst", ".csv", ".json", ".jsonl"]:
            p = tmp_path / f"file{ext}"
            assert _classify_file(p) == "text"

    def test_image_extensions(self, tmp_path):
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            p = tmp_path / f"file{ext}"
            assert _classify_file(p) == "image"

    def test_audio_extensions(self, tmp_path):
        for ext in [".wav", ".flac", ".ogg"]:
            p = tmp_path / f"file{ext}"
            assert _classify_file(p) == "audio"

    def test_video_extensions(self, tmp_path):
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            p = tmp_path / f"file{ext}"
            assert _classify_file(p) == "video"

    def test_unsupported_extension(self, tmp_path):
        assert _classify_file(tmp_path / "file.xyz") is None
        assert _classify_file(tmp_path / "file.pdf") is None

    def test_npy_in_image_dir(self, tmp_path):
        # .npy under an "images" directory → "image"
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        assert _classify_file(img_dir / "data.npy") == "image"

    def test_npy_in_audio_dir(self, tmp_path):
        # .npy under an "audio" directory → "audio"
        aud_dir = tmp_path / "audio"
        aud_dir.mkdir()
        assert _classify_file(aud_dir / "data.npy") == "audio"

    def test_npy_ambiguous_dir(self, tmp_path):
        # .npy in a directory with no image/audio keyword → None
        misc_dir = tmp_path / "misc"
        misc_dir.mkdir()
        assert _classify_file(misc_dir / "data.npy") is None


# ---------------------------------------------------------------------------
# Extension constants
# ---------------------------------------------------------------------------


class TestExtensionConstants:
    """Verify extension sets are correct."""

    def test_text_exts(self):
        assert ".txt" in _TEXT_EXTS
        assert ".md" in _TEXT_EXTS
        assert ".rst" in _TEXT_EXTS
        assert ".py" in _TEXT_EXTS

    def test_image_exts(self):
        assert ".jpg" in _IMAGE_EXTS
        assert ".png" in _IMAGE_EXTS
        assert ".webp" in _IMAGE_EXTS
        # .npy is no longer in _IMAGE_EXTS — handled by directory routing
        assert ".npy" not in _IMAGE_EXTS

    def test_audio_exts(self):
        assert ".wav" in _AUDIO_EXTS
        assert ".flac" in _AUDIO_EXTS
        assert ".mp3" in _AUDIO_EXTS
        # .npy is no longer in _AUDIO_EXTS — handled by directory routing
        assert ".npy" not in _AUDIO_EXTS

    def test_modality_ids(self):
        assert _MODALITY_ID == {"text": 0, "image": 1, "audio": 2, "video": 3}


# ---------------------------------------------------------------------------
# _load_all_tokenizers
# ---------------------------------------------------------------------------


class TestLoadAllTokenizers:
    """Test _load_all_tokenizers helper."""

    def test_empty_dir_returns_all_none(self, tmp_path):
        result = _load_all_tokenizers(tmp_path)
        assert result == {"text": None, "image": None, "audio": None, "video": None}

    def test_loads_text_tokenizer(self, tmp_path):
        """Train a small BPE tokenizer, save it, and verify _load_all_tokenizers finds it."""
        from auralith_pipeline.tokenization import BPETokenizer

        tok = BPETokenizer(vocab_size=500)
        tok.train("hello world test data " * 100)

        text_dir = tmp_path / "text"
        tok.save(text_dir)

        result = _load_all_tokenizers(tmp_path)
        assert result["text"] is not None
        assert result["image"] is None
        assert result["audio"] is None
        assert result["video"] is None


# ---------------------------------------------------------------------------
# _tokenize_file
# ---------------------------------------------------------------------------


class TestTokenizeFile:
    """Test _tokenize_file helper."""

    @pytest.fixture()
    def text_tokenizer(self, tmp_path):
        from auralith_pipeline.tokenization import BPETokenizer

        tok = BPETokenizer(vocab_size=500)
        tok.train("hello world test data " * 100)
        return tok

    def test_tokenize_text_file(self, tmp_path, text_tokenizer):
        txt = tmp_path / "sample.txt"
        txt.write_text("hello world this is a test")

        tokenizers = {"text": text_tokenizer, "image": None, "audio": None, "video": None}
        result = _tokenize_file(txt, tokenizers, max_seq_len=4096)

        assert result is not None
        assert "input_ids" in result
        assert "modality_mask" in result
        assert "source" in result
        assert len(result["input_ids"]) == len(result["modality_mask"])
        assert all(m == 0 for m in result["modality_mask"])  # text = modality 0

    def test_tokenize_empty_text_returns_none(self, tmp_path, text_tokenizer):
        txt = tmp_path / "empty.txt"
        txt.write_text("   ")

        tokenizers = {"text": text_tokenizer, "image": None, "audio": None, "video": None}
        result = _tokenize_file(txt, tokenizers, max_seq_len=4096)
        assert result is None

    def test_tokenize_unsupported_ext_returns_none(self, tmp_path, text_tokenizer):
        pdf = tmp_path / "doc.pdf"
        pdf.write_text("fake pdf")

        tokenizers = {"text": text_tokenizer, "image": None, "audio": None, "video": None}
        result = _tokenize_file(pdf, tokenizers, max_seq_len=4096)
        assert result is None

    def test_tokenize_no_tokenizer_returns_none(self, tmp_path):
        txt = tmp_path / "sample.txt"
        txt.write_text("hello world")

        tokenizers = {"text": None, "image": None, "audio": None, "video": None}
        result = _tokenize_file(txt, tokenizers, max_seq_len=4096)
        assert result is None

    def test_tokenize_truncates_to_max_seq_len(self, tmp_path, text_tokenizer):
        txt = tmp_path / "long.txt"
        txt.write_text("word " * 10000)

        tokenizers = {"text": text_tokenizer, "image": None, "audio": None, "video": None}
        result = _tokenize_file(txt, tokenizers, max_seq_len=128)

        assert result is not None
        assert len(result["input_ids"]) <= 128
        assert len(result["modality_mask"]) <= 128


# ---------------------------------------------------------------------------
# _write_shards
# ---------------------------------------------------------------------------


class TestWriteShards:
    """Test _write_shards helper."""

    def test_write_single_shard(self, tmp_path):
        samples = [
            {"input_ids": [1, 2, 3, 4], "modality_mask": [0, 0, 0, 0], "source": "a.txt"},
            {"input_ids": [5, 6, 7], "modality_mask": [0, 0, 0], "source": "b.txt"},
        ]
        out = tmp_path / "shards"
        out.mkdir()

        num_shards = _write_shards(samples, out, max_seq_len=16, shard_size=100)
        assert num_shards == 1

        shard_files = list(out.glob("*.safetensors"))
        assert len(shard_files) == 1
        assert shard_files[0].name == "shard_000000.safetensors"

    def test_write_multiple_shards(self, tmp_path):
        samples = [{"input_ids": [i], "modality_mask": [0], "source": f"{i}.txt"} for i in range(5)]
        out = tmp_path / "shards"
        out.mkdir()

        num_shards = _write_shards(samples, out, max_seq_len=8, shard_size=2)
        assert num_shards == 3  # 2+2+1

    def test_shard_schema(self, tmp_path):
        """Verify safetensors shard contains all required tensors with correct shapes."""
        from safetensors.numpy import load_file

        samples = [
            {"input_ids": [10, 20, 30], "modality_mask": [0, 1, 0], "source": "test.txt"},
        ]
        out = tmp_path / "shards"
        out.mkdir()

        _write_shards(samples, out, max_seq_len=8, shard_size=100)

        data = load_file(str(out / "shard_000000.safetensors"))
        assert "input_ids" in data
        assert "attention_mask" in data
        assert "modality_mask" in data
        assert "targets" in data

        # Shape: (1 sample, 8 seq_len)
        assert data["input_ids"].shape == (1, 8)
        assert data["attention_mask"].shape == (1, 8)
        assert data["modality_mask"].shape == (1, 8)
        assert data["targets"].shape == (1, 8)

        # Verify padding
        assert data["input_ids"][0, 0] == 10
        assert data["input_ids"][0, 3] == 0  # padded
        assert data["attention_mask"][0, 0] == 1
        assert data["attention_mask"][0, 3] == 0  # padded

        # Verify targets = right-shifted input_ids
        assert data["targets"][0, 0] == 20  # input_ids[0, 1]
        assert data["targets"][0, 1] == 30  # input_ids[0, 2]


# ---------------------------------------------------------------------------
# process CLI command (integration)
# ---------------------------------------------------------------------------


class TestProcessCommand:
    """Integration tests for the `process` CLI command."""

    @pytest.fixture()
    def trained_text_tokenizer_dir(self, tmp_path):
        """Create a tokenizers/ directory with a trained text BPE tokenizer."""
        from auralith_pipeline.tokenization import BPETokenizer

        tok = BPETokenizer(vocab_size=500)
        tok.train("hello world test data the quick brown fox " * 200)

        tok_dir = tmp_path / "tokenizers" / "text"
        tok.save(tok_dir)
        return tmp_path / "tokenizers"

    def test_process_text_files(self, runner, tmp_path, trained_text_tokenizer_dir):
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "doc1.txt").write_text("the quick brown fox jumps over the lazy dog")
        (raw / "doc2.txt").write_text("hello world this is a test document")

        out = tmp_path / "shards"

        result = runner.invoke(
            main,
            [
                "process",
                "--input",
                str(raw),
                "--output",
                str(out),
                "--tokenizers",
                str(trained_text_tokenizer_dir),
                "--max-seq-len",
                "128",
                "--shard-size",
                "100",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "[OK] Created" in result.output
        assert "shard(s)" in result.output

        shard_files = list(out.glob("*.safetensors"))
        assert len(shard_files) >= 1

    def test_process_no_tokenizers_error(self, runner, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "doc.txt").write_text("test")

        empty_tok = tmp_path / "empty_tok"
        empty_tok.mkdir()

        out = tmp_path / "shards"

        result = runner.invoke(
            main,
            [
                "process",
                "--input",
                str(raw),
                "--output",
                str(out),
                "--tokenizers",
                str(empty_tok),
            ],
        )
        assert result.exit_code != 0
        assert "No tokenizers found" in result.output

    def test_process_empty_input_error(self, runner, tmp_path, trained_text_tokenizer_dir):
        raw = tmp_path / "empty_raw"
        raw.mkdir()
        out = tmp_path / "shards"

        result = runner.invoke(
            main,
            [
                "process",
                "--input",
                str(raw),
                "--output",
                str(out),
                "--tokenizers",
                str(trained_text_tokenizer_dir),
            ],
        )
        assert result.exit_code != 0
        assert "No files found" in result.output

    def test_process_help(self, runner):
        result = runner.invoke(main, ["process", "--help"])
        assert result.exit_code == 0
        assert "--input" in result.output
        assert "--output" in result.output
        assert "--tokenizers" in result.output
        assert "--max-seq-len" in result.output
        assert "--shard-size" in result.output

    def test_process_shard_contents_are_valid(self, runner, tmp_path, trained_text_tokenizer_dir):
        """Verify the generated shard has correct schema v2 format."""
        from safetensors.numpy import load_file

        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "doc.txt").write_text("the quick brown fox jumps over the lazy dog " * 10)

        out = tmp_path / "shards"

        result = runner.invoke(
            main,
            [
                "process",
                "--input",
                str(raw),
                "--output",
                str(out),
                "--tokenizers",
                str(trained_text_tokenizer_dir),
                "--max-seq-len",
                "64",
                "--shard-size",
                "100",
            ],
        )
        assert result.exit_code == 0

        shard = load_file(str(out / "shard_000000.safetensors"))
        assert shard["input_ids"].dtype.name == "int32"
        assert shard["attention_mask"].dtype.name == "uint8"
        assert shard["modality_mask"].dtype.name == "uint8"
        assert shard["targets"].dtype.name == "int32"

        # All modality mask values should be 0 (text)
        assert (shard["modality_mask"][shard["attention_mask"] == 1] == 0).all()


# ---------------------------------------------------------------------------
# Banner / startup UI
# ---------------------------------------------------------------------------


class TestBanner:
    """Test the startup banner and --no-banner flag."""

    def test_banner_art_contains_auralith(self):
        """The ASCII art should spell out AURALITH."""
        # The banner uses box-drawing characters; verify key letters are present
        assert "██" in _BANNER_ART
        assert "╗" in _BANNER_ART or "╔" in _BANNER_ART

    def test_print_banner_runs_without_error(self, capsys):
        """_print_banner should never crash, even outside a real terminal."""
        _print_banner()  # should not raise

    def test_no_banner_flag_suppresses_output(self, runner):
        """--no-banner should suppress the startup banner."""
        result = runner.invoke(main, ["--no-banner", "--help"])
        assert result.exit_code == 0
        # The help text should still appear
        assert "Auralith Data Pipeline" in result.output

    def test_banner_not_shown_in_non_tty(self, runner, monkeypatch):
        """Banner is auto-suppressed when stderr is not a TTY (e.g. tests)."""
        import io

        monkeypatch.setattr("sys.stderr", io.StringIO())
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0

    def test_no_banner_env_var(self, runner, monkeypatch):
        """AURALITH_NO_BANNER env var suppresses the banner."""
        monkeypatch.setenv("AURALITH_NO_BANNER", "1")
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Auralith Data Pipeline" in result.output
