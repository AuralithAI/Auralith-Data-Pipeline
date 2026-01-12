"""Unit tests for tokenization modules."""

import tempfile
from pathlib import Path

import numpy as np

from auralith_pipeline.tokenization import (
    AudioTokenizer,
    BPETokenizer,
    ImageTokenizer,
    MultimodalTokenizer,
    TokenizedSample,
    Tokenizer,
    VectorQuantizer,
)


class TestBPETokenizer:
    """Tests for BPE tokenizer."""

    def test_initialization(self):
        tokenizer = BPETokenizer(vocab_size=1000, min_frequency=2)
        assert tokenizer.vocab_size == 1000
        assert tokenizer.min_frequency == 2
        assert len(tokenizer.SPECIAL_TOKENS) == 10
        assert tokenizer.pad_token_id == 0
        assert tokenizer.unk_token_id == 1

    def test_training(self):
        corpus = "the quick brown fox jumps over the lazy dog " * 50
        tokenizer = BPETokenizer(vocab_size=200, min_frequency=2)
        tokenizer.train(corpus, verbose=False)

        assert len(tokenizer.vocab) > 10
        assert len(tokenizer.merge_rules) > 0
        assert tokenizer.get_vocab_size() <= 200

    def test_encode_decode(self):
        corpus = "hello world this is a test " * 100
        tokenizer = BPETokenizer(vocab_size=150, min_frequency=2)
        tokenizer.train(corpus, verbose=False)

        text = "hello world"
        token_ids = tokenizer.encode(text, add_special_tokens=True)

        assert isinstance(token_ids, list)
        assert len(token_ids) > 0
        assert token_ids[0] == tokenizer.bos_token_id
        assert token_ids[-1] == tokenizer.eos_token_id

        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert isinstance(decoded, str)
        assert "hello" in decoded.lower()

    def test_special_tokens(self):
        tokenizer = BPETokenizer(vocab_size=100)
        corpus = "test text"
        tokenizer.train(corpus, verbose=False)

        token_ids = tokenizer.encode("test", add_special_tokens=True)
        assert tokenizer.bos_token_id in token_ids
        assert tokenizer.eos_token_id in token_ids

    def test_truncation(self):
        corpus = "word " * 100
        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.train(corpus, verbose=False)

        long_text = "word " * 1000
        token_ids = tokenizer.encode(long_text, max_length=50)
        assert len(token_ids) <= 50

    def test_save_load(self):
        corpus = "test corpus for save and load " * 50
        tokenizer = BPETokenizer(vocab_size=150, min_frequency=2)
        tokenizer.train(corpus, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "tokenizer"
            tokenizer.save(save_path)

            loaded_tokenizer = BPETokenizer.load(save_path)

            assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
            assert len(loaded_tokenizer.vocab) == len(tokenizer.vocab)
            assert len(loaded_tokenizer.merge_rules) == len(tokenizer.merge_rules)

            text = "test text"
            original_ids = tokenizer.encode(text)
            loaded_ids = loaded_tokenizer.encode(text)
            assert original_ids == loaded_ids

    def test_empty_text(self):
        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.train("test", verbose=False)

        token_ids = tokenizer.encode("", add_special_tokens=True)
        assert token_ids == [tokenizer.bos_token_id, tokenizer.eos_token_id]

    def test_unknown_words(self):
        corpus = "known words only " * 50
        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.train(corpus, verbose=False)

        token_ids = tokenizer.encode("unknownword12345", add_special_tokens=False)
        assert len(token_ids) > 0


class TestVectorQuantizer:
    """Tests for vector quantizer."""

    def test_initialization(self):
        vq = VectorQuantizer(codebook_size=64, max_iters=50)
        assert vq.codebook_size == 64
        assert vq.max_iters == 50
        assert vq.codebook is None

    def test_training(self):
        np.random.seed(42)
        features = np.random.randn(1000, 32).astype(np.float32)

        vq = VectorQuantizer(codebook_size=64, max_iters=20)
        vq.train(features, verbose=False)

        assert vq.codebook is not None
        assert vq.codebook.shape == (64, 32)
        assert vq.feature_dim == 32

    def test_encode_decode(self):
        np.random.seed(42)
        features = np.random.randn(500, 16).astype(np.float32)

        vq = VectorQuantizer(codebook_size=32, max_iters=10)
        vq.train(features, verbose=False)

        test_features = np.random.randn(100, 16).astype(np.float32)
        codes = vq.encode(test_features)

        assert codes.shape == (100,)
        assert codes.min() >= 0
        assert codes.max() < 32

        reconstructed = vq.decode(codes)
        assert reconstructed.shape == test_features.shape

    def test_save_load(self):
        np.random.seed(42)
        features = np.random.randn(200, 8).astype(np.float32)

        vq = VectorQuantizer(codebook_size=16)
        vq.train(features, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "vq.json"
            vq.save(save_path)

            loaded_vq = VectorQuantizer.load(save_path)

            assert loaded_vq.codebook_size == vq.codebook_size
            assert loaded_vq.feature_dim == vq.feature_dim
            assert np.allclose(loaded_vq.codebook, vq.codebook)


class TestImageTokenizer:
    """Tests for image tokenizer."""

    def test_initialization(self):
        tokenizer = ImageTokenizer(image_size=224, patch_size=16, codebook_size=512, channels=3)
        assert tokenizer.image_size == 224
        assert tokenizer.patch_size == 16
        assert tokenizer.codebook_size == 512
        assert tokenizer.num_patches == 196

    def test_patchify(self):
        tokenizer = ImageTokenizer(image_size=64, patch_size=16, codebook_size=64)
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        patches = tokenizer._patchify(image.astype(np.float32) / 255.0)

        expected_patches = (64 // 16) ** 2
        assert patches.shape[0] == expected_patches
        assert patches.shape[1] == 16 * 16 * 3

    def test_preprocess_image(self):
        tokenizer = ImageTokenizer(image_size=32, patch_size=8, codebook_size=16)

        image_uint8 = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        processed = tokenizer._preprocess_image(image_uint8)

        assert processed.shape == (32, 32, 3)
        assert processed.max() <= 1.0
        assert processed.min() >= 0.0

    def test_save_load(self):
        tokenizer = ImageTokenizer(image_size=32, patch_size=8, codebook_size=16)

        np.random.seed(42)
        dummy_images = []
        for _ in range(50):
            img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            dummy_images.append(img)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            for i, img in enumerate(dummy_images):
                np.save(tmpdir_path / f"img_{i}.npy", img)

            image_paths = list(tmpdir_path.glob("*.npy"))
            tokenizer.train(image_paths[:30], sample_size=None)

            save_dir = tmpdir_path / "tokenizer"
            tokenizer.save(save_dir)

            loaded = ImageTokenizer.load(save_dir)

            assert loaded.image_size == tokenizer.image_size
            assert loaded.patch_size == tokenizer.patch_size
            assert loaded.codebook_size == tokenizer.codebook_size


class TestAudioTokenizer:
    """Tests for audio tokenizer."""

    def test_initialization(self):
        tokenizer = AudioTokenizer(sample_rate=16000, n_fft=512, hop_length=256, codebook_size=256)
        assert tokenizer.sample_rate == 16000
        assert tokenizer.n_fft == 512
        assert tokenizer.codebook_size == 256

    def test_compute_spectrogram(self):
        tokenizer = AudioTokenizer(sample_rate=16000, n_fft=256, n_mels=40)

        waveform = np.random.randn(16000).astype(np.float32)
        spectrogram = tokenizer._compute_spectrogram(waveform)

        assert spectrogram.shape[0] == 40
        assert spectrogram.shape[1] > 0

    def test_patchify_spectrogram(self):
        tokenizer = AudioTokenizer(n_mels=40, patch_length=8)

        spectrogram = np.random.randn(40, 80).astype(np.float32)
        patches = tokenizer._patchify_spectrogram(spectrogram)

        expected_patches = 80 // 8
        assert patches.shape[0] == expected_patches
        assert patches.shape[1] == 40 * 8

    def test_save_load(self):
        tokenizer = AudioTokenizer(sample_rate=8000, n_fft=128, codebook_size=32, patch_length=4)

        np.random.seed(42)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            for i in range(20):
                waveform = np.random.randn(8000).astype(np.float32)
                np.save(tmpdir_path / f"audio_{i}.npy", waveform)

            audio_paths = list(tmpdir_path.glob("*.npy"))
            tokenizer.train(audio_paths[:15], sample_size=None)

            save_dir = tmpdir_path / "tokenizer"
            tokenizer.save(save_dir)

            loaded = AudioTokenizer.load(save_dir)

            assert loaded.sample_rate == tokenizer.sample_rate
            assert loaded.codebook_size == tokenizer.codebook_size


class TestMultimodalTokenizer:
    """Tests for multimodal tokenizer."""

    def test_initialization(self):
        text_tok = BPETokenizer(vocab_size=100)
        text_tok.train("test corpus", verbose=False)

        mm_tok = MultimodalTokenizer(text_tokenizer=text_tok)

        assert mm_tok.text_tokenizer == text_tok
        assert mm_tok.image_token_offset == 100000
        assert mm_tok.audio_token_offset == 200000

    def test_encode_text_only(self):
        text_tok = BPETokenizer(vocab_size=100)
        text_tok.train("hello world test", verbose=False)

        mm_tok = MultimodalTokenizer(text_tokenizer=text_tok)

        tokens = mm_tok.encode("hello world", max_length=100)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(t < 100000 for t in tokens)

    def test_get_vocab_size(self):
        text_tok = BPETokenizer(vocab_size=1000)
        text_tok.train("test corpus for vocab size", verbose=False)

        mm_tok = MultimodalTokenizer(
            text_tokenizer=text_tok,
            image_token_offset=100000,
            audio_token_offset=200000,
        )

        total_size = mm_tok.get_total_vocab_size()
        assert total_size >= text_tok.get_vocab_size()


class TestTokenizerWrapper:
    """Tests for Tokenizer wrapper class."""

    def test_initialization(self):
        tokenizer = Tokenizer(vocab_size=500, max_length=1024)
        assert tokenizer.vocab_size == 500
        assert tokenizer.max_length == 1024

    def test_encode_decode_untrained(self):
        tokenizer = Tokenizer(vocab_size=100)
        tokenizer.train("test corpus for encoding", save_path=None)

        token_ids = tokenizer.encode("test")
        assert isinstance(token_ids, list)

        decoded = tokenizer.decode(token_ids)
        assert isinstance(decoded, str)

    def test_save_load_integration(self):
        tokenizer = Tokenizer(vocab_size=200)
        tokenizer.train("integration test corpus " * 50, save_path=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "tokenizer")
            tokenizer._tokenizer.save(save_path)

            loaded_tokenizer = Tokenizer(tokenizer_path=save_path)
            loaded_tokenizer._load_tokenizer()

            text = "integration test"
            original_ids = tokenizer.encode(text)
            loaded_ids = loaded_tokenizer.encode(text)

            assert original_ids == loaded_ids


class TestTokenizedSample:
    """Tests for TokenizedSample dataclass."""

    def test_creation(self):
        sample = TokenizedSample(
            input_ids=[1, 2, 3, 4, 5], attention_mask=[1, 1, 1, 1, 1], metadata={}
        )
        assert sample.length == 5
        assert len(sample.input_ids) == 5

    def test_length_property(self):
        sample = TokenizedSample(
            input_ids=[1, 2, 3], attention_mask=[1, 1, 1], metadata={"source": "test"}
        )
        assert sample.length == 3
        assert sample.metadata["source"] == "test"
