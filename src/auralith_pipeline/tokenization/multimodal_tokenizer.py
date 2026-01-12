"""Multimodal tokenizer for images, audio, and other non-text modalities.

This module implements custom feature extraction and quantization for non-text
modalities, converting them to discrete token sequences that can be fused with
text tokens.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from PIL import Image
from scipy import ndimage
from scipy.signal import resample

logger = logging.getLogger(__name__)


class VectorQuantizer:
    """Vector Quantizer using k-means clustering.

    Converts continuous feature vectors to discrete codes via a learned codebook.
    """

    def __init__(self, codebook_size: int = 1024, max_iters: int = 100):
        """Initialize vector quantizer.

        Args:
            codebook_size: Number of discrete codes (vocabulary size for modality)
            max_iters: Maximum k-means iterations
        """
        self.codebook_size = codebook_size
        self.max_iters = max_iters
        self.codebook: np.ndarray | None = None  # Shape: (codebook_size, feature_dim)
        self.feature_dim: int | None = None

    def train(self, features: np.ndarray, verbose: bool = True) -> None:
        """Train codebook via k-means clustering.

        Args:
            features: Feature vectors, shape (num_samples, feature_dim)
            verbose: Whether to log training progress
        """
        if features.shape[0] < self.codebook_size:
            raise ValueError(
                f"Need at least {self.codebook_size} samples to train codebook, "
                f"got {features.shape[0]}"
            )

        self.feature_dim = features.shape[1]

        if verbose:
            logger.info(f"Training VQ codebook with {self.codebook_size} codes...")
            logger.info(f"Feature dim: {self.feature_dim}, Samples: {features.shape[0]}")

        # Initialize centroids randomly from data
        np.random.seed(42)
        indices = np.random.choice(features.shape[0], self.codebook_size, replace=False)
        self.codebook = features[indices].copy()

        # K-means iterations
        for iteration in range(self.max_iters):
            # Assign each feature to nearest centroid
            distances = self._compute_distances(features)
            assignments = np.argmin(distances, axis=1)

            # Update centroids
            old_codebook = self.codebook.copy()
            for k in range(self.codebook_size):
                mask = assignments == k
                if mask.sum() > 0:
                    self.codebook[k] = features[mask].mean(axis=0)

            # Check convergence
            change = np.abs(self.codebook - old_codebook).max()
            if verbose and (iteration + 1) % 10 == 0:
                logger.info(f"Iteration {iteration + 1}/{self.max_iters}, max change: {change:.6f}")

            if change < 1e-6:
                if verbose:
                    logger.info(f"Converged at iteration {iteration + 1}")
                break

        if verbose:
            logger.info("VQ training complete!")

    def _compute_distances(self, features: np.ndarray) -> np.ndarray:
        """Compute Euclidean distances between features and codebook.

        Args:
            features: Shape (num_samples, feature_dim)

        Returns:
            Distances: Shape (num_samples, codebook_size)
        """
        # Efficient distance computation: ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
        features_sq = (features**2).sum(axis=1, keepdims=True)
        codebook_sq = (self.codebook**2).sum(axis=1, keepdims=True).T
        cross_term = features @ self.codebook.T

        distances = features_sq + codebook_sq - 2 * cross_term
        return distances

    def encode(self, features: np.ndarray) -> np.ndarray:
        """Encode features to discrete codes.

        Args:
            features: Shape (num_samples, feature_dim)

        Returns:
            Codes: Shape (num_samples,), dtype int
        """
        if self.codebook is None:
            raise ValueError("Codebook not trained! Call train() first.")

        distances = self._compute_distances(features)
        codes = np.argmin(distances, axis=1)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode codes back to feature vectors.

        Args:
            codes: Shape (num_samples,), dtype int

        Returns:
            Reconstructed features: Shape (num_samples, feature_dim)
        """
        if self.codebook is None:
            raise ValueError("Codebook not trained!")

        return self.codebook[codes]

    def save(self, save_path: str | Path) -> None:
        """Save codebook to file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "codebook_size": self.codebook_size,
            "feature_dim": self.feature_dim,
            "codebook": self.codebook.tolist() if self.codebook is not None else None,
        }

        with open(save_path, "w") as f:
            json.dump(data, f)

        logger.info(f"VQ codebook saved to {save_path}")

    @classmethod
    def load(cls, load_path: str | Path) -> "VectorQuantizer":
        """Load codebook from file."""
        with open(load_path) as f:
            data = json.load(f)

        vq = cls(codebook_size=data["codebook_size"])
        vq.feature_dim = data["feature_dim"]
        vq.codebook = np.array(data["codebook"]) if data["codebook"] is not None else None

        logger.info(f"VQ codebook loaded from {load_path}")
        return vq


class ImageTokenizer:
    """Tokenize images into discrete token sequences.

    Converts images to patches, extracts features, and quantizes to tokens.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        codebook_size: int = 1024,
        channels: int = 3,
    ):
        """Initialize image tokenizer.

        Args:
            image_size: Target image size (assumes square images)
            patch_size: Size of each patch
            codebook_size: VQ codebook size
            channels: Number of color channels (3 for RGB)
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.codebook_size = codebook_size
        self.channels = channels

        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2
        self.patches_per_side = image_size // patch_size

        # Feature dimension = patch_size^2 * channels
        self.feature_dim = patch_size * patch_size * channels

        # Vector quantizer
        self.vq = VectorQuantizer(codebook_size=codebook_size)

    def _load_image_raw(self, image_path: str | Path) -> np.ndarray:
        """Load image from file.

        Supports .npy files for preprocessed data and common image formats
        (JPEG, PNG, BMP, TIFF) via Pillow.

        Args:
            image_path: Path to image file

        Returns:
            Image array, shape (H, W, C)
        """
        image_path = Path(image_path)

        if image_path.suffix == ".npy":
            image = np.load(image_path)
            return image
        elif image_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                return np.array(img, dtype=np.uint8)
        else:
            raise ValueError(
                f"Unsupported image format: {image_path.suffix}. "
                f"Supported: .npy, .jpg, .jpeg, .png, .bmp, .tiff"
            )

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image: resize and normalize.

        Args:
            image: Input image, shape (H, W, C)

        Returns:
            Preprocessed image, shape (image_size, image_size, channels)
        """
        # Resize (bilinear interpolation)
        if image.shape[:2] != (self.image_size, self.image_size):
            # Simple resize using numpy (production: implement proper bilinear)
            from scipy.ndimage import zoom

            h, w = image.shape[:2]
            zoom_factors = (self.image_size / h, self.image_size / w, 1)
            image = zoom(image, zoom_factors, order=1)

        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0

        # Ensure correct shape
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        if image.shape[2] != self.channels:
            if self.channels == 3 and image.shape[2] == 1:
                # Grayscale to RGB
                image = np.repeat(image, 3, axis=2)
            else:
                raise ValueError(
                    f"Channel mismatch: expected {self.channels}, got {image.shape[2]}"
                )

        return image

    def _patchify(self, image: np.ndarray) -> np.ndarray:
        """Divide image into patches and flatten.

        Args:
            image: Shape (image_size, image_size, channels)

        Returns:
            Patches: Shape (num_patches, feature_dim)
        """
        patches = []
        for i in range(self.patches_per_side):
            for j in range(self.patches_per_side):
                y = i * self.patch_size
                x = j * self.patch_size
                patch = image[y : y + self.patch_size, x : x + self.patch_size, :]
                patches.append(patch.flatten())

        return np.array(patches)

    def train(self, image_paths: list[str | Path], sample_size: int | None = None) -> None:
        """Train VQ codebook on image patches.

        Args:
            image_paths: List of paths to training images
            sample_size: Number of images to sample (None = use all)
        """
        logger.info(f"Training image tokenizer on {len(image_paths)} images...")

        # Sample if needed
        if sample_size and len(image_paths) > sample_size:
            np.random.seed(42)
            image_paths = np.random.choice(image_paths, sample_size, replace=False).tolist()

        # Extract patches from all images
        all_patches = []
        for i, img_path in enumerate(image_paths):
            if (i + 1) % 100 == 0:
                logger.info(f"Processing image {i + 1}/{len(image_paths)}...")

            try:
                image = self._load_image_raw(img_path)
                image = self._preprocess_image(image)
                patches = self._patchify(image)
                all_patches.append(patches)
            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
                continue

        # Concatenate all patches
        all_patches = np.concatenate(all_patches, axis=0)
        logger.info(f"Extracted {len(all_patches)} patches total")

        # Train VQ
        self.vq.train(all_patches, verbose=True)

    def encode(self, image_path: str | Path) -> list[int]:
        """Encode image to token sequence.

        Args:
            image_path: Path to image file

        Returns:
            List of token IDs, length = num_patches
        """
        image = self._load_image_raw(image_path)
        image = self._preprocess_image(image)
        patches = self._patchify(image)

        # Quantize patches to codes
        codes = self.vq.encode(patches)

        return codes.tolist()

    def save(self, save_dir: str | Path) -> None:
        """Save image tokenizer."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config = {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "codebook_size": self.codebook_size,
            "channels": self.channels,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save VQ codebook
        self.vq.save(save_dir / "vq_codebook.json")

        logger.info(f"Image tokenizer saved to {save_dir}")

    @classmethod
    def load(cls, load_dir: str | Path) -> "ImageTokenizer":
        """Load image tokenizer."""
        load_dir = Path(load_dir)

        # Load config
        with open(load_dir / "config.json") as f:
            config = json.load(f)

        tokenizer = cls(**config)

        # Load VQ
        tokenizer.vq = VectorQuantizer.load(load_dir / "vq_codebook.json")

        logger.info(f"Image tokenizer loaded from {load_dir}")
        return tokenizer


class AudioTokenizer:
    """Tokenize audio into discrete token sequences.

    Converts audio waveforms to spectrograms, extracts patches, and quantizes.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
        n_mels: int = 80,
        codebook_size: int = 512,
        patch_length: int = 16,
    ):
        """Initialize audio tokenizer.

        Args:
            sample_rate: Target sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel frequency bins
            codebook_size: VQ codebook size
            patch_length: Number of time frames per patch
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.codebook_size = codebook_size
        self.patch_length = patch_length

        # Feature dimension
        self.feature_dim = n_mels * patch_length

        # Vector quantizer
        self.vq = VectorQuantizer(codebook_size=codebook_size)

    def _load_audio_raw(self, audio_path: str | Path) -> np.ndarray:
        """Load audio from file.

        Supports .npy files for preprocessed data and common audio formats
        (WAV, FLAC, OGG) via soundfile.

        Args:
            audio_path: Path to audio file

        Returns:
            Waveform array, shape (num_samples,)
        """
        audio_path = Path(audio_path)

        if audio_path.suffix == ".npy":
            waveform = np.load(audio_path)
            return waveform
        elif audio_path.suffix.lower() in [".wav", ".flac", ".ogg"]:
            waveform_data, sr = sf.read(audio_path)
            # Resample if needed
            if sr != self.sample_rate:
                waveform_data = resample(
                    waveform_data, int(len(waveform_data) * self.sample_rate / sr)
                )
            waveform = np.asarray(waveform_data, dtype=np.float32)
            return waveform
        else:
            raise ValueError(
                f"Unsupported audio format: {audio_path.suffix}. "
                f"Supported: .npy, .wav, .flac, .ogg"
            )

    def _compute_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram from waveform using STFT.

        Args:
            waveform: Shape (num_samples,)

        Returns:
            Spectrogram: Shape (n_mels, num_frames)
        """
        # Ensure waveform is 1D
        if waveform.ndim > 1:
            waveform = waveform.flatten()

        # Pad waveform if needed
        if len(waveform) < self.n_fft:
            waveform = np.pad(waveform, (0, self.n_fft - len(waveform)))

        num_frames = (len(waveform) - self.n_fft) // self.hop_length + 1

        # Compute STFT with proper windowing
        hann_window = np.hanning(self.n_fft)
        spectrogram = []

        for i in range(num_frames):
            start = i * self.hop_length
            frame = waveform[start : start + self.n_fft]

            # Apply Hann window
            windowed_frame = frame * hann_window

            # Compute FFT
            fft = np.fft.rfft(windowed_frame)
            magnitude = np.abs(fft)

            # Convert power to dB scale
            power = magnitude**2
            power = np.maximum(power, 1e-10)  # Avoid log(0)

            # Apply mel filterbank (simplified linear approximation)
            # For production with librosa: librosa.feature.melspectrogram()
            mel_bins = np.linspace(0, len(magnitude) - 1, self.n_mels, dtype=int)
            mel_spec = power[mel_bins]

            # Convert to log scale (dB)
            mel_spec_db = 10 * np.log10(mel_spec + 1e-10)

            spectrogram.append(mel_spec_db)

        return np.array(spectrogram).T  # Shape: (n_mels, num_frames)

    def _patchify_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """Divide spectrogram into time patches.

        Args:
            spectrogram: Shape (n_mels, num_frames)

        Returns:
            Patches: Shape (num_patches, feature_dim)
        """
        num_frames = spectrogram.shape[1]
        num_patches = num_frames // self.patch_length

        patches = []
        for i in range(num_patches):
            start = i * self.patch_length
            patch = spectrogram[:, start : start + self.patch_length]
            patches.append(patch.flatten())

        return np.array(patches)

    def train(self, audio_paths: list[str | Path], sample_size: int | None = None) -> None:
        """Train VQ codebook on audio patches.

        Args:
            audio_paths: List of paths to training audio files
            sample_size: Number of files to sample
        """
        logger.info(f"Training audio tokenizer on {len(audio_paths)} files...")

        if sample_size and len(audio_paths) > sample_size:
            np.random.seed(42)
            audio_paths = np.random.choice(audio_paths, sample_size, replace=False).tolist()

        # Extract patches
        all_patches = []
        for i, audio_path in enumerate(audio_paths):
            if (i + 1) % 100 == 0:
                logger.info(f"Processing audio {i + 1}/{len(audio_paths)}...")

            try:
                waveform = self._load_audio_raw(audio_path)
                spectrogram = self._compute_spectrogram(waveform)
                patches = self._patchify_spectrogram(spectrogram)
                all_patches.append(patches)
            except Exception as e:
                logger.warning(f"Failed to process {audio_path}: {e}")
                continue

        all_patches = np.concatenate(all_patches, axis=0)
        logger.info(f"Extracted {len(all_patches)} patches total")

        # Train VQ
        self.vq.train(all_patches, verbose=True)

    def encode(self, audio_path: str | Path) -> list[int]:
        """Encode audio to token sequence.

        Args:
            audio_path: Path to audio file

        Returns:
            List of token IDs
        """
        waveform = self._load_audio_raw(audio_path)
        spectrogram = self._compute_spectrogram(waveform)
        patches = self._patchify_spectrogram(spectrogram)

        codes = self.vq.encode(patches)
        return codes.tolist()

    def save(self, save_dir: str | Path) -> None:
        """Save audio tokenizer."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "codebook_size": self.codebook_size,
            "patch_length": self.patch_length,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        self.vq.save(save_dir / "vq_codebook.json")

        logger.info(f"Audio tokenizer saved to {save_dir}")

    @classmethod
    def load(cls, load_dir: str | Path) -> "AudioTokenizer":
        """Load audio tokenizer."""
        load_dir = Path(load_dir)

        with open(load_dir / "config.json") as f:
            config = json.load(f)

        tokenizer = cls(**config)
        tokenizer.vq = VectorQuantizer.load(load_dir / "vq_codebook.json")

        logger.info(f"Audio tokenizer loaded from {load_dir}")
        return tokenizer


class MultimodalTokenizer:
    """Unified tokenizer for text + images + audio.

    Fuses multiple modalities into a single token sequence.
    """

    def __init__(
        self,
        text_tokenizer: Any,  # BPETokenizer instance
        image_tokenizer: ImageTokenizer | None = None,
        audio_tokenizer: AudioTokenizer | None = None,
        image_token_offset: int = 100000,
        audio_token_offset: int = 200000,
    ):
        """Initialize multimodal tokenizer.

        Args:
            text_tokenizer: BPE tokenizer for text
            image_tokenizer: Image tokenizer (optional)
            audio_tokenizer: Audio tokenizer (optional)
            image_token_offset: Offset to add to image token IDs
            audio_token_offset: Offset to add to audio token IDs
        """
        self.text_tokenizer = text_tokenizer
        self.image_tokenizer = image_tokenizer
        self.audio_tokenizer = audio_tokenizer

        # Token offsets to avoid ID collisions
        self.image_token_offset = image_token_offset
        self.audio_token_offset = audio_token_offset

    def encode(
        self,
        text: str,
        image_path: str | Path | None = None,
        audio_path: str | Path | None = None,
        max_length: int = 2048,
    ) -> list[int]:
        """Encode multimodal input to unified token sequence.

        Args:
            text: Text content (can include <image_start>, <audio_start> placeholders)
            image_path: Optional image file path
            audio_path: Optional audio file path
            max_length: Maximum sequence length

        Returns:
            Unified list of token IDs
        """
        # Tokenize text first
        text_tokens = self.text_tokenizer.encode(text, add_special_tokens=True, max_length=None)

        # Process image if provided
        if image_path and self.image_tokenizer:
            image_tokens = self.image_tokenizer.encode(image_path)
            # Add offset to avoid collision with text tokens
            image_tokens = [tid + self.image_token_offset for tid in image_tokens]

            # Insert image tokens at placeholder
            image_start_id = self.text_tokenizer.SPECIAL_TOKENS["<image_start>"]
            image_end_id = self.text_tokenizer.SPECIAL_TOKENS["<image_end>"]

            if image_start_id in text_tokens:
                # Find insertion point
                idx = text_tokens.index(image_start_id)
                # Insert: <image_start> + image_tokens + <image_end>
                text_tokens = (
                    text_tokens[: idx + 1] + image_tokens + [image_end_id] + text_tokens[idx + 1 :]
                )

        # Process audio if provided
        if audio_path and self.audio_tokenizer:
            audio_tokens = self.audio_tokenizer.encode(audio_path)
            audio_tokens = [tid + self.audio_token_offset for tid in audio_tokens]

            audio_start_id = self.text_tokenizer.SPECIAL_TOKENS["<audio_start>"]
            audio_end_id = self.text_tokenizer.SPECIAL_TOKENS["<audio_end>"]

            if audio_start_id in text_tokens:
                idx = text_tokens.index(audio_start_id)
                text_tokens = (
                    text_tokens[: idx + 1] + audio_tokens + [audio_end_id] + text_tokens[idx + 1 :]
                )

        # Truncate to max length
        if len(text_tokens) > max_length:
            text_tokens = text_tokens[:max_length]

        return text_tokens

    def get_total_vocab_size(self) -> int:
        """Get total vocabulary size across all modalities."""
        total = self.text_tokenizer.get_vocab_size()

        if self.image_tokenizer:
            total = max(total, self.image_token_offset + self.image_tokenizer.codebook_size)

        if self.audio_tokenizer:
            total = max(total, self.audio_token_offset + self.audio_tokenizer.codebook_size)

        return total
