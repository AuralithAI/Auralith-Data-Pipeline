"""Video tokenizer — frame-level VQ, reuses ImageTokenizer's patch-and-quantize pattern.

Each video becomes a sequence of VQ token IDs:
    <VIDEO> [frame0_patch0] [frame0_patch1] ... [frame0_patchN]
            [frame1_patch0] ... [frameK_patchN] <VIDEO_END>

The tokens are interleaved into the unified input_ids stream by MultimodalTokenizer.
"""

import json
import logging
from pathlib import Path

import numpy as np

from auralith_pipeline.tokenization.multimodal_tokenizer import ImageTokenizer, VectorQuantizer

logger = logging.getLogger(__name__)


class VideoTokenizer:
    """Tokenize videos into discrete token sequences.

    Extracts frames → patches → VQ codes, mirroring ImageTokenizer
    but operating over a temporal sequence of frames.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        codebook_size: int = 1024,
        channels: int = 3,
        max_frames: int = 64,
        temporal_stride: int = 1,
    ):
        """Initialize video tokenizer.

        Args:
            image_size: Resize each frame to this square size
            patch_size: Spatial patch size
            codebook_size: VQ codebook size
            channels: Colour channels (3 for RGB)
            max_frames: Maximum frames to tokenize per video
            temporal_stride: Take every Nth frame (further temporal subsampling)
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.codebook_size = codebook_size
        self.channels = channels
        self.max_frames = max_frames
        self.temporal_stride = temporal_stride

        # Re-use ImageTokenizer internals for per-frame patch extraction
        self._frame_tokenizer = ImageTokenizer(
            image_size=image_size,
            patch_size=patch_size,
            codebook_size=codebook_size,
            channels=channels,
        )

        # Expose VQ for training
        self.vq = self._frame_tokenizer.vq

        # Per-frame patch count
        self.patches_per_frame = self._frame_tokenizer.num_patches

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        video_frames: list[np.ndarray],
        sample_size: int | None = None,
    ) -> None:
        """Train VQ codebook on video patches.

        Args:
            video_frames: List of frame arrays, each shape (H, W, C)
            sample_size: Max frames to use for training
        """
        if sample_size and len(video_frames) > sample_size:
            np.random.seed(42)
            indices = np.random.choice(len(video_frames), sample_size, replace=False)
            video_frames = [video_frames[i] for i in indices]

        logger.info(f"Training video VQ on {len(video_frames)} frames...")

        all_patches = []
        for frame in video_frames:
            processed = self._frame_tokenizer._preprocess_image(frame)
            patches = self._frame_tokenizer._patchify(processed)
            all_patches.append(patches)

        all_patches_arr = np.concatenate(all_patches, axis=0)
        logger.info(f"Extracted {len(all_patches_arr)} patches from video frames")
        self.vq.train(all_patches_arr, verbose=True)

    def train_from_video_files(
        self,
        video_paths: list[str | Path],
        max_frames_per_video: int = 16,
    ) -> None:
        """Train codebook from video files using frame sampler."""
        from auralith_pipeline.sources.video import VideoFrameSampler

        sampler = VideoFrameSampler(
            max_frames=max_frames_per_video,
            frame_size=(self.image_size, self.image_size),
            strategy="uniform",
        )

        all_frames = []
        for vpath in video_paths:
            try:
                frames = sampler.extract_frames(vpath)
                all_frames.extend(frames)
            except Exception as e:
                logger.warning(f"Skipping {vpath}: {e}")

        if not all_frames:
            raise ValueError("No frames extracted from any video file")

        self.train(all_frames)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, video_path: str | Path) -> list[int]:
        """Encode a video file to a flat list of VQ token IDs.

        The token sequence is all frames concatenated:
            [frame0_patch0, ..., frame0_patchN, frame1_patch0, ..., frameK_patchN]

        Args:
            video_path: Path to video file

        Returns:
            List of VQ code token IDs
        """
        from auralith_pipeline.sources.video import VideoFrameSampler

        sampler = VideoFrameSampler(
            max_frames=self.max_frames,
            frame_size=(self.image_size, self.image_size),
            strategy="uniform",
        )

        frames = sampler.extract_frames(video_path)

        # Apply temporal stride
        if self.temporal_stride > 1:
            frames = frames[:: self.temporal_stride]

        all_codes: list[int] = []
        for frame in frames:
            processed = self._frame_tokenizer._preprocess_image(frame)
            patches = self._frame_tokenizer._patchify(processed)
            codes = self.vq.encode(patches)
            all_codes.extend(codes.tolist())

        return all_codes

    def encode_frames(self, frames: np.ndarray) -> list[int]:
        """Encode pre-extracted frames to token IDs.

        Args:
            frames: Shape (num_frames, H, W, C), uint8

        Returns:
            Flat list of VQ code token IDs
        """
        if self.temporal_stride > 1:
            frames = frames[:: self.temporal_stride]

        all_codes: list[int] = []
        for frame in frames:
            processed = self._frame_tokenizer._preprocess_image(frame)
            patches = self._frame_tokenizer._patchify(processed)
            codes = self.vq.encode(patches)
            all_codes.extend(codes.tolist())

        return all_codes

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, save_dir: str | Path) -> None:
        """Save video tokenizer config + VQ codebook."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "codebook_size": self.codebook_size,
            "channels": self.channels,
            "max_frames": self.max_frames,
            "temporal_stride": self.temporal_stride,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        self.vq.save(save_dir / "vq_codebook.json")
        logger.info(f"Video tokenizer saved to {save_dir}")

    @classmethod
    def load(cls, load_dir: str | Path) -> "VideoTokenizer":
        """Load video tokenizer from directory."""
        load_dir = Path(load_dir)

        with open(load_dir / "config.json") as f:
            config = json.load(f)

        tokenizer = cls(**config)
        tokenizer.vq = VectorQuantizer.load(load_dir / "vq_codebook.json")
        tokenizer._frame_tokenizer.vq = tokenizer.vq

        logger.info(f"Video tokenizer loaded from {load_dir}")
        return tokenizer
