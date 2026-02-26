"""Video data source and frame extraction module.

Supports loading video files, extracting frames via decord (GPU-accelerated),
and producing DataSamples with video metadata for downstream VQ tokenization.
"""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from auralith_pipeline.sources.data_sources import DataSample, DataSource

logger = logging.getLogger(__name__)

# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv"}


class VideoFrameSampler:
    """Extract and sample frames from video files.

    Uses decord for fast, GPU-accelerated frame extraction.
    Falls back to OpenCV when decord is not available.
    """

    def __init__(
        self,
        fps: float = 1.0,
        max_frames: int = 64,
        frame_size: tuple[int, int] = (224, 224),
        strategy: str = "uniform",
    ):
        """Initialize video frame sampler.

        Args:
            fps: Frames per second to sample (ignored for 'uniform' strategy)
            max_frames: Maximum number of frames to extract
            frame_size: Target (height, width) for resized frames
            strategy: Sampling strategy — 'uniform', 'fps', 'keyframe'
        """
        self.fps = fps
        self.max_frames = max_frames
        self.frame_size = frame_size
        self.strategy = strategy

    def extract_frames(self, video_path: str | Path) -> np.ndarray:
        """Extract frames from a video file.

        Args:
            video_path: Path to video file

        Returns:
            Frames array, shape (num_frames, H, W, 3), dtype uint8
        """
        video_path = str(video_path)

        try:
            return self._extract_decord(video_path)
        except ImportError:
            logger.debug("decord not available, falling back to OpenCV")
            return self._extract_opencv(video_path)

    def _extract_decord(self, video_path: str) -> np.ndarray:
        """Extract frames using decord (fast, GPU-ready)."""
        import decord

        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        video_fps = vr.get_avg_fps()

        indices = self._get_frame_indices(total_frames, video_fps)
        frames = vr.get_batch(indices).asnumpy()  # (N, H, W, C)

        return self._resize_frames(frames)

    def _extract_opencv(self, video_path: str) -> np.ndarray:
        """Extract frames using OpenCV (fallback)."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        indices = self._get_frame_indices(total_frames, video_fps)

        frames = []
        for idx in sorted(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()

        if not frames:
            raise ValueError(f"No frames extracted from {video_path}")

        return self._resize_frames(np.array(frames))

    def _get_frame_indices(self, total_frames: int, video_fps: float) -> list[int]:
        """Compute which frame indices to sample."""
        if total_frames <= 0:
            return [0]

        if self.strategy == "uniform":
            n = min(self.max_frames, total_frames)
            return np.linspace(0, total_frames - 1, n, dtype=int).tolist()

        elif self.strategy == "fps":
            step = max(1, int(video_fps / self.fps))
            indices = list(range(0, total_frames, step))
            return indices[: self.max_frames]

        else:
            # Default uniform
            n = min(self.max_frames, total_frames)
            return np.linspace(0, total_frames - 1, n, dtype=int).tolist()

    def _resize_frames(self, frames: np.ndarray) -> np.ndarray:
        """Resize frames to target size."""
        if frames.shape[1:3] == self.frame_size:
            return frames

        from scipy.ndimage import zoom

        h, w = frames.shape[1], frames.shape[2]
        th, tw = self.frame_size
        zoom_factors = (1, th / h, tw / w, 1)
        return zoom(frames, zoom_factors, order=1).astype(np.uint8)

    def get_video_info(self, video_path: str | Path) -> dict[str, Any]:
        """Get video metadata without extracting frames."""
        video_path = str(video_path)

        try:
            import decord

            vr = decord.VideoReader(video_path)
            return {
                "total_frames": len(vr),
                "fps": float(vr.get_avg_fps()),
                "duration_seconds": len(vr) / max(vr.get_avg_fps(), 1),
            }
        except ImportError:
            pass

        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            info = {
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
                "duration_seconds": (
                    int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    / max(cap.get(cv2.CAP_PROP_FPS) or 30.0, 1)
                ),
            }
            cap.release()
            return info
        except ImportError:
            return {"total_frames": -1, "fps": -1, "duration_seconds": -1}


class VideoSource(DataSource):
    """Load video data from a local directory.

    Extracts frames and produces DataSamples with modality='video'
    and the frame data stored in metadata for downstream VQ tokenization.
    """

    def __init__(
        self,
        path: str,
        pattern: str = "**/*.mp4",
        max_samples: int | None = None,
        fps: float = 1.0,
        max_frames: int = 64,
        frame_size: tuple[int, int] = (224, 224),
        caption_file: str | None = None,
    ):
        """Initialize video source.

        Args:
            path: Directory containing video files
            pattern: Glob pattern for video files
            max_samples: Maximum number of videos to process
            fps: Frames per second for extraction
            max_frames: Maximum frames per video
            frame_size: Target frame size (H, W)
            caption_file: Optional JSON/JSONL file mapping video filenames to captions
        """
        self.path = Path(path)
        self.pattern = pattern
        self.max_samples = max_samples
        self._files: list[Path] | None = None

        self.sampler = VideoFrameSampler(
            fps=fps,
            max_frames=max_frames,
            frame_size=frame_size,
        )

        # Load captions if provided
        self._captions: dict[str, str] = {}
        if caption_file:
            self._load_captions(caption_file)

    def _load_captions(self, caption_file: str) -> None:
        """Load video captions from a JSON or JSONL file."""
        import json

        caption_path = Path(caption_file)
        if caption_path.suffix == ".json":
            with open(caption_path) as f:
                self._captions = json.load(f)
        elif caption_path.suffix == ".jsonl":
            with open(caption_path) as f:
                for line in f:
                    item = json.loads(line)
                    filename = item.get("filename", item.get("video", ""))
                    caption = item.get("caption", item.get("text", ""))
                    if filename and caption:
                        self._captions[filename] = caption

    def _get_files(self) -> list[Path]:
        if self._files is None:
            self._files = sorted(
                f for f in self.path.glob(self.pattern) if f.suffix.lower() in VIDEO_EXTENSIONS
            )
        return self._files

    def __iter__(self) -> Iterator[DataSample]:
        count = 0
        for video_path in self._get_files():
            if self.max_samples and count >= self.max_samples:
                break

            try:
                # Get caption or default description
                caption = self._captions.get(
                    video_path.name,
                    self._captions.get(video_path.stem, f"Video: {video_path.name}"),
                )

                # Get video info
                info = self.sampler.get_video_info(video_path)

                yield DataSample(
                    content=caption,
                    source=f"video/{video_path.name}",
                    modality="video",
                    metadata={
                        "video_path": str(video_path),
                        "total_frames": info.get("total_frames", -1),
                        "fps": info.get("fps", -1),
                        "duration_seconds": info.get("duration_seconds", -1),
                        "max_frames_sampled": self.sampler.max_frames,
                    },
                )
                count += 1

            except Exception as e:
                logger.warning(f"Failed to process video {video_path}: {e}")
                continue

    def __len__(self) -> int:
        total = len(self._get_files())
        if self.max_samples:
            return min(self.max_samples, total)
        return total

    @property
    def name(self) -> str:
        return f"video/{self.path}"
