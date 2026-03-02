"""Shared file-type constants and classification logic.

Centralised here so that ``cli.py``, ``worker.py``, and any future
consumer agree on which extensions map to which modality.

**.npy** files are ambiguous — they can represent either image arrays
(H, W, 3) or audio waveforms (1-D / 2-D time-series).  The function
:func:`_classify_file` resolves the ambiguity by inspecting the
**parent directory name**.  If it contains ``image`` or ``img`` the
file is classified as ``"image"``; if it contains ``audio`` or
``speech`` it is ``"audio"``.  When no keyword matches the directory
name, ``None`` is returned so the caller can decide how to handle it.
"""

from __future__ import annotations

import re
from pathlib import Path

# ── extension sets ─────────────────────────────────────────────────────
# .npy is intentionally absent — it is handled by directory-based
# routing inside _classify_file().

TEXT_EXTS: frozenset[str] = frozenset(
    {".txt", ".md", ".rst", ".csv", ".json", ".jsonl", ".tsv", ".xml", ".html", ".py", ".rs"}
)
IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})
AUDIO_EXTS: frozenset[str] = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a"})
VIDEO_EXTS: frozenset[str] = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm"})

# ── modality mask IDs ──────────────────────────────────────────────────

MODALITY_ID: dict[str, int] = {"text": 0, "image": 1, "audio": 2, "video": 3}

# ── token offsets ──────────────────────────────────────────────────────
# Separate ID ranges prevent collisions across modalities inside a
# single vocabulary.

IMAGE_TOKEN_OFFSET: int = 100_000
AUDIO_TOKEN_OFFSET: int = 200_000
VIDEO_TOKEN_OFFSET: int = 300_000

# ── directory keyword patterns for .npy disambiguation ─────────────────

_IMAGE_DIR_RE = re.compile(r"image|img|picture|photo|visual", re.IGNORECASE)
_AUDIO_DIR_RE = re.compile(r"audio|speech|sound|music|waveform", re.IGNORECASE)


def classify_file(file_path: Path) -> str | None:
    """Return the modality name for *file_path*, or ``None`` if unsupported.

    For ``.npy`` files the parent directory name is inspected:

    * ``images/data.npy``  → ``"image"``
    * ``audio/data.npy``   → ``"audio"``
    * ``misc/data.npy``    → ``None`` (ambiguous — caller must decide)

    All other extensions are matched against the canonical sets above.
    """
    ext = file_path.suffix.lower()

    # ── .npy: directory-based routing ──────────────────────────────
    if ext == ".npy":
        parent = file_path.parent.name.lower()
        if _IMAGE_DIR_RE.search(parent):
            return "image"
        if _AUDIO_DIR_RE.search(parent):
            return "audio"
        # Ambiguous — cannot decide from extension alone
        return None

    # ── standard extension lookup ──────────────────────────────────
    if ext in TEXT_EXTS:
        return "text"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in VIDEO_EXTS:
        return "video"
    return None
