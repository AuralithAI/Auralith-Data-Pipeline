"""Data source for compound documents (DOCX, XLSX, DICOM, GeoTIFF, etc.).

Walks a directory tree, extracts modality segments from each compound file
via :class:`~auralith_pipeline.extraction.compound.CompoundDocumentExtractor`,
and yields ``DataSample`` objects that flow through the standard pipeline.

Binary segments (images, audio) are written to a temporary staging area so
that downstream multimodal tokenizers can load them by path.

Usage::

    source = CompoundDocumentSource("data/raw_documents/")
    for sample in source:
        print(sample.modality, sample.source, len(sample.content))
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Iterator
from pathlib import Path

from auralith_pipeline.extraction.compound import (
    COMPOUND_EXTS,
    CompoundDocumentExtractor,
    ModalitySegment,
)
from auralith_pipeline.sources.data_sources import DataSample, DataSource

logger = logging.getLogger(__name__)


class CompoundDocumentSource(DataSource):
    """Iterate over compound files in a directory, yielding per-segment DataSamples.

    Each compound file (DOCX, XLSX, DICOM, GeoTIFF, …) is decomposed
    into its constituent modalities.  Text/table segments produce
    ``DataSample(modality="text")``.  Binary segments (images, audio)
    are staged to disk and referenced via ``metadata["binary_path"]``
    so that downstream tokenizers (ImageTokenizer, AudioTokenizer) can
    load them.

    Parameters
    ----------
    root_dir:
        Root directory to walk for compound files.
    extensions:
        Specific extensions to process.  Defaults to all supported types.
    staging_dir:
        Directory for temporary binary assets.  Defaults to a temp dir
        inside ``root_dir``.
    extract_images:
        Extract embedded images from documents.
    extract_tables:
        Extract tables as Markdown text.
    extract_audio:
        Extract audio tracks from media containers.
    max_samples:
        Stop after emitting this many samples (None = unlimited).
    """

    def __init__(
        self,
        root_dir: str | Path,
        extensions: frozenset[str] | None = None,
        staging_dir: str | Path | None = None,
        extract_images: bool = True,
        extract_tables: bool = True,
        extract_audio: bool = True,
        max_samples: int | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.extensions = extensions or COMPOUND_EXTS
        self.staging_dir = Path(staging_dir) if staging_dir else self.root_dir / ".staging"
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.extract_audio = extract_audio
        self.max_samples = max_samples

        self._extractor = CompoundDocumentExtractor(
            extract_images=extract_images,
            extract_tables=extract_tables,
            extract_audio=extract_audio,
        )

        # Pre-scan for file count
        self._files: list[Path] = list(self._iter_compound_files())
        logger.info(
            "CompoundDocumentSource: found %d compound files in %s",
            len(self._files),
            self.root_dir,
        )

    # ── DataSource interface ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"compound:{self.root_dir.name}"

    def __len__(self) -> int:
        return len(self._files)

    def __iter__(self) -> Iterator[DataSample]:
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        emitted = 0

        for file_path in self._files:
            if self.max_samples is not None and emitted >= self.max_samples:
                return

            try:
                doc = self._extractor.extract(file_path)
            except Exception as exc:
                logger.warning("Skipping %s: %s", file_path, exc)
                continue

            for seg in doc.segments:
                if self.max_samples is not None and emitted >= self.max_samples:
                    return

                sample = self._segment_to_sample(seg, doc.document_id)
                if sample is not None:
                    yield sample
                    emitted += 1

    # ── helpers ───────────────────────────────────────────────────────

    def _iter_compound_files(self) -> Iterator[Path]:
        """Walk the directory tree yielding files with compound extensions."""
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            # Skip hidden / cache directories
            dirnames[:] = [
                d
                for d in dirnames
                if not d.startswith(".") and d not in {"__pycache__", "node_modules"}
            ]
            for fname in filenames:
                fpath = Path(dirpath) / fname
                # Handle .nii.gz
                if fname.lower().endswith(".nii.gz"):
                    if ".nii" in self.extensions:
                        yield fpath
                    continue
                if fpath.suffix.lower() in self.extensions:
                    yield fpath

    def _segment_to_sample(self, seg: ModalitySegment, document_id: str) -> DataSample | None:
        """Convert a ModalitySegment into a DataSample.

        For text segments, the content goes directly into DataSample.content.
        For binary segments (image/audio), the bytes are staged to a temp
        file and the path is stored in metadata so tokenizers can load it.
        """
        metadata = {**seg.metadata, "document_id": document_id}

        if seg.modality == "text":
            if not seg.content_text or not seg.content_text.strip():
                return None
            return DataSample(
                content=seg.content_text,
                source=metadata.get("source", "compound"),
                metadata=metadata,
                modality="text",
            )

        # Binary modality (image / audio / video)
        if seg.content_bytes is None:
            return None

        # Stage binary to disk
        binary_path = self._stage_binary(seg, document_id)
        if binary_path is None:
            return None

        # For binary modalities, content is a placeholder describing the asset.
        # The actual data is accessed via metadata["binary_path"] by the
        # multimodal tokenizer.
        content_desc = (
            f"[{seg.modality.upper()}] "
            f"{metadata.get('type', 'unknown')} from {metadata.get('source', 'unknown')}"
        )

        return DataSample(
            content=content_desc,
            source=metadata.get("source", "compound"),
            metadata={**metadata, "binary_path": str(binary_path)},
            modality=seg.modality,
        )

    def _stage_binary(self, seg: ModalitySegment, document_id: str = "") -> Path | None:
        """Write binary content to a staging file and return the path.

        Args:
            seg: The modality segment whose bytes should be staged.
            document_id: Document ID for lineage (used in path namespacing).
        """
        if seg.content_bytes is None:
            return None

        # Deterministic filename from content hash
        content_hash = hashlib.sha256(seg.content_bytes).hexdigest()[:12]
        ext_map = {"image": ".png", "audio": ".wav", "video": ".mp4"}
        ext = ext_map.get(seg.modality, ".bin")

        subdir = self.staging_dir / seg.modality
        subdir.mkdir(parents=True, exist_ok=True)
        out_path = subdir / f"{content_hash}{ext}"

        if not out_path.exists():
            out_path.write_bytes(seg.content_bytes)
            logger.debug(
                "Staged %s asset: %s (%d bytes)", seg.modality, out_path, len(seg.content_bytes)
            )

        return out_path
