"""Extraction module."""

from auralith_pipeline.extraction.compound import (
    COMPOUND_EXTS,
    CompoundDocument,
    CompoundDocumentExtractor,
    ModalitySegment,
)
from auralith_pipeline.extraction.extractor import (
    ContentExtractor,
    ExtractedContent,
)

__all__ = [
    "ContentExtractor",
    "ExtractedContent",
    "CompoundDocumentExtractor",
    "CompoundDocument",
    "ModalitySegment",
    "COMPOUND_EXTS",
]
