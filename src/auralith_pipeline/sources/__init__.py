"""Data sources module."""

from auralith_pipeline.sources.code import GitHubCodeSource, LocalCodeSource
from auralith_pipeline.sources.compound import CompoundDocumentSource
from auralith_pipeline.sources.data_sources import (
    DATASET_REGISTRY,
    DEPRECATED_DATASETS,
    DataSample,
    DataSource,
    HuggingFaceSource,
    JSONLSource,
    LocalFileSource,
    create_source,
)
from auralith_pipeline.sources.video import VideoFrameSampler, VideoSource

__all__ = [
    "DataSource",
    "DataSample",
    "HuggingFaceSource",
    "LocalFileSource",
    "LocalCodeSource",
    "GitHubCodeSource",
    "CompoundDocumentSource",
    "JSONLSource",
    "VideoSource",
    "VideoFrameSampler",
    "DATASET_REGISTRY",
    "DEPRECATED_DATASETS",
    "create_source",
]
