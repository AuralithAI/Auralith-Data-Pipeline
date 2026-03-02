"""
Auralith Data Pipeline

A production-grade data collection and processing pipeline for training
large language models and multimodal AI systems.

Usage:
    from auralith_pipeline import Pipeline, PipelineConfig

    config = PipelineConfig.from_preset("production")
    pipeline = Pipeline(config)
    pipeline.run()
"""

try:
    from auralith_pipeline._version import __version__
except ImportError:  # editable install / dev checkout without build
    try:
        from importlib.metadata import version

        __version__ = version("auralith-data-pipeline")
    except Exception:
        __version__ = "0.0.0-dev"

__author__ = "AuralithAI"

from auralith_pipeline.config.pipeline_config import PipelineConfig
from auralith_pipeline.pipeline import Pipeline

__all__ = [
    "Pipeline",
    "PipelineConfig",
    "__version__",
]
