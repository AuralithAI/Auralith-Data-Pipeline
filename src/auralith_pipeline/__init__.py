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

__version__ = "1.0.0"
__author__ = "AuralithAI"

from auralith_pipeline.config.pipeline_config import PipelineConfig
from auralith_pipeline.pipeline import Pipeline

__all__ = [
    "Pipeline",
    "PipelineConfig",
    "__version__",
]
