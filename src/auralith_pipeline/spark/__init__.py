"""Apache Spark integration for large-scale distributed processing."""

from auralith_pipeline.spark.config import SparkConfig, SparkJobConfig
from auralith_pipeline.spark.runner import SparkPipelineRunner, run_spark_pipeline
from auralith_pipeline.spark.transforms import (
    spark_deduplicate,
    spark_preprocess,
    spark_tokenize,
)

__all__ = [
    "SparkPipelineRunner",
    "run_spark_pipeline",
    "SparkConfig",
    "SparkJobConfig",
    "spark_preprocess",
    "spark_tokenize",
    "spark_deduplicate",
]
