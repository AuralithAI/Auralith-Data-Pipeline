"""Spark pipeline runner for large-scale processing."""

import logging
from typing import Any

from auralith_pipeline.spark.config import SparkConfig, SparkJobConfig
from auralith_pipeline.spark.transforms import (
    spark_deduplicate,
    spark_preprocess,
    spark_tokenize,
)

logger = logging.getLogger(__name__)


class SparkPipelineRunner:
    """Run Auralith pipeline on Apache Spark for large-scale processing."""

    def __init__(self, spark_config: SparkConfig):
        """
        Initialize Spark pipeline runner.

        Args:
            spark_config: Spark configuration
        """
        self.spark_config = spark_config
        self.spark = None
        self.sc = None

    def initialize_spark(self):
        """Initialize Spark session."""
        try:
            from pyspark.sql import SparkSession

            builder = SparkSession.builder

            # Apply configuration
            for key, value in self.spark_config.to_spark_conf().items():
                builder = builder.config(key, value)

            self.spark = builder.getOrCreate()
            self.sc = self.spark.sparkContext
            self.sc.setLogLevel("WARN")

            logger.info(
                f"Initialized Spark session: {self.spark_config.app_name} "
                f"on {self.spark_config.master}"
            )

        except ImportError:
            raise ImportError("PySpark not installed. Install with: pip install pyspark") from None

    def stop(self):
        """Stop Spark session."""
        if self.spark:
            self.spark.stop()
            logger.info("Spark session stopped")

    def load_data(self, job_config: SparkJobConfig):
        """
        Load data from input path.

        Args:
            job_config: Job configuration

        Returns:
            PySpark DataFrame
        """
        input_path = job_config.input_path

        if input_path.endswith(".parquet"):
            df = self.spark.read.parquet(input_path)
        elif input_path.endswith(".json") or input_path.endswith(".jsonl"):
            df = self.spark.read.json(input_path)
        elif input_path.endswith(".csv"):
            df = self.spark.read.csv(input_path, header=True)
        else:
            # Try to read as text
            df = self.spark.read.text(input_path).toDF("text")

        logger.info(f"Loaded data from {input_path}: {df.count()} rows")
        return df

    def preprocess(self, df, job_config: SparkJobConfig):
        """
        Apply preprocessing transformations.

        Args:
            df: Input DataFrame
            job_config: Job configuration

        Returns:
            Preprocessed DataFrame
        """
        from pyspark.sql.functions import udf
        from pyspark.sql.types import BooleanType, StringType, StructField, StructType

        logger.info("Starting preprocessing...")

        config = {
            "normalize": True,
            "quality_filter": job_config.quality_filter,
            "remove_pii": job_config.remove_pii,
            "min_length": 10,
        }

        # Define schema for preprocessed data
        output_schema = StructType(
            [
                StructField("text", StringType(), True),
                StructField("processed", BooleanType(), True),
            ]
        )

        # Apply preprocessing UDF
        preprocess_udf = udf(lambda row: spark_preprocess(row.asDict(), config), output_schema)

        df = df.select(preprocess_udf(df).alias("processed_data")).select("processed_data.*")

        # Filter out None results
        df = df.filter(df.text.isNotNull())

        logger.info(f"Preprocessing complete: {df.count()} rows")
        return df

    def deduplicate(self, df, job_config: SparkJobConfig):
        """
        Apply deduplication.

        Args:
            df: Input DataFrame
            job_config: Job configuration

        Returns:
            Deduplicated DataFrame
        """
        if not job_config.deduplicate:
            return df

        logger.info("Starting deduplication...")

        config = {"threshold": 0.9, "num_hash_tables": 5}

        # Add unique ID column
        from pyspark.sql.functions import monotonically_increasing_id

        df = df.withColumn("id", monotonically_increasing_id())

        result = spark_deduplicate(df, config)

        logger.info(f"Deduplication complete: {result.count()} rows")
        return result.drop("id")

    def tokenize(self, df, job_config: SparkJobConfig):
        """
        Apply tokenization.

        Args:
            df: Input DataFrame
            job_config: Job configuration

        Returns:
            Tokenized DataFrame
        """
        from pyspark.sql.functions import udf
        from pyspark.sql.types import (
            ArrayType,
            BooleanType,
            IntegerType,
            StringType,
            StructField,
            StructType,
        )

        logger.info("Starting tokenization...")

        config = {
            "tokenizer_name": job_config.tokenizer_name,
            "max_length": job_config.max_length,
        }

        # Define schema for tokenized data
        output_schema = StructType(
            [
                StructField("text", StringType(), True),
                StructField("input_ids", ArrayType(IntegerType()), True),
                StructField("attention_mask", ArrayType(IntegerType()), True),
                StructField("tokenized", BooleanType(), True),
            ]
        )

        # Apply tokenization UDF
        tokenize_udf = udf(lambda row: spark_tokenize(row.asDict(), config), output_schema)

        df = df.select(tokenize_udf(df).alias("tokenized_data")).select("tokenized_data.*")

        # Filter successfully tokenized rows
        df = df.filter(df.tokenized)

        logger.info(f"Tokenization complete: {df.count()} rows")
        return df

    def save_output(self, df, job_config: SparkJobConfig):
        """
        Save processed data to output path.

        Args:
            df: DataFrame to save
            job_config: Job configuration
        """
        output_path = job_config.output_path

        # Repartition if configured
        if job_config.repartition_output:
            df = df.repartition(job_config.num_partitions)

        # Save as parquet (default) or JSON
        if output_path.endswith(".json"):
            df.write.mode("overwrite").json(output_path)
        else:
            # Default to parquet with compression
            df.write.mode("overwrite").parquet(output_path, compression=job_config.compression)

        logger.info(f"Saved output to {output_path}")

    def run(self, job_config: SparkJobConfig) -> dict[str, Any]:
        """
        Run complete Spark pipeline.

        Args:
            job_config: Job configuration

        Returns:
            Job statistics
        """
        if not self.spark:
            self.initialize_spark()

        logger.info(f"Starting Spark pipeline job: {job_config.dataset_name}")

        # Load data
        df = self.load_data(job_config)
        initial_count = df.count()

        # Preprocess
        df = self.preprocess(df, job_config)

        # Deduplicate
        if job_config.deduplicate:
            df = self.deduplicate(df, job_config)

        # Tokenize
        df = self.tokenize(df, job_config)

        # Save output
        self.save_output(df, job_config)

        final_count = df.count()

        stats = {
            "dataset": job_config.dataset_name,
            "initial_samples": initial_count,
            "final_samples": final_count,
            "filtered_samples": initial_count - final_count,
            "output_path": job_config.output_path,
        }

        logger.info(f"Spark pipeline complete: {stats}")
        return stats


# Convenience function
def run_spark_pipeline(
    input_path: str,
    output_path: str,
    dataset_name: str,
    spark_config: SparkConfig | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Run Spark pipeline with simplified interface.

    Args:
        input_path: Path to input data
        output_path: Path to save output
        dataset_name: Name of dataset
        spark_config: Spark configuration (optional)
        **kwargs: Additional job configuration options

    Returns:
        Job statistics
    """
    spark_config = spark_config or SparkConfig()
    job_config = SparkJobConfig(
        input_path=input_path, output_path=output_path, dataset_name=dataset_name, **kwargs
    )

    runner = SparkPipelineRunner(spark_config)

    try:
        stats = runner.run(job_config)
        return stats
    finally:
        runner.stop()
