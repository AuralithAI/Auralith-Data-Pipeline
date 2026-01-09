"""Configuration for Spark integration."""

from dataclasses import dataclass


@dataclass
class SparkConfig:
    """Configuration for Spark pipeline execution."""

    app_name: str = "Auralith-Pipeline"
    master: str = "local[*]"  # local, yarn, spark://host:port

    # Spark configuration
    executor_memory: str = "4g"
    executor_cores: int = 4
    num_executors: int = 2
    driver_memory: str = "2g"

    # Dynamic allocation
    dynamic_allocation: bool = True
    min_executors: int = 1
    max_executors: int = 10

    # Shuffle settings
    shuffle_partitions: int = 200

    # Serialization
    serializer: str = "org.apache.spark.serializer.KryoSerializer"

    # Additional Spark configs
    spark_configs: dict[str, str] | None = None

    def to_spark_conf(self) -> dict[str, str]:
        """Convert to Spark configuration dictionary."""
        config = {
            "spark.app.name": self.app_name,
            "spark.master": self.master,
            "spark.executor.memory": self.executor_memory,
            "spark.executor.cores": str(self.executor_cores),
            "spark.executor.instances": str(self.num_executors),
            "spark.driver.memory": self.driver_memory,
            "spark.sql.shuffle.partitions": str(self.shuffle_partitions),
            "spark.serializer": self.serializer,
        }

        if self.dynamic_allocation:
            config.update(
                {
                    "spark.dynamicAllocation.enabled": "true",
                    "spark.dynamicAllocation.minExecutors": str(self.min_executors),
                    "spark.dynamicAllocation.maxExecutors": str(self.max_executors),
                }
            )

        if self.spark_configs:
            config.update(self.spark_configs)

        return config


@dataclass
class SparkJobConfig:
    """Configuration for a Spark processing job."""

    input_path: str
    output_path: str
    dataset_name: str

    # Processing options
    deduplicate: bool = True
    quality_filter: bool = True
    remove_pii: bool = True

    # Tokenization
    tokenizer_name: str = "gpt2"
    max_length: int = 2048

    # Sharding
    shard_size_mb: int = 1000
    compression: str = "zstd"

    # Partitioning
    num_partitions: int = 200
    repartition_output: bool = True
