"""Spark transformations for pipeline operations."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def spark_preprocess(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """
    Preprocess a single row in Spark.

    Args:
        row: Input data row
        config: Preprocessing configuration

    Returns:
        Preprocessed row
    """
    text = row.get("text", "")

    # Text normalization
    if config.get("normalize", True):
        text = text.strip().lower()

    # Quality filtering
    if config.get("quality_filter", True):
        min_length = config.get("min_length", 10)
        if len(text) < min_length:
            return None

    # PII removal (simplified)
    if config.get("remove_pii", True):
        import re

        # Remove email patterns
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", text)
        # Remove phone patterns
        text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", text)

    row["text"] = text
    row["processed"] = True
    return row


def spark_deduplicate(df, config: dict[str, Any]):
    """
    Deduplicate DataFrame using MinHash LSH.

    Args:
        df: PySpark DataFrame
        config: Deduplication configuration

    Returns:
        Deduplicated DataFrame
    """
    try:
        from pyspark.ml.feature import HashingTF, MinHashLSH, NGram

        # Create n-grams
        ngram = NGram(n=5, inputCol="text", outputCol="ngrams")
        ngram_df = ngram.transform(df)

        # Convert to features
        hashing_tf = HashingTF(inputCol="ngrams", outputCol="features", numFeatures=1000)
        hashed_df = hashing_tf.transform(ngram_df)

        # Apply MinHash LSH
        mh = MinHashLSH(
            inputCol="features", outputCol="hashes", numHashTables=config.get("num_hash_tables", 5)
        )
        model = mh.fit(hashed_df)

        # Find approximate duplicates
        threshold = config.get("threshold", 0.9)
        duplicates = model.approxSimilarityJoin(
            hashed_df, hashed_df, threshold, distCol="distance"
        ).filter("datasetA.id != datasetB.id")

        # Keep only one from each duplicate pair
        dup_ids = duplicates.select("datasetB.id").distinct()
        result = hashed_df.join(dup_ids, hashed_df.id == dup_ids.id, "left_anti")

        logger.info(f"Deduplication removed {hashed_df.count() - result.count()} duplicates")
        return result.select(df.columns)

    except ImportError:
        logger.warning("PySpark ML not available, skipping deduplication")
        return df


def spark_tokenize(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """
    Tokenize text in a Spark row.

    Args:
        row: Input data row
        config: Tokenization configuration

    Returns:
        Row with tokenized data
    """
    text = row.get("text", "")
    tokenizer_name = config.get("tokenizer_name", "gpt2")
    max_length = config.get("max_length", 2048)

    try:
        # Lazy import to avoid loading on executor startup
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokens = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )

        row["input_ids"] = tokens["input_ids"][0].tolist()
        row["attention_mask"] = tokens["attention_mask"][0].tolist()
        row["tokenized"] = True

    except Exception as e:
        logger.error(f"Tokenization failed: {e}")
        row["tokenized"] = False

    return row


def broadcast_tokenizer(spark_context, tokenizer_name: str):
    """
    Broadcast tokenizer to Spark executors.

    Args:
        spark_context: SparkContext
        tokenizer_name: Name of tokenizer to broadcast

    Returns:
        Broadcast variable containing tokenizer
    """
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        return spark_context.broadcast(tokenizer)
    except Exception as e:
        logger.error(f"Failed to broadcast tokenizer: {e}")
        return None
