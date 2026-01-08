"""Main Pipeline class that orchestrates the entire data processing workflow."""

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from auralith_pipeline.config.pipeline_config import PipelineConfig
from auralith_pipeline.preprocessing.preprocessor import DataPreprocessor
from auralith_pipeline.sharding.shard_writer import ShardWriter
from auralith_pipeline.sources.data_sources import DataSample, DataSource
from auralith_pipeline.tokenization.tokenizer import TokenizationPipeline, TokenizedSample
from auralith_pipeline.utils.helpers import format_number, format_size

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Statistics from a pipeline run."""

    total_samples: int = 0
    samples_after_filter: int = 0
    samples_tokenized: int = 0
    duplicates_removed: int = 0
    pii_removed: int = 0
    num_shards: int = 0
    total_tokens: int = 0
    total_size_bytes: int = 0
    elapsed_time_seconds: float = 0

    def summary(self) -> str:
        """Generate summary string."""
        return f"""
Pipeline Statistics:
  Input samples:      {format_number(self.total_samples)}
  After filtering:    {format_number(self.samples_after_filter)}
  Duplicates removed: {format_number(self.duplicates_removed)}
  PII removed:        {format_number(self.pii_removed)}
  Tokens generated:   {format_number(self.total_tokens)}
  Shards created:     {format_number(self.num_shards)}
  Total size:         {format_size(self.total_size_bytes)}
  Elapsed time:       {self.elapsed_time_seconds:.2f}s
  Throughput:         {self.total_samples / max(self.elapsed_time_seconds, 1):.2f} samples/sec
"""


class Pipeline:
    """Main data processing pipeline."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._sources: list[DataSource] = []
        self._preprocessor: DataPreprocessor | None = None
        self._tokenizer: TokenizationPipeline | None = None
        self._shard_writer: ShardWriter | None = None

    def add_source(self, source: DataSource):
        """Add a data source to the pipeline."""
        self._sources.append(source)
        logger.info(f"Added source: {source.name}")

    def _setup_components(self):
        """Initialize pipeline components."""
        # Preprocessor
        self._preprocessor = DataPreprocessor(
            quality_config=self.config.quality,
            dedup_config=self.config.deduplication,
            normalize=self.config.normalize_text,
            remove_pii=self.config.remove_pii,
        )

        # Tokenizer
        self._tokenizer = TokenizationPipeline(
            tokenizer_path=self.config.tokenization.tokenizer_path,
            vocab_size=self.config.tokenization.vocab_size,
            max_length=self.config.tokenization.max_length,
        )

        # Shard writer
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        self._shard_writer = ShardWriter(
            output_dir=self.config.output_dir,
            config=self.config.sharding,
        )

    def _iterate_sources(self) -> Iterator[DataSample]:
        """Iterate over all sources."""
        for source in self._sources:
            logger.info(f"Processing source: {source.name}")
            yield from source

    def run(
        self,
        max_samples: int | None = None,
        progress_callback: callable | None = None,
    ) -> PipelineStats:
        """Run the pipeline."""
        start_time = time.time()
        stats = PipelineStats()

        # Override max_samples from config if provided
        max_samples = max_samples or self.config.max_samples

        # Setup components
        self._setup_components()

        logger.info(f"Starting pipeline: {self.config.name}")
        logger.info(f"Output directory: {self.config.output_dir}")

        # Process samples
        sample_count = 0
        token_count = 0

        try:
            # Create sample iterator
            samples = self._iterate_sources()

            # Apply preprocessing
            if self.config.deduplicate or self.config.quality_filter or self.config.remove_pii:
                samples = self._preprocessor.process(samples)

            # Process each sample
            for sample in samples:
                if max_samples and sample_count >= max_samples:
                    break

                stats.total_samples += 1

                # Tokenize
                for tokenized in self._tokenizer.process(sample.content, sample.metadata):
                    self._shard_writer.add_sample(tokenized)
                    token_count += tokenized.length
                    stats.samples_tokenized += 1

                sample_count += 1

                # Progress callback
                if progress_callback and sample_count % 1000 == 0:
                    progress_callback(sample_count)

                # Log progress
                if sample_count % 10000 == 0:
                    logger.info(f"Processed {format_number(sample_count)} samples...")

            # Finalize shards
            shard_index = self._shard_writer.finalize()

            # Update stats
            stats.samples_after_filter = sample_count
            stats.total_tokens = token_count
            stats.num_shards = shard_index.total_shards
            stats.total_size_bytes = shard_index.total_size_bytes

            if self._preprocessor:
                stats.duplicates_removed = self._preprocessor.stats["duplicates_removed"]
                stats.pii_removed = self._preprocessor.stats["pii_removed"]

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

        stats.elapsed_time_seconds = time.time() - start_time

        logger.info(stats.summary())

        return stats

    def run_streaming(
        self,
        batch_size: int = 100,
    ) -> Iterator[list[TokenizedSample]]:
        """Run pipeline in streaming mode, yielding batches of tokenized samples."""
        self._setup_components()

        batch = []
        samples = self._iterate_sources()

        if self.config.deduplicate or self.config.quality_filter:
            samples = self._preprocessor.process(samples)

        for sample in samples:
            for tokenized in self._tokenizer.process(sample.content, sample.metadata):
                batch.append(tokenized)

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

        if batch:
            yield batch
