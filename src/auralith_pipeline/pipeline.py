"""Main Pipeline class that orchestrates the entire data processing workflow."""

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    perplexity_filtered: int = 0
    llm_judge_filtered: int = 0
    embedding_dedup_removed: int = 0
    license_blocked: int = 0
    synthetic_generated: int = 0
    modalities: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate summary string."""
        lines = f"""
Pipeline Statistics:
  Input samples:        {format_number(self.total_samples)}
  After filtering:      {format_number(self.samples_after_filter)}
  Duplicates removed:   {format_number(self.duplicates_removed)}
  PII removed:          {format_number(self.pii_removed)}
  Tokens generated:     {format_number(self.total_tokens)}
  Shards created:       {format_number(self.num_shards)}
  Total size:           {format_size(self.total_size_bytes)}
  Elapsed time:         {self.elapsed_time_seconds:.2f}s
  Throughput:           {self.total_samples / max(self.elapsed_time_seconds, 1):.2f} samples/sec
"""
        if self.embedding_dedup_removed:
            lines += f"  Embedding dedup:      {format_number(self.embedding_dedup_removed)}\n"
        if self.perplexity_filtered:
            lines += f"  Perplexity filtered:  {format_number(self.perplexity_filtered)}\n"
        if self.llm_judge_filtered:
            lines += f"  LLM judge filtered:   {format_number(self.llm_judge_filtered)}\n"
        if self.license_blocked:
            lines += f"  License blocked:      {format_number(self.license_blocked)}\n"
        if self.synthetic_generated:
            lines += f"  Synthetic generated:  {format_number(self.synthetic_generated)}\n"
        if self.modalities:
            lines += f"  Modalities:           {self.modalities}\n"
        return lines

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to a flat dictionary for tracking."""
        return {
            "total_samples": self.total_samples,
            "samples_after_filter": self.samples_after_filter,
            "samples_tokenized": self.samples_tokenized,
            "duplicates_removed": self.duplicates_removed,
            "pii_removed": self.pii_removed,
            "num_shards": self.num_shards,
            "total_tokens": self.total_tokens,
            "total_size_bytes": self.total_size_bytes,
            "elapsed_time_seconds": self.elapsed_time_seconds,
            "perplexity_filtered": self.perplexity_filtered,
            "llm_judge_filtered": self.llm_judge_filtered,
            "embedding_dedup_removed": self.embedding_dedup_removed,
            "license_blocked": self.license_blocked,
            "synthetic_generated": self.synthetic_generated,
        }


class Pipeline:
    """Main data processing pipeline."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._sources: list[DataSource] = []
        self._preprocessor: DataPreprocessor | None = None
        self._tokenizer: TokenizationPipeline | None = None
        self._shard_writer: ShardWriter | None = None

        self._advanced_quality: Any = None
        self._embedding_dedup: Any = None
        self._license_detector: Any = None
        self._audit_logger: Any = None
        self._experiment_tracker: Any = None
        self._lineage_tracker: Any = None
        self._pii_scrubber: Any = None
        self._data_sanitizer: Any = None
        self._privacy_audit: Any = None

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

        if self.config.advanced_quality.enabled:
            self._setup_advanced_quality()

        if self.config.tracking.enabled:
            self._setup_tracking()

        if self.config.compliance.enabled:
            self._setup_compliance()

        if self.config.security.enabled:
            self._setup_security()

    def _setup_advanced_quality(self) -> None:
        """Set up advanced quality pipeline."""
        try:
            from auralith_pipeline.preprocessing.quality import AdvancedQualityPipeline

            aq_cfg = self.config.advanced_quality
            perplexity_config = (
                {
                    "model_name": aq_cfg.perplexity_model,
                    "max_perplexity": aq_cfg.max_perplexity,
                    "min_perplexity": aq_cfg.min_perplexity,
                }
                if aq_cfg.perplexity_filter
                else None
            )

            llm_judge_config = (
                {
                    "provider": aq_cfg.llm_judge_provider,
                    "model_name": aq_cfg.llm_judge_model,
                }
                if aq_cfg.llm_judge
                else None
            )

            self._advanced_quality = AdvancedQualityPipeline(
                enable_perplexity=aq_cfg.perplexity_filter,
                enable_llm_judge=aq_cfg.llm_judge,
                perplexity_config=perplexity_config,
                llm_judge_config=llm_judge_config,
            )
            logger.info("Advanced quality pipeline enabled")
        except Exception as e:
            logger.warning(f"Could not initialize advanced quality: {e}")

        if self.config.deduplication.method == "embedding":
            try:
                from auralith_pipeline.preprocessing.deduplication import EmbeddingDeduplicator

                self._embedding_dedup = EmbeddingDeduplicator(
                    model_name=self.config.deduplication.embedding_model,
                    similarity_threshold=self.config.deduplication.embedding_threshold,
                )
                logger.info("FAISS embedding deduplication enabled")
            except Exception as e:
                logger.warning(f"Could not initialize embedding dedup: {e}")

    def _setup_tracking(self) -> None:
        """Set up experiment tracking and lineage."""
        from auralith_pipeline.utils.tracking import ExperimentTracker, LineageTracker

        t_cfg = self.config.tracking
        self._experiment_tracker = ExperimentTracker(
            backend=t_cfg.backend,
            project_name=t_cfg.project_name,
            experiment_name=t_cfg.experiment_name,
            run_name=t_cfg.run_name or f"{self.config.name}-{int(time.time())}",
        )
        self._experiment_tracker.start_run()
        self._experiment_tracker.log_params(self.config.to_dict())

        if t_cfg.lineage:
            self._lineage_tracker = LineageTracker(output_dir=self.config.output_dir)

    def _setup_compliance(self) -> None:
        """Set up compliance components."""
        from auralith_pipeline.preprocessing.compliance import AuditLogger, LicenseDetector

        c_cfg = self.config.compliance
        if c_cfg.license_detection:
            self._license_detector = LicenseDetector(
                allow_permissive=c_cfg.allow_permissive,
                allow_copyleft=c_cfg.allow_copyleft,
            )
            logger.info("License detection enabled")

        self._audit_logger = AuditLogger(log_path=c_cfg.audit_log_path)

    def _setup_security(self) -> None:
        """Set up security layer — PII scrubbing + data sanitization.

        This is the dedicated security module that ensures NO private user
        data from any country enters the training pipeline.
        """
        from auralith_pipeline.security.audit import PrivacyAuditLogger
        from auralith_pipeline.security.data_sanitizer import DataSanitizer
        from auralith_pipeline.security.pii_scrubber import PIIScrubber
        from auralith_pipeline.security.privacy_config import PrivacyConfig

        s_cfg = self.config.security

        privacy_config = PrivacyConfig(
            enabled=s_cfg.enabled,
            mode=s_cfg.mode,
            replacement_style=s_cfg.replacement_style,
            rescan_after_processing=s_cfg.rescan_after_processing,
            log_redactions=s_cfg.log_redactions,
            fail_on_pii=s_cfg.fail_on_pii,
        )

        self._pii_scrubber = PIIScrubber(config=privacy_config)
        self._data_sanitizer = DataSanitizer(
            enabled=s_cfg.sanitize_secrets,
            block_internal_urls=s_cfg.block_internal_urls,
        )
        self._privacy_audit = PrivacyAuditLogger(
            log_path=s_cfg.audit_log_path,
            enabled=s_cfg.log_redactions,
        )

        logger.info(
            f"Security layer enabled: mode={s_cfg.mode}, "
            f"rescan={s_cfg.rescan_after_processing}, "
            f"secrets={s_cfg.sanitize_secrets}"
        )

    def _iterate_sources(self) -> Iterator[DataSample]:
        """Iterate over all sources."""
        for source in self._sources:
            logger.info(f"Processing source: {source.name}")
            yield from source

    def _apply_advanced_filters(
        self,
        sample: DataSample,
        stats: PipelineStats,
    ) -> bool:
        """Apply security, advanced quality, and compliance filters. Returns True if sample passes."""

        # ── Security layer: PII scrubbing (runs first) ──
        if self._pii_scrubber:
            scrub_result = self._pii_scrubber.scrub(sample.content)
            if scrub_result.pii_found:
                sample.content = scrub_result.cleaned_text
                stats.pii_removed += 1
                if self._privacy_audit:
                    self._privacy_audit.log_redaction(
                        sample_id=sample.metadata.get("id", f"s_{id(sample)}"),
                        source=sample.source,
                        categories=[c.value for c in scrub_result.categories_found],
                        redaction_count=scrub_result.count,
                    )

        # ── Security layer: credential / secret sanitization ──
        if self._data_sanitizer:
            sanitize_result = self._data_sanitizer.sanitize(sample.content)
            if sanitize_result.had_issues:
                sample.content = sanitize_result.cleaned_text

        # ── Security rescan: fail-safe check after all processing ──
        if self._pii_scrubber and self.config.security.fail_on_pii:
            if self._pii_scrubber.has_pii(sample.content):
                logger.warning(f"PII still found after scrubbing — blocking sample {sample.source}")
                if self._privacy_audit:
                    self._privacy_audit.log_blocked(
                        sample_id=sample.metadata.get("id", f"s_{id(sample)}"),
                        source=sample.source,
                        reason="pii_rescan_failed",
                    )
                return False

        # License check for code samples
        if self._license_detector and getattr(sample, "modality", None) == "code":
            if not self._license_detector.is_allowed(sample):
                stats.license_blocked += 1
                if self._audit_logger:
                    self._audit_logger.log(
                        sample_id=sample.metadata.get("id", "unknown"),
                        action="reject",
                        reason="license_not_allowed",
                        details={"license": sample.metadata.get("detected_license")},
                    )
                return False

        # Advanced quality (perplexity + LLM judge)
        if self._advanced_quality:
            passed, scores = self._advanced_quality.evaluate(sample)
            if not passed:
                if scores.get("perplexity", -1) < 0:
                    stats.perplexity_filtered += 1
                if scores.get("coherence", 1.0) < 0.4:
                    stats.llm_judge_filtered += 1
                return False

        # Embedding dedup
        if self._embedding_dedup:
            if self._embedding_dedup.is_duplicate(sample.content):
                stats.embedding_dedup_removed += 1
                return False

        return True

    def run(
        self,
        max_samples: int | None = None,
        progress_callback: Callable[[int], None] | None = None,
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

                # advanced filters
                if not self._apply_advanced_filters(sample, stats):
                    continue

                # Track modality
                modality = sample.metadata.get("modality", "text")
                stats.modalities[modality] = stats.modalities.get(modality, 0) + 1

                # Lineage record
                lineage_record = None
                if self._lineage_tracker:
                    lineage_record = self._lineage_tracker.create_record(
                        source=sample.source,
                        modality=modality,
                    )

                # Tokenize
                for tokenized in self._tokenizer.process(sample.content, sample.metadata):
                    self._shard_writer.add_sample(tokenized)
                    token_count += tokenized.length
                    stats.samples_tokenized += 1

                    if lineage_record:
                        previous_count = getattr(lineage_record, "token_count", 0) or 0
                        lineage_record.token_count = previous_count + tokenized.length

                # Audit log acceptance
                if self._audit_logger:
                    self._audit_logger.log(
                        sample_id=sample.metadata.get("id", f"sample_{sample_count}"),
                        action="accept",
                        reason="passed_all_filters",
                    )

                sample_count += 1

                # Progress callback
                if progress_callback and sample_count % 1000 == 0:
                    progress_callback(sample_count)

                # Log progress + metrics
                if sample_count % 10000 == 0:
                    logger.info(f"Processed {format_number(sample_count)} samples...")
                    if self._experiment_tracker:
                        self._experiment_tracker.log_metrics(
                            {"samples_processed": sample_count, "tokens": token_count},
                            step=sample_count,
                        )

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
            if self._experiment_tracker:
                self._experiment_tracker.end_run(status="failed")
            raise

        stats.elapsed_time_seconds = time.time() - start_time

        # ------- finalize tracking -------
        self._finalize_tracking(stats)

        # ------- Security: flush privacy audit -------
        if self._privacy_audit:
            self._privacy_audit.close()
            logger.info(f"Privacy audit: {self._privacy_audit.summary()}")
        if self._pii_scrubber:
            logger.info(f"PII scrubber: {self._pii_scrubber.summary()}")
        if self._data_sanitizer:
            logger.info(f"Data sanitizer: {self._data_sanitizer.summary()}")

        logger.info(stats.summary())

        return stats

    def _finalize_tracking(self, stats: PipelineStats) -> None:
        """Finalize tracking: metrics, lineage, data card."""
        if self._experiment_tracker:
            self._experiment_tracker.log_metrics(
                {k: float(v) for k, v in stats.to_dict().items() if isinstance(v, (int, float))}
            )
            self._experiment_tracker.end_run(status="success")

        if self._lineage_tracker:
            self._lineage_tracker.save()
            if self._experiment_tracker:
                lineage_path = Path(self.config.output_dir) / "lineage.jsonl"
                if lineage_path.exists():
                    self._experiment_tracker.log_artifact(str(lineage_path))

        # Auto data card
        if self.config.tracking.enabled and self.config.tracking.data_cards:
            try:
                from auralith_pipeline.utils.tracking import DataCardGenerator

                card_gen = DataCardGenerator()
                card_gen.generate(
                    output_path=str(Path(self.config.output_dir) / "README.md"),
                    config=self.config.to_dict(),
                    stats=stats.to_dict(),
                    lineage_summary=(
                        self._lineage_tracker.summary() if self._lineage_tracker else None
                    ),
                )
            except Exception as e:
                logger.warning(f"Could not generate data card: {e}")

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
