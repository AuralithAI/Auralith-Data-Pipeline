"""Main Pipeline class that orchestrates the entire data processing workflow."""

import itertools
import json
import logging
import random
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

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
        self._sources: list[tuple[DataSource, float]] = []  # (source, weight)
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

    def add_source(self, source: DataSource, weight: float = 1.0) -> None:
        """Add a data source to the pipeline.

        Args:
            source: The data source to add.
            weight: Relative sampling weight used for round-robin interleaving
                    when multiple sources are present.  Default 1.0 (equal share).
        """
        self._sources.append((source, weight))
        logger.info(f"Added source: {source.name} (weight={weight})")

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

        if self.config.deduplication.method == "embedding":
            self._setup_embedding_dedup()

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
                    **(
                        {"model_name": aq_cfg.llm_judge_model}
                        if aq_cfg.llm_judge_model is not None
                        else {}
                    ),
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

    def _setup_embedding_dedup(self) -> None:
        """Set up FAISS embedding deduplication (independent of advanced quality)."""
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
        """Iterate over all sources using weighted round-robin interleaving.

        Each source is assigned a fractional credit equal to its normalised weight.
        At every step the source with the highest accumulated credit is chosen,
        one sample is yielded from it, and its credit is decremented by 1.
        Sources are removed from rotation as they are exhausted.
        """
        if not self._sources:
            return

        if len(self._sources) == 1:
            source, _ = self._sources[0]
            logger.info(f"Processing source: {source.name}")
            yield from source
            return

        total_weight = sum(w for _, w in self._sources)
        norm_weights = [w / total_weight for _, w in self._sources]
        iterators = [iter(src) for src, _ in self._sources]
        exhausted = [False] * len(self._sources)
        rr_credits = [0.0] * len(self._sources)

        for src, _ in self._sources:
            logger.info(f"Processing source: {src.name}")

        while not all(exhausted):
            for i, w in enumerate(norm_weights):
                if not exhausted[i]:
                    rr_credits[i] += w

            active = [i for i in range(len(self._sources)) if not exhausted[i]]
            if not active:
                break
            best = max(active, key=lambda i: rr_credits[i])

            try:
                yield next(iterators[best])
                rr_credits[best] -= 1.0
            except StopIteration:
                exhausted[best] = True

    # ------------------------------------------------------------------ #
    #  Per-sample filter stages (extracted to keep run() simple)          #
    # ------------------------------------------------------------------ #

    def _pii_rescan_blocked(self, sample: DataSample) -> bool:
        """Fail-safe PII rescan after scrubbing. Returns True when sample must be blocked."""
        if not (self._pii_scrubber and self.config.security.fail_on_pii):
            return False
        if not self._pii_scrubber.has_pii(sample.content):
            return False
        logger.warning(f"PII persists after scrub — blocking {sample.source}")
        if self._privacy_audit:
            self._privacy_audit.log_blocked(
                sample_id=sample.metadata.get("id", f"s_{id(sample)}"),
                source=sample.source,
                reason="pii_rescan_failed",
            )
        return True

    def _stage_security(
        self, sample: DataSample, stats: PipelineStats
    ) -> tuple[DataSample, bool]:
        """Stage 3 — PII scrubbing + secrets sanitisation.

        Returns (sample, passed). Sample content is mutated in-place when PII is found.
        """
        if self._pii_scrubber:
            scrub_result = self._pii_scrubber.scrub(sample.content)
            if scrub_result.pii_found:
                sample.content = scrub_result.cleaned_text
                stats.pii_removed += scrub_result.count
                if self._privacy_audit:
                    self._privacy_audit.log_redaction(
                        sample_id=sample.metadata.get("id", f"s_{id(sample)}"),
                        source=sample.source,
                        categories=[c.value for c in scrub_result.categories_found],
                        redaction_count=scrub_result.count,
                    )

        if self._data_sanitizer:
            sanitize_result = self._data_sanitizer.sanitize(sample.content)
            if sanitize_result.had_issues:
                sample.content = sanitize_result.cleaned_text

        return sample, not self._pii_rescan_blocked(sample)

    def _stage_embedding_dedup(self, sample: DataSample, stats: PipelineStats) -> bool:
        """Stage 4 — FAISS embedding dedup. Returns True when sample is a duplicate."""
        if self._embedding_dedup and self._embedding_dedup.is_duplicate(sample.content):
            stats.embedding_dedup_removed += 1
            return True
        return False

    def _stage_advanced_quality(self, sample: DataSample, stats: PipelineStats) -> bool:
        """Stage 5 — Perplexity filter + LLM judge. Returns True when sample passes."""
        if not self._advanced_quality:
            return True
        prev_ppl = self._advanced_quality.stats["failed_perplexity"]
        prev_judge = self._advanced_quality.stats["failed_llm_judge"]
        passed, _ = self._advanced_quality.evaluate(sample)
        if not passed:
            if self._advanced_quality.stats["failed_perplexity"] > prev_ppl:
                stats.perplexity_filtered += 1
            if self._advanced_quality.stats["failed_llm_judge"] > prev_judge:
                stats.llm_judge_filtered += 1
        return passed

    def _stage_compliance(self, sample: DataSample, stats: PipelineStats) -> bool:
        """Stage 6 — Licence detection. Returns True when sample is allowed."""
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
        return True

    def _tokenize_and_shard(
        self, sample: DataSample, lineage_record: Any, stats: PipelineStats
    ) -> int:
        """Tokenise one sample and write to shard. Returns number of tokens added."""
        assert self._tokenizer is not None
        assert self._shard_writer is not None
        added = 0
        for tokenized in self._tokenizer.process(sample.content, sample.metadata):
            self._shard_writer.add_sample(tokenized)
            added += tokenized.length
            stats.samples_tokenized += 1
            if lineage_record is not None:
                previous = getattr(lineage_record, "token_count", 0) or 0
                lineage_record.token_count = previous + tokenized.length
        return added

    def _run_one_sample(
        self, raw_sample: DataSample, stats: PipelineStats, sample_idx: int
    ) -> tuple[bool, int]:
        """Run all 6 filter stages + tokenise for a single raw sample.

        Returns (accepted, tokens_added). tokens_added is 0 when rejected.
        """
        assert self._preprocessor is not None

        # Stage 1+2: normalise + basic quality
        processed = list(self._preprocessor.process(iter([raw_sample])))
        if not processed:
            return False, 0
        sample = processed[0]

        # Stage 3: security
        sample, security_ok = self._stage_security(sample, stats)
        if not security_ok:
            return False, 0

        # Stage 4: embedding dedup
        if self._stage_embedding_dedup(sample, stats):
            return False, 0

        # Stage 5: advanced quality
        if not self._stage_advanced_quality(sample, stats):
            return False, 0

        # Stage 6: compliance
        if not self._stage_compliance(sample, stats):
            return False, 0

        # Passed all filters — tokenise and shard
        modality = sample.modality
        stats.modalities[modality] = stats.modalities.get(modality, 0) + 1

        lineage_record = None
        if self._lineage_tracker:
            lineage_record = self._lineage_tracker.create_record(
                source=sample.source, modality=modality
            )

        tokens = self._tokenize_and_shard(sample, lineage_record, stats)

        if self._audit_logger:
            self._audit_logger.log(
                sample_id=sample.metadata.get("id", f"sample_{sample_idx}"),
                action="accept",
                reason="passed_all_filters",
            )

        return True, tokens

    def _periodic_checkpoint(
        self, n_accepted: int, token_count: int, raw_total: int
    ) -> None:
        """Save checkpoint and emit metrics every checkpoint_every accepted samples."""
        self._save_checkpoint(n_accepted, raw_total)
        logger.info(f"Processed {format_number(n_accepted)} samples…")
        if self._experiment_tracker:
            self._experiment_tracker.log_metrics(
                {"samples_processed": n_accepted, "tokens": token_count},
                step=n_accepted,
            )

    def _flush_security_logs(self) -> None:
        """Close and log security audit outputs at pipeline end."""
        if self._privacy_audit:
            self._privacy_audit.close()
            logger.info(f"Privacy audit: {self._privacy_audit.summary()}")
        if self._pii_scrubber:
            logger.info(f"PII scrubber: {self._pii_scrubber.summary()}")
        if self._data_sanitizer:
            logger.info(f"Data sanitizer: {self._data_sanitizer.summary()}")

    # ------------------------------------------------------------------ #
    #  Checkpoint helpers                                                  #
    # ------------------------------------------------------------------ #

    def _checkpoint_path(self) -> Path:
        return Path(
            self.config.checkpoint_path
            or Path(self.config.output_dir) / ".pipeline_checkpoint.json"
        )

    def _save_checkpoint(self, samples_accepted: int, raw_total: int) -> None:
        rng_state = np.random.get_state()  # tuple (str, ndarray, int, int, float)
        ckpt = {
            "samples_accepted": samples_accepted,
            "raw_total": raw_total,
            "seed": self.config.seed,
            "numpy_rng_state": rng_state[1].tolist(),  # type: ignore[index]
        }
        path = self._checkpoint_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ckpt))
        logger.debug(f"Checkpoint saved ({samples_accepted} accepted so far)")

    def _load_checkpoint(self) -> dict | None:
        path = self._checkpoint_path()
        if path.exists():
            try:
                ckpt = json.loads(path.read_text())
                logger.info(
                    f"Resuming from checkpoint: {ckpt['samples_accepted']} samples already accepted"
                )
                return ckpt
            except Exception as e:
                logger.warning(f"Could not load checkpoint, starting fresh: {e}")
        return None

    def _restore_rng_state(self, checkpoint: dict) -> None:
        """Best-effort restoration of numpy RNG state from a checkpoint dict."""
        try:
            state = list(np.random.get_state())   # mutable copy of the tuple
            state[1] = np.array(checkpoint["numpy_rng_state"], dtype=np.uint32)  # type: ignore[index]
            np.random.set_state(tuple(state))      # type: ignore[arg-type]
        except Exception as exc:
            logger.warning(f"RNG state restoration skipped: {exc}")

    def _init_from_checkpoint(self) -> tuple[int, int]:
        """Load checkpoint if present, restore RNG, return (skip_count, samples_accepted)."""
        checkpoint = self._load_checkpoint()
        if not checkpoint:
            return 0, 0
        self._restore_rng_state(checkpoint)
        return checkpoint["raw_total"], checkpoint["samples_accepted"]

    # ------------------------------------------------------------------ #
    #  Main run                                                            #
    # ------------------------------------------------------------------ #

    def run(
        self,
        max_samples: int | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> PipelineStats:
        """Run the pipeline.

        Filter execution order (per sample):
          Stage 1+2 — Normalise + basic quality filter  (preprocessor)
          Stage 3   — Security scrubbing  (PII + secrets, before dedup fingerprint)
          Stage 4   — Embedding dedup    (FAISS)
          Stage 5   — Advanced quality   (perplexity, LLM judge)
          Stage 6   — Compliance         (licence detection)
          → Tokenise → Shard
        """
        start_time = time.time()
        stats = PipelineStats()

        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

        max_samples = max_samples or self.config.max_samples

        skip_count, samples_accepted = self._init_from_checkpoint()

        self._setup_components()
        assert self._preprocessor is not None, "_setup_components must initialise _preprocessor"

        logger.info(f"Starting pipeline: {self.config.name}")
        logger.info(f"Output: {self.config.output_dir} | resume offset: {skip_count} samples")

        token_count = 0

        try:
            raw_iter = self._iterate_sources()
            deque(itertools.islice(raw_iter, skip_count), maxlen=0)

            for raw_sample in raw_iter:
                if max_samples and samples_accepted >= max_samples:
                    break

                stats.total_samples += 1
                accepted, tokens = self._run_one_sample(raw_sample, stats, samples_accepted)
                if not accepted:
                    continue

                token_count += tokens
                samples_accepted += 1

                if progress_callback and samples_accepted % 1000 == 0:
                    progress_callback(samples_accepted)

                if samples_accepted % self.config.checkpoint_every == 0:
                    self._periodic_checkpoint(samples_accepted, token_count, stats.total_samples)

            assert self._shard_writer is not None
            shard_index = self._shard_writer.finalize()

            stats.samples_after_filter = samples_accepted
            stats.total_tokens = token_count
            stats.num_shards = shard_index.total_shards
            stats.total_size_bytes = shard_index.total_size_bytes
            stats.duplicates_removed = self._preprocessor.stats.get("duplicates_removed", 0)
            stats.pii_removed += self._preprocessor.stats.get("pii_removed", 0)
            self._checkpoint_path().unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            if self._experiment_tracker:
                self._experiment_tracker.end_run(status="failed")
            raise

        stats.elapsed_time_seconds = time.time() - start_time
        self._finalize_tracking(stats)
        self._flush_security_logs()
        logger.info(stats.summary())
        return stats

    def _finalize_tracking(self, stats: PipelineStats) -> None:
        """Finalize tracking: metrics, lineage, data card."""
        if self._experiment_tracker:
            self._experiment_tracker.log_metrics(
                {k: float(v) for k, v in stats.to_dict().items() if isinstance(v, int | float)}
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
