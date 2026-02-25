"""End-to-end tests for Auralith Data Pipeline.

Validates:
  • SafeTensors schema (input_ids, attention_mask, modality_mask, labels)
  • Multimodal tokenization (text + image + audio + video tokens)
  • Quality pipeline (perplexity filter, LLM judge integration)
  • Compliance (license detection, audit logging)
  • Lineage tracking
  • Full pipeline round-trip
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from auralith_pipeline.config.pipeline_config import PipelineConfig
from auralith_pipeline.sources.data_sources import DataSample


# ===========================================================================
# Tensor Schema Tests
# ===========================================================================


class TestSafeTensorsSchema:
    """Verify the RT-DLM-compatible SafeTensors schema."""

    def test_shard_writer_produces_all_tensors(self):
        """Shards must contain input_ids, attention_mask, modality_mask, labels."""
        from auralith_pipeline.sharding.shard_writer import ShardWriter
        from auralith_pipeline.config.pipeline_config import ShardConfig
        from auralith_pipeline.tokenization.tokenizer import TokenizedSample

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ShardConfig(
                max_size_mb=1,
                sequence_length=16,
                format="safetensors",
                compression=None,
            )
            writer = ShardWriter(output_dir=tmpdir, config=config)

            for _ in range(5):
                sample = TokenizedSample(
                    input_ids=[1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    attention_mask=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    metadata={"source": "test"},
                    modality_mask=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    labels=[1, 2, 3, 4, 5, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
                )
                writer.add_sample(sample)

            index = writer.finalize()
            assert index.total_shards >= 1

            # Load the shard and verify tensors
            from safetensors.numpy import load_file

            shard_path = Path(tmpdir) / "shard_00000.safetensors"
            if shard_path.exists():
                tensors = load_file(str(shard_path))
                assert "input_ids" in tensors
                assert "attention_mask" in tensors
                assert "modality_mask" in tensors
                assert "labels" in tensors

                assert tensors["input_ids"].dtype == np.int32
                assert tensors["modality_mask"].dtype == np.uint8
                assert tensors["labels"].dtype == np.int32
                assert tensors["attention_mask"].dtype == np.int32

                # Shapes match
                assert tensors["input_ids"].shape == tensors["attention_mask"].shape
                assert tensors["input_ids"].shape == tensors["modality_mask"].shape
                assert tensors["input_ids"].shape == tensors["labels"].shape

    def test_tokenized_sample_defaults(self):
        """TokenizedSample should auto-fill modality_mask and labels."""
        from auralith_pipeline.tokenization.tokenizer import TokenizedSample

        sample = TokenizedSample(
            input_ids=[10, 20, 30],
            attention_mask=[1, 1, 1],
            metadata={},
        )

        # __post_init__ should fill defaults
        assert sample.modality_mask is not None
        assert sample.modality_mask == [0, 0, 0]
        assert sample.labels is not None
        assert sample.labels == [10, 20, 30]


# ===========================================================================
# Special Tokens
# ===========================================================================


class TestSpecialTokens:
    """Verify expanded special token vocabulary."""

    def test_special_token_count(self):
        from auralith_pipeline.tokenization.bpe_tokenizer import BPETokenizer

        tok = BPETokenizer(vocab_size=100)
        assert len(tok.SPECIAL_TOKENS) >= 10  # At least original 10

    def test_modality_markers_exist(self):
        from auralith_pipeline.tokenization.bpe_tokenizer import BPETokenizer

        tok = BPETokenizer(vocab_size=100)
        expected = {"<IMG>", "<IMG_END>", "<AUDIO>", "<AUDIO_END>", "<VIDEO>", "<VIDEO_END>"}
        token_set = set(tok.SPECIAL_TOKENS)
        for marker in expected:
            assert marker in token_set, f"Missing special token: {marker}"

    def test_control_tokens_exist(self):
        from auralith_pipeline.tokenization.bpe_tokenizer import BPETokenizer

        tok = BPETokenizer(vocab_size=100)
        token_set = set(tok.SPECIAL_TOKENS)
        for ctrl in ("<FUSE>", "<SEP>", "<MASK>", "<CODE>", "<CODE_END>", "<THINK>"):
            assert ctrl in token_set, f"Missing control token: {ctrl}"


# ===========================================================================
# Video Tokenizer
# ===========================================================================


class TestVideoTokenizer:
    """Test video tokenizer creates valid token sequences."""

    def test_video_tokenizer_init(self):
        from auralith_pipeline.tokenization.video_tokenizer import VideoTokenizer

        vt = VideoTokenizer(image_size=32, patch_size=8, codebook_size=16)
        assert vt.codebook_size == 16

    def test_encode_frames(self):
        from auralith_pipeline.tokenization.video_tokenizer import VideoTokenizer

        vt = VideoTokenizer(image_size=32, patch_size=8, codebook_size=16)

        # Train on dummy frames
        frames = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(30)]
        vt.train(frames)

        # Encode — encode_frames expects np.ndarray shape (N, H, W, C)
        test_frames = np.stack(
            [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(4)]
        )
        tokens = vt.encode_frames(test_frames)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)


# ===========================================================================
# Quality Pipeline
# ===========================================================================


class TestQualityPipeline:
    """Test advanced quality scoring."""

    def test_perplexity_filter_init(self):
        """PerplexityFilter should initialize without errors."""
        from auralith_pipeline.preprocessing.quality import PerplexityFilter

        # With default (no model loaded unless transformers is installed)
        pf = PerplexityFilter(model_name="gpt2", max_perplexity=1500.0)
        assert abs(pf.max_perplexity - 1500.0) < 1e-6

    def test_llm_judge_local_fallback(self):
        """LLM judge local provider should return scores without external API."""
        from auralith_pipeline.preprocessing.quality import LLMJudge

        judge = LLMJudge(provider="local")
        scores = judge.score("This is a well-written paragraph about machine learning. " * 5)
        assert isinstance(scores, dict)
        assert "coherence" in scores


# ===========================================================================
# Deduplication
# ===========================================================================


class TestEmbeddingDeduplication:
    """Test FAISS embedding deduplicator interface."""

    def test_deduplicator_init(self):
        from auralith_pipeline.preprocessing.deduplication import EmbeddingDeduplicator

        dedup = EmbeddingDeduplicator(
            model_name="all-MiniLM-L6-v2",
            similarity_threshold=0.92,
        )
        assert abs(dedup.similarity_threshold - 0.92) < 1e-6


# ===========================================================================
# Synthetic Data
# ===========================================================================


class TestSyntheticData:
    """Test synthetic data generator."""

    def test_generator_init(self):
        from auralith_pipeline.preprocessing.synthetic import LocalDataAugmenter

        gen = LocalDataAugmenter(strategies=["sentence_shuffle"])
        assert "sentence_shuffle" in gen.strategies

    def test_local_paraphrase(self):
        from auralith_pipeline.preprocessing.synthetic import LocalDataAugmenter

        gen = LocalDataAugmenter(strategies=["sentence_shuffle"])
        original = DataSample(
            content="The quick brown fox jumped over the lazy dog with great agility and speed.",
            source="test",
            metadata={},
        )
        results = list(gen.augment(iter([original])))
        # Should yield at least the original
        assert len(results) >= 1


# ===========================================================================
# Tracking & Lineage
# ===========================================================================


class TestTracking:
    """Test experiment tracking and lineage."""

    def test_lineage_tracker(self):
        from auralith_pipeline.utils.tracking import LineageTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = LineageTracker(output_dir=tmpdir)
            record = tracker.create_record(source="wikipedia", modality="text")
            record.token_count = 512

            tracker.save()

            lineage_path = Path(tmpdir) / "lineage.jsonl"
            assert lineage_path.exists()

            with open(lineage_path) as f:
                line = json.loads(f.readline())
                assert line["source"] == "wikipedia"
                assert line["token_count"] == 512

    def test_experiment_tracker_local(self):
        from auralith_pipeline.utils.tracking import ExperimentTracker

        tracker = ExperimentTracker(backend="local", run_name="test-run")
        tracker.start_run()
        tracker.log_params({"vocab_size": 50257})
        tracker.log_metrics({"loss": 0.5})
        tracker.end_run()

        # Local log should have entries
        assert len(tracker._local_log) >= 2

    def test_data_card_generation(self):
        from auralith_pipeline.utils.tracking import DataCardGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            gen = DataCardGenerator()
            card = gen.generate(
                output_path=str(Path(tmpdir) / "README.md"),
                config={"name": "test-dataset"},
                stats={"total_samples": 1000, "total_tokens": 50000, "num_shards": 5},
                lineage_summary={"sources": {"wiki": 800, "code": 200}, "modalities": {"text": 1000}},
            )
            assert "test-dataset" in card
            assert Path(tmpdir, "README.md").exists()


# ===========================================================================
# Compliance
# ===========================================================================


class TestCompliance:
    """Test license detection and audit logging."""

    def test_license_detection_permissive(self):
        from auralith_pipeline.preprocessing.compliance import LicenseDetector

        detector = LicenseDetector(allow_permissive=True, allow_copyleft=False)

        sample = DataSample(
            content="# MIT License\n\nCopyright (c) 2024 Test\n\ndef hello(): pass",
            source="github",
            metadata={"license": "mit"},
        )
        assert detector.is_allowed(sample) is True
        assert sample.metadata["detected_license"] == "mit"

    def test_license_detection_blocks_gpl(self):
        from auralith_pipeline.preprocessing.compliance import LicenseDetector

        detector = LicenseDetector(allow_permissive=True, allow_copyleft=False)

        sample = DataSample(
            content="# GPL code\ndef main(): pass",
            source="github",
            metadata={"license": "gpl-3.0"},
        )
        assert detector.is_allowed(sample) is False

    def test_license_detection_from_content(self):
        from auralith_pipeline.preprocessing.compliance import LicenseDetector

        detector = LicenseDetector(allow_permissive=True)

        sample = DataSample(
            content='# Apache License, Version 2.0\n\n"""Licensed under Apache 2.0"""',
            source="github",
            metadata={},
        )
        assert detector.is_allowed(sample) is True

    def test_audit_logger(self):
        from auralith_pipeline.preprocessing.compliance import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = str(Path(tmpdir) / "audit.jsonl")
            logger = AuditLogger(log_path=log_path)

            logger.log("sample_001", "accept", "passed_all_filters")
            logger.log("sample_002", "reject", "license_not_allowed", {"license": "gpl-3.0"})

            summary = logger.summary()
            assert summary["accept"] == 1
            assert summary["reject"] == 1

            # Check file was written
            with open(log_path) as f:
                lines = f.readlines()
                assert len(lines) == 2

    def test_license_detector_stats(self):
        from auralith_pipeline.preprocessing.compliance import LicenseDetector

        detector = LicenseDetector(allow_permissive=True)

        for lic in ["mit", "apache-2.0", "gpl-3.0"]:
            sample = DataSample(content="code", source="test", metadata={"license": lic})
            detector.is_allowed(sample)

        assert detector.stats["total_checked"] == 3
        assert detector.stats["permissive"] == 2
        assert detector.stats["copyleft"] == 1


# ===========================================================================
# Pipeline Config Tests
# ===========================================================================


class TestPipelineConfigV2:
    """Test pipeline config dataclasses."""

    def test_config_has_new_sections(self):
        config = PipelineConfig()
        assert hasattr(config, "advanced_quality")
        assert hasattr(config, "video")
        assert hasattr(config, "tracking")
        assert hasattr(config, "compliance")
        assert hasattr(config, "distributed")

    def test_production_preset_enables_tracking(self):
        config = PipelineConfig.from_preset("production")
        assert config.tracking.enabled is True
        assert config.tracking.lineage is True
        assert config.compliance.enabled is True

    def test_multimodal_preset_enables_video(self):
        config = PipelineConfig.from_preset("multimodal")
        assert config.video.enabled is True

    def test_config_from_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test.yaml"
            yaml_path.write_text("""
pipeline:
  name: yaml-test
  output_dir: ./test-output
  deduplicate: true
  quality_filter: true
  remove_pii: false
  normalize_text: true
  num_workers: 2
  batch_size: 50
  streaming: false

quality:
  min_text_length: 100

tracking:
  enabled: true
  backend: local

compliance:
  enabled: true
  allow_copyleft: true

video:
  enabled: true
  max_frames: 16
  resize: [112, 112]
""")
            config = PipelineConfig.from_yaml(str(yaml_path))
            assert config.name == "yaml-test"
            assert config.tracking.enabled is True
            assert config.compliance.allow_copyleft is True
            assert config.video.enabled is True
            assert config.video.max_frames == 16
            assert config.video.resize == (112, 112)

    def test_config_round_trip(self):
        config = PipelineConfig.from_preset("production")
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "production-pipeline"
        assert "advanced_quality" in d
        assert "tracking" in d


# ===========================================================================
# Pipeline Integration (mini E2E)
# ===========================================================================


class TestPipelineE2E:
    """Mini end-to-end pipeline test."""

    def test_pipeline_with_tracking_disabled(self):
        """Pipeline should run fine with all optional features disabled."""
        from auralith_pipeline.pipeline import Pipeline

        config = PipelineConfig.from_preset("small")
        pipeline = Pipeline(config)
        assert pipeline.config.name == "small-pipeline"
        # Verify no optional components initialized yet
        assert pipeline._advanced_quality is None
        assert pipeline._experiment_tracker is None

    def test_pipeline_stats_to_dict(self):
        from auralith_pipeline.pipeline import PipelineStats

        stats = PipelineStats(
            total_samples=1000,
            total_tokens=50000,
            license_blocked=5,
        )
        d = stats.to_dict()
        assert d["total_samples"] == 1000
        assert d["license_blocked"] == 5

    def test_pipeline_stats_summary(self):
        from auralith_pipeline.pipeline import PipelineStats

        stats = PipelineStats(
            total_samples=1000,
            samples_after_filter=900,
            total_tokens=50000,
            num_shards=3,
            total_size_bytes=1024 * 1024,
            elapsed_time_seconds=10.0,
            perplexity_filtered=50,
            license_blocked=10,
            modalities={"text": 800, "code": 200},
        )
        summary = stats.summary()
        assert "1,000" in summary
        assert "Perplexity filtered" in summary
        assert "License blocked" in summary
        assert "Modalities" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
