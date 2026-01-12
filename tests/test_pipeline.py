"""Tests for the Auralith Data Pipeline."""

import tempfile
from pathlib import Path

import pytest


class TestPipelineConfig:
    """Tests for pipeline configuration."""

    def test_default_config(self):
        from auralith_pipeline.config import PipelineConfig

        config = PipelineConfig()
        assert config.name == "default-pipeline"
        assert config.deduplicate is True
        assert config.quality_filter is True

    def test_preset_config(self):
        from auralith_pipeline.config import PipelineConfig

        config = PipelineConfig.from_preset("production")
        assert config.name == "production-pipeline"
        assert config.num_workers == 8

    def test_config_to_dict(self):
        from auralith_pipeline.config import PipelineConfig

        config = PipelineConfig()
        data = config.to_dict()
        assert isinstance(data, dict)
        assert "name" in data
        assert "quality" in data


class TestDataSources:
    """Tests for data sources."""

    def test_data_sample(self):
        from auralith_pipeline.sources import DataSample

        sample = DataSample(
            content="Hello world",
            source="test",
            metadata={"key": "value"},
        )
        assert sample.word_count == 2
        assert sample.char_count == 11

    def test_dataset_registry(self):
        from auralith_pipeline.sources import DATASET_REGISTRY, DEPRECATED_DATASETS

        # Test that registry contains working datasets
        assert "wikipedia" in DATASET_REGISTRY
        assert "c4" in DATASET_REGISTRY
        assert "redpajama" in DATASET_REGISTRY
        assert "the_pile" in DATASET_REGISTRY

        # Test that deprecated datasets are not in registry
        assert "arxiv" not in DATASET_REGISTRY

        # Test that deprecated datasets are tracked
        assert "arxiv" in DEPRECATED_DATASETS


class TestPreprocessing:
    """Tests for preprocessing."""

    def test_text_normalizer(self):
        from auralith_pipeline.preprocessing import TextNormalizer

        normalizer = TextNormalizer()
        text = "Hello   world\n\n\n\ntest"
        normalized = normalizer.normalize(text)
        assert "   " not in normalized

    def test_pii_remover(self):
        from auralith_pipeline.preprocessing import PIIRemover

        remover = PIIRemover()
        text = "Contact me at test@example.com or 123-456-7890"
        cleaned = remover.remove_pii(text)
        assert "[EMAIL]" in cleaned
        assert "[PHONE]" in cleaned
        assert "test@example.com" not in cleaned

    def test_quality_filter(self):
        from auralith_pipeline.config import QualityConfig
        from auralith_pipeline.preprocessing import QualityFilter
        from auralith_pipeline.sources import DataSample

        config = QualityConfig(min_text_length=10, min_word_count=2)
        filter = QualityFilter(config)

        # Should pass
        good_sample = DataSample(content="This is a good sample with enough text", source="test")
        assert filter.passes_filter(good_sample) is True

        # Should fail - too short
        bad_sample = DataSample(content="Short", source="test")
        assert filter.passes_filter(bad_sample) is False


class TestTokenization:
    """Tests for tokenization."""

    def test_tokenized_sample(self):
        from auralith_pipeline.tokenization import TokenizedSample

        sample = TokenizedSample(
            input_ids=[1, 2, 3, 4, 5],
            attention_mask=[1, 1, 1, 1, 1],
            metadata={},
        )
        assert sample.length == 5


class TestSharding:
    """Tests for sharding."""

    def test_shard_index_save_load(self):
        from auralith_pipeline.sharding import ShardIndex, ShardMetadata

        metadata = ShardMetadata(
            shard_id=0,
            num_samples=100,
            size_bytes=1024,
            sequence_length=2048,
            created_at="2024-01-01T00:00:00",
            checksum="abc123",
        )

        index = ShardIndex(
            total_shards=1,
            total_samples=100,
            total_size_bytes=1024,
            sequence_length=2048,
            shards=[metadata],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "index.json"
            index.save(str(path))

            loaded = ShardIndex.load(str(path))
            assert loaded.total_shards == 1
            assert loaded.total_samples == 100


class TestStorage:
    """Tests for storage backends."""

    def test_create_storage_backend(self):
        from auralith_pipeline.storage import create_storage_backend

        backend = create_storage_backend("huggingface", repo_id="test/repo")
        assert backend.repo_id == "test/repo"


class TestPipeline:
    """Tests for the main pipeline."""

    def test_pipeline_creation(self):
        from auralith_pipeline import Pipeline, PipelineConfig

        config = PipelineConfig.from_preset("small")
        pipeline = Pipeline(config)

        assert pipeline.config.name == "small-pipeline"


class TestUtils:
    """Tests for utility functions."""

    def test_format_size(self):
        from auralith_pipeline.utils import format_size

        assert "B" in format_size(100)
        assert "KB" in format_size(1024)
        assert "MB" in format_size(1024 * 1024)
        assert "GB" in format_size(1024 * 1024 * 1024)

    def test_format_number(self):
        from auralith_pipeline.utils import format_number

        assert format_number(1000) == "1,000"
        assert format_number(1000000) == "1,000,000"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
