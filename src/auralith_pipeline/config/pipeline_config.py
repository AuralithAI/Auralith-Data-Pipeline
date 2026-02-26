"""Pipeline configuration module."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class QualityConfig:
    """Quality filtering configuration."""

    min_text_length: int = 50
    max_text_length: int = 100000
    min_word_count: int = 10
    max_word_count: int = 50000
    allowed_languages: list[str] = field(default_factory=lambda: ["en"])
    max_special_char_ratio: float = 0.3
    max_digit_ratio: float = 0.3
    max_uppercase_ratio: float = 0.4
    filter_toxic: bool = True
    toxic_threshold: float = 0.7


@dataclass
class AdvancedQualityConfig:
    """Advanced quality pipeline configuration."""

    enabled: bool = False
    perplexity_filter: bool = False
    perplexity_model: str = "gpt2"
    max_perplexity: float = 1500.0
    min_perplexity: float = 5.0
    llm_judge: bool = False
    llm_judge_provider: str = "local"
    llm_judge_model: str | None = None
    min_llm_score: float = 0.5


@dataclass
class DeduplicationConfig:
    """Deduplication configuration."""

    enabled: bool = True
    method: Literal["minhash", "exact", "simhash", "embedding"] = "minhash"
    minhash_threshold: float = 0.85
    minhash_num_perm: int = 128
    minhash_bands: int = 32
    cache_size: int = 1000000
    # Embedding-based (FAISS) dedup
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_threshold: float = 0.92


@dataclass
class ShardConfig:
    """Shard configuration."""

    max_size_mb: int = 500
    sequence_length: int = 2048
    format: Literal["safetensors", "parquet", "jsonl"] = "safetensors"
    compression: Literal["zstd", "lz4", "gzip"] | None = "zstd"
    include_metadata: bool = True
    create_index: bool = True


@dataclass
class TokenizationConfig:
    """Tokenization configuration."""

    tokenizer_path: str | None = None
    vocab_size: int = 50257
    model_type: Literal["bpe", "unigram", "word"] = "bpe"
    add_special_tokens: bool = True
    padding: bool = True
    truncation: bool = True
    max_length: int = 2048


@dataclass
class VideoConfig:
    """Video processing configuration."""

    enabled: bool = False
    frame_strategy: Literal["uniform", "fps", "keyframe"] = "uniform"
    max_frames: int = 32
    target_fps: float = 1.0
    resize: tuple[int, int] = (224, 224)
    codebook_size: int = 1024
    video_token_offset: int = 300000


@dataclass
class StorageConfig:
    """Storage configuration."""

    backend: Literal["local", "huggingface", "s3", "gcs", "azure"] = "local"
    path: str = "./data/shards"
    repo_id: str | None = None
    bucket: str | None = None
    container: str | None = None
    prefix: str = ""


@dataclass
class TrackingConfig:
    """Observability / experiment tracking."""

    enabled: bool = False
    backend: Literal["local", "mlflow", "wandb"] = "local"
    project_name: str = "auralith-data-pipeline"
    experiment_name: str | None = None
    run_name: str | None = None
    lineage: bool = True
    data_cards: bool = True


@dataclass
class ComplianceConfig:
    """Compliance and license detection."""

    enabled: bool = False
    license_detection: bool = True
    allow_permissive: bool = True
    allow_copyleft: bool = False
    audit_log_path: str | None = None
    pii_rescan: bool = False


@dataclass
class SecurityConfig:
    """Security — PII scrubbing and data sanitization.

    This is the dedicated security layer that ensures no private user data
    from any jurisdiction worldwide enters the training pipeline.
    """

    enabled: bool = True
    mode: Literal["strict", "jurisdiction"] = "strict"
    replacement_style: Literal["tag", "hash", "remove"] = "tag"
    rescan_after_processing: bool = True
    log_redactions: bool = True
    fail_on_pii: bool = False
    audit_log_path: str | None = None
    # Data sanitizer (credentials, secrets, API keys)
    sanitize_secrets: bool = True
    block_internal_urls: bool = True


@dataclass
class DistributedConfig:
    """Distributed processing configuration."""

    enabled: bool = False
    backend: Literal["ray", "spark", "local"] = "local"
    ray_address: str = "auto"
    num_cpus: int | None = None
    num_gpus: int | None = None


def _convert_video_data(data: dict[str, Any]) -> dict[str, Any]:
    """Convert YAML-loaded video config, coercing resize list to tuple."""
    result = dict(data)
    if "resize" in result and isinstance(result["resize"], list):
        result["resize"] = tuple(result["resize"])
    return result


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""

    name: str = "default-pipeline"
    output_dir: str = "./data/shards"
    temp_dir: str = "./data/temp"

    # Preprocessing
    deduplicate: bool = True
    quality_filter: bool = True
    remove_pii: bool = True
    normalize_text: bool = True

    # Processing
    num_workers: int = 4
    batch_size: int = 1000
    streaming: bool = True
    max_samples: int | None = None

    # Reproducibility & fault-tolerance
    seed: int = 42
    checkpoint_every: int = 10_000  # save resume checkpoint every N accepted samples
    checkpoint_path: str | None = None  # defaults to {output_dir}/.pipeline_checkpoint.json

    # Sub-configs
    quality: QualityConfig = field(default_factory=QualityConfig)
    advanced_quality: AdvancedQualityConfig = field(default_factory=AdvancedQualityConfig)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    sharding: ShardConfig = field(default_factory=ShardConfig)
    tokenization: TokenizationConfig = field(default_factory=TokenizationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    def __post_init__(self):
        """Validate that token ID spaces for each modality never overlap."""
        max_text_id = self.tokenization.vocab_size - 1
        image_offset = 100_000
        audio_offset = 200_000
        video_offset = self.video.video_token_offset  # 300_000 default

        img_end = image_offset + 1024 - 1  # image codebook_size = 1024
        audio_end = audio_offset + 512 - 1  # audio codebook_size = 512

        if max_text_id >= image_offset:
            raise ValueError(
                f"vocab_size={self.tokenization.vocab_size} overlaps image token space "
                f"(starts at {image_offset}). Reduce vocab_size or increase image_token_offset."
            )
        if img_end >= audio_offset:
            raise ValueError(
                f"Image tokens (up to {img_end}) overlap audio token space (starts at {audio_offset})."
            )
        if audio_end >= video_offset:
            raise ValueError(
                f"Audio tokens (up to {audio_end}) overlap video token space (starts at {video_offset})."
            )
        if self.tokenization.max_length != self.sharding.sequence_length:
            # Warn rather than hard-error; user might intentionally differ
            import logging

            logging.getLogger(__name__).warning(
                f"tokenization.max_length ({self.tokenization.max_length}) != "
                f"sharding.sequence_length ({self.sharding.sequence_length}). "
                "Shard sequences will use sharding.sequence_length for padding."
            )

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        """Create configuration from dictionary."""
        # Extract nested config sections
        pipeline_data = data.pop("pipeline", None)
        if pipeline_data:
            data.update(pipeline_data)

        quality_data = data.pop("quality", {})
        advanced_quality_data = data.pop("advanced_quality", {})
        dedup_data = data.pop("deduplication", {})
        shard_data = data.pop("sharding", {})
        token_data = data.pop("tokenization", {})
        storage_data = data.pop("storage", {})
        video_data = data.pop("video", {})
        tracking_data = data.pop("tracking", {})
        compliance_data = data.pop("compliance", {})
        security_data = data.pop("security", {})
        distributed_data = data.pop("distributed", {})

        # Remove non-config keys (e.g. 'sources')
        data.pop("sources", None)

        return cls(
            **data,
            quality=QualityConfig(**quality_data),
            advanced_quality=AdvancedQualityConfig(**advanced_quality_data),
            deduplication=DeduplicationConfig(**dedup_data),
            sharding=ShardConfig(**shard_data),
            tokenization=TokenizationConfig(**token_data),
            storage=StorageConfig(**storage_data),
            video=VideoConfig(**_convert_video_data(video_data)),
            tracking=TrackingConfig(**tracking_data),
            compliance=ComplianceConfig(**compliance_data),
            security=SecurityConfig(**security_data),
            distributed=DistributedConfig(**distributed_data),
        )

    @classmethod
    def from_preset(cls, preset: str) -> "PipelineConfig":
        """Create configuration from preset."""
        presets = {
            "small": cls(
                name="small-pipeline",
                batch_size=100,
                max_samples=10000,
                sharding=ShardConfig(max_size_mb=100),
            ),
            "medium": cls(
                name="medium-pipeline",
                batch_size=500,
                max_samples=100000,
                num_workers=4,
                sharding=ShardConfig(max_size_mb=250),
            ),
            "production": cls(
                name="production-pipeline",
                batch_size=1000,
                num_workers=8,
                streaming=True,
                deduplicate=True,
                quality_filter=True,
                remove_pii=True,
                sharding=ShardConfig(max_size_mb=1000, compression="zstd"),
                deduplication=DeduplicationConfig(
                    minhash_threshold=0.85,
                    minhash_num_perm=256,
                ),
                advanced_quality=AdvancedQualityConfig(enabled=True, perplexity_filter=True),
                tracking=TrackingConfig(enabled=True, lineage=True, data_cards=True),
                compliance=ComplianceConfig(enabled=True),
                security=SecurityConfig(
                    enabled=True,
                    mode="strict",
                    rescan_after_processing=True,
                    log_redactions=True,
                    sanitize_secrets=True,
                ),
            ),
            "multimodal": cls(
                name="multimodal-pipeline",
                batch_size=100,
                num_workers=4,
                sharding=ShardConfig(max_size_mb=500),
                video=VideoConfig(enabled=True),
            ),
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")

        return presets[preset]

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict

        return asdict(self)
