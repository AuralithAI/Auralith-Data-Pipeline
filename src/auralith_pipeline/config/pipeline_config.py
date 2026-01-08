"""Pipeline configuration module."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
from pathlib import Path
import yaml


@dataclass
class QualityConfig:
    """Quality filtering configuration."""
    
    min_text_length: int = 50
    max_text_length: int = 100000
    min_word_count: int = 10
    max_word_count: int = 50000
    allowed_languages: List[str] = field(default_factory=lambda: ["en"])
    max_special_char_ratio: float = 0.3
    max_digit_ratio: float = 0.3
    max_uppercase_ratio: float = 0.4
    filter_toxic: bool = True
    toxic_threshold: float = 0.7


@dataclass
class DeduplicationConfig:
    """Deduplication configuration."""
    
    enabled: bool = True
    method: Literal["minhash", "exact", "simhash"] = "minhash"
    minhash_threshold: float = 0.85
    minhash_num_perm: int = 128
    minhash_bands: int = 32
    cache_size: int = 1000000


@dataclass
class ShardConfig:
    """Shard configuration."""
    
    max_size_mb: int = 500
    sequence_length: int = 2048
    format: Literal["safetensors", "parquet", "jsonl"] = "safetensors"
    compression: Optional[Literal["zstd", "lz4", "gzip"]] = "zstd"
    include_metadata: bool = True
    create_index: bool = True


@dataclass
class TokenizationConfig:
    """Tokenization configuration."""
    
    tokenizer_path: Optional[str] = None
    vocab_size: int = 50257
    model_type: Literal["bpe", "unigram", "word"] = "bpe"
    add_special_tokens: bool = True
    padding: bool = True
    truncation: bool = True
    max_length: int = 2048


@dataclass
class StorageConfig:
    """Storage configuration."""
    
    backend: Literal["local", "huggingface", "s3", "gcs", "azure"] = "local"
    path: str = "./data/shards"
    repo_id: Optional[str] = None
    bucket: Optional[str] = None
    container: Optional[str] = None
    prefix: str = ""


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
    max_samples: Optional[int] = None
    
    # Sub-configs
    quality: QualityConfig = field(default_factory=QualityConfig)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    sharding: ShardConfig = field(default_factory=ShardConfig)
    tokenization: TokenizationConfig = field(default_factory=TokenizationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create configuration from dictionary."""
        quality_data = data.pop("quality", {})
        dedup_data = data.pop("deduplication", {})
        shard_data = data.pop("sharding", {})
        token_data = data.pop("tokenization", {})
        storage_data = data.pop("storage", {})
        
        return cls(
            **data,
            quality=QualityConfig(**quality_data),
            deduplication=DeduplicationConfig(**dedup_data),
            sharding=ShardConfig(**shard_data),
            tokenization=TokenizationConfig(**token_data),
            storage=StorageConfig(**storage_data),
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
            ),
            "multimodal": cls(
                name="multimodal-pipeline",
                batch_size=100,
                num_workers=4,
                sharding=ShardConfig(max_size_mb=500),
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)
