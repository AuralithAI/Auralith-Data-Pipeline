"""Sharding module for creating and reading SafeTensors shards."""

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from auralith_pipeline.config.pipeline_config import ShardConfig
from auralith_pipeline.tokenization.tokenizer import TokenizedSample

logger = logging.getLogger(__name__)


@dataclass
class ShardMetadata:
    """Metadata for a shard file."""

    shard_id: int
    num_samples: int
    size_bytes: int
    sequence_length: int
    created_at: str
    checksum: str
    compression: str | None = None
    source_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class ShardIndex:
    """Index for a collection of shards."""

    total_shards: int
    total_samples: int
    total_size_bytes: int
    sequence_length: int
    shards: list[ShardMetadata]
    config: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str):
        """Save index to JSON file."""
        data = {
            "total_shards": self.total_shards,
            "total_samples": self.total_samples,
            "total_size_bytes": self.total_size_bytes,
            "sequence_length": self.sequence_length,
            "config": self.config,
            "shards": [
                {
                    "shard_id": s.shard_id,
                    "num_samples": s.num_samples,
                    "size_bytes": s.size_bytes,
                    "sequence_length": s.sequence_length,
                    "created_at": s.created_at,
                    "checksum": s.checksum,
                    "compression": s.compression,
                    "source_info": s.source_info,
                }
                for s in self.shards
            ],
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ShardIndex":
        """Load index from JSON file."""
        with open(path) as f:
            data = json.load(f)

        shards = [ShardMetadata(**s) for s in data.get("shards", [])]

        return cls(
            total_shards=data["total_shards"],
            total_samples=data["total_samples"],
            total_size_bytes=data["total_size_bytes"],
            sequence_length=data["sequence_length"],
            shards=shards,
            config=data.get("config", {}),
        )


class ShardWriter:
    """Write tokenized samples to shards."""

    def __init__(
        self,
        output_dir: str,
        config: ShardConfig | None = None,
        prefix: str = "shard",
    ):
        self.output_dir = Path(output_dir)
        self.config = config or ShardConfig()
        self.prefix = prefix

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._current_samples: list[TokenizedSample] = []
        self._current_size = 0
        self._shard_count = 0
        self._total_samples = 0
        self._shards_metadata: list[ShardMetadata] = []

    def add_sample(self, sample: TokenizedSample):
        """Add a sample to the current shard."""
        sample_size = len(sample.input_ids) * 4  # Approximate size in bytes

        # Check if we need to flush
        if self._current_size + sample_size > self.config.max_size_mb * 1024 * 1024:
            self._flush_shard()

        self._current_samples.append(sample)
        self._current_size += sample_size
        self._total_samples += 1

    def _flush_shard(self):
        """Write current samples to a shard file."""
        if not self._current_samples:
            return

        shard_path = self.output_dir / f"{self.prefix}_{self._shard_count:05d}.safetensors"

        # Convert samples to arrays
        input_ids = np.array([s.input_ids for s in self._current_samples], dtype=np.int32)
        attention_mask = np.array([s.attention_mask for s in self._current_samples], dtype=np.int32)

        # Write SafeTensors
        try:
            from safetensors.numpy import save_file

            tensors = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            save_file(tensors, str(shard_path))

            # Calculate checksum
            import hashlib

            checksum = hashlib.md5(shard_path.read_bytes()).hexdigest()

            # Create metadata
            from datetime import datetime

            metadata = ShardMetadata(
                shard_id=self._shard_count,
                num_samples=len(self._current_samples),
                size_bytes=shard_path.stat().st_size,
                sequence_length=len(self._current_samples[0].input_ids),
                created_at=datetime.now().isoformat(),
                checksum=checksum,
                compression=self.config.compression,
            )
            self._shards_metadata.append(metadata)

            logger.info(f"Wrote shard {shard_path.name}: {len(self._current_samples)} samples")

        except ImportError:
            logger.error("safetensors not installed. Install with: pip install safetensors")
            raise

        # Reset current shard
        self._current_samples = []
        self._current_size = 0
        self._shard_count += 1

    def finalize(self) -> ShardIndex:
        """Finalize writing and return the shard index."""
        # Flush remaining samples
        self._flush_shard()

        # Create index
        index = ShardIndex(
            total_shards=self._shard_count,
            total_samples=self._total_samples,
            total_size_bytes=sum(s.size_bytes for s in self._shards_metadata),
            sequence_length=self.config.sequence_length,
            shards=self._shards_metadata,
            config={
                "max_size_mb": self.config.max_size_mb,
                "format": self.config.format,
                "compression": self.config.compression,
            },
        )

        # Save index
        index_path = self.output_dir / "index.json"
        index.save(str(index_path))
        logger.info(f"Saved shard index to {index_path}")

        return index


class ShardReader:
    """Read samples from shards."""

    def __init__(self, shard_dir: str):
        self.shard_dir = Path(shard_dir)
        self._index = None

    def _load_index(self) -> ShardIndex:
        """Load the shard index."""
        if self._index is None:
            index_path = self.shard_dir / "index.json"
            if index_path.exists():
                self._index = ShardIndex.load(str(index_path))
            else:
                raise FileNotFoundError(f"No index.json found in {self.shard_dir}")
        return self._index

    def get_shard_files(self) -> list[Path]:
        """Get list of shard files."""
        return sorted(self.shard_dir.glob("*.safetensors"))

    def read_shard(self, shard_path: str) -> dict[str, np.ndarray]:
        """Read a single shard file."""
        try:
            from safetensors.numpy import load_file

            return load_file(shard_path)
        except ImportError:
            logger.error("safetensors not installed")
            raise

    def iterate_samples(self) -> Iterator[TokenizedSample]:
        """Iterate over all samples in all shards."""
        for shard_path in self.get_shard_files():
            data = self.read_shard(str(shard_path))
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]

            for i in range(len(input_ids)):
                yield TokenizedSample(
                    input_ids=input_ids[i].tolist(),
                    attention_mask=attention_mask[i].tolist(),
                    metadata={"shard": shard_path.name},
                )

    @property
    def total_samples(self) -> int:
        """Get total number of samples."""
        return self._load_index().total_samples

    @property
    def total_shards(self) -> int:
        """Get total number of shards."""
        return self._load_index().total_shards
