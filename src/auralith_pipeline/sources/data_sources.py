"""Data sources module for ingesting data from various sources."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass
class DataSample:
    """A single data sample."""

    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    modality: Literal["text", "image", "audio", "video", "code"] = "text"

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    @property
    def char_count(self) -> int:
        return len(self.content)


class DataSource(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    def __iter__(self) -> Iterator[DataSample]:
        """Iterate over data samples."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return estimated number of samples."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return source name."""
        pass


class HuggingFaceSource(DataSource):
    """Load data from HuggingFace datasets."""

    def __init__(
        self,
        path: str,
        name: str | None = None,
        split: str = "train",
        text_column: str = "text",
        streaming: bool = True,
        max_samples: int | None = None,
        **kwargs,
    ):
        self.path = path
        self.dataset_name = name
        self.split = split
        self.text_column = text_column
        self.streaming = streaming
        self.max_samples = max_samples
        self.kwargs = kwargs
        self._dataset = None
        self._len = None

    def _load_dataset(self):
        """Lazy load the dataset."""
        if self._dataset is None:
            import time

            from datasets import load_dataset

            logger.info(f"Loading dataset: {self.path}")

            max_retries = 3
            retry_delay = 5

            for attempt in range(max_retries):
                try:
                    self._dataset = load_dataset(
                        self.path,
                        self.dataset_name,
                        split=self.split,
                        streaming=self.streaming,
                        **self.kwargs,
                    )
                    return self._dataset
                except Exception as e:
                    error_msg = str(e).lower()

                    # Check for deprecated dataset scripts
                    if "dataset scripts are no longer supported" in error_msg:
                        logger.error(
                            f"Dataset '{self.path}' uses deprecated Python scripts. "
                            f"Try using 'wikimedia/wikipedia' instead of 'wikipedia'."
                        )
                        raise

                    # Retry on timeout errors
                    if (
                        "timeout" in error_msg or "timed out" in error_msg
                    ) and attempt < max_retries - 1:
                        logger.warning(
                            f"Timeout loading dataset (attempt {attempt + 1}/{max_retries}). "
                            f"Retrying in {retry_delay}s..."
                        )
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue

                    # Re-raise if not a timeout or last attempt
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to load dataset after {max_retries} attempts: {e}")
                        raise

        return self._dataset

    def __iter__(self) -> Iterator[DataSample]:
        dataset = self._load_dataset()
        count = 0

        for item in dataset:
            if self.max_samples and count >= self.max_samples:
                break

            text = item.get(self.text_column, "")
            if not text:
                continue

            yield DataSample(
                content=text,
                source=f"huggingface/{self.path}",
                metadata={
                    "split": self.split,
                    **{k: v for k, v in item.items() if k != self.text_column},
                },
            )
            count += 1

    def __len__(self) -> int:
        if self._len is None:
            if self.max_samples:
                self._len = self.max_samples
            elif not self.streaming:
                self._len = len(self._load_dataset())
            else:
                self._len = -1  # Unknown for streaming
        return self._len

    @property
    def name(self) -> str:
        return f"huggingface/{self.path}"


class LocalFileSource(DataSource):
    """Load data from local files."""

    def __init__(
        self,
        path: str,
        pattern: str = "**/*.txt",
        encoding: str = "utf-8",
        max_samples: int | None = None,
    ):
        from pathlib import Path

        self.path = Path(path)
        self.pattern = pattern
        self.encoding = encoding
        self.max_samples = max_samples
        self._files = None

    def _get_files(self) -> list:
        if self._files is None:
            self._files = list(self.path.glob(self.pattern))
        return self._files

    def __iter__(self) -> Iterator[DataSample]:
        count = 0

        for file_path in self._get_files():
            if self.max_samples and count >= self.max_samples:
                break

            try:
                content = file_path.read_text(encoding=self.encoding)
                yield DataSample(
                    content=content,
                    source=f"local/{file_path.name}",
                    metadata={
                        "path": str(file_path),
                        "size": file_path.stat().st_size,
                    },
                )
                count += 1
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                continue

    def __len__(self) -> int:
        if self.max_samples:
            return min(self.max_samples, len(self._get_files()))
        return len(self._get_files())

    @property
    def name(self) -> str:
        return f"local/{self.path}"


class JSONLSource(DataSource):
    """Load data from JSONL files."""

    def __init__(
        self,
        path: str,
        text_field: str = "text",
        max_samples: int | None = None,
    ):
        self.path = path
        self.text_field = text_field
        self.max_samples = max_samples

    def __iter__(self) -> Iterator[DataSample]:
        import json

        count = 0
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if self.max_samples and count >= self.max_samples:
                    break

                try:
                    item = json.loads(line)
                    text = item.get(self.text_field, "")
                    if text:
                        yield DataSample(
                            content=text,
                            source=f"jsonl/{self.path}",
                            metadata={k: v for k, v in item.items() if k != self.text_field},
                        )
                        count += 1
                except json.JSONDecodeError:
                    continue

    def __len__(self) -> int:
        if self.max_samples:
            return self.max_samples
        # Estimate line count
        with open(self.path) as f:
            return sum(1 for _ in f)

    @property
    def name(self) -> str:
        return f"jsonl/{self.path}"


# Registry of available datasets
DATASET_REGISTRY = {
    "wikipedia": {
        "path": "wikimedia/wikipedia",
        "name": "20231101.en",
        "text_column": "text",
        "description": "English Wikipedia (20GB)",
        "split": "train",
    },
    "c4": {
        "path": "allenai/c4",
        "name": "en",
        "text_column": "text",
        "description": "C4 - Common Crawl cleaned (750GB)",
        "split": "train",
    },
    "redpajama": {
        "path": "togethercomputer/RedPajama-Data-1T",
        "text_column": "text",
        "description": "RedPajama - LLaMA reproduction (1.2TB)",
        "split": "train",
    },
    "openwebtext": {
        "path": "Skylion007/openwebtext",
        "text_column": "text",
        "description": "OpenWebText - Reddit links (40GB)",
        "split": "train",
    },
    "bookcorpus": {
        "path": "bookcorpus",
        "text_column": "text",
        "description": "BookCorpus - 11k books (5GB)",
        "split": "train",
    },
    "wikitext": {
        "path": "wikitext",
        "name": "wikitext-103-v1",
        "text_column": "text",
        "description": "Wikitext-103 - Wikipedia subset (500MB)",
        "split": "train",
    },
    "dolly": {
        "path": "databricks/databricks-dolly-15k",
        "text_column": "response",
        "description": "Dolly-15k - instruction following (15MB)",
        "split": "train",
    },
    "the_stack": {
        "path": "bigcode/the-stack-dedup",
        "name": "data",
        "text_column": "content",
        "description": "The Stack (deduplicated) - source code (3TB)",
        "split": "train",
    },
    "the_pile": {
        "path": "monology/pile-uncopyrighted",
        "text_column": "text",
        "description": "The Pile (uncopyrighted subset) - diverse text corpus (825GB)",
        "split": "train",
    },
    "arxiv": {
        "path": "CShorten/ML-ArXiv-Papers",
        "text_column": "abstract",
        "description": "ML ArXiv Papers - machine learning papers from arXiv (117k papers)",
        "split": "train",
    },
}

# Deprecated datasets (no longer accessible via new HuggingFace API)
DEPRECATED_DATASETS: dict[str, str] = {
    "scientific_papers": "Uses deprecated scripts - use 'arxiv' instead",
}


def create_source(
    name: str,
    streaming: bool = True,
    max_samples: int | None = None,
    **kwargs,
) -> HuggingFaceSource:
    """Create a data source from the registry."""
    # Check if the dataset is deprecated
    if name in DEPRECATED_DATASETS:
        raise ValueError(f"Dataset '{name}' is deprecated: {DEPRECATED_DATASETS[name]}")

    if name not in DATASET_REGISTRY:
        available = list(DATASET_REGISTRY.keys())
        deprecated = list(DEPRECATED_DATASETS.keys())
        raise ValueError(
            f"Unknown dataset: {name}.\n" f"Available: {available}\n" f"Deprecated: {deprecated}"
        )

    config = DATASET_REGISTRY[name].copy()
    config.pop("description", None)
    config.update(kwargs)

    return HuggingFaceSource(
        streaming=streaming,
        max_samples=max_samples,
        **config,
    )
