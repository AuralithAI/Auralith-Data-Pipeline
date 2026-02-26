"""Tokenization module for text and multimodal content.

This module provides a unified interface to the custom BPE tokenizer
and multimodal tokenizers for images and audio.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from auralith_pipeline.tokenization.bpe_tokenizer import BPETokenizer

logger = logging.getLogger(__name__)


@dataclass
class TokenizedSample:
    """A tokenized sample ready for training.

    Fields align with RT-DLM's expected SafeTensors schema:
        input_ids:      All tokens (text + image + audio + video)
        attention_mask:  1 = real token, 0 = padding
        modality_mask:   Per-token modality (0=text, 1=image, 2=audio, 3=video, 4=code)
        labels:          Causal LM labels (-100 = ignore in loss)
        metadata:        Source, quality scores, lineage, etc.
    """

    input_ids: list[int]
    attention_mask: list[int]
    metadata: dict[str, Any]
    modality_mask: list[int] | None = None
    labels: list[int] | None = None

    @property
    def length(self) -> int:
        return len(self.input_ids)

    def __post_init__(self):
        """Fill defaults for modality_mask and labels if not provided.

        Labels default to input_ids for real tokens and -100 for pad
        positions (where attention_mask == 0), matching the documented
        schema semantics so pad tokens are excluded from the loss.
        """
        if self.modality_mask is None:
            self.modality_mask = [0] * len(self.input_ids)  # Default: all text
        if self.labels is None:
            self.labels = [
                tid if mask == 1 else -100 for tid, mask in zip(self.input_ids, self.attention_mask)
            ]


class Tokenizer:
    """Text tokenizer using custom BPE implementation.

    This is a wrapper around BPETokenizer for backward compatibility.
    """

    def __init__(
        self,
        tokenizer_path: str | None = None,
        vocab_size: int = 32000,
        max_length: int = 2048,
        add_special_tokens: bool = True,
        padding: bool = True,
        truncation: bool = True,
    ):
        self.tokenizer_path = tokenizer_path
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.padding = padding
        self.truncation = truncation

        self._tokenizer: BPETokenizer | None = None

    def _load_tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is not None:
            return

        if self.tokenizer_path:
            # Try to load pre-trained tokenizer
            try:
                self._tokenizer = BPETokenizer.load(self.tokenizer_path)
                logger.info(f"Loaded BPE tokenizer from {self.tokenizer_path}")
                return
            except Exception as e:
                logger.debug(f"Failed to load tokenizer from {self.tokenizer_path}: {e}")

        # Create new tokenizer (needs training)
        self._tokenizer = BPETokenizer(vocab_size=self.vocab_size)

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids."""
        self._load_tokenizer()

        max_len = self.max_length if self.truncation else None

        return self._tokenizer.encode(
            text,
            add_special_tokens=self.add_special_tokens,
            max_length=max_len,
        )

    def decode(self, ids: list[int]) -> str:
        """Decode token ids to text."""
        self._load_tokenizer()
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    def train(self, corpus: str, save_path: str | None = None) -> None:
        """Train the BPE tokenizer on a corpus.

        Args:
            corpus: Text corpus to train on
            save_path: Optional path to save trained tokenizer
        """
        self._load_tokenizer()
        logger.info("Training BPE tokenizer...")
        self._tokenizer.train(corpus, verbose=True)

        if save_path:
            self._tokenizer.save(save_path)
            logger.info(f"Tokenizer saved to {save_path}")

    def tokenize(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> TokenizedSample:
        """Tokenize text into a training sample."""
        input_ids = self.encode(text)

        # Padding
        if self.padding and len(input_ids) < self.max_length:
            pad_length = self.max_length - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * pad_length
            input_ids = input_ids + [0] * pad_length
        else:
            attention_mask = [1] * len(input_ids)

        return TokenizedSample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            metadata=metadata or {},
        )

    def batch_tokenize(
        self,
        texts: list[str],
        metadata_list: list[dict[str, Any]] | None = None,
    ) -> list[TokenizedSample]:
        """Tokenize a batch of texts."""
        if metadata_list is None:
            metadata_list = [{}] * len(texts)

        return [self.tokenize(text, meta) for text, meta in zip(texts, metadata_list)]


class TokenizationPipeline:
    """Pipeline for tokenizing data samples."""

    def __init__(
        self,
        tokenizer_path: str | None = None,
        vocab_size: int = 50257,
        max_length: int = 2048,
        chunk_overlap: int = 128,
    ):
        self.tokenizer = Tokenizer(
            tokenizer_path=tokenizer_path,
            vocab_size=vocab_size,
            max_length=max_length,
        )
        self.max_length = max_length
        self.chunk_overlap = chunk_overlap

    def process(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> Iterator[TokenizedSample]:
        """Process text into tokenized samples, chunking if necessary."""

        # Get initial tokens to check length
        ids = self.tokenizer.encode(text)

        if len(ids) <= self.max_length:
            # Single sample
            yield self.tokenizer.tokenize(text, metadata)
        else:
            # Need to chunk
            for chunk in self._chunk_text(text):
                yield self.tokenizer.tokenize(chunk, metadata)

    def _chunk_text(self, text: str) -> Iterator[str]:
        """Chunk text for long documents."""
        words = text.split()

        # Estimate words per chunk based on average token per word ratio
        words_per_chunk = int(self.max_length * 0.7)  # Conservative estimate
        overlap_words = int(self.chunk_overlap * 0.7)

        start = 0
        while start < len(words):
            end = min(start + words_per_chunk, len(words))
            chunk = " ".join(words[start:end])
            yield chunk

            if end >= len(words):
                break

            start = end - overlap_words
