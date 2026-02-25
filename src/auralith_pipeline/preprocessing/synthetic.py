"""Safe local-only data augmentation
This module provides safe, deterministic, local-only augmentation:
  • Sentence-level shuffling (preserving meaning)
  • Paragraph extraction (split long documents into chunks)
  • Token-level noise (typo simulation for robustness)
  • Back-translation placeholder (requires local MarianMT model)

All transformations are auditable and produce no copyright concerns.
"""

import logging
import random
import re
from collections.abc import Iterator
from typing import Any

from auralith_pipeline.sources.data_sources import DataSample

logger = logging.getLogger(__name__)


class LocalDataAugmenter:
    """Local-only data augmentation."""

    def __init__(
        self,
        strategies: list[str] | None = None,
        augmentation_factor: int = 1,
        min_text_length: int = 100,
        seed: int | None = None,
    ):
        """Initialize local augmenter.

        Args:
            strategies: List of strategies to apply.
                Options: 'sentence_shuffle', 'paragraph_extract',
                         'token_noise', 'back_translate'
            augmentation_factor: How many augmented copies per original.
            min_text_length: Skip augmentation for very short texts.
            seed: Random seed for reproducibility.
        """
        self.strategies = strategies or ["sentence_shuffle", "paragraph_extract"]
        self.augmentation_factor = augmentation_factor
        self.min_text_length = min_text_length
        self.rng = random.Random(seed)

        self.stats = {
            "original_samples": 0,
            "augmented_samples": 0,
            "skipped_too_short": 0,
        }

    def augment(self, samples: Iterator[DataSample]) -> Iterator[DataSample]:
        """Augment a stream of samples.

        Always yields the original sample first, then augmented copies.
        """
        for sample in samples:
            self.stats["original_samples"] += 1

            # Always yield original
            yield sample

            # Skip augmentation for short texts
            if len(sample.content) < self.min_text_length:
                self.stats["skipped_too_short"] += 1
                continue

            for i in range(self.augmentation_factor):
                strategy = self.strategies[i % len(self.strategies)]
                augmented = self._apply_strategy(sample, strategy)
                if augmented:
                    self.stats["augmented_samples"] += 1
                    yield augmented

    def _apply_strategy(
        self, sample: DataSample, strategy: str
    ) -> DataSample | None:
        """Apply a single augmentation strategy."""
        if strategy == "sentence_shuffle":
            return self._sentence_shuffle(sample)
        elif strategy == "paragraph_extract":
            return self._paragraph_extract(sample)
        elif strategy == "token_noise":
            return self._token_noise(sample)
        elif strategy == "back_translate":
            return self._back_translate(sample)
        else:
            logger.warning(f"Unknown augmentation strategy: {strategy}")
            return None

    def _sentence_shuffle(self, sample: DataSample) -> DataSample | None:
        """Shuffle sentences in middle of text (keep first and last)."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", sample.content) if s.strip()]
        if len(sentences) < 4:
            return None

        # Keep first and last sentence, shuffle middle
        middle = sentences[1:-1]
        self.rng.shuffle(middle)
        new_text = " ".join([sentences[0]] + middle + [sentences[-1]])

        return DataSample(
            content=new_text,
            source=f"augmented/sentence_shuffle/{sample.source}",
            metadata={
                **sample.metadata,
                "augmentation": "sentence_shuffle",
                "original_source": sample.source,
            },
            modality=sample.modality,
        )

    def _paragraph_extract(self, sample: DataSample) -> DataSample | None:
        """Extract a random contiguous block of paragraphs."""
        paragraphs = [p.strip() for p in sample.content.split("\n\n") if p.strip()]
        if len(paragraphs) < 3:
            return None

        # Pick a random contiguous slice (at least 2 paragraphs)
        max_start = len(paragraphs) - 2
        start = self.rng.randint(0, max_start)
        length = self.rng.randint(2, min(len(paragraphs) - start, 5))
        extracted = "\n\n".join(paragraphs[start : start + length])

        if len(extracted) < self.min_text_length:
            return None

        return DataSample(
            content=extracted,
            source=f"augmented/paragraph_extract/{sample.source}",
            metadata={
                **sample.metadata,
                "augmentation": "paragraph_extract",
                "original_source": sample.source,
                "paragraph_range": f"{start}-{start + length}",
            },
            modality=sample.modality,
        )

    def _token_noise(self, sample: DataSample) -> DataSample | None:
        """Add minor token-level noise (swap adjacent chars, ~1% of words).

        This improves model robustness to typos without changing meaning.
        """
        words = sample.content.split()
        if len(words) < 20:
            return None

        noisy_words = []
        for word in words:
            if len(word) > 3 and self.rng.random() < 0.01:
                # Swap two adjacent characters
                idx = self.rng.randint(1, len(word) - 2)
                chars = list(word)
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
                noisy_words.append("".join(chars))
            else:
                noisy_words.append(word)

        new_text = " ".join(noisy_words)
        if new_text == sample.content:
            return None

        return DataSample(
            content=new_text,
            source=f"augmented/token_noise/{sample.source}",
            metadata={
                **sample.metadata,
                "augmentation": "token_noise",
                "original_source": sample.source,
            },
            modality=sample.modality,
        )

    def _back_translate(self, sample: DataSample) -> DataSample | None:
        """Back-translate using local MarianMT model (EN→DE→EN).

        Requires: pip install transformers sentencepiece
        Falls back gracefully if models are not available.
        """
        try:
            from transformers import MarianMTModel, MarianTokenizer

            # EN → DE
            en_de_name = "Helsinki-NLP/opus-mt-en-de"
            de_en_name = "Helsinki-NLP/opus-mt-de-en"

            tokenizer_en_de = MarianTokenizer.from_pretrained(en_de_name)
            model_en_de = MarianMTModel.from_pretrained(en_de_name)

            tokenizer_de_en = MarianTokenizer.from_pretrained(de_en_name)
            model_de_en = MarianMTModel.from_pretrained(de_en_name)

            # Truncate to avoid OOM
            text = sample.content[:512]

            # EN → DE
            inputs = tokenizer_en_de(text, return_tensors="pt", truncation=True, max_length=512)
            translated = model_en_de.generate(**inputs)
            german = tokenizer_en_de.decode(translated[0], skip_special_tokens=True)

            # DE → EN
            inputs = tokenizer_de_en(german, return_tensors="pt", truncation=True, max_length=512)
            back_translated = model_de_en.generate(**inputs)
            result = tokenizer_de_en.decode(back_translated[0], skip_special_tokens=True)

            if len(result.strip()) < 20:
                return None

            return DataSample(
                content=result.strip(),
                source=f"augmented/back_translate/{sample.source}",
                metadata={
                    **sample.metadata,
                    "augmentation": "back_translate",
                    "original_source": sample.source,
                    "pivot_language": "de",
                },
                modality=sample.modality,
            )

        except (ImportError, OSError) as e:
            logger.debug(f"Back-translation unavailable: {e}")
            return None

    def summary(self) -> dict[str, Any]:
        """Return augmentation statistics."""
        return dict(self.stats)
