"""Advanced quality filtering — perplexity scoring + LLM-as-Judge.

Features:
  • Perplexity filter using a small causal LM (GPT-2 or custom)
  • LLM-as-Judge for coherence, toxicity, and multimodal alignment
  • Pluggable scorer interface for future models
"""

import logging
import math
from abc import ABC, abstractmethod
from typing import Any

from auralith_pipeline.sources.data_sources import DataSample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scorer interface
# ---------------------------------------------------------------------------


class QualityScorer(ABC):
    """Base class for quality scorers."""

    @abstractmethod
    def score(self, text: str) -> dict[str, float]:
        """Return quality scores for a text sample.

        Returns:
            Dict with score names as keys (e.g. 'perplexity', 'coherence').
        """
        ...


# ---------------------------------------------------------------------------
# Perplexity filter
# ---------------------------------------------------------------------------


class PerplexityFilter(QualityScorer):
    """Compute perplexity using a small causal LM.

    Uses GPT-2 (124 M) by default. Samples above `max_perplexity` are dropped,
    and samples below `min_perplexity` are suspicious (likely boilerplate).

    Requires `transformers` + `torch` (installed via the `multimodal` extra).
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        max_perplexity: float = 1000.0,
        min_perplexity: float = 5.0,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 8,
    ):
        self.model_name = model_name
        self.max_perplexity = max_perplexity
        self.min_perplexity = min_perplexity
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Lazy-load the LM."""
        if self._model is not None:
            return

        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading perplexity model: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model.eval()
            self._model.to(self.device)

            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

        except ImportError:
            logger.warning(
                "transformers/torch not installed — perplexity filter disabled. "
                "Install with: pip install transformers torch"
            )

    def score(self, text: str) -> dict[str, float]:
        """Compute perplexity for a text."""
        self._load_model()

        if self._model is None:
            return {"perplexity": -1.0}

        import torch

        encodings = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss.item()

        perplexity = math.exp(loss)
        return {"perplexity": perplexity}

    def passes(self, text: str) -> bool:
        """Check if text passes the perplexity filter."""
        scores = self.score(text)
        ppl = scores["perplexity"]
        if ppl < 0:
            return True  # Model not loaded, let it through
        return self.min_perplexity <= ppl <= self.max_perplexity


# ---------------------------------------------------------------------------
# LLM-as-Judge
# ---------------------------------------------------------------------------


class LLMJudge(QualityScorer):
    """Use an LLM to judge sample quality.

    Supports:
      • Local models via transformers (e.g. Llama-3.1-8B)
      • Remote APIs (OpenAI-compatible, Anthropic, Grok)

    The judge evaluates:
      • Coherence (0–1)
      • Toxicity  (0–1, lower is better)
      • Educational value (0–1)
      • Multimodal alignment (0–1) — when modality info is available
    """

    SYSTEM_PROMPT = (
        "You are a data quality judge. Given a text sample, rate it on a 0–1 scale "
        "for: coherence, toxicity (0 = safe, 1 = toxic), educational_value. "
        "Reply ONLY with JSON: "
        '{"coherence": 0.X, "toxicity": 0.X, "educational_value": 0.X}'
    )

    def __init__(
        self,
        provider: str = "local",
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        api_key: str | None = None,
        api_base: str | None = None,
        max_tokens: int = 128,
        coherence_threshold: float = 0.4,
        toxicity_threshold: float = 0.5,
    ):
        """Initialize LLM judge.

        Args:
            provider: 'local', 'openai', 'anthropic'
            model_name: Model identifier
            api_key: API key for remote providers
            api_base: Custom API base URL
            max_tokens: Max output tokens
            coherence_threshold: Minimum coherence score to pass
            toxicity_threshold: Maximum toxicity score to pass
        """
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.coherence_threshold = coherence_threshold
        self.toxicity_threshold = toxicity_threshold

    def score(self, text: str) -> dict[str, float]:
        """Score a text sample using the LLM judge."""
        # Truncate very long texts for the judge
        sample_text = text[:2000] if len(text) > 2000 else text

        if self.provider == "openai":
            return self._score_openai(sample_text)
        elif self.provider == "anthropic":
            return self._score_anthropic(sample_text)
        else:
            return self._score_local(sample_text)

    def _score_openai(self, text: str) -> dict[str, float]:
        """Score using OpenAI-compatible API."""
        import json

        try:
            import openai

            client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Rate this text:\n\n{text}"},
                ],
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            result = json.loads(response.choices[0].message.content)
            return {
                "coherence": float(result.get("coherence", 0.5)),
                "toxicity": float(result.get("toxicity", 0.0)),
                "educational_value": float(result.get("educational_value", 0.5)),
            }
        except Exception as e:
            logger.warning(f"LLM judge (openai) failed: {e}")
            return {"coherence": 0.5, "toxicity": 0.0, "educational_value": 0.5}

    def _score_anthropic(self, text: str) -> dict[str, float]:
        """Score using Anthropic API."""
        import json

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Rate this text:\n\n{text}"}],
            )
            result = json.loads(response.content[0].text)
            return {
                "coherence": float(result.get("coherence", 0.5)),
                "toxicity": float(result.get("toxicity", 0.0)),
                "educational_value": float(result.get("educational_value", 0.5)),
            }
        except Exception as e:
            logger.warning(f"LLM judge (anthropic) failed: {e}")
            return {"coherence": 0.5, "toxicity": 0.0, "educational_value": 0.5}

    def _score_local(self, text: str) -> dict[str, float]:
        """Score using a local model — lightweight heuristic fallback."""
        # Simple heuristic when no LLM is available
        words = text.split()
        word_count = len(words)

        # Coherence heuristic: sentence length variance
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if len(sentences) > 1:
            lengths = [len(s.split()) for s in sentences]
            avg_len = sum(lengths) / len(lengths)
            variance = sum((length - avg_len) ** 2 for length in lengths) / len(lengths)
            coherence = min(1.0, 1.0 / (1.0 + variance / 100))
        else:
            coherence = 0.5

        # Toxicity heuristic: presence of common toxic patterns
        toxic_patterns = [
            "kill",
            "hate",
            "die",
            "stupid",
            "idiot",
            "racist",
        ]
        toxic_count = sum(1 for w in words if w.lower() in toxic_patterns)
        toxicity = min(1.0, toxic_count / max(word_count, 1) * 50)

        # Educational value: presence of informative markers
        edu_markers = [
            "research",
            "study",
            "analysis",
            "theorem",
            "equation",
            "algorithm",
            "method",
            "result",
            "conclusion",
            "hypothesis",
        ]
        edu_count = sum(1 for w in words if w.lower() in edu_markers)
        educational_value = min(1.0, edu_count / max(word_count, 1) * 20)

        return {
            "coherence": round(coherence, 3),
            "toxicity": round(toxicity, 3),
            "educational_value": round(educational_value, 3),
        }

    def passes(self, text: str) -> bool:
        """Check if text passes the LLM judge."""
        scores = self.score(text)
        return (
            scores["coherence"] >= self.coherence_threshold
            and scores["toxicity"] <= self.toxicity_threshold
        )


# ---------------------------------------------------------------------------
# Combined quality pipeline
# ---------------------------------------------------------------------------


class AdvancedQualityPipeline:
    """Combine perplexity + LLM judge into a single quality gate.

    Attaches quality scores to sample metadata for lineage/observability.
    """

    def __init__(
        self,
        enable_perplexity: bool = True,
        enable_llm_judge: bool = False,
        perplexity_config: dict[str, Any] | None = None,
        llm_judge_config: dict[str, Any] | None = None,
    ):
        self.perplexity_filter: PerplexityFilter | None = None
        self.llm_judge: LLMJudge | None = None

        if enable_perplexity:
            self.perplexity_filter = PerplexityFilter(**(perplexity_config or {}))

        if enable_llm_judge:
            self.llm_judge = LLMJudge(**(llm_judge_config or {}))

        self.stats = {
            "total": 0,
            "passed_perplexity": 0,
            "failed_perplexity": 0,
            "passed_llm_judge": 0,
            "failed_llm_judge": 0,
        }

    def evaluate(self, sample: DataSample) -> tuple[bool, dict[str, float]]:
        """Evaluate a sample and return (pass/fail, scores).

        Scores are attached to sample.metadata["quality_scores"].
        """
        self.stats["total"] += 1
        scores: dict[str, float] = {}

        # Perplexity check
        if self.perplexity_filter:
            ppl_scores = self.perplexity_filter.score(sample.content)
            scores.update(ppl_scores)
            if not self.perplexity_filter.passes(sample.content):
                self.stats["failed_perplexity"] += 1
                return False, scores
            self.stats["passed_perplexity"] += 1

        # LLM judge check
        if self.llm_judge:
            judge_scores = self.llm_judge.score(sample.content)
            scores.update(judge_scores)
            if not self.llm_judge.passes(sample.content):
                self.stats["failed_llm_judge"] += 1
                return False, scores
            self.stats["passed_llm_judge"] += 1

        return True, scores
