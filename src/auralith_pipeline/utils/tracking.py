"""Observability, lineage tracking, and experiment versioning.

Integrates with MLflow and Weights & Biases for:
  • Pipeline run tracking (params, metrics, artifacts)
  • Per-sample lineage (source → filters → shard)
  • Data versioning via DVC / HF Datasets
  • Auto Data Cards
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lineage record
# ---------------------------------------------------------------------------

@dataclass
class SampleLineage:
    """Full provenance record for a single sample."""

    sample_id: str
    source: str
    source_index: int | None = None
    filters_applied: list[str] = field(default_factory=list)
    filters_passed: list[str] = field(default_factory=list)
    quality_scores: dict[str, float] = field(default_factory=dict)
    dedup_hash: str | None = None
    pii_removed: bool = False
    synthetic: bool = False
    synthetic_strategy: str | None = None
    shard_id: int | None = None
    token_count: int = 0
    modality: str = "text"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source": self.source,
            "source_index": self.source_index,
            "filters_applied": self.filters_applied,
            "filters_passed": self.filters_passed,
            "quality_scores": self.quality_scores,
            "dedup_hash": self.dedup_hash,
            "pii_removed": self.pii_removed,
            "synthetic": self.synthetic,
            "synthetic_strategy": self.synthetic_strategy,
            "shard_id": self.shard_id,
            "token_count": self.token_count,
            "modality": self.modality,
            "timestamp": self.timestamp,
        }


class LineageTracker:
    """Track provenance for every sample through the pipeline."""

    def __init__(self, output_dir: str | None = None):
        self.records: list[SampleLineage] = []
        self.output_dir = Path(output_dir) if output_dir else None
        self._counter = 0

    def create_record(self, source: str, **kwargs: Any) -> SampleLineage:
        """Create a new lineage record."""
        self._counter += 1
        record = SampleLineage(
            sample_id=f"sample_{self._counter:08d}",
            source=source,
            **kwargs,
        )
        self.records.append(record)
        return record

    def save(self, path: str | None = None) -> None:
        """Save lineage records to JSONL."""
        output = Path(path) if path else (self.output_dir / "lineage.jsonl" if self.output_dir else None)
        if output is None:
            logger.warning("No output path for lineage — skipping save")
            return

        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            for record in self.records:
                f.write(json.dumps(record.to_dict()) + "\n")

        logger.info(f"Saved {len(self.records)} lineage records to {output}")

    def summary(self) -> dict[str, Any]:
        """Summarize lineage statistics."""
        sources: dict[str, int] = {}
        modalities: dict[str, int] = {}
        for r in self.records:
            sources[r.source] = sources.get(r.source, 0) + 1
            modalities[r.modality] = modalities.get(r.modality, 0) + 1

        return {
            "total_samples": len(self.records),
            "sources": sources,
            "modalities": modalities,
            "synthetic_count": sum(1 for r in self.records if r.synthetic),
            "pii_removed_count": sum(1 for r in self.records if r.pii_removed),
        }


# ---------------------------------------------------------------------------
# Experiment tracker (MLflow / W&B)
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """Unified experiment tracking for pipeline runs.

    Supports MLflow and Weights & Biases as backends.
    Falls back to local JSON logging when neither is available.
    """

    def __init__(
        self,
        backend: str = "local",
        project_name: str = "auralith-data-pipeline",
        experiment_name: str | None = None,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ):
        """Initialize experiment tracker.

        Args:
            backend: 'mlflow', 'wandb', or 'local'
            project_name: Project / experiment group name
            experiment_name: MLflow experiment name
            run_name: Name for this specific run
            tags: Key-value tags for the run
        """
        self.backend = backend
        self.project_name = project_name
        self.experiment_name = experiment_name or project_name
        self.run_name = run_name
        self.tags = tags or {}

        self._run = None
        self._local_log: list[dict[str, Any]] = []
        self._start_time = time.time()

    def start_run(self) -> None:
        """Start a tracking run."""
        if self.backend == "mlflow":
            self._start_mlflow()
        elif self.backend == "wandb":
            self._start_wandb()
        else:
            logger.info(f"Local tracking run: {self.run_name}")

    def _start_mlflow(self) -> None:
        try:
            import mlflow

            mlflow.set_experiment(self.experiment_name)
            self._run = mlflow.start_run(run_name=self.run_name, tags=self.tags)
            logger.info(f"MLflow run started: {self._run.info.run_id}")
        except ImportError:
            logger.warning("mlflow not installed — falling back to local tracking")
            self.backend = "local"

    def _start_wandb(self) -> None:
        try:
            import wandb

            self._run = wandb.init(
                project=self.project_name,
                name=self.run_name,
                tags=list(self.tags.values()),
                config=self.tags,
            )
            logger.info(f"W&B run started: {self._run.id}")
        except ImportError:
            logger.warning("wandb not installed — falling back to local tracking")
            self.backend = "local"

    def log_params(self, params: dict[str, Any]) -> None:
        """Log pipeline parameters."""
        if self.backend == "mlflow":
            import mlflow
            mlflow.log_params({k: str(v) for k, v in params.items()})
        elif self.backend == "wandb":
            import wandb
            wandb.config.update(params)
        else:
            self._local_log.append({"type": "params", "data": params})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log pipeline metrics."""
        if self.backend == "mlflow":
            import mlflow
            mlflow.log_metrics(metrics, step=step)
        elif self.backend == "wandb":
            import wandb
            wandb.log(metrics, step=step)
        else:
            self._local_log.append({"type": "metrics", "step": step, "data": metrics})

    def log_artifact(self, path: str, artifact_type: str = "data") -> None:
        """Log a file/directory as an artifact."""
        if self.backend == "mlflow":
            import mlflow
            mlflow.log_artifact(path)
        elif self.backend == "wandb":
            import wandb
            artifact = wandb.Artifact(name=artifact_type, type=artifact_type)
            if Path(path).is_dir():
                artifact.add_dir(path)
            else:
                artifact.add_file(path)
            wandb.log_artifact(artifact)
        else:
            self._local_log.append({"type": "artifact", "path": path})

    def end_run(self, status: str = "success") -> None:
        """End the tracking run."""
        elapsed = time.time() - self._start_time
        self.log_metrics({"elapsed_seconds": elapsed})

        if self.backend == "mlflow":
            import mlflow
            mlflow.end_run(status="FINISHED" if status == "success" else "FAILED")
        elif self.backend == "wandb":
            import wandb
            wandb.finish(exit_code=0 if status == "success" else 1)
        else:
            logger.info(f"Local run complete: {status} ({elapsed:.1f}s)")

    def save_local_log(self, path: str) -> None:
        """Save local log to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._local_log, f, indent=2, default=str)
        logger.info(f"Saved local tracking log to {path}")


# ---------------------------------------------------------------------------
# Auto Data Card generator
# ---------------------------------------------------------------------------

class DataCardGenerator:
    """Generate HuggingFace-style Data Cards for processed datasets."""

    def __init__(self, pipeline_name: str = "Auralith Data Pipeline"):
        self.pipeline_name = pipeline_name

    def generate(
        self,
        output_path: str,
        config: dict[str, Any],
        stats: dict[str, Any],
        lineage_summary: dict[str, Any] | None = None,
    ) -> str:
        """Generate a Markdown data card.

        Args:
            output_path: Where to save the README.md
            config: Pipeline configuration dict
            stats: Pipeline run statistics
            lineage_summary: Optional lineage summary

        Returns:
            The generated Markdown string
        """
        sources = lineage_summary.get("sources", {}) if lineage_summary else {}
        modalities = lineage_summary.get("modalities", {}) if lineage_summary else {}

        def _fmt(val: Any) -> str:
            """Format a value, using comma separator for integers."""
            if isinstance(val, int):
                return f"{val:,}"
            return str(val)

        card = f"""---
language: en
license: apache-2.0
tags:
  - auralith
  - rt-dlm
  - safetensors
  - multimodal
---

# {config.get('name', 'Auralith Dataset')}

Generated by **{self.pipeline_name}**

## Dataset Summary

| Metric | Value |
|--------|-------|
| Total samples | {_fmt(stats.get('total_samples', 'N/A'))} |
| Total tokens | {_fmt(stats.get('total_tokens', 'N/A'))} |
| Shards | {_fmt(stats.get('num_shards', 'N/A'))} |
| Total size | {_fmt(stats.get('total_size', 'N/A'))} |
| Duplicates removed | {_fmt(stats.get('duplicates_removed', 'N/A'))} |
| PII scrubbed | {_fmt(stats.get('pii_removed', 'N/A'))} |

## SafeTensors Schema

Each `.safetensors` shard contains:

| Tensor | Dtype | Shape | Description |
|--------|-------|-------|-------------|
| `input_ids` | int32 | (batch, seq_len) | All tokens (text + image + audio + video) |
| `attention_mask` | int32 | (batch, seq_len) | 1 = real token, 0 = padding |
| `modality_mask` | uint8 | (batch, seq_len) | 0=text, 1=image, 2=audio, 3=video |
| `labels` | int32 | (batch, seq_len) | Causal LM labels (-100 = ignore) |

## Sources

{self._format_sources(sources)}

## Modalities

{self._format_dict(modalities)}

## Processing Pipeline

1. **Ingestion**: HuggingFace Datasets (streaming)
2. **Normalization**: ftfy + Unicode cleanup
3. **Quality filtering**: Length, language, character ratios
4. **Deduplication**: MinHash LSH + FAISS embedding similarity
5. **PII removal**: Regex-based scrubbing
6. **Tokenization**: Custom BPE + VQ for images/audio/video
7. **Sharding**: SafeTensors + Zstd compression

## Configuration

```yaml
{self._format_config(config)}
```

## Usage with RT-DLM

```python
from safetensors.numpy import load_file

shard = load_file("shard_00000.safetensors")
input_ids = shard["input_ids"]       # (batch, seq_len)
modality_mask = shard["modality_mask"] # (batch, seq_len)
labels = shard["labels"]              # (batch, seq_len)
```

## License

Apache 2.0 — see [LICENSE](LICENSE)
"""
        # Save
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(card)
        logger.info(f"Data card saved to {output_path}")
        return card

    def _format_sources(self, sources: dict[str, int]) -> str:
        if not sources:
            return "No source information available."
        lines = ["| Source | Samples |", "|--------|---------|"]
        for src, count in sources.items():
            lines.append(f"| {src} | {count:,} |")
        return "\n".join(lines)

    def _format_dict(self, d: dict[str, Any]) -> str:
        if not d:
            return "N/A"
        lines = []
        for k, v in d.items():
            lines.append(f"- **{k}**: {v}")
        return "\n".join(lines)

    def _format_config(self, config: dict[str, Any]) -> str:
        import yaml
        return yaml.dump(config, default_flow_style=False)
