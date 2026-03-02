# Auralith Data Pipeline

**Production-grade multimodal data processing pipeline for training [RT-DLM](https://github.com/AuralithAI/RT-DLM) and large-scale AI systems.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://github.com/AuralithAI/Auralith-Data-Pipeline/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI](https://img.shields.io/pypi/v/auralith-data-pipeline)](https://pypi.org/project/auralith-data-pipeline/)

---

## Overview

Auralith Data Pipeline ingests raw **text, images, audio, video, and code**, applies production-quality curation (perplexity filtering, LLM-as-Judge scoring, FAISS deduplication, PII scrubbing, license detection), tokenizes everything through **BPE + Vector Quantization**, and outputs **SafeTensors shards** ready for distributed model training.

### Pipeline Stages

| Stage | What happens |
|-------|-------------|
| **Ingestion** | Text (HuggingFace, Common Crawl, local), images (`.npy`/JPEG/PNG), audio (`.wav`/`.npy`), video (`.mp4`), code (TheStack) |
| **Quality Curation** | GPT-2 perplexity filter, LLM-as-Judge scoring, FAISS embedding dedup, license detection |
| **Tokenization** | BPE for text, patch + VQ for images, mel + VQ for audio, frame + VQ for video |
| **Sharding** | SafeTensors v2 schema with `input_ids`, `attention_mask`, `modality_mask`, `targets` |
| **Observability** | MLflow / W&B tracking, per-sample lineage, auto data cards |
| **Orchestration** | Argo Workflows, Helm/K8s, Ray, distributed coordinator + workers |

---

## Installation

```bash
# Core text pipeline
pip install auralith-data-pipeline

# With all extras (multimodal, cloud, distributed, dev tools)
pip install "auralith-data-pipeline[all]"

# Pick only what you need
pip install "auralith-data-pipeline[quality]"        # + perplexity filter + FAISS dedup
pip install "auralith-data-pipeline[distributed]"    # + Ray
pip install "auralith-data-pipeline[cloud,pdf]"      # + S3/GCS/Azure + PDF extraction
pip install "auralith-data-pipeline[multimodal]"     # + video/image/audio (PyTorch)
pip install "auralith-data-pipeline[tracking]"       # + MLflow + W&B
```

---

## Quick Start

### CLI

```bash
# List available datasets
auralith-pipeline list-datasets

# Process Wikipedia dataset
auralith-pipeline collect \
  --dataset wikipedia \
  --output ./data/shards \
  --max-samples 100000 \
  --preset production
```

### End-to-End Workflow

```bash
# 1. Train tokenizers (BPE + VQ codebooks)
auralith-pipeline train-tokenizer all \
  --corpus  data/corpus/ \
  --images  data/images/ \
  --audio   data/audio/ \
  --videos  data/videos/ \
  --output  tokenizers/ \
  --vocab-size 32000 \
  --codebook-size 1024

# 2. Process raw data into SafeTensors shards
auralith-pipeline process \
  --input  data/raw/ \
  --output shards/ \
  --tokenizers tokenizers/ \
  --max-seq-len 4096 \
  --shard-size 10000

# 3. Upload to cloud storage or HuggingFace Hub
auralith-pipeline upload --source shards/ --dest s3://my-bucket/training-data/
```

### Python API

```python
from auralith_pipeline import Pipeline, PipelineConfig
from auralith_pipeline.sources import create_source

config = PipelineConfig.from_preset("production")
pipeline = Pipeline(config)
source = create_source("wikipedia", streaming=True, max_samples=1_000_000)
pipeline.add_source(source)

stats = pipeline.run()
print(stats.summary())
```

---

## Key Features

### Data Processing
- Multi-source ingestion (HuggingFace, Common Crawl, local files, video)
- Weighted round-robin interleaving across multiple sources
- MinHash + FAISS embedding deduplication
- Quality filtering (length, language, perplexity, LLM-as-Judge)
- PII removal (multi-jurisdiction, 15+ countries)
- License compliance scanning for code data
- Document extraction (PDF, DOCX, HTML, Markdown)
- SafeTensors sharding with Zstd compression and SHA-256 checksums
- Streaming checkpointing with seeded reproducibility

### Tokenization
- Custom BPE tokenizer with 16 special tokens and byte-level fallback
- Vector quantization for images, audio, and video
- Multimodal token fusion with `encode_with_mask()`
- Configurable vocab size (32k-128k)

### Distributed Processing
- **Embedded mode** — in-process coordinator + workers (no Redis needed)
- **External mode** — multi-machine with Redis state store
- Worker failure detection + automatic task requeue
- Linear scaling up to 64+ workers

### Observability & Compliance
- MLflow / Weights & Biases experiment tracking
- Per-sample lineage (source to shard provenance)
- Auto-generated data cards (HuggingFace-compatible)
- Full audit logging (JSONL) for accept/reject decisions
- Credential and secret sanitization

---

## SafeTensors Schema (v2)

Every output shard is directly compatible with RT-DLM training.

| Tensor | Dtype | Shape | Description |
|--------|-------|-------|-------------|
| `input_ids` | int32 | (batch, seq_len) | All tokens (text + image + audio + video + code) |
| `attention_mask` | uint8 | (batch, seq_len) | 1 = real token, 0 = padding |
| `modality_mask` | uint8 | (batch, seq_len) | 0=text, 1=image, 2=audio, 3=video, 4=code |
| `targets` | int32 | (batch, seq_len) | Right-shifted `input_ids` for causal LM |

### Special Tokens

| ID | Token | Purpose |
|----|-------|---------|
| 0 | `<PAD>` | Padding |
| 1 | `<UNK>` | Unknown |
| 2 | `<BOS>` | Beginning of sequence |
| 3 | `<EOS>` | End of sequence |
| 4-5 | `<IMG>` / `<IMG_END>` | Image region |
| 6-7 | `<AUDIO>` / `<AUDIO_END>` | Audio region |
| 8-9 | `<VIDEO>` / `<VIDEO_END>` | Video region |
| 10 | `<FUSE>` | Cross-modal fusion |
| 11 | `<SEP>` | Separator |
| 12 | `<MASK>` | Masked LM |
| 13-14 | `<CODE>` / `<CODE_END>` | Code block |
| 15 | `<THINK>` | Chain-of-thought |

---

## Available Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| wikipedia | 20 GB | English Wikipedia |
| c4 | 750 GB | Cleaned Common Crawl |
| redpajama | 1.2 TB | LLaMA training data |
| openwebtext | 40 GB | Reddit links |
| bookcorpus | 5 GB | 11k books |
| the_stack | 3 TB | Source code (deduplicated) |

---

## Performance

| Operation | Speed |
|-----------|-------|
| Text preprocessing | 10k samples/sec |
| MinHash deduplication | 5k samples/sec |
| FAISS dedup | 3k samples/sec |
| BPE encoding | < 1 ms/sample |
| SafeTensors writing | 50 MB/s |
| Image tokenization | 50 ms/image |
| Video tokenization | 200 ms/video |

---

## Configuration

```yaml
# configs/production.yaml
pipeline:
  name: production-pipeline
  output_dir: ./data/shards
  deduplicate: true
  quality_filter: true
  remove_pii: true
  seed: 42
  checkpoint_every: 10000

advanced_quality:
  enabled: true
  perplexity_filter: true
  max_perplexity: 1500.0
```

---

## Documentation

For the full documentation, architecture diagrams, distributed processing guide, and contributor guide, visit the [GitHub repository](https://github.com/AuralithAI/Auralith-Data-Pipeline).

- [Architecture](https://github.com/AuralithAI/Auralith-Data-Pipeline/blob/main/docs/ARCHITECTURE.md)
- [Contributing](https://github.com/AuralithAI/Auralith-Data-Pipeline/blob/main/docs/CONTRIBUTING.md)
- [Distributed Processing](https://github.com/AuralithAI/Auralith-Data-Pipeline/blob/main/docs/DISTRIBUTED_PROCESSING.md)
- [Changelog](https://github.com/AuralithAI/Auralith-Data-Pipeline/releases)

---

## License

Apache License 2.0 — see [LICENSE](https://github.com/AuralithAI/Auralith-Data-Pipeline/blob/main/LICENSE).

---

Built by [AuralithAI](https://github.com/AuralithAI) for [RT-DLM](https://github.com/AuralithAI/RT-DLM).
