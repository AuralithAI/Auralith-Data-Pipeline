# Auralith Data Pipeline

Production-grade multimodal data processing pipeline for training [RT-DLM](https://github.com/AuralithAI/RT-DLM) and large-scale AI systems.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Auralith Data Pipeline                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
│  │   Text   │  │  Images   │  │  Audio    │  │  Video    │  │   Code    │   │
│  │  (HF/CC) │  │  (.npy)   │  │  (.wav)   │  │  (.mp4)   │  │(TheStack) │   │
│  └────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘   │
│       │              │              │              │              │         │
│       ▼              ▼              ▼              ▼              ▼         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       Quality Curation                              │    │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────────┐     │    │
│  │  │Perplexity │  │ LLM-as-   │  │   FAISS   │  │   License     │     │    │
│  │  │  Filter   │  │  Judge    │  │  DeDup    │  │  Detection    │     │    │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────────┘     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   Tokenization (BPE + VQ)                           │    │
│  │  Text → BPE  │  Img → Patch+VQ  │  Audio → Mel+VQ  │  Video → VQ │  │    │
│  │                                                                     │    │
│  │  Special Tokens: <IMG> <AUDIO> <VIDEO> <FUSE> <CODE> <THINK>        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              SafeTensors Shards (RT-DLM Compatible)                 │    │
│  │  ┌───────────┐ ┌───────────────┐ ┌──────────────┐ ┌────────────┐    │    │
│  │  │ input_ids │ │attention_mask │ │modality_mask │ │   labels   │    │    │
│  │  │  int32    │ │    int32      │ │    uint8     │ │   int32    │    │    │
│  │  └───────────┘ └───────────────┘ └──────────────┘ └────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                     │
│  ┌────┴──────────────────────────────────────────────────────────────┐      │
│  │  Observability            │  Orchestration                        │      │
│  │  MLflow / W&B / Local     │  Argo Workflows / Ray / Helm          │      │
│  │  Lineage + Data Cards     │  K8s + DGX Cloud                      │      │
│  └───────────────────────────┴───────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │     RT-DLM      │
                           │  (JAX / Haiku)  │
                           │  Distributed    │
                           │  Training       │
                           └─────────────────┘
```

## Capabilities

| Category | Feature | Status |
|----------|---------|--------|
| **Schema** | SafeTensors 4-tensor schema (`modality_mask`, `labels`) | ✅ |
| **Schema** | Video frame extraction + VQ tokenizer | ✅ |
| **Schema** | 16 special tokens (`<IMG>`, `<VIDEO>`, `<FUSE>`, `<THINK>`, etc.) | ✅ |
| **Quality** | GPT-2 perplexity filter | ✅ |
| **Quality** | LLM-as-Judge quality scoring | ✅ |
| **Quality** | FAISS embedding deduplication | ✅ |
| **Quality** | Local data augmentation (sentence shuffle, noise, back-translate) | ✅ |
| **Observability** | MLflow / W&B experiment tracking | ✅ |
| **Observability** | Per-sample lineage (source → shard provenance) | ✅ |
| **Observability** | Auto data card generation | ✅ |
| **Orchestration** | Argo Workflows DAG orchestration | ✅ |
| **Orchestration** | Helm chart for K8s deployment | ✅ |
| **Orchestration** | Ray distributed pipeline runner | ✅ |
| **Compliance** | License detection (permissive/copyleft) | ✅ |
| **Compliance** | Full audit logging (JSONL) | ✅ |
| **Compliance** | E2E schema validation tests | ✅ |
| **Security** | Multi-jurisdiction PII scrubbing (15+ countries) | ✅ |
| **Security** | Credential / secret sanitization | ✅ |
| **Security** | IRSA / Workload Identity (no static keys) | ✅ |

## Installation

```bash
git clone https://github.com/AuralithAI/Auralith-Data-Pipeline.git
cd Auralith-Data-Pipeline

python -m venv venv
source venv/bin/activate  # Linux/Mac
# or .\venv\Scripts\activate  # Windows

# Core installation
pip install -e .

# With specific extras
pip install -e ".[quality]"        # Perplexity filter + FAISS dedup
pip install -e ".[tracking]"       # MLflow + W&B
pip install -e ".[distributed]"    # Ray
pip install -e ".[multimodal]"     # Video + image + audio
pip install -e ".[cloud,pdf]"      # Cloud storage + PDF extraction
pip install -e ".[all]"            # Everything (~3 GB with PyTorch)
```

## Quick Start

### CLI Usage

```bash
# List available datasets
auralith-pipeline list-datasets

# Process Wikipedia dataset (production preset)
auralith-pipeline collect \
  --dataset wikipedia \
  --output ./data/shards \
  --max-samples 100000 \
  --preset production

# Train custom BPE tokenizer
python scripts/train_tokenizer.py text \
  --corpus data/corpus.txt \
  --output tokenizers/bpe_32k \
  --vocab-size 32000
```

### Python API

```python
from auralith_pipeline import Pipeline, PipelineConfig
from auralith_pipeline.sources import create_source

# Configure pipeline
config = PipelineConfig.from_preset("production")

# Create and run pipeline
pipeline = Pipeline(config)
source = create_source("wikipedia", streaming=True, max_samples=1_000_000)
pipeline.add_source(source)

stats = pipeline.run()
print(stats.summary())
```

### Using Processed Shards with RT-DLM

```python
from safetensors.numpy import load_file

shard = load_file("data/shards/shard_00000.safetensors")

input_ids      = shard["input_ids"]       # (batch, seq_len) — int32
attention_mask = shard["attention_mask"]   # (batch, seq_len) — int32
modality_mask  = shard["modality_mask"]    # (batch, seq_len) — uint8 (0=text,1=img,2=aud,3=vid)
labels         = shard["labels"]          # (batch, seq_len) — int32 (-100 = ignore)

# Feed directly to RT-DLM training
# python src/train.py --data-dir ./data/shards
```

## SafeTensors Schema

Every shard is RT-DLM compatible:

| Tensor | Dtype | Shape | Description |
|--------|-------|-------|-------------|
| `input_ids` | int32 | (batch, seq_len) | All tokens (text + image + audio + video) |
| `attention_mask` | int32 | (batch, seq_len) | 1 = real token, 0 = padding |
| `modality_mask` | uint8 | (batch, seq_len) | 0=text, 1=image, 2=audio, 3=video |
| `labels` | int32 | (batch, seq_len) | Causal LM labels (-100 = ignore special tokens) |

### Special Tokens (IDs 0–15)

| ID | Token | Purpose |
|----|-------|---------|
| 0 | `<PAD>` | Padding |
| 1 | `<UNK>` | Unknown |
| 2 | `<BOS>` | Beginning of sequence |
| 3 | `<EOS>` | End of sequence |
| 4 | `<IMG>` | Image region start |
| 5 | `<IMG_END>` | Image region end |
| 6 | `<AUDIO>` | Audio region start |
| 7 | `<AUDIO_END>` | Audio region end |
| 8 | `<VIDEO>` | Video region start |
| 9 | `<VIDEO_END>` | Video region end |
| 10 | `<FUSE>` | Cross-modal fusion |
| 11 | `<SEP>` | Separator |
| 12 | `<MASK>` | Masked LM |
| 13 | `<CODE>` | Code block start |
| 14 | `<CODE_END>` | Code block end |
| 15 | `<THINK>` | Chain-of-thought |

## Features

### Data Processing
- Multi-source ingestion (HuggingFace, Common Crawl, local files, video)
- MinHash + FAISS embedding deduplication
- Quality filtering (length, language, perplexity, LLM-as-Judge)
- PII removal (automatic detection and redaction)
- License compliance scanning for code data
- Document extraction (PDF, DOCX, HTML, Markdown)
- SafeTensors sharding with Zstd compression

### Tokenization
- Custom BPE tokenizer (16 special tokens, no external dependency)
- Vector quantization for images, audio, and video
- Multimodal token fusion with `encode_with_mask()`
- Character-level fallback for OOV handling
- Configurable vocab size (32k–128k)

### Quality & Compliance
- **Perplexity filter**: GPT-2 based scoring with configurable thresholds
- **LLM-as-Judge**: Score coherence, toxicity, educational value
- **FAISS dedup**: Cosine similarity with IVFFlat/IVFPQ indexes
- **License detection**: Permissive vs copyleft classification
- **Audit logging**: Full accept/reject decisions to JSONL
- **Local augmentation**: Sentence shuffle, paragraph extract, token noise, back-translate

### Observability
- **MLflow / W&B** experiment tracking (params, metrics, artifacts)
- **Per-sample lineage** — track every sample from source to shard
- **Auto data cards** — HuggingFace-compatible README.md generation

### Orchestration
- **Argo Workflows** — DAG-based parallel dataset processing
- **Helm chart** — deploy on any K8s cluster or DGX Cloud
- **Ray** — horizontal scaling across machines

### Storage & Deployment
- Cloud storage (HuggingFace Hub, S3, GCS, Azure Blob)
- Docker support for containerized deployment
- CI via GitHub Actions (lint, test, build)

## Configuration

```yaml
# configs/production.yaml
pipeline:
  name: production-pipeline
  output_dir: ./data/shards
  deduplicate: true
  quality_filter: true
  remove_pii: true

advanced_quality:
  enabled: true
  perplexity_filter: true
  max_perplexity: 1500.0

deduplication:
  method: minhash    # or: embedding (FAISS)
  minhash_threshold: 0.85

tracking:
  enabled: true
  backend: local     # or: mlflow, wandb

compliance:
  enabled: true
  license_detection: true
  allow_copyleft: false
  audit_log_path: ./data/audit/audit.jsonl

video:
  enabled: false
  frame_strategy: uniform
  max_frames: 32
```

See [`configs/production.yaml`](configs/production.yaml) for the full configuration reference.

## Deploy to DGX Cloud in 5 Steps

```bash
# 1. Build container
docker build -t auralith-pipeline:latest .

# 2. Push to registry
docker tag auralith-pipeline:latest nvcr.io/YOUR_ORG/auralith-pipeline:latest
docker push nvcr.io/YOUR_ORG/auralith-pipeline:latest

# 3. Install Helm chart
helm install auralith docker/kubernetes/helm/ \
  --set image.repository=nvcr.io/YOUR_ORG/auralith-pipeline \
  --set image.tag=latest \
  --set pipeline.config=production

# 4. Submit Argo workflow (parallel datasets)
argo submit docker/kubernetes/argo-workflow.yaml

# 5. Monitor with Ray dashboard
ray dashboard  # http://localhost:8265
```

## Available Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| wikipedia | 20GB | English Wikipedia (verified) |
| c4 | 750GB | Cleaned Common Crawl |
| redpajama | 1.2TB | LLaMA training data |
| openwebtext | 40GB | Reddit links |
| bookcorpus | 5GB | 11k books |
| wikitext | 500MB | Wikipedia subset |
| dolly | 15MB | Instruction following |
| the_stack | 3TB | Source code (deduplicated) |

## Tokenization

### Train BPE Tokenizer

```bash
# Text tokenizer
python scripts/train_tokenizer.py text \
  --corpus data/train.txt \
  --output tokenizers/bpe_32k \
  --vocab-size 32000

# Image tokenizer (VQ codebook)
python scripts/train_tokenizer.py image \
  --images data/images/ \
  --output tokenizers/image_vq \
  --codebook-size 1024

# Audio tokenizer
python scripts/train_tokenizer.py audio \
  --audio data/audio/ \
  --output tokenizers/audio_vq \
  --codebook-size 512
```

## Performance

| Operation | Speed | Notes |
|-----------|-------|-------|
| Text preprocessing | 10k samples/sec | Single core |
| MinHash deduplication | 5k samples/sec | With LSH index |
| FAISS dedup | 3k samples/sec | IVFFlat on CPU |
| Perplexity filter | 500 samples/sec | GPT-2, GPU |
| BPE encoding | <1 ms/sample | With caching |
| SafeTensors writing | 50 MB/s | Zstd compressed |
| Image tokenization | 50 ms/image | 224×224, 196 patches |
| Video tokenization | 200 ms/video | 32 frames, uniform |
| Ray distributed | Linear scaling | Up to 64 workers |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run E2E schema validation
pytest tests/test_e2e_schema.py -v

# Code quality
black src/ tests/ scripts/
ruff check src/ tests/ scripts/
mypy src/
```

## Project Structure

```
auralith_pipeline/
├── cli.py                        # CLI commands
├── pipeline.py                   # Main pipeline (tracking, compliance, security)
├── config/                       # Configuration management
│   └── pipeline_config.py        # All config dataclasses
├── sources/
│   ├── data_sources.py           # HuggingFace, local, JSONL sources
│   └── video.py                  # Video frame sampler
├── preprocessing/
│   ├── preprocessor.py           # Text normalization, MinHash, PII
│   ├── quality.py                # Perplexity + LLM judge
│   ├── deduplication.py          # FAISS embedding dedup
│   ├── synthetic.py              # Local data augmentation
│   └── compliance.py             # License detection + audit
├── tokenization/
│   ├── bpe_tokenizer.py          # Custom BPE (16 special tokens)
│   ├── multimodal_tokenizer.py   # Text+Image+Audio+Video fusion
│   ├── video_tokenizer.py        # Video VQ tokenizer
│   └── tokenizer.py              # TokenizedSample + pipeline wrapper
├── sharding/
│   └── shard_writer.py           # SafeTensors writer (4-tensor schema)
├── security/
│   ├── pii_scrubber.py           # Multi-jurisdiction PII detection
│   ├── data_sanitizer.py         # Credential / secret sanitization
│   ├── privacy_config.py         # Privacy policies + PII categories
│   └── audit.py                  # Privacy audit logger
├── storage/
│   └── backends.py               # HF Hub, S3, GCS, Azure
├── distributed/
│   └── ray_runner.py             # Ray distributed runner
├── spark/                        # Apache Spark transforms
└── utils/
    ├── helpers.py                # Formatting utilities
    └── tracking.py               # MLflow/W&B + lineage

docker/kubernetes/
├── argo-workflow.yaml            # Argo DAG
└── helm/                         # Helm chart
    ├── Chart.yaml
    ├── values.yaml
    └── templates/

tests/
├── test_pipeline.py              # Core pipeline tests
├── test_tokenization.py          # Tokenizer tests
├── test_e2e_schema.py            # E2E validation
└── test_security.py              # Security & PII tests
```

## Environment Variables

```bash
# HuggingFace Hub (required for upload)
export HF_TOKEN=hf_xxxxxxxxxxxxx

# AWS S3 (optional)
export AWS_ACCESS_KEY_ID=xxxxx
export AWS_SECRET_ACCESS_KEY=xxxxx

# MLflow (optional)
export MLFLOW_TRACKING_URI=http://mlflow.internal:5000

# W&B (optional)
export WANDB_API_KEY=xxxxx
```

## License

Apache License 2.0 — See [LICENSE](LICENSE)

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md)

---

Built by [AuralithAI](https://github.com/AuralithAI) for [RT-DLM](https://github.com/AuralithAI/RT-DLM)
