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
│  │  │ input_ids │ │attention_mask │ │modality_mask │ │  targets   │    │    │
│  │  │  int32    │ │    uint8      │ │    uint8     │ │   int32    │    │    │
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
| **Schema** | SafeTensors 4-tensor schema v2 (`targets`, uint8 masks) | ✅ |
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
| **Pipeline** | `process` command: raw data → production `.safetensors` shards | ✅ |

## Installation

```bash
git clone https://github.com/AuralithAI/Auralith-Data-Pipeline.git
cd Auralith-Data-Pipeline

python -m venv venv
source venv/bin/activate  # Linux/Mac
# or .\venv\Scripts\activate  # Windows

# Recommended — install everything (includes multimodal, cloud, dev tools)
pip install -e ".[all]"            # ~3 GB with PyTorch

# Or install only what you need
pip install -e .                   # Core only (text pipeline)
pip install -e ".[quality]"        # + Perplexity filter + FAISS dedup
pip install -e ".[tracking]"       # + MLflow + W&B
pip install -e ".[distributed]"    # + Ray
pip install -e ".[multimodal]"     # + Video + image + audio (PyTorch)
pip install -e ".[cloud,pdf]"      # + Cloud storage + PDF extraction
```

> **Tip:** If you plan to use the full CLI (tokenizer training, multimodal
> processing, distributed jobs), install with `[all]` to avoid missing
> dependency errors at runtime.

## Quick Start

### CLI Usage

When you run any `auralith-pipeline` command, a vibrant startup banner is
displayed with version and environment info.  Suppress it with `--no-banner`
or set `AURALITH_NO_BANNER=1`.

```bash
# List available datasets
auralith-pipeline list-datasets

# Process Wikipedia dataset (production preset)
auralith-pipeline collect \
  --dataset wikipedia \
  --output ./data/shards \
  --max-samples 100000 \
  --preset production
```

### End-to-End Workflow: Raw Data → Trained Tokenizers → Production Shards

The full pipeline has three stages:

1. **Prepare raw data** — gather text, images, audio, and video into a folder
2. **Train tokenizers** — learn BPE vocabulary + VQ codebooks from your data
3. **Process** — tokenize everything and produce `.safetensors` shards for RT-DLM

```
  data/raw/              tokenizers/                    shards/
  ├── docs/*.txt    ──►  ├── text/   (BPE)         ──►  ├── shard_000000.safetensors
  ├── imgs/*.npy    ──►  ├── image/  (VQ codebook) ──►  ├── shard_000001.safetensors
  ├── audio/*.npy   ──►  ├── audio/  (VQ codebook) ──►  └── ...
  └── videos/*.mp4  ──►  └── video/  (VQ codebook) ──►
```

#### Step 1 — Prepare Raw Data

Organise your data into a single directory. The pipeline auto-detects file types:

| Modality | Accepted formats |
|----------|-----------------|
| Text | `.txt`, `.md`, `.rst`, `.csv`, `.json`, `.jsonl` |
| Image | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.npy` |
| Audio | `.wav`, `.flac`, `.ogg`, `.npy` |
| Video | `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm` |

```bash
data/raw/
├── corpus/
│   ├── wikipedia.txt
│   └── books.txt
├── images/
│   ├── img_001.npy
│   └── img_002.npy
├── audio/
│   ├── speech_001.npy
│   └── speech_002.npy
└── videos/
    └── lecture_001.mp4
```

#### Step 2 — Train Tokenizers

Train all modality tokenizers in one command:

```bash
auralith-pipeline train-tokenizer all \
  --corpus  data/raw/corpus/ \
  --images  data/raw/images/ \
  --audio   data/raw/audio/ \
  --videos  data/raw/videos/ \
  --output  tokenizers/ \
  --vocab-size 32000 \
  --codebook-size 1024 \
  --audio-codebook-size 512
```

This creates:

```
tokenizers/
├── text/          # BPE tokenizer (vocab.json, merges.txt, config.json)
├── image/         # Image VQ tokenizer (config.json, vq_codebook.json)
├── audio/         # Audio VQ tokenizer (config.json, vq_codebook.json)
└── video/         # Video VQ tokenizer (config.json, vq_codebook.json)
```

Or train each modality separately for finer control:

```bash
# Text BPE tokenizer
auralith-pipeline train-tokenizer text \
  --corpus data/raw/corpus/ \
  --output tokenizers/text \
  --vocab-size 32000

# Image VQ tokenizer
auralith-pipeline train-tokenizer image \
  --images data/raw/images/ \
  --output tokenizers/image \
  --codebook-size 1024 \
  --image-size 224 \
  --patch-size 16

# Audio VQ tokenizer
auralith-pipeline train-tokenizer audio \
  --audio data/raw/audio/ \
  --output tokenizers/audio \
  --codebook-size 512 \
  --sample-rate 16000

# Video VQ tokenizer
auralith-pipeline train-tokenizer video \
  --videos data/raw/videos/ \
  --output tokenizers/video \
  --codebook-size 1024 \
  --max-frames 32
```

> **Tip:** Store trained tokenizers in version control or cold storage (S3/GCS).
> They are small (~2 MB each) and must stay frozen for the lifetime of a model.

#### Step 3 — Process Raw Data into Shards

```bash
auralith-pipeline process \
  --input  data/raw/ \
  --output shards/ \
  --tokenizers tokenizers/ \
  --max-seq-len 4096 \
  --shard-size 10000
```

Each shard is a `.safetensors` file with the [schema v2](#safetensors-schema-v2) tensors
(`input_ids`, `attention_mask`, `modality_mask`, `targets`), ready for RT-DLM training.

#### Step 4 — Feed into RT-DLM

```bash
# Upload shards to cloud storage
auralith-pipeline upload --source shards/ --dest s3://my-bucket/training-data/

# Or upload to HuggingFace Hub
auralith-pipeline upload --source shards/ --dest hf://AuralithAI/training-shards

# Train RT-DLM
python src/train.py --data-dir shards/
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
attention_mask = shard["attention_mask"]   # (batch, seq_len) — uint8 (1=real, 0=pad)
modality_mask  = shard["modality_mask"]    # (batch, seq_len) — uint8 (0=text,1=img,2=aud,3=vid,4=code)
targets        = shard["targets"]         # (batch, seq_len) — int32, right-shifted input_ids

# targets[:, t] == input_ids[:, t+1]  (causal LM next-token prediction)
# JAX uses attention_mask to zero out padding in the loss — no -100 ignore index.

# Feed directly to RT-DLM training
# python src/train.py --data-dir ./data/shards
```

## SafeTensors Schema (v2)

Every shard is RT-DLM compatible. All sequences are padded/truncated to a fixed `seq_len` (default 2048) for JAX compatibility.

| Tensor | Dtype | Shape | Description |
|--------|-------|-------|-------------|
| `input_ids` | int32 | (batch, seq_len) | All tokens (text + image + audio + video + code) |
| `attention_mask` | uint8 | (batch, seq_len) | 1 = real token, 0 = padding |
| `modality_mask` | uint8 | (batch, seq_len) | 0=text, 1=image, 2=audio, 3=video, 4=code |
| `targets` | int32 | (batch, seq_len) | Right-shifted `input_ids` for causal LM (next-token prediction) |

> **Schema v2 changes** (from v1): `labels` → `targets` (right-shifted, no −100 ignore index),
> `attention_mask` dtype int32 → uint8 (4× memory savings), fixed-length padding,
> SHA-256 checksums (was MD5).

### Token ID Layout

| Range | Purpose |
|-------|---------|
| 0–15 | Special tokens (see below) |
| 16–271 | Byte tokens (`<byte_00>` – `<byte_ff>`) — lossless UTF-8 fallback |
| 272+ | BPE merge tokens (learned vocabulary) |

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
- Weighted round-robin interleaving across multiple sources
- MinHash + FAISS embedding deduplication
- Quality filtering (length, language, perplexity, LLM-as-Judge)
- PII removal (automatic detection and redaction)
- License compliance scanning for code data
- Document extraction (PDF, DOCX, HTML, Markdown)
- SafeTensors sharding with Zstd compression and SHA-256 checksums
- Streaming checkpointing with seeded reproducibility (numpy + stdlib RNG)
- Deterministic resumption from checkpoint (skip-ahead + RNG state restore)

### Tokenization
- Custom BPE tokenizer (16 special tokens, byte-level fallback, no external dependency)
- 256 byte tokens (IDs 16–271) for lossless UTF-8 encoding of any input
- LRU-bounded merge cache (100k entries) for fast encoding
- Vector quantization for images, audio, and video
- Multimodal token fusion with `encode_with_mask()`
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
  seed: 42                       # Reproducibility (numpy + stdlib RNG)
  checkpoint_every: 10000        # Save resume checkpoint every N accepted samples

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

### Training Tokenizers (Detailed Guide)

Before you can process raw files into shards, you need trained tokenizers for each
modality your model will consume. These tokenizers are **frozen artifacts** — once
trained, they must not change for the entire model's lifecycle.

#### Why train your own tokenizers?

- **Text (BPE):** Learns subword units tuned to your domain vocabulary (e.g. medical, legal, code).
- **Image (VQ):** Learns a discrete codebook that maps image patches → token IDs.
- **Audio (VQ):** Learns a codebook over mel-spectrogram patches.
- **Video (VQ):** Same as image, but trained on video frames for temporal consistency.

#### Recommended Training Data Sizes

| Modality | Minimum | Recommended | Notes |
|----------|---------|-------------|-------|
| Text | 1 MB | 1–10 GB | More data = better subword coverage |
| Image | 100 images | 10k+ images | `.npy` arrays (H, W, 3) or JPEG/PNG |
| Audio | 100 files | 10k+ files | `.npy` waveforms or `.wav/.flac` |
| Video | 50 videos | 1k+ videos | `.mp4/.avi/.mov` — frames extracted automatically |

#### Training Commands

```bash
# All at once (recommended)
auralith-pipeline train-tokenizer all \
  --corpus  data/corpus/ \
  --images  data/images/ \
  --audio   data/audio/ \
  --videos  data/videos/ \
  --output  tokenizers/ \
  --vocab-size 32000 \
  --codebook-size 1024 \
  --audio-codebook-size 512 \
  --image-size 224 \
  --patch-size 16 \
  --sample-rate 16000 \
  --max-frames 32

# Or individually
auralith-pipeline train-tokenizer text  --corpus data/corpus.txt --output tokenizers/text --vocab-size 32000
auralith-pipeline train-tokenizer image --images data/images/    --output tokenizers/image --codebook-size 1024
auralith-pipeline train-tokenizer audio --audio  data/audio/     --output tokenizers/audio --codebook-size 512
auralith-pipeline train-tokenizer video --videos data/videos/    --output tokenizers/video --codebook-size 1024
```

#### Output Structure

```
tokenizers/
├── text/
│   ├── vocab.json       # Token → ID mapping
│   ├── merges.txt       # BPE merge rules (ordered)
│   └── config.json      # Tokenizer hyperparameters
├── image/
│   ├── config.json      # image_size, patch_size, codebook_size
│   └── vq_codebook.json # Learned VQ centroids
├── audio/
│   ├── config.json      # sample_rate, n_fft, codebook_size
│   └── vq_codebook.json
└── video/
    ├── config.json      # image_size, patch_size, max_frames
    └── vq_codebook.json
```

> **Cold Storage:** Archive `tokenizers/` to S3/GCS alongside your model checkpoints.
> The `process` command reads these frozen tokenizers at inference time.

### Processing Raw Data into Shards

Once tokenizers are trained, use `process` to convert raw files into production shards:

```bash
auralith-pipeline process \
  --input  data/raw/ \
  --output shards/ \
  --tokenizers tokenizers/ \
  --max-seq-len 4096 \
  --shard-size 10000
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | — | Folder with raw `.txt/.jpg/.wav/.mp4` etc. |
| `--output` | — | Where `.safetensors` shards are written |
| `--tokenizers` | — | Root folder containing `text/`, `image/`, `audio/`, `video/` subdirs |
| `--max-seq-len` | 4096 | Maximum token sequence length per sample |
| `--shard-size` | 10000 | Maximum samples per shard file |

Each shard contains 4 tensors matching the [SafeTensors Schema v2](#safetensors-schema-v2):
`input_ids`, `attention_mask`, `modality_mask`, and `targets`.

### Token ID Layout

| Range | Purpose |
|-------|---------|
| 0–15 | Special tokens (see below) |
| 16–271 | Byte tokens (`<byte_00>` – `<byte_ff>`) — lossless UTF-8 fallback |
| 272+ | BPE merge tokens (learned vocabulary) |
| 100,000+ | Image VQ codes (offset to avoid collisions) |
| 200,000+ | Audio VQ codes |
| 300,000+ | Video VQ codes |

## Performance

| Operation | Speed | Notes |
|-----------|-------|-------|
| Text preprocessing | 10k samples/sec | Single core |
| MinHash deduplication | 5k samples/sec | With LSH index |
| FAISS dedup | 3k samples/sec | IVFFlat on CPU |
| Perplexity filter | 500 samples/sec | GPT-2, GPU |
| BPE encoding | <1 ms/sample | LRU-cached (100k entries) |
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
├── cli.py                        # CLI commands (collect, process, train-tokenizer, etc.)
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
│   └── shard_writer.py           # SafeTensors writer (4-tensor schema v2)
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
├── test_cli.py                   # CLI command tests (process, train-tokenizer, etc.)
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
