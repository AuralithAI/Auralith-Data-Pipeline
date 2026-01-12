# Auralith Data Pipeline

Production-grade data processing pipeline for training large language models and multimodal AI systems.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

Auralith Data Pipeline is a modular system for:

- Collecting data from 20+ open-source datasets (HuggingFace, Common Crawl)
- Preprocessing with deduplication, quality filtering, and PII removal
- Tokenizing text and multimodal content (custom BPE, image/audio VQ)
- Sharding into SafeTensors format for distributed training
- Storing on HuggingFace Hub, S3, GCS, or local storage

## Installation

```bash
git clone https://github.com/AuralithAI/Auralith-Data-Pipeline.git
cd Auralith-Data-Pipeline

python -m venv venv
source venv/bin/activate  # Linux/Mac
# or .\venv\Scripts\activate  # Windows

pip install -e ".[cloud,pdf]"  # Core + cloud + PDF support
# or pip install -e ".[all]"  # All dependencies (~2-3 GB with PyTorch)
```

## Quick Start

### CLI Usage

```bash
# List available datasets
auralith-pipeline list-datasets

# Process Wikipedia dataset
auralith-pipeline collect \
  --dataset wikipedia \
  --output ./data/shards \
  --max-samples 100000 \
  --deduplicate \
  --quality-filter \
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
print(f"Processed {stats.total_samples:,} samples")
```

## Features

### Data Processing
- Multi-source ingestion (HuggingFace, Common Crawl, local files)
- MinHash deduplication (near-duplicate removal at scale)
- Quality filtering (length, language, toxicity, special char ratio)
- PII removal (automatic detection and redaction)
- Document extraction (PDF, DOCX, HTML, Markdown)
- SafeTensors sharding (fast, memory-mapped format)

### Tokenization
- Custom BPE tokenizer (no Transformers dependency)
- Vector quantization for images and audio
- Multimodal token fusion (text + images + audio)
- Character-level fallback for OOV handling
- Configurable vocab size (32k-128k)

### Storage & Deployment
- Cloud storage (HuggingFace Hub, S3, GCS, Azure Blob)
- Distributed processing (multi-machine with Redis orchestration)
- GitHub Actions automation (scheduled weekly runs)
- Docker support for containerized deployment

## Configuration

```yaml
# configs/production.yaml
pipeline:
  output_dir: ./data/shards
  
sources:
  - type: huggingface
    path: wikipedia
    max_samples: 10000000
    
preprocessing:
  deduplicate: true
  quality_filter: true
  remove_pii: true
  
tokenization:
  tokenizer_path: tokenizers/bpe_32k
  vocab_size: 32000
  max_length: 2048
  
sharding:
  format: safetensors
  max_size_mb: 1000
  compression: zstd
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

**Deprecated**: `the_pile`, `arxiv` (use alternatives listed above)

## Tokenization

### Train BPE Tokenizer

```bash
# Text tokenizer
python scripts/train_tokenizer.py text \
  --corpus data/train.txt \
  --output tokenizers/bpe_32k \
  --vocab-size 32000 \
  --min-frequency 2

# Image tokenizer (requires .npy format)
python scripts/train_tokenizer.py image \
  --images data/images/ \
  --output tokenizers/image_vq \
  --codebook-size 1024 \
  --image-size 224 \
  --patch-size 16

# Audio tokenizer (requires .npy format)
python scripts/train_tokenizer.py audio \
  --audio data/audio/ \
  --output tokenizers/audio_vq \
  --codebook-size 512 \
  --sample-rate 16000
```

### Python API

```python
from auralith_pipeline.tokenization import BPETokenizer

# Train tokenizer
tokenizer = BPETokenizer(vocab_size=32000, min_frequency=2)
tokenizer.train(corpus, verbose=True)
tokenizer.save("tokenizers/bpe_32k")

# Load and use
tokenizer = BPETokenizer.load("tokenizers/bpe_32k")
token_ids = tokenizer.encode("Hello world", add_special_tokens=True)
text = tokenizer.decode(token_ids)
```

## Environment Variables

```bash
# HuggingFace Hub (required for upload)
export HF_TOKEN=hf_xxxxxxxxxxxxx

# AWS S3 (optional)
export AWS_ACCESS_KEY_ID=xxxxx
export AWS_SECRET_ACCESS_KEY=xxxxx
export AWS_DEFAULT_REGION=us-east-1
```

## GitHub Actions Pipeline

Automated data processing runs weekly (Sunday 2 AM UTC):

1. Collect data from configured datasets
2. Apply preprocessing and quality filters
3. Generate SafeTensors shards
4. Upload to S3 storage

**Setup**: Add secrets to repository settings:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_S3_BUCKET`

**Manual trigger**: Go to Actions tab and click "Run workflow"

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run specific test file
pytest tests/test_tokenization.py -v

# Code quality checks
black src/ tests/ scripts/
ruff check src/ tests/ scripts/
mypy src/

# Type checking
mypy src/auralith_pipeline/
```

## Project Structure

```
auralith_pipeline/
├── cli.py                    # CLI commands
├── pipeline.py               # Main pipeline orchestration
├── config/                   # Configuration management
├── sources/                  # Data source adapters
├── extraction/               # Content extraction (PDF, etc.)
├── preprocessing/            # Text cleaning, deduplication
├── tokenization/            # BPE tokenizer, multimodal VQ
├── sharding/                # SafeTensors shard writer
├── storage/                 # Cloud storage backends
└── utils/                   # Helper utilities

tests/                       # Unit tests
scripts/                     # Training scripts
configs/                     # YAML configurations
```

## Performance

| Operation | Speed | Notes |
|-----------|-------|-------|
| Text preprocessing | 10k samples/sec | Single core |
| MinHash deduplication | 5k samples/sec | With LSH index |
| BPE encoding | <1 ms/sample | With caching |
| SafeTensors writing | 50 MB/s | Compressed |
| Image tokenization | 50 ms/image | 224x224, 196 patches |

## License

Apache License 2.0 - See [LICENSE](LICENSE)

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md)

---

Built by [AuralithAI](https://github.com/AuralithAI)
