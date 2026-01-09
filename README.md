# Auralith Data Pipeline

A production-grade, scalable data collection and processing pipeline for training large language models and multimodal AI systems.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

Auralith Data Pipeline is a modular, enterprise-ready system for:

- **Collecting** data from 20+ open-source datasets (HuggingFace, Common Crawl, etc.)
- **Preprocessing** with deduplication, quality filtering, and PII removal
- **Extracting** content from PDFs, documents, images, audio, and video
- **Tokenizing** text and multimodal content
- **Sharding** into efficient SafeTensors format for distributed training
- **Storing** on HuggingFace Hub, S3, GCS, or local storage

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AuralithAI/Auralith-Data-Pipeline.git
cd Auralith-Data-Pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[all]"
```

### Deployment Options

**Local Development:**
```bash
pip install -e ".[all]"
```

**Docker (Recommended for Testing):**
```bash
docker-compose up -d
```

**GitHub Actions (Scheduled Pipeline):**
See [GitHub Actions Pipeline Guide](docs/GITHUB_ACTIONS_PIPELINE.md) for automated data processing.

### CLI Usage

**Basic Processing:**
```bash
# List available datasets
auralith-pipeline list-datasets

# Collect and process Wikipedia
auralith-pipeline collect \
  --dataset wikipedia \
  --output ./data/shards \
  --max-samples 100000 \
  --deduplicate \
  --quality-filter \
  --preset production

# Upload to S3
aws s3 sync ./data/shards s3://your-bucket/datasets/wikipedia/
```

**Large-Scale Processing with Spark:**
```bash
# Process large datasets with Spark
auralith-pipeline spark-submit \
  --input s3://bucket/raw-data \
  --output s3://bucket/processed \
  --dataset-name wikipedia \
  --master local[*] \
  --executor-memory 8g \
  --deduplicate \
  --quality-filter
```

**Scheduled Automation:**
The pipeline runs automatically via GitHub Actions:
- **Weekly**: Every Sunday at 2 AM UTC
- **Manual**: Trigger from Actions tab
- **Output**: SafeTensors shards uploaded to S3

See [GitHub Actions Pipeline](docs/GITHUB_ACTIONS_PIPELINE.md) for setup.
docker-compose up -d --scale worker=5

# See docker/README.md for details
```

### Python API

```python
from auralith_pipeline import Pipeline, PipelineConfig
from auralith_pipeline.sources import HuggingFaceSource

# Configure pipeline
config = PipelineConfig.from_preset("production")

# Create and run pipeline
pipeline = Pipeline(config)
pipeline.add_source(HuggingFaceSource("wikipedia", split="train"))

stats = pipeline.run(max_samples=1_000_000)
print(f"Processed {stats.total_samples:,} samples")
```

## Features

| Feature | Description |
|---------|-------------|
| Multi-source ingestion | HuggingFace, Common Crawl, web scraping, local files |
| MinHash deduplication | Remove near-duplicate content at scale |
| Quality filtering | Length, language, toxicity, special char ratio |
| PII removal | Automatic detection and redaction |
| Document extraction | PDF, DOCX, PPTX, XLSX, HTML, Markdown |
| Multimodal support | Images, audio, video with embedding generation |
| SafeTensors shards | Fast, memory-mapped, secure format |
| Cloud storage | HuggingFace Hub, S3, GCS, Azure Blob |
| Distributed processing | Multi-machine orchestration with Redis |
| CI/CD ready | GitHub Actions workflows included |

## Architecture & Documentation

### Core Documentation

- **[Architecture Documentation](docs/ARCHITECTURE.md)** - System design with visual diagrams, component descriptions, and data flow examples
- **[Distributed Processing Guide](docs/DISTRIBUTED_PROCESSING.md)** - Multi-machine processing setup, orchestration, monitoring, and cloud deployment

### What You'll Learn

**Architecture Documentation**:
- High-level system design with visual diagrams
- Detailed component descriptions and data formats
- Complete data flow examples
- Performance considerations and optimization tips
- Extension points for custom components
- Deployment and security guidelines

**Distributed Processing**:
- Multi-machine architecture and task distribution
- Coordinator and worker setup
- Redis/Etcd state management
- Cloud deployment (AWS, GCP, Kubernetes)
- Monitoring dashboards and CLI tools
- Fault tolerance and recovery mechanisms
- Performance optimization strategies
- Troubleshooting guide and best practices

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
  
sharding:
  format: safetensors
  max_size_mb: 1000
  compression: zstd
```

## Available Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| wikipedia | 20GB | English Wikipedia |
| the_pile | 800GB | Diverse text mixture |
| redpajama | 1.2TB | LLaMA training data |
| c4 | 750GB | Cleaned Common Crawl |
| the_stack | 3TB | Source code |
| arxiv | 50GB | Research papers |

## Environment Variables

```bash
# HuggingFace Hub (required for upload)
export HF_TOKEN=hf_xxxxxxxxxxxxx

# AWS S3 (optional)
export AWS_ACCESS_KEY_ID=xxxxx
export AWS_SECRET_ACCESS_KEY=xxxxx
```

## Development

```bash
# Setup dev environment
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
black src/ tests/
ruff check src/ tests/
```

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines on contributing to this project.

---

Built by [AuralithAI](https://github.com/AuralithAI)
