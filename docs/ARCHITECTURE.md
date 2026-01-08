# Architecture Documentation

## Overview

The Auralith Data Pipeline is designed as a modular, scalable system for collecting, processing, and storing large-scale datasets for training large language models and multimodal AI systems. The architecture follows a stage-based pipeline pattern with pluggable components.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                 │
│  (HuggingFace, Common Crawl, Local Files, Web Scraping, etc.)      │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING STAGE                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │  Normalize   │→ │ Deduplicate  │→ │   Quality    │             │
│  │    Text      │  │  (MinHash)   │  │   Filter    │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│                                              │                      │
│  ┌──────────────┐                           │                      │
│  │ PII Removal  │◀─────────────────────────┘                      │
│  └──────────────┘                                                  │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     EXTRACTION STAGE                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │     PDF      │  │    Images    │  │     Audio    │             │
│  │ Extraction   │  │ Embedding    │  │ Transcription│             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐                               │
│  │   DOCX       │  │   Video      │                               │
│  │ Extraction   │  │ Processing   │                               │
│  └──────────────┘  └──────────────┘                               │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    TOKENIZATION STAGE                               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Convert text to token IDs using tokenizer                   │  │
│  │  Support for multiple tokenizers (BPE, WordPiece, etc.)      │  │
│  │  Create attention masks and token metadata                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SHARDING STAGE                                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Organize tokenized samples into efficient SafeTensors       │  │
│  │  Configurable shard size (default: 1GB)                      │  │
│  │  Metadata tracking (statistics, checksums, etc.)             │  │
│  │  Optional compression (zstd, gzip)                           │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STORAGE STAGE                                   │
│  (HuggingFace Hub, S3, GCS, Azure Blob, Local Filesystem)          │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Data Sources (`sources/`)

**Purpose**: Abstracts data ingestion from various sources

**Key Classes**:
- `DataSource` (ABC): Base class for all data sources
- `HuggingFaceSource`: Loads datasets from Hugging Face Hub
- `LocalFileSource`: Reads from local text files with pattern matching
- `JSONLSource`: Handles JSONL formatted files

**Data Format**:
```python
@dataclass
class DataSample:
    content: str                          # Text content
    source: str                           # Source identifier
    metadata: Dict[str, Any]              # Custom metadata
    modality: Literal[...] = "text"       # Content type
```

### 2. Preprocessing (`preprocessing/`)

**Purpose**: Clean, filter, and deduplicate data

**Key Components**:

#### TextNormalizer
- Lowercasing and whitespace normalization
- Unicode normalization (NFKC)
- Removes control characters

#### QualityFilter
- **Length filtering**: Min/max character counts
- **Language detection**: Identifies and filters non-target languages
- **Toxicity scoring**: Detects harmful content
- **Special character ratio**: Prevents token-heavy content

#### MinHashDeduplicator
- Uses LSH (Locality Sensitive Hashing) for near-duplicate detection
- Configurable threshold (default: 0.9 similarity)
- Efficient for large-scale processing

#### PIIRemover
- Pattern-based detection for:
  - Email addresses
  - Phone numbers
  - Credit card numbers
  - Social Security Numbers
  - IP addresses
- Redacts sensitive information

### 3. Extraction (`extraction/`)

**Purpose**: Extract content from multiple document formats

**Supported Formats**:
- **PDF**: Text extraction, table detection, metadata
- **DOCX**: Text, tables, hyperlinks
- **PPTX**: Slide text extraction
- **XLSX**: Tabular data extraction
- **Images**: OCR and embedding generation
- **Audio**: Transcription via speech-to-text
- **Video**: Frame extraction and analysis

**Example Usage**:
```python
from auralith_pipeline.extraction import Extractor

extractor = Extractor()
result = extractor.extract("document.pdf")
# Returns: {text, tables, images, metadata}
```

### 4. Tokenization (`tokenization/`)

**Purpose**: Convert text to token sequences for model training

**Key Classes**:

#### Tokenizer
- Wraps HuggingFace tokenizers (or custom implementations)
- Handles encoding/decoding
- Supports multiple vocabularies

#### TokenizationPipeline
- **Chunking**: Splits long texts into training sequences
- **Padding**: Handles variable-length inputs
- **Attention masks**: Generates masks for padding
- **Metadata preservation**: Maintains sample provenance

**Output Format**:
```python
@dataclass
class TokenizedSample:
    input_ids: list[int]          # Token IDs
    attention_mask: list[int]      # Attention mask
    metadata: dict[str, Any]       # Original sample metadata
```

### 5. Sharding (`sharding/`)

**Purpose**: Organize tokenized data into efficient storage format

**Key Components**:

#### ShardWriter
- **Buffering**: Accumulates samples before writing
- **Size management**: Flushes shards at configured size limit
- **Format**: SafeTensors (secure, fast, language-agnostic)
- **Compression**: Optional zstd/gzip compression

#### ShardMetadata
Tracks per-shard information:
```python
@dataclass
class ShardMetadata:
    shard_id: int
    num_samples: int
    total_bytes: int
    sequence_length: int
    compression: Optional[str]
    created_at: str
    checksum: str
```

#### ShardIndex
Global index for all shards:
```python
@dataclass
class ShardIndex:
    total_samples: int
    total_size_bytes: int
    sequence_length: int
    shards: list[ShardMetadata]
```

### 6. Storage (`storage/`)

**Purpose**: Upload/download shards to cloud or local storage

**Backends**:

| Backend | Features |
|---------|----------|
| **HuggingFace Hub** | Version control, public/private repos |
| **S3** | Scalable cloud storage, lifecycle policies |
| **GCS** | Google Cloud integration, object versioning |
| **Azure Blob** | Enterprise storage, compliance options |
| **Filesystem** | Local storage, network mounts |

### 7. Configuration (`config/`)

**Purpose**: Centralized configuration management

**Configuration Structure**:
```yaml
pipeline:
  output_dir: ./data/shards

sources:
  - type: huggingface
    path: dataset_name
    max_samples: 1000000

preprocessing:
  normalize: true
  deduplicate: true
  quality_filter: true
  remove_pii: true

tokenization:
  tokenizer_path: "gpt2"
  max_length: 2048
  
sharding:
  format: safetensors
  max_size_mb: 1000
  compression: zstd

storage:
  backend: huggingface
  repo_id: "org/dataset"
```

## Data Flow Example

### Complete Pipeline Run

```python
from auralith_pipeline import Pipeline, PipelineConfig

# 1. Load configuration
config = PipelineConfig.from_preset("production")

# 2. Create pipeline
pipeline = Pipeline(config)

# 3. Add data sources
pipeline.add_source(HuggingFaceSource("wikipedia", split="train"))
pipeline.add_source(LocalFileSource("./data/custom", pattern="*.txt"))

# 4. Run pipeline
stats = pipeline.run(max_samples=1_000_000)

# 5. Results
print(f"Processed: {stats.total_samples:,} samples")
print(f"Output: {stats.output_dir}")
print(f"Shards: {stats.shard_count}")
```

### Step-by-Step Flow

1. **Collection**
   - Data sources iterate and yield `DataSample` objects
   - Content, metadata, and modality information included

2. **Preprocessing**
   - Text normalization applied
   - Quality filtering removes low-quality samples
   - Deduplication identifies and removes duplicates
   - PII redaction masks sensitive information

3. **Extraction**
   - Multi-modal content extracted from documents
   - Images processed to embeddings
   - Audio transcribed to text
   - Metadata enriched

4. **Tokenization**
   - Text converted to token sequences
   - Long sequences chunked with overlap
   - Attention masks created
   - Training samples prepared

5. **Sharding**
   - TokenizedSamples buffered in memory
   - Shards written when size threshold reached
   - Metadata and index created
   - Checksums computed

6. **Storage**
   - Shards uploaded to configured backend
   - Index file uploaded
   - Verification against checksums
   - Dataset ready for training

## Streaming vs Batch Modes

### Batch Mode
- Processes entire dataset at once
- Higher memory usage, better for exploration
- Returns `PipelineStats` with summary

### Streaming Mode
- Yields results incrementally
- Memory-efficient, suitable for large datasets
- Allows real-time processing

```python
# Streaming mode
for batch in pipeline.stream(batch_size=100):
    # batch: List[TokenizedSample]
    # Process in real-time
    train_model(batch)
```

## Performance Considerations

### Memory Management
- **Default shard size**: 1GB (configurable)
- **Buffering**: Samples accumulated before write
- **Streaming**: Iterative processing avoids full load

### Deduplication Efficiency
- **LSH**: O(1) lookup via hash tables
- **Shingle size**: k=5 (configurable)
- **Threshold**: 0.9 similarity (configurable)

### Tokenization Speed
- **Batch processing**: 10,000+ samples/second
- **Vectorized operations**: NumPy/PyTorch
- **Parallel tokenizers**: Multi-threaded by default

### Storage Optimization
- **SafeTensors**: Minimal overhead, zero-copy reads
- **Compression**: Optional zstd (20-30% reduction)
- **Chunking**: Efficient for distributed training

## Extension Points

### Adding Custom Sources
```python
from auralith_pipeline.sources import DataSource

class CustomSource(DataSource):
    def __init__(self, config):
        self.config = config
    
    def __iter__(self):
        # Yield DataSample objects
        pass
```

### Custom Preprocessing
```python
from auralith_pipeline.preprocessing import DataPreprocessor

class CustomPreprocessor(DataPreprocessor):
    def process(self, sample):
        # Custom logic
        return sample
```

### Custom Storage Backend
```python
from auralith_pipeline.storage import StorageBackend

class CustomStorage(StorageBackend):
    def upload(self, local_path, remote_path):
        # Upload logic
        pass
```

## Monitoring & Debugging

### Logging
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- Per-component loggers
- Performance metrics tracked

### Statistics Tracking
```python
class PipelineStats:
    total_samples: int
    deduplicated_count: int
    filtered_count: int
    total_tokens: int
    output_dir: str
    shard_count: int
    total_size_bytes: int
    processing_time_seconds: float
```

### Error Handling
- Graceful degradation on source failures
- Retries with exponential backoff
- Detailed error logs and recovery options

## Deployment

### Local Development
```bash
pip install -e ".[dev]"
python -m auralith_pipeline.cli collect --dataset wikipedia
```

### Docker Deployment
```bash
docker build -t auralith-pipeline .
docker run -v /data:/data auralith-pipeline collect --dataset wikipedia
```

### Cloud Deployment
- Kubernetes support via Dockerfile
- HuggingFace Hub integration for model sharing
- S3/GCS credentials via environment variables
- Distributed processing across multiple workers

## Security Considerations

- **PII Protection**: Automatic detection and redaction
- **Data Validation**: Schema validation for all inputs
- **Safe Tensor Format**: Cryptographic checksums
- **Access Control**: Token-based authentication
- **Audit Logging**: Complete processing history

## Future Enhancements

- [ ] Distributed processing across multiple machines (see [Distributed Processing Guide](DISTRIBUTED_PROCESSING.md))
- [ ] Apache Spark integration for large-scale jobs
- [ ] Real-time streaming with Kafka/Pub-Sub
- [ ] Advanced multimodal embeddings
- [ ] Active learning feedback loop
- [ ] A/B testing framework for preprocessing
