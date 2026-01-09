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

## Distributed Processing

### 8. Distributed Module (`distributed/`)

**Purpose**: Enable multi-machine processing for large-scale operations

**Key Components**:

#### Coordinator
- **Job Management**: Task scheduling and distribution
- **Worker Monitoring**: Heartbeat tracking and health checks
- **State Management**: Redis-based state store
- **Fault Tolerance**: Automatic task reassignment on worker failure

#### Worker
- **Task Execution**: Processes assigned pipeline tasks
- **Heartbeat**: Regular status updates to coordinator
- **Resource Reporting**: CPU, memory, and processing metrics
- **Local Caching**: Efficient temporary storage

#### State Store
- **Redis Implementation**: Primary state storage backend
- **Task Queues**: Distributed work queue management
- **Heartbeat Tracking**: Worker liveness detection
- **Metadata Storage**: Job and task state persistence

#### Distribution Strategies
- **Round Robin**: Evenly distributes tasks across workers
- **Least Busy**: Assigns to worker with smallest queue
- **Dynamic**: Considers CPU, memory, and queue depth

**Distributed Configuration**:
```yaml
coordinator:
  host: coordinator.internal
  port: 8080
  state_store_type: redis
  heartbeat_interval: 10

workers:
  - name: collection
    worker_ids: [worker-1, worker-2]
    batch_size: 1000
  - name: processing
    worker_ids: [worker-3, worker-4]
    batch_size: 500
```

**CLI Commands**:
```bash
# Start coordinator
auralith-pipeline coordinator --config distributed.yaml

# Start worker
auralith-pipeline worker \
  --coordinator coordinator:8080 \
  --worker-id worker-1

# Submit job
auralith-pipeline submit-job \
  --coordinator coordinator:8080 \
  --job-name my-job \
  --dataset wikipedia

# Monitor status
auralith-pipeline status --coordinator coordinator:8080
```

For detailed distributed processing setup and deployment, see [Distributed Processing Guide](DISTRIBUTED_PROCESSING.md).

## 9. Apache Spark Integration

For processing datasets at massive scale (petabytes), the pipeline integrates with Apache Spark:

**Components**:
- `SparkPipelineRunner`: Main entry point for Spark jobs
- `SparkConfig`: Cluster configuration (executors, memory, cores)
- `SparkJobConfig`: Job-specific settings (input/output, transformations)
- `spark.transforms`: Distributed transformations (preprocess, deduplicate, tokenize)

**Architecture**:
```
┌──────────────────────────────────────────────────────────────┐
│                    SPARK CLUSTER                             │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐       │
│  │   Driver   │────▶│  Executor  │     │  Executor  │       │
│  │            │     │            │     │            │       │
│  │  Pipeline  │     │  Worker    │     │  Worker    │       │
│  │  Runner    │     │  Tasks     │     │  Tasks     │       │
│  └────────────┘     └────────────┘     └────────────┘       │
│         │                  │                  │              │
│         ▼                  ▼                  ▼              │
│  ┌──────────────────────────────────────────────────┐       │
│  │        Distributed DataFrame Processing          │       │
│  │                                                   │       │
│  │  • Read from S3/HDFS/Parquet                    │       │
│  │  • Preprocess (normalize, quality filter, PII)  │       │
│  │  • Deduplicate (MinHashLSH)                     │       │
│  │  • Tokenize (broadcast tokenizer)               │       │
│  │  • Write shards to output                       │       │
│  └──────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────┘
```

**Configuration Example**:
```python
from auralith_pipeline.spark import (
    SparkConfig, 
    SparkJobConfig, 
    SparkPipelineRunner
)

# Cluster configuration
spark_config = SparkConfig(
    app_name="wikipedia-processing",
    master="spark://spark-master:7077",  # or yarn, mesos, k8s
    executor_memory="8g",
    driver_memory="4g",
    num_executors=10,
    executor_cores=4,
    dynamic_allocation=True
)

# Job configuration
job_config = SparkJobConfig(
    input_path="s3://bucket/raw/wikipedia/*.parquet",
    output_path="s3://bucket/processed/wikipedia",
    dataset_name="wikipedia",
    tokenizer_name="gpt2",
    max_length=2048,
    deduplicate=True,
    quality_filter=True,
    remove_pii=True,
    num_partitions=1000,
    compression="snappy"
)

# Run job
runner = SparkPipelineRunner(spark_config)
stats = runner.run(job_config)
runner.stop()
```

**CLI Usage**:
```bash
# Submit to Spark cluster
auralith-pipeline spark-submit \
  --input s3://bucket/raw-data \
  --output s3://bucket/processed \
  --dataset-name wikipedia \
  --master spark://spark-master:7077 \
  --executor-memory 8g \
  --driver-memory 4g \
  --num-executors 10 \
  --executor-cores 4 \
  --tokenizer gpt2 \
  --max-length 2048 \
  --deduplicate \
  --quality-filter \
  --remove-pii \
  --num-partitions 1000
```

**Features**:
- Scales to petabyte-scale datasets
- Fault-tolerant with task retry
- Dynamic resource allocation
- Broadcast tokenizer for efficient processing
- MinHashLSH for distributed deduplication
- Configurable output partitioning

**Deployment**:
```bash
# Using Docker Compose
docker-compose up -d spark-master spark-worker

# Kubernetes
kubectl apply -f docker/kubernetes/spark-deployment.yaml

# Submit job
auralith-pipeline spark-submit --master spark://spark-master:7077 ...
```

## 10. Docker Deployment

Complete containerized deployment with Docker and Kubernetes support:

**Docker Compose Services**:
- Redis: State store for distributed coordination
- Coordinator: Job manager and orchestrator
- Workers: Parallel processing nodes (scalable)
- Spark Master: Spark cluster manager
- Spark Workers: Spark execution nodes (scalable)

**Quick Start**:
```bash
# Start everything
docker-compose up -d

# Scale workers
docker-compose up -d --scale worker=5
docker-compose up -d --scale spark-worker=4

# View logs
docker-compose logs -f coordinator

# Stop everything
docker-compose down
```

**Kubernetes Deployment**:
```bash
# Deploy to cluster
kubectl apply -f docker/kubernetes/deployment.yaml

# Scale workers
kubectl scale deployment worker -n auralith --replicas=10

# Auto-scaling enabled (3-10 replicas based on CPU/memory)
```

See [docker/README.md](../docker/README.md) for detailed deployment instructions.

## Future Enhancements

- [x] Distributed processing across multiple machines (implemented)
- [x] Apache Spark integration for large-scale jobs (implemented)
- [x] Docker and Kubernetes deployment (implemented)
- [ ] Real-time streaming with Kafka/Pub-Sub
- [ ] Advanced multimodal embeddings
- [ ] Active learning feedback loop
- [ ] A/B testing framework for preprocessing
