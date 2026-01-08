# Distributed Processing Guide

## Overview

The Auralith Data Pipeline supports distributed processing across multiple machines, enabling massive-scale data collection and processing operations. This guide covers architecture, setup, deployment, and best practices.

## Architecture

### Distributed Pipeline Components

```
┌────────────────────────────────────────────────────────────────────┐
│                     COORDINATOR NODE                               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Job Manager                                                 │  │
│  │  - Task scheduling                                           │  │
│  │  - Worker management                                         │  │
│  │  - Progress tracking                                         │  │
│  │  - Fault tolerance                                           │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  State Store (Redis/Etcd)                                    │  │
│  │  - Task queue                                                │  │
│  │  - Worker heartbeats                                         │  │
│  │  - Progress state                                            │  │
│  │  - Metadata                                                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
         │                     │                       │
         ▼                     ▼                       ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  WORKER NODE 1   │  │  WORKER NODE 2   │  │  WORKER NODE N   │
│                  │  │                  │  │                  │
│ ┌──────────────┐ │  │ ┌──────────────┐ │  │ ┌──────────────┐ │
│ │ Collection   │ │  │ │ Extraction   │ │  │ │ Tokenization │ │
│ │ + Preprocess │ │  │ │ + Preprocess │ │  │ │ + Sharding   │ │
│ └──────────────┘ │  │ └──────────────┘ │  │ └──────────────┘ │
│                  │  │                  │  │                  │
│ Local Cache      │  │ Local Cache      │  │ Local Cache      │
│ Temp Storage     │  │ Temp Storage     │  │ Temp Storage     │
└──────────────────┘  └──────────────────┘  └──────────────────┘
         │                     │                       │
         └─────────────────────┼───────────────────────┘
                               │
                               ▼
                    ┌────────────────────────┐
                    │  SHARED STORAGE        │
                    │  (S3, GCS, NFS, etc.)  │
                    │  - Input datasets      │
                    │  - Output shards       │
                    │  - Checkpoints         │
                    └────────────────────────┘
```

## Setup & Configuration

### Prerequisites

- Python 3.10+
- Redis or Etcd (for state management)
- Shared storage (S3, GCS, NFS, or similar)
- Network connectivity between all nodes
- SSH access for deployment (optional)

### Installation

```bash
# On all nodes
pip install -e ".[distributed]"

# Install Redis (optional, for local development)
# Or use managed Redis service (AWS ElastiCache, Redis Cloud, etc.)
```

### Configuration

Create a distributed configuration file:

```yaml
# configs/distributed.yaml
coordinator:
  # Coordinator node address
  host: coordinator.internal
  port: 8080
  
  # State store configuration
  state_store:
    type: redis  # or etcd
    host: redis.internal
    port: 6379
    db: 0
    
  # Heartbeat settings
  heartbeat_interval: 10  # seconds
  heartbeat_timeout: 30   # seconds
  
  # Task configuration
  task_queue:
    max_retries: 3
    retry_backoff: 5  # seconds
    task_timeout: 3600  # seconds (1 hour)

workers:
  # Worker pool settings
  worker_processes: 4
  batch_size: 100
  
  # Local caching
  cache_dir: /tmp/auralith_cache
  max_cache_size: 50GB
  
  # Temp storage
  temp_dir: /tmp/auralith_temp
  cleanup_on_exit: true

storage:
  # Shared storage backend
  backend: s3
  
  # S3 configuration
  s3:
    bucket: auralith-datasets
    region: us-west-2
    prefix: distributed/
    
  # Or GCS configuration
  # gcs:
  #   bucket: auralith-datasets
  #   project_id: my-project
  
  # Or NFS configuration
  # nfs:
  #   mount_point: /mnt/shared-storage
  #   host: nfs-server.internal

pipeline:
  # Pipeline configuration
  output_dir: ./data/shards
  
  sources:
    - type: huggingface
      path: wikipedia
      max_samples: 10000000
      # Distribute across workers
      distributed: true
      
  preprocessing:
    deduplicate: true
    quality_filter: true
    remove_pii: true
    
  tokenization:
    tokenizer_path: gpt2
    max_length: 2048
    
  sharding:
    format: safetensors
    max_size_mb: 1000
    compression: zstd

scaling:
  # Auto-scaling settings (optional)
  min_workers: 2
  max_workers: 10
  scale_up_threshold: 0.8  # CPU/Memory usage
  scale_down_threshold: 0.2
```

## Launching Distributed Pipelines

### 1. Start the Coordinator

```bash
# On coordinator node
auralith-pipeline coordinator --config configs/distributed.yaml
```

This starts:
- Job manager listening on port 8080
- Connection to state store (Redis/Etcd)
- Worker registry and heartbeat monitor

### 2. Start Worker Nodes

```bash
# On each worker node
auralith-pipeline worker \
  --coordinator coordinator.internal:8080 \
  --worker-id worker-1 \
  --config configs/distributed.yaml
```

Or use a deployment script:

```bash
#!/bin/bash
# deploy_workers.sh

COORDINATOR="coordinator.internal:8080"
WORKER_NODES=("worker1.internal" "worker2.internal" "worker3.internal")

for node in "${WORKER_NODES[@]}"; do
  ssh $node "cd /opt/auralith && \
    auralith-pipeline worker \
    --coordinator $COORDINATOR \
    --worker-id $(hostname) \
    --config configs/distributed.yaml &"
done
```

### 3. Submit Pipeline Job

```bash
# On any machine with network access to coordinator
auralith-pipeline submit-job \
  --coordinator coordinator.internal:8080 \
  --config configs/distributed.yaml \
  --job-name "wikipedia-processing" \
  --output-dir s3://auralith-datasets/wikipedia/shards
```

Or programmatically:

```python
from auralith_pipeline.distributed import DistributedPipeline, JobConfig

# Configure job
job_config = JobConfig(
    name="wikipedia-processing",
    coordinator_host="coordinator.internal",
    coordinator_port=8080,
    num_workers=5,
    timeout=86400,  # 24 hours
)

# Create distributed pipeline
pipeline = DistributedPipeline(job_config)

# Add sources
pipeline.add_source(HuggingFaceSource("wikipedia"))

# Run distributed job
stats = pipeline.run(
    output_path="s3://auralith-datasets/wikipedia/shards",
    monitor=True  # Enable web monitoring
)

print(f"Processed {stats.total_samples:,} samples")
print(f"Total time: {stats.processing_time_seconds}s")
print(f"Throughput: {stats.total_samples / stats.processing_time_seconds:.0f} samples/sec")
```

## Task Distribution Strategies

### 1. Source-Based Distribution

Each worker processes different data sources:

```python
from auralith_pipeline.distributed import DistributedPipeline

pipeline = DistributedPipeline(job_config)

# Worker 1: Wikipedia
pipeline.add_source(
    HuggingFaceSource("wikipedia"),
    target_worker="worker-1"
)

# Worker 2: Common Crawl
pipeline.add_source(
    HuggingFaceSource("cc_news"),
    target_worker="worker-2"
)

# Worker 3: ArXiv
pipeline.add_source(
    HuggingFaceSource("arxiv"),
    target_worker="worker-3"
)

stats = pipeline.run()
```

### 2. Partition-Based Distribution

Split data source across multiple workers:

```python
pipeline = DistributedPipeline(job_config)

# Split Wikipedia into 5 partitions
source = HuggingFaceSource(
    "wikipedia",
    partitions=5,  # Split across workers
    distributed=True
)

pipeline.add_source(source)
stats = pipeline.run()
```

### 3. Pipeline Stage Distribution

Different workers handle different processing stages:

```python
from auralith_pipeline.distributed import StageWorkerPool

pipeline = DistributedPipeline(job_config)

# Collection workers
collection_pool = StageWorkerPool(
    stage="collection",
    worker_ids=["worker-1", "worker-2"],
)

# Processing workers
processing_pool = StageWorkerPool(
    stage="preprocessing",
    worker_ids=["worker-3", "worker-4"],
)

# Tokenization workers
tokenization_pool = StageWorkerPool(
    stage="tokenization",
    worker_ids=["worker-5", "worker-6"],
)

pipeline.set_worker_pools(
    collection_pool,
    processing_pool,
    tokenization_pool
)

stats = pipeline.run()
```

## Monitoring & Management

### Web Dashboard

Access the monitoring dashboard:

```
http://coordinator.internal:8080/dashboard
```

Shows:
- Real-time worker status
- Task queue depth
- Processing throughput
- Error rates
- Resource utilization

### CLI Monitoring

```bash
# Check worker status
auralith-pipeline status --coordinator coordinator.internal:8080

# Monitor specific job
auralith-pipeline job-status \
  --coordinator coordinator.internal:8080 \
  --job-id wikipedia-processing

# Stream logs
auralith-pipeline logs \
  --coordinator coordinator.internal:8080 \
  --job-id wikipedia-processing \
  --follow
```

### Programmatic Monitoring

```python
from auralith_pipeline.distributed import DistributedClient

client = DistributedClient("coordinator.internal:8080")

# Get job status
job = client.get_job("wikipedia-processing")
print(f"Status: {job.status}")
print(f"Progress: {job.progress}%")
print(f"Processed: {job.samples_processed:,}")
print(f"Failed: {job.samples_failed}")

# List active workers
workers = client.list_workers()
for worker in workers:
    print(f"{worker.id}: {worker.status} ({worker.cpu_usage}% CPU)")

# Get metrics
metrics = client.get_metrics("wikipedia-processing")
print(f"Throughput: {metrics.throughput:.0f} samples/sec")
print(f"Error rate: {metrics.error_rate:.2%}")
```

## Fault Tolerance & Recovery

### Checkpointing

Automatic checkpointing of progress:

```yaml
coordinator:
  checkpointing:
    enabled: true
    interval: 300  # Every 5 minutes
    checkpoint_dir: s3://auralith-datasets/checkpoints/
    keep_last_n: 5  # Keep last 5 checkpoints
```

### Recovery from Failures

```python
from auralith_pipeline.distributed import DistributedPipeline

# Resume from checkpoint
pipeline = DistributedPipeline(job_config)
pipeline.resume_from_checkpoint(
    checkpoint_path="s3://auralith-datasets/checkpoints/wikipedia-processing-001"
)

stats = pipeline.run()
```

### Worker Failure Handling

- **Automatic Detection**: Coordinator detects failed workers via heartbeat
- **Task Reassignment**: Failed tasks automatically reassigned to healthy workers
- **State Recovery**: State recovered from checkpoint store
- **Exponential Backoff**: Retry with increasing delays

```yaml
coordinator:
  failure_handling:
    max_retries: 3
    initial_backoff: 5  # seconds
    max_backoff: 300    # seconds
    backoff_multiplier: 2
    
  worker_monitoring:
    heartbeat_interval: 10
    heartbeat_timeout: 30
    max_consecutive_failures: 3
```

## Performance Optimization

### Load Balancing

```python
from auralith_pipeline.distributed import LoadBalancingStrategy

pipeline = DistributedPipeline(job_config)
pipeline.set_load_balancing(
    strategy=LoadBalancingStrategy.DYNAMIC,  # or ROUND_ROBIN, LEAST_BUSY
    rebalance_interval=60,  # seconds
    target_queue_depth=100,
)
```

### Resource Allocation

```python
# Allocate resources per worker
worker_config = {
    "cpu_cores": 4,
    "memory_gb": 16,
    "gpu_count": 1,  # Optional
}

pipeline = DistributedPipeline(job_config, worker_config=worker_config)
```

### Batch Processing Optimization

```python
pipeline = DistributedPipeline(job_config)
pipeline.configure_batching(
    collection_batch_size=1000,
    preprocessing_batch_size=500,
    tokenization_batch_size=100,
    sharding_batch_size=50,
)
```

## Cloud Deployment

### AWS EC2

```bash
#!/bin/bash
# deploy_aws.sh

# Create coordinator instance
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.xlarge \
  --key-name my-key \
  --security-groups auralith-sg \
  --iam-instance-profile Name=auralith-role \
  --user-data file://coordinator-init.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=auralith-coordinator}]'

# Create worker instances (scaling group)
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name auralith-workers \
  --launch-configuration-name auralith-worker-lc \
  --min-size 2 \
  --max-size 10 \
  --desired-capacity 5 \
  --availability-zones us-west-2a us-west-2b
```

### Kubernetes

```yaml
# kubernetes/coordinator-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: auralith-coordinator
spec:
  replicas: 1
  selector:
    matchLabels:
      app: auralith-coordinator
  template:
    metadata:
      labels:
        app: auralith-coordinator
    spec:
      containers:
      - name: coordinator
        image: auralith/pipeline:latest
        command: ["auralith-pipeline", "coordinator", "--config", "/config/distributed.yaml"]
        ports:
        - containerPort: 8080
        env:
        - name: REDIS_HOST
          value: redis-service
        - name: S3_BUCKET
          value: auralith-datasets
        volumeMounts:
        - name: config
          mountPath: /config
      volumes:
      - name: config
        configMap:
          name: auralith-config
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: auralith-workers
spec:
  serviceName: auralith-workers
  replicas: 5
  selector:
    matchLabels:
      app: auralith-worker
  template:
    metadata:
      labels:
        app: auralith-worker
    spec:
      containers:
      - name: worker
        image: auralith/pipeline:latest
        command: ["auralith-pipeline", "worker", "--coordinator", "auralith-coordinator:8080"]
        resources:
          requests:
            cpu: "4"
            memory: "16Gi"
          limits:
            cpu: "8"
            memory: "32Gi"
        env:
        - name: COORDINATOR_HOST
          value: auralith-coordinator
        - name: WORKER_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        volumeMounts:
        - name: cache
          mountPath: /tmp/auralith_cache
      volumes:
      - name: cache
        emptyDir:
          sizeLimit: 50Gi
```

Deploy to Kubernetes:

```bash
# Create namespace
kubectl create namespace auralith

# Apply configurations
kubectl apply -f kubernetes/ -n auralith

# Scale workers
kubectl scale statefulset auralith-workers --replicas=10 -n auralith

# Check status
kubectl get pods -n auralith
kubectl logs -f pod/auralith-workers-0 -n auralith
```

### Google Cloud Dataflow

```python
import apache_beam as beam
from auralith_pipeline.distributed.beam import PipelineRunner

# Create Beam pipeline
beam_pipeline = beam.Pipeline(runner='DataflowRunner')

# Configure pipeline
runner = PipelineRunner(
    project='my-gcp-project',
    region='us-west1',
    num_workers=10,
    machine_type='n1-standard-4',
)

# Run pipeline
runner.run(
    beam_pipeline,
    config_path='configs/distributed.yaml',
    output_bucket='gs://auralith-datasets/shards'
)
```

## Troubleshooting

### Common Issues

#### 1. Worker Connection Failures

```bash
# Check coordinator is running
auralith-pipeline coordinator-status

# Check network connectivity
ping coordinator.internal
telnet coordinator.internal 8080

# Check logs
tail -f /var/log/auralith/worker.log
```

#### 2. Slow Throughput

```python
# Monitor performance
client = DistributedClient("coordinator.internal:8080")
metrics = client.get_metrics("job-id")

# Check bottlenecks
print(f"Collection rate: {metrics.collection_rate} samples/sec")
print(f"Preprocessing rate: {metrics.preprocessing_rate} samples/sec")
print(f"Tokenization rate: {metrics.tokenization_rate} samples/sec")

# Increase batch sizes if bottleneck identified
```

#### 3. Memory Issues

```yaml
# Reduce batch sizes and cache
workers:
  batch_size: 50  # Reduce from 100
  
  cache_dir: /tmp/auralith_cache
  max_cache_size: 25GB  # Reduce from 50GB
```

#### 4. Storage Access Issues

```bash
# Verify credentials
aws s3 ls s3://auralith-datasets/

# Check permissions
aws iam get-user
aws s3api get-object-acl --bucket auralith-datasets --key test-file
```

## Best Practices

1. **Resource Planning**: Allocate 1 coordinator per 50-100 workers
2. **Network**: Use high-speed network (10Gbps+) between nodes
3. **Storage**: Use cloud storage (S3/GCS) for production deployments
4. **Monitoring**: Always enable monitoring dashboard for large jobs
5. **Checkpointing**: Enable checkpointing for long-running jobs (>1 hour)
6. **Testing**: Test on small dataset before running large jobs
7. **Scaling**: Start with 2-3 workers, gradually increase
8. **Logging**: Enable debug logging during initial troubleshooting

## Example: Complete Distributed Setup

```python
from auralith_pipeline.distributed import (
    DistributedPipeline,
    JobConfig,
    WorkerPoolConfig,
)
from auralith_pipeline.sources import HuggingFaceSource, LocalFileSource
from auralith_pipeline.storage import S3Storage

# Configure job
job_config = JobConfig(
    name="massive-dataset-processing",
    coordinator_host="coordinator.internal",
    coordinator_port=8080,
    timeout=86400 * 7,  # 7 days
)

# Configure worker pools
collection_pool = WorkerPoolConfig(
    name="collection",
    worker_ids=["worker-1", "worker-2", "worker-3"],
    batch_size=1000,
)

processing_pool = WorkerPoolConfig(
    name="processing",
    worker_ids=["worker-4", "worker-5", "worker-6"],
    batch_size=500,
)

tokenization_pool = WorkerPoolConfig(
    name="tokenization",
    worker_ids=["worker-7", "worker-8", "worker-9"],
    batch_size=100,
)

# Create pipeline
pipeline = DistributedPipeline(
    job_config,
    worker_pools=[collection_pool, processing_pool, tokenization_pool],
)

# Add multiple sources
pipeline.add_source(HuggingFaceSource("wikipedia", partitions=3))
pipeline.add_source(HuggingFaceSource("cc_news", partitions=3))
pipeline.add_source(LocalFileSource("./data/custom/*.txt"))

# Configure storage
storage = S3Storage(
    bucket="auralith-datasets",
    region="us-west-2",
)

# Run pipeline
stats = pipeline.run(
    output_path="s3://auralith-datasets/massive/shards",
    storage_backend=storage,
    monitor=True,
    enable_checkpointing=True,
)

print(f"✓ Processing complete!")
print(f"  Samples: {stats.total_samples:,}")
print(f"  Shards: {stats.shard_count}")
print(f"  Size: {stats.total_size_bytes / 1e9:.1f} GB")
print(f"  Time: {stats.processing_time_seconds / 3600:.1f} hours")
print(f"  Throughput: {stats.total_samples / stats.processing_time_seconds:.0f} samples/sec")
```

## Next Steps

- Review [Architecture Documentation](ARCHITECTURE.md) for pipeline details
- Check [Contributing Guide](CONTRIBUTING.md) for development setup
- Explore example configurations in `configs/`
- Join our community for support and discussions
