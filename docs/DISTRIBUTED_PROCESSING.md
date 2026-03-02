# Distributed Processing Guide

## Overview

The Auralith Data Pipeline distributes file-level tokenization and sharding
across multiple workers.  Two deployment modes are supported:

| Mode | State Store | Machines | Redis Required |
|------|-------------|----------|----------------|
| **Embedded** | In-memory | 1 | No |
| **External** | Redis | 1+ | Yes |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      COORDINATOR NODE                               │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  JobManager                                                   │  │
│  │  • create_job / submit_tasks / record_task_result             │  │
│  │  • _monitor_workers  (heartbeat watchdog thread)              │  │
│  │  • _requeue_pending_tasks (pending → workers thread)          │  │
│  │  • _handle_worker_failure (drain queue → retry or fail)       │  │
│  │  • Pluggable strategy: round_robin | least_busy | dynamic     │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  StateStore (Redis or InMemory)                               │  │
│  │  • heartbeat:<worker_id>   (TTL-based liveness)               │  │
│  │  • tasks:<worker_id>       (per-worker task queue)            │  │
│  │  • pending_tasks           (unassigned tasks)                 │  │
│  │  • job:<job_id>            (job metadata + counters)          │  │
│  │  • task:<task_id>          (task status + result)             │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
         │                    │                      │
         ▼                    ▼                      ▼
┌────────────────┐  ┌────────────────┐     ┌────────────────┐
│  WORKER 1      │  │  WORKER 2      │ ... │  WORKER N      │
│                │  │                │     │                │
│ poll tasks:w1  │  │ poll tasks:w2  │     │ poll tasks:wN  │
│ tokenize files │  │ tokenize files │     │ tokenize files │
│ write shards   │  │ write shards   │     │ write shards   │
│ heartbeat ♥    │  │ heartbeat ♥    │     │ heartbeat ♥    │
└────────────────┘  └────────────────┘     └────────────────┘
```

## Quick Start

### Embedded Mode (Single Machine, No Redis)

Best for local development or a single beefy server:

```bash
auralith-pipeline submit-job \
  -c configs/distributed.yaml \
  -i data/raw/ \
  -o shards/ \
  -t tokenizers/ \
  --embedded \
  -w 8                   # 8 worker threads
```

This spins up a `Coordinator` + 8 `Worker` instances in the same process,
using an `InMemoryStateStore`.  No Redis required.

### External Mode (Multi-Machine, Redis)

#### 1. Start Redis

Use a managed instance (ElastiCache, Memorystore) or local:

```bash
redis-server --port 6379
```

#### 2. Start the Coordinator

```bash
auralith-pipeline coordinator \
  -c configs/distributed.yaml \
  --host 0.0.0.0 --port 8080
```

#### 3. Start Workers (one per machine or terminal)

```bash
auralith-pipeline worker \
  -c configs/distributed.yaml \
  --coordinator 10.0.0.1:8080 \
  --worker-id worker-1

# On another machine:
auralith-pipeline worker \
  -c configs/distributed.yaml \
  --coordinator 10.0.0.1:8080 \
  --worker-id worker-2
```

#### 4. Submit the Job

```bash
auralith-pipeline submit-job \
  -c configs/distributed.yaml \
  -i data/raw/ -o shards/ -t tokenizers/ \
  --external
```

## Configuration

```yaml
# configs/distributed.yaml
coordinator:
  host: "0.0.0.0"
  port: 8080

  # "redis" for multi-machine, "memory" for embedded
  state_store_type: redis
  state_store_host: redis.internal
  state_store_port: 6379
  state_store_db: 0
  # state_store_password: null   # For cloud Redis (ElastiCache, etc.)

  heartbeat_interval: 10    # seconds between heartbeat checks
  heartbeat_timeout: 30     # seconds before worker is considered dead
  max_retries: 3            # max task retries before marking failed
  retry_backoff: 5          # seconds between requeue attempts

workers:
  - name: processing
    worker_ids:
      - worker-1
      - worker-2
      - worker-3
      - worker-4
    batch_size: 500
    num_processes: 4

job:
  name: distributed-processing
  num_workers: 4
  timeout: 86400             # 24 hours
  output_dir: ./shards
```

## How It Works

1. **File Discovery** — `DistributedPipeline` scans the input directory
   (`rglob("*")`) for all raw files.

2. **Task Splitting** — Files are chunked into tasks (default 500 files
   per task).  Each task is a self-contained unit with file paths,
   tokenizer directory, output directory, and shard parameters.

3. **Task Submission** — Tasks are pushed to the coordinator's
   `JobManager`, which assigns them to workers via the configured
   distribution strategy.

4. **Worker Execution** — Each worker polls its personal queue
   (`tasks:<worker_id>`), loads the tokenizers, tokenizes every file,
   and writes SafeTensors shards to the output directory.

5. **Heartbeat Monitoring** — Workers send heartbeats every 10 seconds.
   The coordinator's monitor thread checks all heartbeats and, if a
   worker is unresponsive, calls `_handle_worker_failure()`.

6. **Failure Recovery** — Dead worker's task queue is drained.  Each
   task's retry counter is incremented.  Tasks below `max_retries` are
   re-queued to `pending_tasks`; tasks that exceed the limit are marked
   `failed` and the job's failure counter is updated.

7. **Completion** — `DistributedPipeline.run()` polls the job status
   until all tasks are completed (or the timeout is reached).

## Task Distribution Strategies

| Strategy | Class | Description |
|----------|-------|-------------|
| `round_robin` | `RoundRobinDistribution` | Cycle through workers in order |
| `least_busy` | `LeastBusyDistribution` | Assign to worker with shortest queue |
| `dynamic` | `DynamicDistribution` | Weighted score: queue length + CPU + memory |

Set at runtime via the coordinator's `JobManager.set_strategy("least_busy")`.

## Python API

```python
from auralith_pipeline.distributed import DistributedPipeline, JobConfig

job_cfg = JobConfig(
    name="my-processing-job",
    num_workers=4,
    timeout=3600,
)

pipeline = DistributedPipeline(job_cfg, embedded=True, num_workers=4)

result = pipeline.run(
    input_dir="data/raw/",
    output_dir="shards/",
    tokenizers_dir="tokenizers/",
    max_seq_len=4096,
    shard_size=10_000,
    files_per_task=500,
)

print(f"Status:    {result['status']}")
print(f"Tasks:     {result['completed_tasks']}/{result['total_tasks']}")
print(f"Failed:    {result['failed_tasks']}")
print(f"Elapsed:   {result['elapsed_seconds']:.1f}s")
```

## Monitoring

### CLI

```bash
# System status (active workers)
auralith-pipeline status --coordinator 10.0.0.1:8080

# Job progress
auralith-pipeline job-status --coordinator 10.0.0.1:8080 --job-id <id>
```

### Python Client

```python
from auralith_pipeline.distributed import DistributedClient

client = DistributedClient("10.0.0.1:8080")

# List workers
for w in client.list_workers():
    print(f"  {w['id']}: {w['status']}")

# Job metrics
m = client.get_metrics("my-job-id")
print(f"Progress: {m['completed_tasks']}/{m['total_tasks']}")

# Cancel a job
client.cancel_job("my-job-id")
client.close()
```

## Cloud Redis

For production deployments across machines, use a managed Redis service:

| Cloud | Service |
|-------|---------|
| AWS | ElastiCache for Redis |
| GCP | Memorystore for Redis |
| Azure | Azure Cache for Redis |

```yaml
coordinator:
  state_store_type: redis
  state_store_host: my-redis.xxxx.cache.amazonaws.com
  state_store_port: 6379
  state_store_password: "<your-auth-token>"
```

## Troubleshooting

### Worker can't connect

```bash
# Verify Redis is reachable
redis-cli -h redis.internal -p 6379 ping

# Check coordinator logs for heartbeat registrations
auralith-pipeline coordinator -c configs/distributed.yaml 2>&1 | grep heartbeat
```

### Tasks stuck in pending

No workers have registered heartbeats.  Start workers first, then submit
the job.  The coordinator's `_requeue_pending_tasks` thread will
automatically assign pending tasks to newly available workers.

### Job completes with errors

Check individual task results:

```python
from auralith_pipeline.distributed.state import RedisStateStore

store = RedisStateStore(host="redis.internal")
store.connect()

task = store.get("task:<task-id>")
print(task["status"], task.get("error"))
```
