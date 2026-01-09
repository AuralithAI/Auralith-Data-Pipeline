# GitHub Actions Scheduled Pipeline

## Overview

The Auralith Data Pipeline runs as a **scheduled GitHub Actions workflow** that:
1. Triggers weekly (or on-demand)
2. Processes datasets (Wikipedia, The Pile, C4, ArXiv)
3. Creates SafeTensors shards
4. Uploads to S3
5. Generates reports

**No infrastructure needed!** Everything runs in GitHub Actions runners.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              GitHub Actions (Scheduled)                 │
│                                                         │
│  Trigger: Weekly (Sunday 2 AM) or Manual               │
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │  1. Install Python & Dependencies               │  │
│  │  2. Configure AWS Credentials                   │  │
│  │  3. Run: auralith-pipeline collect              │  │
│  │  4. Process: Dedupe → Filter → Tokenize         │  │
│  │  5. Create: SafeTensors shards                  │  │
│  │  6. Upload: aws s3 sync → S3 Bucket             │  │
│  │  7. Generate: Processing report                 │  │
│  └─────────────────────────────────────────────────┘  │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
          ┌────────────────────┐
          │   AWS S3 Bucket    │
          │                    │
          │  /datasets/        │
          │    /wikipedia/     │
          │    /the_pile/      │
          │    /c4/            │
          │    /arxiv/         │
          │  /reports/         │
          └────────────────────┘
```

## Setup

### 1. Configure GitHub Secrets

Go to your repository → Settings → Secrets and variables → Actions

Add these secrets:

```
AWS_ACCESS_KEY_ID=AKIAXXXXXXXXXXXXXXXX
AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
S3_BUCKET=your-bucket-name
AWS_REGION=us-east-1  (optional, defaults to us-east-1)
```

### 2. Create S3 Bucket

```bash
# Create bucket
aws s3 mb s3://auralith-training-data

# Set lifecycle policy (optional - archive old data)
aws s3api put-bucket-lifecycle-configuration \
  --bucket auralith-training-data \
  --lifecycle-configuration '{
    "Rules": [{
      "Id": "ArchiveOldData",
      "Status": "Enabled",
      "Transitions": [{
        "Days": 90,
        "StorageClass": "GLACIER"
      }]
    }]
  }'
```

### 3. Push Code to GitHub

```bash
git add .
git commit -m "Add scheduled data pipeline"
git push origin main
```

The workflow will now run:
- **Automatically**: Every Sunday at 2 AM UTC
- **Manually**: Go to Actions tab → "Data Processing Pipeline" → "Run workflow"

## Workflow Configuration

### Schedule
Edit `.github/workflows/data-pipeline.yml`:

```yaml
schedule:
  - cron: '0 2 * * 0'  # Every Sunday at 2 AM UTC
  # - cron: '0 0 1 * *'  # First day of every month
  # - cron: '0 0 * * 1'  # Every Monday
```

### Datasets to Process

The workflow processes datasets based on:
- **Scheduled runs**: Wikipedia (default)
- **Manual runs**: Choose from dropdown

To change the default scheduled dataset, edit:

```yaml
- name: Process Wikipedia Dataset
  if: github.event_name == 'schedule'  # Runs on schedule
  run: |
    auralith-pipeline collect \
      --dataset wikipedia \
      --max-samples 100000
```

### Resource Limits

GitHub Actions runners have:
- 2-core CPU
- 7 GB RAM
- 14 GB disk space
- 6 hours max runtime per job

For larger datasets, use the Spark job (runs on more powerful runners):

```yaml
jobs:
  process-spark-large-scale:
    runs-on: ubuntu-latest
    # Or use self-hosted runner for unlimited resources
    # runs-on: self-hosted
```

## Manual Trigger

### From GitHub UI

1. Go to **Actions** tab
2. Select **"Data Processing Pipeline"**
3. Click **"Run workflow"**
4. Choose:
   - Dataset: wikipedia, the_pile, c4, arxiv, or all
   - Max samples: (optional, leave empty for full dataset)
5. Click **"Run workflow"**

### From Command Line

```bash
# Using GitHub CLI
gh workflow run data-pipeline.yml \
  -f dataset=wikipedia \
  -f max_samples=50000

# Check status
gh run list --workflow=data-pipeline.yml
```

## Output Structure

Data is stored in S3 with this structure:

```
s3://your-bucket/
├── datasets/
│   ├── wikipedia/
│   │   ├── shard_0000.safetensors
│   │   ├── shard_0001.safetensors
│   │   └── metadata.json
│   ├── the_pile/
│   │   └── ...
│   ├── c4/
│   │   └── ...
│   └── arxiv/
│       └── ...
└── reports/
    ├── 2026-01-08-report.md
    ├── 2026-01-15-report.md
    └── ...
```

## Using with Spark (Large Datasets)

For very large datasets (>100GB), use the Spark job:

### Option 1: Local Spark (GitHub Runner)

The workflow includes a Spark job that runs in local mode:

```yaml
- name: Process with Spark (Local mode)
  env:
    SPARK_MASTER: local[*]
  run: |
    auralith-pipeline spark-submit \
      --input s3://$S3_BUCKET/raw-data/ \
      --output s3://$S3_BUCKET/processed/ \
      --master local[*]
```

### Option 2: External Spark Cluster

If you have a Spark cluster (EMR, Databricks, etc.), set the secret:

```
SPARK_MASTER=spark://your-spark-master:7077
```

The workflow will submit jobs to your cluster:

```yaml
env:
  SPARK_MASTER: ${{ secrets.SPARK_MASTER || 'local[*]' }}
```

## Configuration via Environment Variables

You can configure the pipeline without editing code:

**GitHub Secrets/Variables:**
```
S3_BUCKET=auralith-training-data
AWS_REGION=us-east-1
SPARK_MASTER=spark://spark-master:7077  (optional)
HF_TOKEN=hf_xxxxxxxxxxxxx  (for HuggingFace datasets)
```

**In workflow file:**
```yaml
env:
  S3_BUCKET: ${{ secrets.S3_BUCKET }}
  AWS_REGION: ${{ secrets.AWS_REGION || 'us-east-1' }}
  SPARK_MASTER: ${{ secrets.SPARK_MASTER || 'local[*]' }}
```

## Monitoring

### View Logs

1. Go to **Actions** tab
2. Click on the workflow run
3. Click on the job name
4. View step-by-step logs

### Processing Reports

Reports are automatically generated and uploaded to S3:

```bash
# Download latest report
aws s3 cp s3://your-bucket/reports/$(date +%Y-%m-%d)-report.md ./

# View in browser
cat report.md
```

### Email Notifications

GitHub automatically sends emails on workflow failures. You can also add Slack/Discord webhooks:

```yaml
- name: Notify Slack
  if: failure()
  uses: slackapi/slack-github-action@v1
  with:
    webhook-url: ${{ secrets.SLACK_WEBHOOK }}
    payload: |
      {
        "text": "Data pipeline failed!"
      }
```

## Cost Estimation

### GitHub Actions (Free Tier)

- **Public repos**: Unlimited minutes ✅ FREE
- **Private repos**: 2,000 minutes/month free, then $0.008/min

**Example calculation:**
- 1 weekly run × 2 hours = 2 hours/week
- 8 hours/month × 60 min = 480 minutes/month
- **Cost**: FREE (under 2,000 min limit)

### AWS S3

- Storage: $0.023/GB/month (first 50 TB)
- PUT requests: $0.005 per 1,000 requests
- GET requests: $0.0004 per 1,000 requests

**Example calculation:**
- 100 GB processed data = $2.30/month
- 10,000 PUT requests = $0.05/month
- **Total**: ~$3-5/month

### Total Cost: ~$3-5/month ✅

## Advanced: Self-Hosted Runners

For unlimited compute or private datasets:

### 1. Set Up Self-Hosted Runner

```bash
# On your server (EC2, home machine, etc.)
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf ./actions-runner-linux.tar.gz

# Configure
./config.sh --url https://github.com/YOUR_ORG/YOUR_REPO --token YOUR_TOKEN
./run.sh
```

### 2. Update Workflow

```yaml
jobs:
  process-data:
    runs-on: self-hosted  # Use your own machine
```

**Benefits:**
- Unlimited compute time
- More RAM/CPU
- Access to local data
- No cost limits

## Troubleshooting

### Pipeline Fails with "Out of Memory"

Reduce batch size:

```yaml
--max-samples 10000  # Process fewer samples
```

Or use self-hosted runner with more RAM.

### S3 Upload Fails

Check AWS credentials:

```bash
# Test credentials
aws s3 ls s3://your-bucket/
```

Ensure IAM policy has:
```json
{
  "Effect": "Allow",
  "Action": [
    "s3:PutObject",
    "s3:GetObject",
    "s3:ListBucket"
  ],
  "Resource": [
    "arn:aws:s3:::your-bucket",
    "arn:aws:s3:::your-bucket/*"
  ]
}
```

### Workflow Times Out

Increase timeout:

```yaml
timeout-minutes: 360  # 6 hours (default)
timeout-minutes: 1440  # 24 hours (max for self-hosted)
```

## Best Practices

1. **Start Small**: Test with `--max-samples 1000` first
2. **Monitor Costs**: Check S3 and GitHub Actions usage monthly
3. **Use Lifecycle Policies**: Archive old data to Glacier after 90 days
4. **Error Handling**: Check reports after each run
5. **Incremental Processing**: Don't reprocess existing data

## Summary

✅ **Simple Setup**: Just configure GitHub secrets  
✅ **Low Cost**: ~$3-5/month  
✅ **Automated**: Runs on schedule  
✅ **Scalable**: Use Spark for large datasets  
✅ **No Infrastructure**: No servers to manage  
✅ **Reliable**: GitHub's infrastructure  

The pipeline is **production-ready** and requires **zero infrastructure management**!
