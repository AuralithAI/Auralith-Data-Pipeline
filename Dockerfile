FROM python:3.14-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements/ ./requirements/
COPY pyproject.toml README.md LICENSE ./

# Install dependencies and package
RUN pip install --no-cache-dir -r requirements/full.txt && \
    pip install --no-cache-dir -e .

# Copy package files
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Create data directories
RUN mkdir -p /app/data/shards /app/data/cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Default command
ENTRYPOINT ["auralith-pipeline"]
CMD ["--help"]

