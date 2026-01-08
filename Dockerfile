FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Install package
RUN pip install --no-cache-dir -e ".[all]"

# Create data directory
RUN mkdir -p /data

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
ENTRYPOINT ["python", "-m", "auralith_pipeline.cli"]
CMD ["--help"]
