#!/usr/bin/env python3
"""
Quick data collection script.

Usage:
    python scripts/collect.py wikipedia --max-samples 10000
    python scripts/collect.py arxiv --max-samples 5000 --output ./my-data
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from auralith_pipeline import Pipeline, PipelineConfig
from auralith_pipeline.sources import DATASET_REGISTRY, create_source
from auralith_pipeline.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Collect and process datasets")
    parser.add_argument(
        "dataset",
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset to collect",
    )
    parser.add_argument(
        "--max-samples",
        "-n",
        type=int,
        default=10000,
        help="Maximum samples to collect",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./data/shards",
        help="Output directory",
    )
    parser.add_argument(
        "--preset",
        choices=["small", "medium", "production"],
        default="medium",
        help="Configuration preset",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable deduplication",
    )
    parser.add_argument(
        "--no-quality-filter",
        action="store_true",
        help="Disable quality filtering",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging("DEBUG" if args.verbose else "INFO")

    # Create config
    config = PipelineConfig.from_preset(args.preset)
    config.output_dir = args.output
    config.deduplicate = not args.no_dedup
    config.quality_filter = not args.no_quality_filter

    # Create pipeline
    pipeline = Pipeline(config)

    # Add source
    source = create_source(
        args.dataset,
        streaming=True,
        max_samples=args.max_samples,
    )
    pipeline.add_source(source)

    # Run
    print(f"\nCollecting {args.dataset} (max {args.max_samples:,} samples)")
    print(f"Output: {args.output}\n")

    stats = pipeline.run(max_samples=args.max_samples)

    print(stats.summary())


if __name__ == "__main__":
    main()
