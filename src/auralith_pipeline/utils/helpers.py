"""Utility functions and helpers."""

import logging
import sys


def setup_logging(
    level: str = "INFO",
    format_str: str | None = None,
) -> logging.Logger:
    """Setup logging configuration."""
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    return logging.getLogger("auralith_pipeline")


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def format_number(num: int) -> str:
    """Format number with commas."""
    return f"{num:,}"
