"""Centralized configuration for the CAM project.

Reads dataset directory from the DATASETS_DIRECTORY environment variable
(set by config.sh), with a sensible default for reproducibility.
"""

import os
from pathlib import Path

DEFAULT_DATASETS_DIRECTORY = "/mnt/data/Dataset/public/SOSD"


def get_datasets_directory() -> str:
    """Return the datasets directory path (with trailing slash)."""
    path = os.environ.get("DATASETS_DIRECTORY", DEFAULT_DATASETS_DIRECTORY)
    return path.rstrip("/") + "/"


def get_datasets_path() -> Path:
    """Return the datasets directory as a pathlib.Path."""
    return Path(get_datasets_directory())
