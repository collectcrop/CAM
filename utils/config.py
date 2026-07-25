"""Centralized configuration for the CAM project.

Reads local paths from environment variables set by config.sh, with
repository-local defaults for reproducibility.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS_DIRECTORY = REPO_ROOT / "data" / "datasets" / "SOSD"
DEFAULT_LOG_DIRECTORY = REPO_ROOT / "build" / "log"
DEFAULT_REAL_DIRECTORY = REPO_ROOT / "build" / "log" / "real"


def _directory_from_env(primary: str, legacy: str, default: Path) -> str:
    path = os.environ.get(primary) or os.environ.get(legacy) or str(default)
    return path.rstrip("/") + "/"


def get_datasets_directory() -> str:
    """Return the datasets directory path (with trailing slash)."""
    return _directory_from_env("DATASETS_DIRECTORY", "DATASETS_DIRECTORY", DEFAULT_DATASETS_DIRECTORY)


def get_datasets_path() -> Path:
    """Return the datasets directory as a pathlib.Path."""
    return Path(get_datasets_directory())


def get_log_directory() -> str:
    """Return the experiment log directory path (with trailing slash)."""
    return _directory_from_env("CAM_LOG_DIRECTORY", "LOG_DIRECTORY", DEFAULT_LOG_DIRECTORY)


def get_real_directory() -> str:
    """Return the real-measurement directory path (with trailing slash)."""
    return _directory_from_env("CAM_REAL_DIRECTORY", "REAL_DIRECTORY", DEFAULT_REAL_DIRECTORY)
