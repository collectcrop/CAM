# config.sh - Single entry point for local environment configuration.
# Source this file in your shell before running experiments, or edit the defaults below.
#
# Usage:
#   source config.sh
#   bash exp/run_point_io_exp.sh
#
# Or override on the command line:
#   DATASETS_DIRECTORY=/your/data/dir PYTHON_BIN=/path/to/python bash exp/run_point_io_exp.sh

CAM_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CAM_REPO_ROOT

export DATASETS_DIRECTORY="${DATASETS_DIRECTORY:-$CAM_REPO_ROOT/data/datasets/SOSD}"

if [ -z "${PYTHON_BIN:-}" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="python3"
  fi
fi
export PYTHON_BIN

export CAM_LOG_DIRECTORY="${CAM_LOG_DIRECTORY:-$CAM_REPO_ROOT/build/log}"
export CAM_REAL_DIRECTORY="${CAM_REAL_DIRECTORY:-$CAM_REPO_ROOT/build/log/real}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$CAM_REPO_ROOT/build/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CAM_REPO_ROOT/build/xdg-cache}"
export CAM_PLOT_USETEX="${CAM_PLOT_USETEX:-0}"
