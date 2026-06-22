# config.sh — Single entry point for dataset path configuration.
# Source this file in your shell before running experiments, or edit the default below.
#
# Usage:
#   source config.sh
#   bash exp/run_point_io_exp.sh
#
# Or override on the command line:
#   DATASETS_DIRECTORY=/your/data/dir bash exp/run_point_io_exp.sh

export DATASETS_DIRECTORY="${DATASETS_DIRECTORY:-/mnt/data/Dataset/public/SOSD}"
