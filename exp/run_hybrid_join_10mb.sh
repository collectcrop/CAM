#!/usr/bin/env bash
set -euo pipefail

# End-to-end 10 MiB inner-relation experiment:
#   1. take an exact 10 MiB prefix of SOURCE_DATASET;
#   2. generate w1-w6 outer workloads and hybrid metadata for all configured sizes;
#   3. run every join strategy;
#   4. aggregate size-specific bar charts into one PDF.
#
# The default sweep includes 10K, 100K, 1M, 10M, 50M, and 100M outer tuples.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
[ -f config.sh ] && source config.sh

SOURCE_DATASET="${SOURCE_DATASET:-books_200M_uint64_unique}"
DATASET="${DATASET:-books_10MB_uint64_unique}"
INNER_SIZE_MIB="${INNER_SIZE_MIB:-10}"
RECORD_BYTES="${RECORD_BYTES:-8}"
OUTER_SIZES="${OUTER_SIZES:-10K:10000 100K:100000 1M:1000000 10M:10000000 50M:50000000 100M:100000000}"
QUERY_TAG_LIST="${QUERY_TAG_LIST:-10K 100K 1M 10M 50M 100M}"
TABLE_LIST="${TABLE_LIST:-1 2 3 4 5 6}"
MODE_LIST="${MODE_LIST:-hybrid point range inlj hash sortmerge}"

OUT_DIR="${OUT_DIR:-build/log/hybrid_join_10mb}"
WORK_DIR="${WORK_DIR:-build/tmp/hybrid_join_10mb_sort}"
PLOT_OUTPUT_DIR="${PLOT_OUTPUT_DIR:-data/outputs/figures/hybrid_join_10mb}"
PLOT_BASENAME="${PLOT_BASENAME:-${DATASET}_end_to_end_time}"
PLOT_FORMATS="${PLOT_FORMATS:-pdf}"
PLOT_PYTHON="${PLOT_PYTHON:-$PYTHON_BIN}"

source_path="${DATASETS_DIRECTORY}/${SOURCE_DATASET}"
target_path="${DATASETS_DIRECTORY}/${DATASET}"

if [ "${SKIP_PREPARE:-0}" != "1" ]; then
    force_args=()
    if [ "${FORCE_INNER_PREFIX:-0}" = "1" ]; then
        force_args+=(--force)
    fi
    "$PYTHON_BIN" exp/prepare_join_inner_prefix.py \
        --source "$source_path" \
        --output "$target_path" \
        --size-mib "$INNER_SIZE_MIB" \
        --record-bytes "$RECORD_BYTES" \
        "${force_args[@]}"
fi

if [ "${SKIP_GENERATE:-0}" != "1" ]; then
    DATASETS_DIRECTORY="$DATASETS_DIRECTORY" \
    DATASET="$DATASET" \
    OUTER_SIZES="$OUTER_SIZES" \
        bash exp/run_join_workloads.sh
fi

if [ "${SKIP_BUILD:-0}" != "1" ]; then
    cmake --build build --target pgm_hybrid_join
fi

if [ "${SKIP_RUN:-0}" != "1" ]; then
    DATASETS_DIRECTORY="$DATASETS_DIRECTORY" \
    DATASET="$DATASET" \
    QUERY_TAG_LIST="$QUERY_TAG_LIST" \
    TABLE_LIST="$TABLE_LIST" \
    MODE_LIST="$MODE_LIST" \
    OUT_DIR="$OUT_DIR" \
    WORK_DIR="$WORK_DIR" \
    NUM_KEYS= \
        bash exp/run_hybrid_join.sh
fi

if [ "${SKIP_PLOT:-0}" != "1" ]; then
    plot_args=(
        --input-dir "$OUT_DIR"
        --dataset-filter "$DATASET"
        --output-dir "$PLOT_OUTPUT_DIR"
        --basename "$PLOT_BASENAME"
        --formats
    )
    for format in $PLOT_FORMATS; do
        plot_args+=("$format")
    done
    if [ "${ALLOW_INCOMPLETE_PLOT:-0}" = "1" ]; then
        plot_args+=(--allow-incomplete)
    fi
    "$PLOT_PYTHON" visualize/plot_hybrid_join_time.py "${plot_args[@]}"
fi

printf '[+] 10 MiB hybrid-join pipeline complete\n'
printf '    inner: %s\n' "$target_path"
printf '    logs:  %s\n' "$OUT_DIR"
printf '    plot:  %s/%s.pdf\n' "$PLOT_OUTPUT_DIR" "$PLOT_BASENAME"
