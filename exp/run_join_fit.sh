#!/usr/bin/env bash
set -euo pipefail

# Run join-fitting experiments.
# Point mode: sweep number of query keys (N) — aggregate timing per N.
# Range mode: synthesize single range probes with varied page spans for
# CPU-side range parameter fitting.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
[ -f config.sh ] && source config.sh

BINARY="${BINARY:-./build/pgm_join_fit}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASETS_DIRECTORY="${DATASETS_DIRECTORY:-$REPO_ROOT/data/datasets/SOSD}"
DATA_DIR="${DATA_DIR:-$DATASETS_DIRECTORY}"
OUT_DIR="${OUT_DIR:-build/log/join_fit}"
DATASET="${DATASET:-books_10M_uint64_unique}"
NUM_KEYS="${NUM_KEYS:-10000000}"
EPSILON="${EPSILON:-16}"
MEMORY_MIB="${MEMORY_MIB:-16}"
POLICY="${POLICY:-LRU}"

# Point sweep: N values in millions
POINT_N_LIST="${POINT_N_LIST:-1 2 5 10 20}"
# Range calibration: target page spans and samples per span
RANGE_PAGE_SPANS="${RANGE_PAGE_SPANS:-1 2 4 8 16 32 64}"
RANGE_REPEATS="${RANGE_REPEATS:-8}"
FIT_AFTER_RUN="${FIT_AFTER_RUN:-1}"
FIT_SCRIPT="${FIT_SCRIPT:-utils/fit_join_cost_model.py}"
FIT_OUTPUT_PREFIX="${FIT_OUTPUT_PREFIX:-$OUT_DIR/${DATASET}_join_cost_params}"

mkdir -p "$OUT_DIR"

# ------------------------------------------------------------------
# Compute max N and ensure point query file exists
# ------------------------------------------------------------------
ensure_point_query_file() {
    local max_point_n
    max_point_n=$(echo "$POINT_N_LIST" | tr ' ' '\n' | sort -n | tail -1)
    POINT_QUERY_FILE="${POINT_QUERY_FILE:-${DATA_DIR}/${DATASET}.${max_point_n}Mquery.bin}"

    if [ ! -f "$POINT_QUERY_FILE" ]; then
        echo "[*] Generating point query file with $((max_point_n * 1000000)) queries -> $(basename "$POINT_QUERY_FILE")"
        "$PYTHON_BIN" -c "
import sys; sys.path.insert(0, 'utils')
import numpy as np
from generate_query import generate_realistic_queries_from_data
keys = np.fromfile('$DATA_DIR/$DATASET', dtype=np.uint64)
queries = generate_realistic_queries_from_data(keys, num_queries=$((max_point_n * 1000000)), seed=42)
queries.tofile('$POINT_QUERY_FILE')
print(f'[+] Generated {len(queries)} point queries')
"
    fi
}

# ------------------------------------------------------------------
# Run point sweep
# ------------------------------------------------------------------
run_point() {
    local N_KEYS="$1"
    local out_csv="$OUT_DIR/${DATASET}_${N_KEYS}Mquery_join.point.csv"
    echo "[*] point mode: N=${N_KEYS}x1e6 -> $(basename "$out_csv")"
    "$BINARY" \
        --mode point \
        --data "$DATA_DIR/$DATASET" \
        --query "$POINT_QUERY_FILE" \
        --output "$out_csv" \
        --epsilon "$EPSILON" \
        --M "$MEMORY_MIB" \
        --keys "$NUM_KEYS" \
        --num-queries "$N_KEYS"000000 \
        --policy "$POLICY"
}

# ------------------------------------------------------------------
# Run range: one CSV row per synthesized single range probe
# ------------------------------------------------------------------
run_range() {
    local out_csv="$OUT_DIR/${DATASET}_query_join.range.csv"
    echo "[*] range mode: spans={$RANGE_PAGE_SPANS}, repeats=$RANGE_REPEATS -> $(basename "$out_csv")"
    "$BINARY" \
        --mode range \
        --data "$DATA_DIR/$DATASET" \
        --output "$out_csv" \
        --epsilon "$EPSILON" \
        --M "$MEMORY_MIB" \
        --keys "$NUM_KEYS" \
        --range-page-spans "$RANGE_PAGE_SPANS" \
        --range-repeats "$RANGE_REPEATS" \
        --policy "$POLICY"
}

fit_params() {
    local mode="$1"
    if [ "$FIT_AFTER_RUN" != "1" ]; then
        return 0
    fi
    echo "=== Fitting cost model parameters ==="
    "$PYTHON_BIN" "$FIT_SCRIPT" \
        --data-dir "$OUT_DIR" \
        --dataset "$DATASET" \
        --epsilon "$EPSILON" \
        --mode "$mode" \
        --output-json "${FIT_OUTPUT_PREFIX}.json" \
        --output-csv "${FIT_OUTPUT_PREFIX}.csv" \
        --output-env "${FIT_OUTPUT_PREFIX}.env"
}

# ------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------
case "${1:-all}" in
    point)
        ensure_point_query_file
        for N in $POINT_N_LIST; do
            run_point "$N"
        done
        fit_params point
        ;;
    range)
        run_range
        fit_params range
        ;;
    all)
        echo "=== Point sweep ==="
        ensure_point_query_file
        for N in $POINT_N_LIST; do
            run_point "$N"
        done
        echo "=== Range sweep ==="
        run_range
        fit_params all
        ;;
    *)
        echo "Usage: $0 {point|range|all}"
        exit 1
        ;;
esac

echo "[+] Done. Results in $OUT_DIR/"
