#!/usr/bin/env bash
set -euo pipefail

# Run join-fitting experiments.
# Point mode: sweep number of query keys (N) — aggregate timing per N.
# Range mode: synthesize single range probes with varied page spans for
# CPU-side range parameter fitting.

BINARY="${BINARY:-./build/pgm_join_fit}"
DATA_DIR="${DATA_DIR:-/mnt/data/Dataset/public/SOSD}"
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

mkdir -p "$OUT_DIR"

# ------------------------------------------------------------------
# Compute max N and ensure point query file exists
# ------------------------------------------------------------------
MAX_POINT_N=$(echo "$POINT_N_LIST" | tr ' ' '\n' | sort -n | tail -1)
POINT_QUERY_FILE="${POINT_QUERY_FILE:-${DATA_DIR}/${DATASET}.${MAX_POINT_N}Mquery.bin}"

if [ ! -f "$POINT_QUERY_FILE" ]; then
    echo "[*] Generating point query file with $((MAX_POINT_N * 1000000)) queries -> $(basename "$POINT_QUERY_FILE")"
    python3 -c "
import sys; sys.path.insert(0, 'utils')
import numpy as np
from generate_query import generate_realistic_queries_from_data
keys = np.fromfile('$DATA_DIR/$DATASET', dtype=np.uint64)
queries = generate_realistic_queries_from_data(keys, num_queries=$((MAX_POINT_N * 1000000)), seed=42)
queries.tofile('$POINT_QUERY_FILE')
print(f'[+] Generated {len(queries)} point queries')
"
fi

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

# ------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------
case "${1:-all}" in
    point)
        for N in $POINT_N_LIST; do
            run_point "$N"
        done
        ;;
    range)
        run_range
        ;;
    all)
        echo "=== Point sweep ==="
        for N in $POINT_N_LIST; do
            run_point "$N"
        done
        echo "=== Range sweep ==="
        run_range
        ;;
    *)
        echo "Usage: $0 {point|range|all}"
        exit 1
        ;;
esac

echo "[+] Done. Results in $OUT_DIR/"
