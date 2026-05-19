#!/usr/bin/env sh
set -eu

# Fixed-epsilon memory sweep for cache hit ratio.
# Usage example:
#   FIXED_EPSILON=32 M_LIST="10 20 40 60" DATASET_SIZE_MB=200 sh exp.sh
#
# Optional overrides:
#   DATA_DIR=/mnt/data/Dataset/public/SOSD
#   DATASET_SIZE_MB=10|100|200
#   TOTAL_KEYS=<int>
#   FIXED_EPSILON=<even number in [2,128]>
#   M_LIST="10 20 40 60"
#   POLICIES=FIFO,LRU,LFU
#   STRATEGIES=all_in_once
#   BUDGET_MODE=raw|estimated|measured
#   OUT_DIR=build/log/cmp

DATA_DIR="${DATA_DIR:-/mnt/data/Dataset/public/SOSD}"
DATASET_SIZE_MB="${DATASET_SIZE_MB:-10}"
FIXED_EPSILON="${FIXED_EPSILON:-32}"
M_LIST="${M_LIST:-10 20 40 60}"
POLICIES="${POLICIES:-FIFO,LRU,LFU}"
STRATEGIES="${STRATEGIES:-all_in_once}"
BUDGET_MODE="${BUDGET_MODE:-estimated}"
OUT_DIR="${OUT_DIR:-build/log/cmp}"

case "$DATASET_SIZE_MB" in
  10) DEFAULT_KEYS=10000000 ;;
  100) DEFAULT_KEYS=100000000 ;;
  200) DEFAULT_KEYS=200000000 ;;
  *)
    echo "[error] unsupported DATASET_SIZE_MB=$DATASET_SIZE_MB (expected 10/100/200)." >&2
    exit 1
    ;;
esac
TOTAL_KEYS="${TOTAL_KEYS:-$DEFAULT_KEYS}"

DATA_BASENAME="books_${DATASET_SIZE_MB}M_uint64_unique"
DATA_PATH="${DATA_DIR}/${DATA_BASENAME}"
QUERY_PATH="${DATA_DIR}/${DATA_BASENAME}.query.bin"

mkdir -p "$OUT_DIR"

echo "[info] dataset=${DATASET_SIZE_MB}M, epsilon=${FIXED_EPSILON}, M_LIST=${M_LIST}, budget_mode=${BUDGET_MODE}"

for M in $M_LIST; do
  OUT_FILE="${OUT_DIR}/books_${DATASET_SIZE_MB}M_M${M}_eps${FIXED_EPSILON}_memory_sweep_sim.csv"
  echo "[run] M=${M} -> ${OUT_FILE}"

  ./build/pgm_cache_simulate \
    --data "$DATA_PATH" \
    --queries "$QUERY_PATH" \
    --keys "$TOTAL_KEYS" \
    --M "$M" \
    --epsilons "$FIXED_EPSILON" \
    --policies "$POLICIES" \
    --strategies "$STRATEGIES" \
    --budget-mode "$BUDGET_MODE" \
    --summary-out "$OUT_FILE"
done

# Merge per-memory CSVs for easier plotting.
MERGED_OUT="${OUT_DIR}/books_${DATASET_SIZE_MB}M_eps${FIXED_EPSILON}_memory_sweep_sim_merged.csv"
FIRST=1
for M in $M_LIST; do
  CUR_FILE="${OUT_DIR}/books_${DATASET_SIZE_MB}M_M${M}_eps${FIXED_EPSILON}_memory_sweep_sim.csv"
  if [ "$FIRST" -eq 1 ]; then
    cat "$CUR_FILE" > "$MERGED_OUT"
    FIRST=0
  else
    tail -n +2 "$CUR_FILE" >> "$MERGED_OUT"
  fi
done

echo "[done] merged summary: ${MERGED_OUT}"
