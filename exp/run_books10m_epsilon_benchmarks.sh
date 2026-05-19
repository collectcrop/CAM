#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/bin/python}"
DATASETS_DIRECTORY="${DATASETS_DIRECTORY:-/mnt/data/Dataset/public/SOSD}"
DATA_FILE="${DATA_FILE:-books_10M_uint64_unique}"
QUERY_FILE="${QUERY_FILE:-books_10M_uint64_unique.query.bin}"
DATASET_TAG="${DATASET_TAG:-books_10M}"
N_KEYS="${N_KEYS:-10000000}"
MEMORY_LIST="${MEMORY_LIST:-10 20 40 60}"
EPS_LIST="${EPS_LIST:-8,10,12,14,16,20,24,32,64}"
STRATEGY="${STRATEGY:-all_in_once}"

if [ -n "${POLICY:-}" ]; then
  POLICIES="$POLICY"
else
  POLICIES="${POLICIES:-FIFO LRU LFU}"
fi

LOG_DIR="${LOG_DIR:-build/log}"
FITCAM_ROOT="${FITCAM_ROOT:-$LOG_DIR/fitcam_q30}"
OUTPUT_DIR="${OUTPUT_DIR:-data/outputs/figures/epsilon_analysis}"
PGM_BENCH_BIN="${PGM_BENCH_BIN:-./build/pgm_bench}"
CAM_BIN="${CAM_BIN:-./build/pgm_cam_covariance}"
COLD_START_CORRECTION="${COLD_START_CORRECTION:-1}"
ESTIMATE_QUERY_FRACTION="${ESTIMATE_QUERY_FRACTION:-0.3}"

SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_ESTIMATE="${SKIP_ESTIMATE:-0}"
SKIP_BENCH="${SKIP_BENCH:-0}"
SKIP_FITCAM="${SKIP_FITCAM:-0}"
SKIP_PLOT="${SKIP_PLOT:-0}"

DATA_PATH="${DATA_PATH:-$DATASETS_DIRECTORY/$DATA_FILE}"
QUERY_PATH="${QUERY_PATH:-$DATASETS_DIRECTORY/$QUERY_FILE}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

echo "[config] DATA_PATH=$DATA_PATH"
echo "[config] QUERY_PATH=$QUERY_PATH"
echo "[config] DATASET_TAG=$DATASET_TAG"
echo "[config] MEMORY_LIST=$MEMORY_LIST"
echo "[config] POLICIES=$POLICIES"
echo "[config] COLD_START_CORRECTION=$COLD_START_CORRECTION"
echo "[config] ESTIMATE_QUERY_FRACTION=$ESTIMATE_QUERY_FRACTION"

if [ "$SKIP_BUILD" != "1" ]; then
  cmake --build build --target pgm_bench pgm_cam_covariance
fi

if [ ! -x "$PGM_BENCH_BIN" ]; then
  echo "[error] pgm_bench is not executable: $PGM_BENCH_BIN" >&2
  exit 1
fi

if [ ! -x "$CAM_BIN" ]; then
  echo "[error] pgm_cam_covariance is not executable: $CAM_BIN" >&2
  exit 1
fi

if [ "$SKIP_ESTIMATE" != "1" ]; then
  for POLICY_NAME in $POLICIES; do
    EST_LOG="$LOG_DIR/${DATA_FILE}_${POLICY_NAME}.log"
    rm -f "$EST_LOG"
    echo "[estimate] policy=$POLICY_NAME -> $EST_LOG"
    "$PYTHON_BIN" - \
      "$DATASETS_DIRECTORY" \
      "$DATA_FILE" \
      "$QUERY_FILE" \
      "$EST_LOG" \
      "$MEMORY_LIST" \
      "$POLICY_NAME" \
      "$N_KEYS" \
      "$COLD_START_CORRECTION" \
      "$ESTIMATE_QUERY_FRACTION" <<'PY'
import sys

sys.path.insert(0, "utils")
import optimalEpsilon

(
    datasets_directory,
    data_file,
    query_file,
    log_path,
    memory_list,
    policy,
    n_keys,
    cold_start_raw,
    estimate_query_fraction,
) = sys.argv[1:10]

optimalEpsilon.DATASETS_DIRECTORY = datasets_directory.rstrip("/") + "/"
optimalEpsilon.BUDGET_MODE = "ESTIMATED"
estimate_query_fraction = float(estimate_query_fraction)
optimalEpsilon.LEARNING_QUERY_FRACTION = estimate_query_fraction
cold_start_correction = cold_start_raw == "1"
first_touch_scale = 1.0 if estimate_query_fraction >= 1.0 else 1.0 / max(estimate_query_fraction, 1e-12)

for m in [int(token) for token in memory_list.split()]:
    optimalEpsilon.getExpectedCostPerEpsilon(
        ipp=512,
        seg_size=16,
        M=m * 1024 * 1024,
        n=int(n_keys),
        ps=4096,
        data_file=data_file,
        query_file=query_file,
        s="all_in_once",
        cache_policy=policy,
        log_path=log_path,
        cold_start_correction=cold_start_correction,
        first_touch_scale=first_touch_scale,
    )
PY
  done
fi

if [ "$SKIP_BENCH" != "1" ]; then
  for M in $MEMORY_LIST; do
    BENCH_CSV="$LOG_DIR/${DATASET_TAG}_M${M}_bench.csv"
    echo "[bench] M=${M}MiB -> $BENCH_CSV"
    "$PGM_BENCH_BIN" \
      --data "$DATA_PATH" \
      --queries "$QUERY_PATH" \
      --keys "$N_KEYS" \
      --M "$M" \
      --strategies "$STRATEGY" \
      > "$BENCH_CSV"
  done
fi

if [ "$SKIP_FITCAM" != "1" ]; then
  echo "[fitCAM] root=$FITCAM_ROOT/$DATASET_TAG"
  PYTHON_BIN="$PYTHON_BIN" \
  CAM_BIN="$CAM_BIN" \
  DATASETS_DIRECTORY="$DATASETS_DIRECTORY" \
  DATA_FILE="$DATA_FILE" \
  QUERY_FILE="$QUERY_FILE" \
  DATASET_TAG="$DATASET_TAG" \
  N_KEYS="$N_KEYS" \
  EPS_LIST="$EPS_LIST" \
  TRAIN_M_LIST="$MEMORY_LIST" \
  POLICIES="$POLICIES" \
  ROOT_DIR="$FITCAM_ROOT/$DATASET_TAG" \
  COLD_START_CORRECTION="$COLD_START_CORRECTION" \
    bash exp/run_books10m_fitcam_q30.sh
fi

if [ "$SKIP_PLOT" != "1" ]; then
  estimate_paths=()
  bench_paths=()

  for POLICY_NAME in $POLICIES; do
    estimate_paths+=("$LOG_DIR/${DATA_FILE}_${POLICY_NAME}.log")
  done

  for M in $MEMORY_LIST; do
    bench_paths+=("$LOG_DIR/${DATASET_TAG}_M${M}_bench.csv")
  done

  echo "[plot] -> $OUTPUT_DIR"
  "$PYTHON_BIN" visualize/plot_epsilon_benchmarks.py \
    --estimate-paths "${estimate_paths[@]}" \
    --bench-paths "${bench_paths[@]}" \
    --fitcam-root "$FITCAM_ROOT" \
    --output-dir "$OUTPUT_DIR"
fi

echo "[done] estimate logs: $LOG_DIR/${DATA_FILE}_{POLICY}.log"
echo "[done] bench csvs:    $LOG_DIR/${DATASET_TAG}_M{M}_bench.csv"
echo "[done] fitCAM root:    $FITCAM_ROOT/$DATASET_TAG"
echo "[done] figures:        $OUTPUT_DIR"
