#!/bin/sh
set -eu

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/bin/python}"
CAM_BIN="${CAM_BIN:-./build/pgm_cam_covariance}"

DATASETS_DIRECTORY="${DATASETS_DIRECTORY:-/mnt/data/Dataset/public/SOSD}"
DATA_FILE="${DATA_FILE:-books_10M_uint64_unique}"
QUERY_FILE="${QUERY_FILE:-books_10M_uint64_unique.query.bin}"
DATA_PATH="${DATA_PATH:-$DATASETS_DIRECTORY/$DATA_FILE}"
QUERY_PATH="${QUERY_PATH:-$DATASETS_DIRECTORY/$QUERY_FILE}"

DATASET_TAG="${DATASET_TAG:-books_10M}"
N_KEYS="${N_KEYS:-10000000}"
EPS_LIST="${EPS_LIST:-8,10,12,14,16,20,24,32,64}"
TRAIN_M_LIST="${TRAIN_M_LIST:-10 20 40 60}"
# 5 10 15 20 25 30 35 40 45 50 55 60
if [ -n "${POLICY:-}" ]; then
  POLICIES="$POLICY"
else
  POLICIES="${POLICIES:-FIFO LRU LFU}"
fi
POLICIES_CSV=$(printf '%s\n' "$POLICIES" | tr ' ' ',' | tr -s ',')

ROOT_DIR="${ROOT_DIR:-build/log/fitcam_q30/$DATASET_TAG}"
REAL_DIR="$ROOT_DIR/real_summary"
EST_DIR="$ROOT_DIR/estimate"
FIT_DIR="$ROOT_DIR/fit_output"
COLD_START_CORRECTION="${COLD_START_CORRECTION:-1}"

FITCAM_COLD_FLAG=""
if [ "$COLD_START_CORRECTION" = "1" ]; then
  FITCAM_COLD_FLAG="--cold-start-correction"
fi

mkdir -p "$REAL_DIR" "$EST_DIR" "$FIT_DIR"

QUERY_BYTES=$(wc -c < "$QUERY_PATH")
TOTAL_QUERIES=$((QUERY_BYTES / 8))
QUERY_LIMIT=$((TOTAL_QUERIES * 30 / 100))

echo "[q30] DATA_PATH=$DATA_PATH"
echo "[q30] QUERY_PATH=$QUERY_PATH"
echo "[q30] TOTAL_QUERIES=$TOTAL_QUERIES"
echo "[q30] QUERY_LIMIT=$QUERY_LIMIT"
echo "[q30] ROOT_DIR=$ROOT_DIR"
echo "[q30] POLICIES=$POLICIES"
echo "[q30] COLD_START_CORRECTION=$COLD_START_CORRECTION"

for M in $TRAIN_M_LIST; do
  OUT_CSV="$REAL_DIR/${DATASET_TAG}_M${M}_q30_summary.csv"
  echo "[real] M=${M}MiB -> $OUT_CSV"
  "$CAM_BIN" \
    --data "$DATA_PATH" \
    --queries "$QUERY_PATH" \
    --keys "$N_KEYS" \
    --M "$M" \
    --epsilons "$EPS_LIST" \
    --policies "$POLICIES_CSV" \
    --budget-mode estimated \
    --query-limit "$QUERY_LIMIT" \
    --summary-out "$OUT_CSV"
done

for POLICY in $POLICIES; do
EST_LOG="$EST_DIR/${DATA_FILE}_${POLICY}_q30.log"
rm -f "$EST_LOG"

echo "[estimate][$POLICY] -> $EST_LOG"
"$PYTHON_BIN" - "$DATASETS_DIRECTORY" "$DATA_FILE" "$QUERY_FILE" "$EST_LOG" "$TRAIN_M_LIST" "$POLICY" "$COLD_START_CORRECTION" <<'PY'
import sys

sys.path.insert(0, "utils")
import optimalEpsilon

datasets_directory, data_file, query_file, log_path, train_m_list, policy, cold_start_raw = sys.argv[1:8]
optimalEpsilon.DATASETS_DIRECTORY = datasets_directory.rstrip("/") + "/"
cold_start_correction = cold_start_raw == "1"

memory_budgets = [int(token) for token in train_m_list.split()]

for m in memory_budgets:
    optimalEpsilon.getExpectedCostPerEpsilon(
        ipp=512,
        seg_size=16,
        M=m * 1024 * 1024,
        n=int(1e7),
        ps=4096,
        data_file=data_file,
        query_file=query_file,
        s="all_in_once",
        cache_policy=policy,
        log_path=log_path,
        cold_start_correction=cold_start_correction,
    )
PY

POLICY_FIT_DIR="$FIT_DIR/$POLICY"
PATCH_LOG="build/log/${DATA_FILE}_${POLICY}_revision.log"

echo "[fitCAM][$POLICY] -> $POLICY_FIT_DIR"
"$PYTHON_BIN" utils/fitcam_local_runner.py \
  --datasets-directory "$DATASETS_DIRECTORY" \
  --real-summary-dir "$REAL_DIR" \
  --real-summary-pattern "{dataset_tag}_M{M}_q30_summary.csv" \
  --estimate-log "$EST_LOG" \
  --apply-estimate-log "build/log/${DATA_FILE}_${POLICY}.log" \
  --apply-output-log "$PATCH_LOG" \
  --output-dir "$POLICY_FIT_DIR" \
  --dataset-tag "$DATASET_TAG" \
  --data-file "$DATA_FILE" \
  --query-file "$QUERY_FILE" \
  --policy "$POLICY" \
  --strategy all_in_once \
  --train-m $TRAIN_M_LIST \
  --n "$N_KEYS" \
  --seg-size 16 \
  --ipp 512 \
  --ps 4096 \
  --type sample \
  --mode point \
  --fetch-strategy all_in_once \
  $FITCAM_COLD_FLAG \
  --max-eps 64 \
  --eps0 0.0 \
  --ridge-lambda 1e-6 \
  --comparison-csv-name "${DATASET_TAG}_${POLICY}_q30_fitcam_corrected_vs_real.csv" \
  --coef-name "${DATASET_TAG}_${POLICY}_q30_fitcam_coef.txt"
done
