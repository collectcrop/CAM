#!/bin/sh
set -eu

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/bin/python}"
RMI_BIN="${RMI_BIN:-./build/rmi_bench}"

DATA_PATH="${DATA_PATH:-src/rmi/dataset/books_10M_uint64_unique_fixed}"
QUERY_PATH="${QUERY_PATH:-/mnt/data/Dataset/public/SOSD/books_10M_uint64_unique.query.bin}"
RMI_DATA_DIR="${RMI_DATA_DIR:-src/rmi/rmi_data}"
RMI_RESULTS_DIR="${RMI_RESULTS_DIR:-src/rmi/rmi_eval/results}"

DATASET_TAG="${DATASET_TAG:-books_10M}"
N_KEYS="${N_KEYS:-10000000}"
IPP="${IPP:-512}"
HEADER_MODE="${HEADER_MODE:-yes}"
STRATEGY="${STRATEGY:-all_in_once}"
TRAIN_M_LIST="${TRAIN_M_LIST:-4 8 16 24 32 48 64}"
RESULT_M_LIST="${RESULT_M_LIST:-8 16 32 64}"
BF_LIST="${BF_LIST:-64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152}"

if [ -n "${POLICY:-}" ]; then
  POLICIES="$POLICY"
else
  POLICIES="${POLICIES:-FIFO LRU LFU}"
fi
POLICIES_CSV=$(printf '%s\n' "$POLICIES" | tr ' ' ',' | tr -s ',')
BF_CSV=$(printf '%s\n' "$BF_LIST" | tr ' ' ',' | tr -s ',')

ESTIMATE_MODE="${ESTIMATE_MODE:-global}"
EPS_TRANSFORM="${EPS_TRANSFORM:-cap}"
EPS_TRANSFORM_Q="${EPS_TRANSFORM_Q:-0.9}"
ACTUAL_COLUMN="${ACTUAL_COLUMN:-avg_logical_ios}"
MIN_BF="${MIN_BF:-64}"
MAX_BF="${MAX_BF:-524288}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1e-6}"
PLOT_OUTPUT_DIR="${PLOT_OUTPUT_DIR:-data/outputs/figures/rmi_fitrmi/$DATASET_TAG}"
PLOT_PER_QUERY="${PLOT_PER_QUERY:-0}"

ROOT_DIR="${ROOT_DIR:-build/log/fitrmi_q30/$DATASET_TAG}"
REAL_DIR="$ROOT_DIR/real_summary"
RECORD_DIR="$ROOT_DIR/estimate_records"
EST_DIR="$ROOT_DIR/estimate"
NPZ_DIR="$ROOT_DIR/npz"
FIT_DIR="$ROOT_DIR/fit_output"

mkdir -p "$REAL_DIR" "$RECORD_DIR" "$EST_DIR" "$NPZ_DIR" "$FIT_DIR"

RUN_M_LIST=""
for M in $TRAIN_M_LIST $RESULT_M_LIST; do
  case " $RUN_M_LIST " in
    *" $M "*) ;;
    *) RUN_M_LIST="${RUN_M_LIST:+$RUN_M_LIST }$M" ;;
  esac
done

QUERY_BYTES=$(wc -c < "$QUERY_PATH")
TOTAL_QUERIES=$((QUERY_BYTES / 8))
QUERY_LIMIT=$((TOTAL_QUERIES * 30 / 100))

echo "[rmi-q30] DATA_PATH=$DATA_PATH"
echo "[rmi-q30] QUERY_PATH=$QUERY_PATH"
echo "[rmi-q30] TOTAL_QUERIES=$TOTAL_QUERIES"
echo "[rmi-q30] QUERY_LIMIT=$QUERY_LIMIT"
echo "[rmi-q30] ROOT_DIR=$ROOT_DIR"
echo "[rmi-q30] POLICIES=$POLICIES"
echo "[rmi-q30] TRAIN_M_LIST=$TRAIN_M_LIST"
echo "[rmi-q30] RESULT_M_LIST=$RESULT_M_LIST"
echo "[rmi-q30] RUN_M_LIST=$RUN_M_LIST"
echo "[rmi-q30] BF range=[$MIN_BF, $MAX_BF]"

for M in $RUN_M_LIST; do
  OUT_CSV="$REAL_DIR/${DATASET_TAG}_M${M}_rmi_q30_bench.csv"
  echo "[real][rmi] M=${M}MiB -> $OUT_CSV"
  "$RMI_BIN" \
    --data "$DATA_PATH" \
    --queries "$QUERY_PATH" \
    --rmi-data-dir "$RMI_DATA_DIR" \
    --keys "$N_KEYS" \
    --M "$M" \
    --header "$HEADER_MODE" \
    --strategies "$STRATEGY" \
    --policies "$POLICIES_CSV" \
    --branch-factors "$BF_CSV" \
    --query-limit "$QUERY_LIMIT" \
    > "$OUT_CSV"
done

for BF in $BF_LIST; do
  SRC_RECORD="$RMI_RESULTS_DIR/books_rmi_linear_spline_linear_${BF}.csv"
  DST_RECORD="$RECORD_DIR/books_rmi_linear_spline_linear_${BF}_q30.csv"
  echo "[records][rmi] BF=${BF} -> $DST_RECORD"
  "$PYTHON_BIN" utils/limit_rmi_records.py \
    --input "$SRC_RECORD" \
    --output "$DST_RECORD" \
    --limit "$QUERY_LIMIT"
done

for M in $RUN_M_LIST; do
  EST_LOG="$EST_DIR/${DATASET_TAG}_M${M}_rmi_q30_optimalBF_summary.log"
  rm -f "$EST_LOG"
  echo "[estimate][rmi] M=${M}MiB -> $EST_LOG"
  for BF in $BF_LIST; do
    RECORD="$RECORD_DIR/books_rmi_linear_spline_linear_${BF}_q30.csv"
    NPZ_OUT="$NPZ_DIR/books_rmi_linear_spline_linear_${BF}_M${M}_q30_optimalBF.npz"
    "$PYTHON_BIN" utils/optimalBF.py \
      "$RECORD" "$N_KEYS" \
      --ipp "$IPP" \
      --strategy "$STRATEGY" \
      --memory-mib "$M" \
      --policies "$POLICIES_CSV" \
      --header-mode branch_factor \
      --log-path "$EST_LOG" \
      --out "$NPZ_OUT" \
      --mode "$ESTIMATE_MODE" \
      --eps-transform "$EPS_TRANSFORM" \
      --eps-transform-q "$EPS_TRANSFORM_Q"
  done
done

for POLICY in $POLICIES; do
  POLICY_FIT_DIR="$FIT_DIR/$POLICY"
  echo "[fitRMI][$POLICY] -> $POLICY_FIT_DIR"
  "$PYTHON_BIN" utils/fitrmi_local_runner.py \
    --bench-dir "$REAL_DIR" \
    --estimate-dir "$EST_DIR" \
    --output-dir "$POLICY_FIT_DIR" \
    --dataset-tag "$DATASET_TAG" \
    --policy "$POLICY" \
    --strategy "$STRATEGY" \
    --train-m $TRAIN_M_LIST \
    --output-m $RESULT_M_LIST \
    --actual-column "$ACTUAL_COLUMN" \
    --min-bf "$MIN_BF" \
    --max-bf "$MAX_BF" \
    --ridge-lambda "$RIDGE_LAMBDA" \
    --comparison-csv-name "${DATASET_TAG}_${POLICY}_q30_fitrmi_corrected_vs_real.csv" \
    --coef-name "${DATASET_TAG}_${POLICY}_q30_fitrmi_coef.txt"

  PLOT_CSV="$POLICY_FIT_DIR/${DATASET_TAG}_${POLICY}_q30_fitrmi_corrected_vs_real.csv"
  PLOT_ARGS=""
  if [ "$PLOT_PER_QUERY" = "1" ]; then
    PLOT_ARGS="--per-query"
  fi
  "$PYTHON_BIN" utils/plot_rmi_fitrmi_compare.py \
    --comparison-csv "$PLOT_CSV" \
    --output-dir "$PLOT_OUTPUT_DIR" \
    --dataset-tag "$DATASET_TAG" \
    --policies "$POLICY" \
    --m-values $RESULT_M_LIST \
    --min-bf "$MIN_BF" \
    --max-bf "$MAX_BF" \
    $PLOT_ARGS
done
