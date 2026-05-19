#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/bin/python}"
SIM_BIN="${SIM_BIN:-./build/pgm_cache_simulate}"
DATASETS_DIRECTORY="${DATASETS_DIRECTORY:-/mnt/data/Dataset/public/SOSD}"

ROOT_DIR="${ROOT_DIR:-build/log/exp}"
WORKLOAD_DIR="${WORKLOAD_DIR:-$ROOT_DIR/workloads}"
ACTUAL_DIR="${ACTUAL_DIR:-$ROOT_DIR/actual}"
ESTIMATE_DIR="${ESTIMATE_DIR:-$ROOT_DIR/estimate}"
SUMMARY_DIR="${SUMMARY_DIR:-$ROOT_DIR/summary}"
FIGURE_DIR="${FIGURE_DIR:-$ROOT_DIR/figures}"

DATASETS="${DATASETS:-books_200M_uint64_unique fb_200M_uint64_unique wiki_ts_200M_uint64_unique osm_cellids_200M_uint64_unique}"
WORKLOADS="${WORKLOADS:-w1 w2 w3 w4 w5 w6}"
MEMORY_LIST="${MEMORY_LIST:-64 96 128 160}"
EPS_LIST="${EPS_LIST:-8,10,12,14,16,20,24,32,64}"
POLICY="${POLICY:-LRU}"
STRATEGY="${STRATEGY:-all_in_once}"
NUM_QUERIES="${NUM_QUERIES:-1000000}"
QUERY_LIMIT="${QUERY_LIMIT:-0}"
LEARNING_FRACTION="${LEARNING_FRACTION:-0.3}"
COLD_START_CORRECTION="${COLD_START_CORRECTION:-1}"
ACTUAL_WARMUP_PAGES="${ACTUAL_WARMUP_PAGES:-0}"
ACTUAL_WARMUP_SEED="${ACTUAL_WARMUP_SEED:-42}"
SEED="${SEED:-42}"
FORCE_WORKLOADS="${FORCE_WORKLOADS:-0}"
SKIP_GENERATE="${SKIP_GENERATE:-0}"
SKIP_ACTUAL="${SKIP_ACTUAL:-0}"
SKIP_ESTIMATE="${SKIP_ESTIMATE:-0}"
SKIP_SUMMARIZE="${SKIP_SUMMARIZE:-0}"
SKIP_PLOT="${SKIP_PLOT:-0}"

mkdir -p "$ROOT_DIR" "$WORKLOAD_DIR" "$ACTUAL_DIR" "$ESTIMATE_DIR" "$SUMMARY_DIR" "$FIGURE_DIR"

if [ ! -x "$SIM_BIN" ]; then
  echo "[error] simulator is not executable: $SIM_BIN" >&2
  echo "        Build it first, e.g. cmake --build build --target pgm_cache_simulate" >&2
  exit 1
fi

dataset_args=($DATASETS)
workload_args=($WORKLOADS)
memory_args=($MEMORY_LIST)

echo "[config] ROOT_DIR=$ROOT_DIR"
echo "[config] DATASETS=${dataset_args[*]}"
echo "[config] WORKLOADS=${workload_args[*]}"
echo "[config] MEMORY_LIST=${memory_args[*]}"
echo "[config] EPS_LIST=$EPS_LIST"
echo "[config] POLICY=$POLICY STRATEGY=$STRATEGY"
echo "[config] NUM_QUERIES=$NUM_QUERIES QUERY_LIMIT=$QUERY_LIMIT LEARNING_FRACTION=$LEARNING_FRACTION"
echo "[config] COLD_START_CORRECTION=$COLD_START_CORRECTION"
echo "[config] ACTUAL_WARMUP_PAGES=$ACTUAL_WARMUP_PAGES ACTUAL_WARMUP_SEED=$ACTUAL_WARMUP_SEED"

if [ "$SKIP_GENERATE" != "1" ]; then
  gen_args=()
  if [ "$FORCE_WORKLOADS" = "1" ]; then
    gen_args+=(--force)
  fi
  "$PYTHON_BIN" exp/point_io_exp.py generate \
    --datasets-directory "$DATASETS_DIRECTORY" \
    --datasets "${dataset_args[@]}" \
    --workloads "${workload_args[@]}" \
    --output-dir "$WORKLOAD_DIR" \
    --num-queries "$NUM_QUERIES" \
    --seed "$SEED" \
    "${gen_args[@]}"
fi

if [ "$SKIP_ACTUAL" != "1" ]; then
  for dataset in "${dataset_args[@]}"; do
    data_path="$DATASETS_DIRECTORY/$dataset"
    if [ ! -f "$data_path" ]; then
      echo "[error] missing dataset: $data_path" >&2
      exit 1
    fi

    for workload in "${workload_args[@]}"; do
      query_path="$WORKLOAD_DIR/$dataset/$dataset.$workload.bin"
      if [ ! -f "$query_path" ]; then
        echo "[error] missing workload file: $query_path" >&2
        exit 1
      fi

      for M in "${memory_args[@]}"; do
        out_csv="$ACTUAL_DIR/$dataset/${dataset}_${workload}_M${M}_${POLICY}_actual.csv"
        mkdir -p "$(dirname "$out_csv")"
        actual_args=()
        if [ "$ACTUAL_WARMUP_PAGES" != "0" ]; then
          actual_args+=(--warmup-pages "$ACTUAL_WARMUP_PAGES" --warmup-seed "$ACTUAL_WARMUP_SEED")
        fi
        echo "[actual] dataset=$dataset workload=$workload M=${M}MiB -> $out_csv"
        "$SIM_BIN" \
          --data "$data_path" \
          --queries "$query_path" \
          --M "$M" \
          --epsilons "$EPS_LIST" \
          --policies "$POLICY" \
          --strategies "$STRATEGY" \
          --budget-mode estimated \
          --query-limit "$QUERY_LIMIT" \
          --summary-out "$out_csv" \
          "${actual_args[@]}"
      done
    done
  done
fi

if [ "$SKIP_ESTIMATE" != "1" ]; then
  estimate_args=()
  if [ "$COLD_START_CORRECTION" = "1" ]; then
    estimate_args+=(--cold-start-correction)
  fi

  "$PYTHON_BIN" exp/point_io_exp.py estimate \
    --datasets-directory "$DATASETS_DIRECTORY" \
    --datasets "${dataset_args[@]}" \
    --workloads "${workload_args[@]}" \
    --workload-dir "$WORKLOAD_DIR" \
    --output-dir "$ESTIMATE_DIR" \
    --memory-list "${memory_args[@]}" \
    --epsilons "$EPS_LIST" \
    --policy "$POLICY" \
    --strategy "$STRATEGY" \
    --query-limit "$QUERY_LIMIT" \
    --learning-fraction "$LEARNING_FRACTION" \
    "${estimate_args[@]}"
fi

if [ "$SKIP_SUMMARIZE" != "1" ]; then
  "$PYTHON_BIN" exp/point_io_exp.py summarize \
    --datasets-directory "$DATASETS_DIRECTORY" \
    --datasets "${dataset_args[@]}" \
    --workloads "${workload_args[@]}" \
    --actual-dir "$ACTUAL_DIR" \
    --estimate-dir "$ESTIMATE_DIR" \
    --output-dir "$SUMMARY_DIR" \
    --memory-list "${memory_args[@]}" \
    --epsilons "$EPS_LIST" \
    --policy "$POLICY"
fi

if [ "$SKIP_PLOT" != "1" ]; then
  "$PYTHON_BIN" exp/plot_point_io_exp.py \
    --summary-csv "$SUMMARY_DIR/point_io_accuracy_summary.csv" \
    --output-dir "$FIGURE_DIR" \
    --formats pdf
fi

echo "[done] summary: $SUMMARY_DIR/point_io_accuracy_summary.csv"
echo "[done] merged:  $SUMMARY_DIR/point_io_merged.csv"
echo "[done] figures: $FIGURE_DIR"
