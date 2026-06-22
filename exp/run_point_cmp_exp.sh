#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
[ -f config.sh ] && source config.sh

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/bin/python}"
SIM_BIN="${SIM_BIN:-./build/pgm_cache_simulate}"
WORKLOAD="${WORKLOAD:-w1}"
DATASETS_DIRECTORY="${DATASETS_DIRECTORY:-/mnt/data/Dataset/public/SOSD}"

OUTPUT_ROOT="${OUTPUT_ROOT:-build/log/point_cmp}"
ROOT_DIR="${ROOT_DIR:-$OUTPUT_ROOT/$WORKLOAD}"
WORKLOAD_DIR="${WORKLOAD_DIR:-$ROOT_DIR/workloads}"
PREFIX_DIR="${PREFIX_DIR:-$ROOT_DIR/prefixes}"
ACTUAL_DIR="${ACTUAL_DIR:-$ROOT_DIR/actual}"
REPLAY_DIR="${REPLAY_DIR:-$ROOT_DIR/replay}"
CAM_DIR="${CAM_DIR:-$ROOT_DIR/cam}"
SUMMARY_DIR="${SUMMARY_DIR:-$ROOT_DIR/summary}"

DATASETS="${DATASETS:-books_200M_uint64_unique fb_200M_uint64_unique wiki_ts_200M_uint64 osm_cellids_200M_uint64_unique}"
MEMORY_LIST="${MEMORY_LIST:-64 96 128 160}"
EPS_LIST="${EPS_LIST:-8,10,12,14,16,20,24,32,64}"
SAMPLE_RATES="${SAMPLE_RATES:-0.1 1 5 10 20 30 50 80 100}"
POLICY="${POLICY:-LRU}"
STRATEGY="${STRATEGY:-all_in_once}"
NUM_QUERIES="${NUM_QUERIES:-1000000}"
QUERY_LIMIT="${QUERY_LIMIT:-0}"
COLD_START_CORRECTION="${COLD_START_CORRECTION:-1}"
ORDER_MODE="${ORDER_MODE:-global_shuffle}"
WINDOW_SIZE="${WINDOW_SIZE:-100000}"
WINDOW_RATIO_JITTER="${WINDOW_RATIO_JITTER:-0.3}"
SEED="${SEED:-42}"
FORCE_WORKLOADS="${FORCE_WORKLOADS:-0}"
FORCE_PREFIXES="${FORCE_PREFIXES:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_GENERATE="${SKIP_GENERATE:-0}"
SKIP_ACTUAL="${SKIP_ACTUAL:-0}"
SKIP_PREFIX="${SKIP_PREFIX:-0}"
SKIP_REPLAY="${SKIP_REPLAY:-0}"
SKIP_CAM="${SKIP_CAM:-0}"
SKIP_SUMMARIZE="${SKIP_SUMMARIZE:-0}"

mkdir -p "$ROOT_DIR" "$WORKLOAD_DIR" "$PREFIX_DIR" "$ACTUAL_DIR" "$REPLAY_DIR" "$CAM_DIR" "$SUMMARY_DIR"

dataset_args=($DATASETS)
memory_args=($MEMORY_LIST)
sample_rate_args=($SAMPLE_RATES)

echo "[config] ROOT_DIR=$ROOT_DIR"
echo "[config] DATASETS=${dataset_args[*]}"
echo "[config] WORKLOAD=$WORKLOAD"
echo "[config] MEMORY_LIST=${memory_args[*]}"
echo "[config] EPS_LIST=$EPS_LIST"
echo "[config] SAMPLE_RATES=${sample_rate_args[*]}"
echo "[config] POLICY=$POLICY STRATEGY=$STRATEGY"
echo "[config] NUM_QUERIES=$NUM_QUERIES QUERY_LIMIT=$QUERY_LIMIT"
echo "[config] COLD_START_CORRECTION=$COLD_START_CORRECTION"
echo "[config] ORDER_MODE=$ORDER_MODE WINDOW_SIZE=$WINDOW_SIZE WINDOW_RATIO_JITTER=$WINDOW_RATIO_JITTER"

if [ "$SKIP_BUILD" != "1" ]; then
  cmake --build build --target pgm_cache_simulate
fi

if [ ! -x "$SIM_BIN" ]; then
  echo "[error] simulator is not executable: $SIM_BIN" >&2
  echo "        Build it first, e.g. cmake --build build --target pgm_cache_simulate" >&2
  exit 1
fi

if [ "$SKIP_GENERATE" != "1" ]; then
  gen_args=()
  if [ "$FORCE_WORKLOADS" = "1" ]; then
    gen_args+=(--force)
  fi
  "$PYTHON_BIN" exp/point_io_exp.py generate \
    --datasets-directory "$DATASETS_DIRECTORY" \
    --datasets "${dataset_args[@]}" \
    --workloads "$WORKLOAD" \
    --output-dir "$WORKLOAD_DIR" \
    --num-queries "$NUM_QUERIES" \
    --seed "$SEED" \
    --order-mode "$ORDER_MODE" \
    --window-size "$WINDOW_SIZE" \
    --window-ratio-jitter "$WINDOW_RATIO_JITTER" \
    "${gen_args[@]}"
fi

if [ "$SKIP_ACTUAL" != "1" ]; then
  for dataset in "${dataset_args[@]}"; do
    data_path="$DATASETS_DIRECTORY/$dataset"
    query_path="$WORKLOAD_DIR/$dataset/$dataset.$WORKLOAD.bin"
    if [ ! -f "$data_path" ]; then
      echo "[error] missing dataset: $data_path" >&2
      exit 1
    fi
    if [ ! -f "$query_path" ]; then
      echo "[error] missing workload file: $query_path" >&2
      exit 1
    fi

    for M in "${memory_args[@]}"; do
      out_csv="$ACTUAL_DIR/$dataset/${dataset}_${WORKLOAD}_M${M}_${POLICY}_actual.csv"
      mkdir -p "$(dirname "$out_csv")"
      echo "[actual] dataset=$dataset workload=$WORKLOAD M=${M}MiB -> $out_csv"
      "$SIM_BIN" \
        --data "$data_path" \
        --queries "$query_path" \
        --M "$M" \
        --epsilons "$EPS_LIST" \
        --policies "$POLICY" \
        --strategies "$STRATEGY" \
        --budget-mode estimated \
        --query-limit "$QUERY_LIMIT" \
        --summary-out "$out_csv"
    done
  done
fi

if [ "$SKIP_PREFIX" != "1" ]; then
  prefix_args=()
  if [ "$FORCE_PREFIXES" = "1" ]; then
    prefix_args+=(--force)
  fi
  "$PYTHON_BIN" exp/point_cmp_exp.py make-prefixes \
    --datasets-directory "$DATASETS_DIRECTORY" \
    --datasets "${dataset_args[@]}" \
    --workload "$WORKLOAD" \
    --workload-dir "$WORKLOAD_DIR" \
    --output-dir "$PREFIX_DIR" \
    --sample-rates "${sample_rate_args[@]}" \
    --query-limit "$QUERY_LIMIT" \
    "${prefix_args[@]}"
fi

if [ "$SKIP_REPLAY" != "1" ]; then
  for dataset in "${dataset_args[@]}"; do
    data_path="$DATASETS_DIRECTORY/$dataset"
    if [ ! -f "$data_path" ]; then
      echo "[error] missing dataset: $data_path" >&2
      exit 1
    fi

    for rate in "${sample_rate_args[@]}"; do
      rate_label="$($PYTHON_BIN exp/point_cmp_exp.py sample-label "$rate")"
      query_path="$PREFIX_DIR/$dataset/$dataset.$WORKLOAD.p${rate_label}.bin"
      if [ ! -f "$query_path" ]; then
        echo "[error] missing sampled workload file: $query_path" >&2
        exit 1
      fi

      for M in "${memory_args[@]}"; do
        out_csv="$REPLAY_DIR/$dataset/${dataset}_${WORKLOAD}_p${rate_label}_M${M}_${POLICY}_replay.csv"
        mkdir -p "$(dirname "$out_csv")"
        echo "[replay] dataset=$dataset workload=$WORKLOAD p${rate_label} M=${M}MiB -> $out_csv"
        "$SIM_BIN" \
          --data "$data_path" \
          --queries "$query_path" \
          --M "$M" \
          --epsilons "$EPS_LIST" \
          --policies "$POLICY" \
          --strategies "$STRATEGY" \
          --budget-mode estimated \
          --summary-out "$out_csv"
      done
    done
  done
fi

if [ "$SKIP_CAM" != "1" ]; then
  cam_args=()
  if [ "$COLD_START_CORRECTION" = "1" ]; then
    cam_args+=(--cold-start-correction)
  fi
  "$PYTHON_BIN" exp/point_cmp_exp.py cam-estimate \
    --datasets-directory "$DATASETS_DIRECTORY" \
    --datasets "${dataset_args[@]}" \
    --workload "$WORKLOAD" \
    --workload-dir "$WORKLOAD_DIR" \
    --output-csv "$CAM_DIR/cam_estimates.csv" \
    --memory-list "${memory_args[@]}" \
    --epsilons "$EPS_LIST" \
    --sample-rates "${sample_rate_args[@]}" \
    --policy "$POLICY" \
    --strategy "$STRATEGY" \
    --query-limit "$QUERY_LIMIT" \
    "${cam_args[@]}"
fi

if [ "$SKIP_SUMMARIZE" != "1" ]; then
  "$PYTHON_BIN" exp/point_cmp_exp.py summarize \
    --datasets-directory "$DATASETS_DIRECTORY" \
    --datasets "${dataset_args[@]}" \
    --workload "$WORKLOAD" \
    --actual-dir "$ACTUAL_DIR" \
    --replay-dir "$REPLAY_DIR" \
    --cam-csv "$CAM_DIR/cam_estimates.csv" \
    --output-dir "$SUMMARY_DIR" \
    --memory-list "${memory_args[@]}" \
    --epsilons "$EPS_LIST" \
    --sample-rates "${sample_rate_args[@]}" \
    --policy "$POLICY" \
    --strategy "$STRATEGY"
fi

echo "[done] prefix manifest: $PREFIX_DIR/prefix_manifest.csv"
echo "[done] CAM estimates:   $CAM_DIR/cam_estimates.csv"
echo "[done] detail:          $SUMMARY_DIR/point_cmp_detail.csv"
echo "[done] summary:         $SUMMARY_DIR/point_cmp_summary.csv"
