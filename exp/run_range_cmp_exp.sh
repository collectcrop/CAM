#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/bin/python}"
SIM_BIN="${SIM_BIN:-./build/pgm_range_cache_simulate}"
WORKLOAD="${WORKLOAD:-w6}"
DATASETS_DIRECTORY="${DATASETS_DIRECTORY:-/mnt/data/Dataset/public/SOSD}"

OUTPUT_ROOT="${OUTPUT_ROOT:-build/log/range_cmp}"
ROOT_DIR="${ROOT_DIR:-$OUTPUT_ROOT/$WORKLOAD}"
WORKLOAD_DIR="${WORKLOAD_DIR:-$ROOT_DIR/workloads}"
PREFIX_DIR="${PREFIX_DIR:-$ROOT_DIR/prefixes}"
ACTUAL_DIR="${ACTUAL_DIR:-$ROOT_DIR/actual}"
REPLAY_DIR="${REPLAY_DIR:-$ROOT_DIR/replay}"
CAM_DIR="${CAM_DIR:-$ROOT_DIR/cam}"
SUMMARY_DIR="${SUMMARY_DIR:-$ROOT_DIR/summary}"

DATASETS="${DATASETS:-books_200M_uint64_unique fb_200M_uint64_unique wiki_ts_200M_uint64 osm_cellids_200M_uint64_unique}"
MEMORY_LIST="${MEMORY_LIST:-128}"
EPS_LIST="${EPS_LIST:-8,10,12,14,16,20,24,32,64}"
SAMPLE_RATES="${SAMPLE_RATES:-10 30 50 100}"
POLICY="${POLICY:-LRU}"
STRATEGY="${STRATEGY:-all_in_once}"
NUM_QUERIES="${NUM_QUERIES:-1000000}"
QUERY_LIMIT="${QUERY_LIMIT:-0}"
RANGE_MIN_LENGTH_KEYS="${RANGE_MIN_LENGTH_KEYS:-1}"
RANGE_MAX_LENGTH_KEYS="${RANGE_MAX_LENGTH_KEYS:-1024}"
COLD_START_CORRECTION="${COLD_START_CORRECTION:-1}"
CONSERVATIVE_RANGE_ESTIMATE="${CONSERVATIVE_RANGE_ESTIMATE:-1}"
ESTIMATE_WARMUP_REPEATS="${ESTIMATE_WARMUP_REPEATS:-1}"
ESTIMATE_TIMING_REPEATS="${ESTIMATE_TIMING_REPEATS:-3}"
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

echo "[config][range-cmp] ROOT_DIR=$ROOT_DIR"
echo "[config][range-cmp] DATASETS=${dataset_args[*]}"
echo "[config][range-cmp] WORKLOAD=$WORKLOAD"
echo "[config][range-cmp] MEMORY_LIST=${memory_args[*]}"
echo "[config][range-cmp] EPS_LIST=$EPS_LIST"
echo "[config][range-cmp] SAMPLE_RATES=${sample_rate_args[*]}"
echo "[config][range-cmp] POLICY=$POLICY STRATEGY=$STRATEGY"
echo "[config][range-cmp] NUM_QUERIES=$NUM_QUERIES QUERY_LIMIT=$QUERY_LIMIT"
echo "[config][range-cmp] RANGE_LENGTH_KEYS=[$RANGE_MIN_LENGTH_KEYS,$RANGE_MAX_LENGTH_KEYS] uniform"
echo "[config][range-cmp] COLD_START_CORRECTION=$COLD_START_CORRECTION CONSERVATIVE_RANGE_ESTIMATE=$CONSERVATIVE_RANGE_ESTIMATE"
echo "[config][range-cmp] ESTIMATE_WARMUP_REPEATS=$ESTIMATE_WARMUP_REPEATS ESTIMATE_TIMING_REPEATS=$ESTIMATE_TIMING_REPEATS"

if [ "$SKIP_BUILD" != "1" ]; then
  cmake --build build --target pgm_range_cache_simulate
fi

if [ ! -x "$SIM_BIN" ]; then
  echo "[error] range simulator is not executable: $SIM_BIN" >&2
  echo "        Build it first, e.g. cmake --build build --target pgm_range_cache_simulate" >&2
  exit 1
fi

if [ "$SKIP_GENERATE" != "1" ]; then
  gen_args=()
  if [ "$FORCE_WORKLOADS" = "1" ]; then
    gen_args+=(--force)
  fi
  "$PYTHON_BIN" exp/range_io_exp.py generate \
    --datasets-directory "$DATASETS_DIRECTORY" \
    --datasets "${dataset_args[@]}" \
    --workloads "$WORKLOAD" \
    --output-dir "$WORKLOAD_DIR" \
    --num-queries "$NUM_QUERIES" \
    --min-length-keys "$RANGE_MIN_LENGTH_KEYS" \
    --max-length-keys "$RANGE_MAX_LENGTH_KEYS" \
    --seed "$SEED" \
    "${gen_args[@]}"
fi

if [ "$SKIP_ACTUAL" != "1" ]; then
  for dataset in "${dataset_args[@]}"; do
    data_path="$DATASETS_DIRECTORY/$dataset"
    query_path="$WORKLOAD_DIR/$dataset/$dataset.$WORKLOAD.range.bin"
    if [ ! -f "$data_path" ]; then
      echo "[error] missing dataset: $data_path" >&2
      exit 1
    fi
    if [ ! -f "$query_path" ]; then
      echo "[error] missing range workload file: $query_path" >&2
      exit 1
    fi

    for M in "${memory_args[@]}"; do
      out_csv="$ACTUAL_DIR/$dataset/${dataset}_${WORKLOAD}_M${M}_${POLICY}_range_actual.csv"
      mkdir -p "$(dirname "$out_csv")"
      echo "[actual][range] dataset=$dataset workload=$WORKLOAD M=${M}MiB -> $out_csv"
      "$SIM_BIN" \
        --data "$data_path" \
        --queries "$query_path" \
        --M "$M" \
        --epsilons "$EPS_LIST" \
        --policies "$POLICY" \
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
  "$PYTHON_BIN" exp/range_cmp_exp.py make-prefixes \
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
      rate_label="$($PYTHON_BIN exp/range_cmp_exp.py sample-label "$rate")"
      query_path="$PREFIX_DIR/$dataset/$dataset.$WORKLOAD.p${rate_label}.range.bin"
      if [ ! -f "$query_path" ]; then
        echo "[error] missing sampled range workload file: $query_path" >&2
        exit 1
      fi

      for M in "${memory_args[@]}"; do
        out_csv="$REPLAY_DIR/$dataset/${dataset}_${WORKLOAD}_p${rate_label}_M${M}_${POLICY}_range_replay.csv"
        mkdir -p "$(dirname "$out_csv")"
        echo "[replay][range] dataset=$dataset workload=$WORKLOAD p${rate_label} M=${M}MiB -> $out_csv"
        "$SIM_BIN" \
          --data "$data_path" \
          --queries "$query_path" \
          --M "$M" \
          --epsilons "$EPS_LIST" \
          --policies "$POLICY" \
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
  if [ "$CONSERVATIVE_RANGE_ESTIMATE" != "1" ]; then
    cam_args+=(--non-conservative)
  fi
  cam_args+=(
    --warmup-repeats "$ESTIMATE_WARMUP_REPEATS"
    --timing-repeats "$ESTIMATE_TIMING_REPEATS"
  )

  "$PYTHON_BIN" exp/range_cmp_exp.py cam-estimate \
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
  "$PYTHON_BIN" exp/range_cmp_exp.py summarize \
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

echo "[done][range-cmp] prefix manifest: $PREFIX_DIR/prefix_manifest.csv"
echo "[done][range-cmp] CAM estimates:   $CAM_DIR/cam_estimates.csv"
echo "[done][range-cmp] detail:          $SUMMARY_DIR/range_cmp_detail.csv"
echo "[done][range-cmp] summary:         $SUMMARY_DIR/range_cmp_summary.csv"
