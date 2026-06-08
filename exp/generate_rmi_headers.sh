#!/usr/bin/env bash
set -euo pipefail

# Generate CDFShop/RMI C++ artifacts for the branch factors compiled by
# exp/rmi_bench.cpp. The default namespace pattern is:
#   books_rmi_linear_spline_linear_<BF>
#
# Main outputs:
#   src/rmi/rmi_eval/generated/<namespace>.h
#   src/rmi/rmi_eval/generated/<namespace>.cpp
#   src/rmi/rmi_eval/generated/<namespace>_data.h
#   src/rmi/rmi_data/<namespace>_L1_PARAMETERS
#
# Optional outputs when COLLECT_RECORDS=1:
#   src/rmi/rmi_eval/results/<namespace>.csv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RMI_REPO="${RMI_REPO:-src/rmi}"
RMI_EVAL_DIR="${RMI_EVAL_DIR:-src/rmi/rmi_eval}"
GENERATED_DIR="${GENERATED_DIR:-$RMI_EVAL_DIR/generated}"
RESULTS_DIR="${RESULTS_DIR:-$RMI_EVAL_DIR/results}"
RMI_DATA_DIR="${RMI_DATA_DIR:-$RMI_REPO/rmi_data}"
BUILD_DIR="${BUILD_DIR:-build}"

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${DATA_PATH:-src/rmi/dataset/books_200M_uint64_unique_fixed}}"
COLLECT_DATA_PATH="${COLLECT_DATA_PATH:-$TRAIN_DATA_PATH}"
QUERY_PATH="${QUERY_PATH:-/mnt/data/Dataset/public/SOSD/books_200M_uint64_unique.query.bin}"
COLLECT_DATA_HEADER="${COLLECT_DATA_HEADER:-yes}"

MODELS="${MODELS:-linear_spline,linear}"
NAMESPACE_PREFIX="${NAMESPACE_PREFIX:-books_rmi}"
BF_LIST="${BF_LIST:-64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152}"

THREADS="${THREADS:-0}"
ZERO_BUILD_TIME="${ZERO_BUILD_TIME:-0}"
COLLECT_RECORDS="${COLLECT_RECORDS:-1}"
QUERY_LIMIT="${QUERY_LIMIT:-0}"
BUILD_RMI_BENCH="${BUILD_RMI_BENCH:-1}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc 2>/dev/null || printf '4')}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_RMI_DATASET_MISMATCH="${ALLOW_RMI_DATASET_MISMATCH:-0}"

abspath() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$REPO_ROOT/$1" ;;
  esac
}

model_tag() {
  printf '%s' "$1" | sed 's/[,-]/_/g'
}

dataset_token_from_path() {
  local base token
  base="$(basename "$1")"
  base="$(printf '%s' "$base" | tr '[:upper:]' '[:lower:]')"
  token="${base%%_*}"
  printf '%s\n' "$token"
}

namespace_token() {
  local prefix
  prefix="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  prefix="${prefix%_rmi}"
  printf '%s\n' "${prefix%%_*}"
}

is_guarded_dataset_token() {
  case "$1" in
    books|fb) return 0 ;;
    *) return 1 ;;
  esac
}

validate_namespace_matches_training_data() {
  local ns_token train_token
  ns_token="$(namespace_token "$NAMESPACE_PREFIX")"
  train_token="$(dataset_token_from_path "$TRAIN_DATA_PATH")"

  if [ "$ns_token" = "$train_token" ]; then
    return 0
  fi
  if ! is_guarded_dataset_token "$ns_token" && ! is_guarded_dataset_token "$train_token"; then
    return 0
  fi
  if [ "$ALLOW_RMI_DATASET_MISMATCH" = "1" ]; then
    echo "[warn] allowing RMI namespace/training-data token mismatch: prefix=$NAMESPACE_PREFIX train=$TRAIN_DATA_PATH" >&2
    return 0
  fi

  echo "[error] RMI namespace prefix and training data look inconsistent." >&2
  echo "        NAMESPACE_PREFIX=$NAMESPACE_PREFIX (token=$ns_token)" >&2
  echo "        TRAIN_DATA_PATH=$TRAIN_DATA_PATH (token=$train_token)" >&2
  echo "        Set both values for the same dataset, or set ALLOW_RMI_DATASET_MISMATCH=1 to override." >&2
  exit 1
}

run_cmd() {
  printf '[cmd]'
  printf ' %q' "$@"
  printf '\n'
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  "$@"
}

write_wrapper() {
  local namespace="$1"
  local wrapper="$GENERATED_DIR_ABS/rmi_wrapper.h"
  echo "[wrapper] $wrapper -> $namespace"
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  {
    printf '#pragma once\n'
    printf '#include "%s.h"\n' "$namespace"
    printf 'namespace rmi_ns = %s;\n' "$namespace"
  } > "$wrapper"
}

write_artifact_meta() {
  local namespace="$1"
  local branch_factor="$2"
  local train_size train_mtime train_real collect_real generated_meta data_meta

  generated_meta="$GENERATED_DIR_ABS/$namespace.meta"
  data_meta="$RMI_DATA_DIR_ABS/$namespace.meta"
  echo "[meta] $generated_meta"
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi

  train_size="$(stat -c '%s' "$TRAIN_DATA_ABS" 2>/dev/null || printf '')"
  train_mtime="$(stat -c '%Y' "$TRAIN_DATA_ABS" 2>/dev/null || printf '')"
  train_real="$(readlink -f "$TRAIN_DATA_ABS" 2>/dev/null || printf '%s' "$TRAIN_DATA_ABS")"
  collect_real="$(readlink -f "$COLLECT_DATA_ABS" 2>/dev/null || printf '%s' "$COLLECT_DATA_ABS")"

  {
    printf 'artifact_format_version=1\n'
    printf 'namespace=%s\n' "$namespace"
    printf 'namespace_prefix=%s\n' "$NAMESPACE_PREFIX"
    printf 'namespace_token=%s\n' "$(namespace_token "$NAMESPACE_PREFIX")"
    printf 'models=%s\n' "$MODELS"
    printf 'model_tag=%s\n' "$(model_tag "$MODELS")"
    printf 'branch_factor=%s\n' "$branch_factor"
    printf 'train_data_path=%s\n' "$TRAIN_DATA_ABS"
    printf 'train_data_realpath=%s\n' "$train_real"
    printf 'train_data_token=%s\n' "$(dataset_token_from_path "$TRAIN_DATA_PATH")"
    printf 'train_data_size_bytes=%s\n' "$train_size"
    printf 'train_data_mtime_epoch=%s\n' "$train_mtime"
    printf 'collect_data_path=%s\n' "$COLLECT_DATA_ABS"
    printf 'collect_data_realpath=%s\n' "$collect_real"
    printf 'collect_data_header=%s\n' "$COLLECT_DATA_HEADER"
    printf 'query_path=%s\n' "$QUERY_ABS"
    printf 'rmi_data_dir=%s\n' "$RMI_DATA_DIR_ABS"
    printf 'generated_dir=%s\n' "$GENERATED_DIR_ABS"
    printf 'generated_at_epoch=%s\n' "$(date +%s)"
  } > "$generated_meta"
  cp "$generated_meta" "$data_meta"
}

train_one_rmi() {
  local branch_factor="$1"
  local tag namespace
  tag="$(model_tag "$MODELS")"
  namespace="${NAMESPACE_PREFIX}_${tag}_${branch_factor}"

  echo "[generate][rmi] namespace=$namespace models=$MODELS BF=$branch_factor"

  local cargo_cmd=(
    cargo run --release --
    "$TRAIN_DATA_ABS"
    "$namespace"
    "$MODELS"
    "$branch_factor"
    --data-path "$RMI_DATA_DIR_ABS"
  )
  if [ "$THREADS" -gt 0 ]; then
    cargo_cmd+=(--threads "$THREADS")
  fi
  if [ "$ZERO_BUILD_TIME" = "1" ]; then
    cargo_cmd+=(--zero-build-time)
  fi

  (
    cd "$RMI_REPO_ABS"
    run_cmd "${cargo_cmd[@]}"
  )

  local ext src dst
  for ext in .h .cpp _data.h; do
    src="$RMI_REPO_ABS/$namespace$ext"
    dst="$GENERATED_DIR_ABS/$namespace$ext"
    if [ "$DRY_RUN" != "1" ] && [ ! -f "$src" ]; then
      echo "[error] expected generated file missing: $src" >&2
      exit 1
    fi
    run_cmd mv "$src" "$dst"
  done
  write_artifact_meta "$namespace" "$branch_factor"

  if [ "$COLLECT_RECORDS" = "1" ]; then
    write_wrapper "$namespace"
    local collector_bin out_csv
    collector_bin="$RMI_EVAL_DIR_ABS/rmi_collector"
    out_csv="$RESULTS_DIR_ABS/$namespace.csv"
    run_cmd g++ -O3 -std=c++17 \
      "$RMI_EVAL_DIR_ABS/rmi_collector.cpp" \
      "$GENERATED_DIR_ABS/$namespace.cpp" \
      -I "$GENERATED_DIR_ABS" \
      -o "$collector_bin"

    local collector_cmd=(
      "$collector_bin"
      "$COLLECT_DATA_ABS"
      "$RMI_DATA_DIR_ABS"
      "$QUERY_ABS"
      "$out_csv"
    )
    case "$(printf '%s' "$COLLECT_DATA_HEADER" | tr '[:upper:]' '[:lower:]')" in
      no|false|0) collector_cmd+=(--no-header) ;;
      yes|true|1) ;;
      *) echo "[error] COLLECT_DATA_HEADER must be yes or no" >&2; exit 1 ;;
    esac
    if [ "$QUERY_LIMIT" -gt 0 ]; then
      collector_cmd+=(--query-limit "$QUERY_LIMIT")
    fi
    run_cmd "${collector_cmd[@]}"
  fi
}

RMI_REPO_ABS="$(abspath "$RMI_REPO")"
RMI_EVAL_DIR_ABS="$(abspath "$RMI_EVAL_DIR")"
GENERATED_DIR_ABS="$(abspath "$GENERATED_DIR")"
RESULTS_DIR_ABS="$(abspath "$RESULTS_DIR")"
RMI_DATA_DIR_ABS="$(abspath "$RMI_DATA_DIR")"
BUILD_DIR_ABS="$(abspath "$BUILD_DIR")"
TRAIN_DATA_ABS="$(abspath "$TRAIN_DATA_PATH")"
COLLECT_DATA_ABS="$(abspath "$COLLECT_DATA_PATH")"
QUERY_ABS="$(abspath "$QUERY_PATH")"

if [ "$NAMESPACE_PREFIX" != "books_rmi" ] || [ "$MODELS" != "linear_spline,linear" ]; then
  echo "[warn] exp/rmi_bench.cpp is currently wired to books_rmi_linear_spline_linear_<BF>."
  echo "[warn] Different NAMESPACE_PREFIX/MODELS can generate collector records, but rmi_bench will not use them until CMake/rmi_bench.cpp are updated."
fi

echo "[config] TRAIN_DATA_PATH=$TRAIN_DATA_ABS"
echo "[config] COLLECT_DATA_PATH=$COLLECT_DATA_ABS"
echo "[config] QUERY_PATH=$QUERY_ABS"
echo "[config] RMI_DATA_DIR=$RMI_DATA_DIR_ABS"
echo "[config] GENERATED_DIR=$GENERATED_DIR_ABS"
echo "[config] RESULTS_DIR=$RESULTS_DIR_ABS"
echo "[config] MODELS=$MODELS"
echo "[config] BF_LIST=$BF_LIST"
echo "[config] COLLECT_RECORDS=$COLLECT_RECORDS"
echo "[config] BUILD_RMI_BENCH=$BUILD_RMI_BENCH"
echo "[config] ALLOW_RMI_DATASET_MISMATCH=$ALLOW_RMI_DATASET_MISMATCH"

if [ "$DRY_RUN" != "1" ]; then
  mkdir -p "$GENERATED_DIR_ABS" "$RESULTS_DIR_ABS" "$RMI_DATA_DIR_ABS"
fi

validate_namespace_matches_training_data

for BF in $BF_LIST; do
  train_one_rmi "$BF"
done

if [ "$BUILD_RMI_BENCH" = "1" ]; then
  echo "[build] rmi_bench"
  run_cmd cmake --build "$BUILD_DIR_ABS" --target rmi_bench -j "$BUILD_JOBS"
fi

echo "[done] RMI artifacts generated."
