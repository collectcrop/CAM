#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $0 [workload]

Run range-cmp no-cache actual measurements only.

Examples:
  $0 w4
  $0 4
  WORKLOAD=w5 $0

By default this reuses existing range workloads from:
  build/log/range_cmp/<workload>/workloads
and writes actual outputs to:
  build/log/range_cmp/<workload>/wocache/actual

If the workload files do not exist yet, run with SKIP_GENERATE=0.
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ "$#" -gt 1 ]; then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

requested_workload="${1:-${WORKLOAD:-w4}}"
if [[ "$requested_workload" =~ ^[0-9]+$ ]]; then
  requested_workload="w${requested_workload}"
fi
if [[ ! "$requested_workload" =~ ^w[0-9]+$ ]]; then
  echo "[error] workload should look like w4, w5, ...; got: $requested_workload" >&2
  exit 2
fi

export WORKLOAD="$requested_workload"
export OUTPUT_ROOT="${OUTPUT_ROOT:-build/log/range_cmp}"
export ROOT_DIR="${ROOT_DIR:-$OUTPUT_ROOT/$WORKLOAD/wocache}"
export WORKLOAD_DIR="${WORKLOAD_DIR:-$OUTPUT_ROOT/$WORKLOAD/workloads}"
export ACTUAL_DIR="${ACTUAL_DIR:-$ROOT_DIR/actual}"

export POLICY="NONE"

export SKIP_GENERATE="${SKIP_GENERATE:-1}"
export SKIP_ACTUAL="${SKIP_ACTUAL:-0}"
export SKIP_PREFIX="${SKIP_PREFIX:-1}"
export SKIP_REPLAY="${SKIP_REPLAY:-1}"
export SKIP_CAM="${SKIP_CAM:-1}"
export SKIP_SUMMARIZE="${SKIP_SUMMARIZE:-1}"

echo "[wocache][range] WORKLOAD=$WORKLOAD"
echo "[wocache][range] WORKLOAD_DIR=$WORKLOAD_DIR"
echo "[wocache][range] ACTUAL_DIR=$ACTUAL_DIR"
echo "[wocache][range] POLICY=$POLICY"
echo "[wocache][range] SKIP_GENERATE=$SKIP_GENERATE SKIP_ACTUAL=$SKIP_ACTUAL"
echo "[wocache][range] SKIP_PREFIX=$SKIP_PREFIX SKIP_REPLAY=$SKIP_REPLAY SKIP_CAM=$SKIP_CAM SKIP_SUMMARIZE=$SKIP_SUMMARIZE"

bash "$SCRIPT_DIR/run_range_cmp_exp.sh"
