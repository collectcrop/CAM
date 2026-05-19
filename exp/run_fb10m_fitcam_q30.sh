#!/bin/sh
set -eu

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/bin/python}"
CAM_BIN="${CAM_BIN:-./build/pgm_range_bench}"

DATASETS_DIRECTORY="${DATASETS_DIRECTORY:-/mnt/backup_disk/Dataset/public/SOSD}"
DATA_FILE="${DATA_FILE:-fb_10M_uint64_unique}"
QUERY_FILE="${QUERY_FILE:-fb_10M_uint64_unique.range.bin}"
DATA_PATH="${DATA_PATH:-$DATASETS_DIRECTORY/$DATA_FILE}"
QUERY_PATH="${QUERY_PATH:-$DATASETS_DIRECTORY/$QUERY_FILE}"

DATASET_TAG="${DATASET_TAG:-fb_10M}"
N_KEYS="${N_KEYS:-10000000}"
EPS_LIST="${EPS_LIST:-8,10,12,14,16,20,24,32,64}"
TRAIN_M_LIST="${TRAIN_M_LIST:-10 20 40 60}"

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
QUERY_PREFIX_DIR="$ROOT_DIR/query_prefix"

mkdir -p "$REAL_DIR" "$EST_DIR" "$FIT_DIR" "$QUERY_PREFIX_DIR"

QUERY_BYTES=$(wc -c < "$QUERY_PATH")
RANGE_QUERY_BYTES=16
if [ $((QUERY_BYTES % RANGE_QUERY_BYTES)) -ne 0 ]; then
  echo "[error] range query file size is not a multiple of ${RANGE_QUERY_BYTES} bytes: $QUERY_PATH" >&2
  exit 1
fi
TOTAL_QUERIES=$((QUERY_BYTES / RANGE_QUERY_BYTES))
QUERY_LIMIT=$((TOTAL_QUERIES * 30 / 100))
if [ "$TOTAL_QUERIES" -gt 0 ] && [ "$QUERY_LIMIT" -lt 1 ]; then
  QUERY_LIMIT=1
fi
Q30_QUERY_PATH="$QUERY_PREFIX_DIR/${QUERY_FILE%.bin}.q30.bin"
# dd if="$QUERY_PATH" of="$Q30_QUERY_PATH" bs="$RANGE_QUERY_BYTES" count="$QUERY_LIMIT" 2>/dev/null

echo "[q30] DATA_PATH=$DATA_PATH"
echo "[q30] QUERY_PATH=$QUERY_PATH"
echo "[q30] Q30_QUERY_PATH=$Q30_QUERY_PATH"
echo "[q30] TOTAL_QUERIES=$TOTAL_QUERIES"
echo "[q30] QUERY_LIMIT=$QUERY_LIMIT"
echo "[q30] ROOT_DIR=$ROOT_DIR"
echo "[q30] POLICIES=$POLICIES"

# for M in $TRAIN_M_LIST; do
#   OUT_CSV="$REAL_DIR/${DATASET_TAG}_M${M}_q30_summary.csv"
#   RAW_CSV="$REAL_DIR/${DATASET_TAG}_M${M}_q30_summary.raw.csv"
#   echo "[real][range] M=${M}MiB -> $OUT_CSV"
#   "$CAM_BIN" \
#     --data "$DATA_PATH" \
#     --queries "$Q30_QUERY_PATH" \
#     --keys "$N_KEYS" \
#     --M "$M" \
#     --policies "$POLICIES_CSV" \
#     > "$RAW_CSV"

#   "$PYTHON_BIN" - "$RAW_CSV" "$OUT_CSV" "$EPS_LIST" <<'PY'
# import csv
# import sys

# raw_path, out_path, eps_list = sys.argv[1:4]
# allowed_eps = {int(token) for token in eps_list.split(",") if token.strip()}

# with open(raw_path, newline="") as src:
#     reader = csv.DictReader(src)
#     rows = [row for row in reader if int(row["epsilon"]) in allowed_eps]
#     fieldnames = reader.fieldnames

# if not rows:
#     raise SystemExit(f"no rows remained after filtering {raw_path} by eps={sorted(allowed_eps)}")

# with open(out_path, "w", newline="") as dst:
#     writer = csv.DictWriter(dst, fieldnames=fieldnames)
#     writer.writeheader()
#     writer.writerows(rows)
# PY
# done

for POLICY in $POLICIES; do
EST_LOG="$EST_DIR/${DATA_FILE}_${POLICY}_q30.log"
# rm -f "$EST_LOG"

# echo "[estimate][$POLICY] -> $EST_LOG"
# "$PYTHON_BIN" - "$DATASETS_DIRECTORY" "$DATA_FILE" "$QUERY_FILE" "$EST_LOG" "$TRAIN_M_LIST" "$POLICY" "$N_KEYS" <<'PY'
# import sys

# sys.path.insert(0, "utils")
# import optimalEpsilon

# datasets_directory, data_file, query_file, log_path, train_m_list, policy, n_keys = sys.argv[1:8]
# optimalEpsilon.DATASETS_DIRECTORY = datasets_directory.rstrip("/") + "/"
# optimalEpsilon.LEARNING_QUERY_FRACTION = 0.3

# memory_budgets = [int(token) for token in train_m_list.split()]

# for m in memory_budgets:
#     optimalEpsilon.getExpectedRangeCostPerEpsilon(
#         ipp=512,
#         seg_size=16,
#         M=m * 1024 * 1024,
#         n=int(n_keys),
#         ps=4096,
#         data_file=data_file,
#         query_file=query_file,
#         cache_policy=policy,
#         log_path=log_path,
#     )
# PY

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
  --mode range \
  --fetch-strategy all_in_once \
  --max-eps 64 \
  --eps0 0.0 \
  --ridge-lambda 1e-6 \
  --comparison-csv-name "${DATASET_TAG}_${POLICY}_q30_fitcam_corrected_vs_real.csv" \
  --coef-name "${DATASET_TAG}_${POLICY}_q30_fitcam_coef.txt"
done
