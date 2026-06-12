#!/usr/bin/env bash
set -euo pipefail

# Compare three join execution modes over books_200M table workloads:
#   hybrid: use .par/.bitmap, point partitions call point query, range partitions call range query
#   point : externally sort the workload, then probe every key through the point query interface
#   range : externally sort the workload, then use one range probe from min(workload) to max(workload)
#   inlj  : unsorted workload point lookup baseline
#
# QUERY_TAG=1M is about 8 MiB of uint64 keys. Keep SORT_MEMORY_MIB below
# that size (for example 0.5 or 1) if you want a real external-sort path
# instead of one in-memory initial run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BINARY="${BINARY:-./build/pgm_hybrid_join}"
DATASETS_DIRECTORY="${DATASETS_DIRECTORY:-/mnt/data/Dataset/public/SOSD}"
DATASET="${DATASET:-books_200M_uint64_unique}"
QUERY_TAG="${QUERY_TAG:-1M}"
TABLE_LIST="${TABLE_LIST:-1 2 3 4 5 6}"
MODE_LIST="${MODE_LIST:-hybrid point range inlj}"
OUT_CSV="${OUT_CSV:-build/log/hybrid_join/books_200M_join_compare.csv}"
WORK_DIR="${WORK_DIR:-build/tmp/hybrid_join_sort}"

NUM_KEYS="${NUM_KEYS:-200000000}"
EPSILON="${EPSILON:-16}"
MEMORY_MIB="${MEMORY_MIB:-16}"
SORT_MEMORY_MIB="${SORT_MEMORY_MIB:-0.5}"
POLICY="${POLICY:-LRU}"

mkdir -p "$(dirname "$OUT_CSV")"

first=1
for table_id in $TABLE_LIST; do
    query_file="${DATASETS_DIRECTORY}/${DATASET}.${QUERY_TAG}table${table_id}.bin"
    par_file="${DATASETS_DIRECTORY}/${DATASET}.${QUERY_TAG}table${table_id}.par"
    bitmap_file="${DATASETS_DIRECTORY}/${DATASET}.${QUERY_TAG}table${table_id}.bitmap"

    if [ ! -f "$query_file" ]; then
        echo "[error] missing input file: $query_file" >&2
        exit 1
    fi

    for mode in $MODE_LIST; do
        append_args=()
        if [ "$first" -eq 0 ]; then
            append_args+=(--append)
        fi

        mode_args=()
        if [ "$mode" = "hybrid" ]; then
            for path in "$par_file" "$bitmap_file"; do
                if [ ! -f "$path" ]; then
                    echo "[error] missing input file: $path" >&2
                    exit 1
                fi
            done
            mode_args+=(--par "$par_file" --bitmap "$bitmap_file")
        fi

        echo "[*] ${mode} join table${table_id} -> $OUT_CSV"
        "$BINARY" \
            --mode "$mode" \
            --data "${DATASETS_DIRECTORY}/${DATASET}" \
            --queries "$query_file" \
            "${mode_args[@]}" \
            --output "$OUT_CSV" \
            --label "table${table_id}" \
            --epsilon "$EPSILON" \
            --M "$MEMORY_MIB" \
            --sort-M "$SORT_MEMORY_MIB" \
            --work-dir "$WORK_DIR" \
            --keys "$NUM_KEYS" \
            --policy "$POLICY" \
            "${append_args[@]}"

        first=0
    done
done

echo "[+] Done. Results: $OUT_CSV"
