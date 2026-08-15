#!/usr/bin/env bash
set -euo pipefail

# Compare join execution modes over table workloads:
#   hybrid: use .par/.bitmap, point partitions call point query, range partitions call range query
#   point : externally sort the workload, then probe every key through the point query interface
#   range : externally sort the workload, then use one range probe from min(workload) to max(workload)
#   inlj  : unsorted workload point lookup baseline
#   hash  : build a tuple-per-entry hash table over outer, then scan/probe the inner dataset
#   sortmerge: externally sort the outer workload, then merge it with the sorted inner dataset
# Dataset loading and PGM construction are excluded from indexed-mode latency:
# the dataset and learned index are treated as a prebuilt inner access structure.
# Hybrid wall time includes runtime loading/validation of .par and .bitmap.
#
# By default this first generates w1-w6 workloads and hybrid partition metadata
# from the 200M-key inner dataset, then runs every generated outer size and
# writes one CSV per size. Set SKIP_GENERATE=1 to reuse existing workloads.
# Override QUERY_TAG_LIST and OUTER_SIZES together to run a subset, for example:
#   QUERY_TAG_LIST="10K 100K" OUTER_SIZES="10K:10000 100K:100000" \
#     bash exp/run_hybrid_join.sh
# To reuse those generated files:
#   SKIP_GENERATE=1 QUERY_TAG_LIST="10K 100K 1M" bash exp/run_hybrid_join.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
[ -f config.sh ] && source config.sh

BINARY="${BINARY:-./build/pgm_hybrid_join}"
DATASETS_DIRECTORY="${DATASETS_DIRECTORY:-$REPO_ROOT/data/datasets/SOSD}"
# The default inner relation remains the SOSD 200M-key dataset.
DATASET="${DATASET:-books_200M_uint64_unique}"
QUERY_TAG_LIST="${QUERY_TAG_LIST:-${QUERY_TAG:-1K 10K 100K 1M 10M 100M}}"
OUTER_SIZES="${OUTER_SIZES:-1K:1000 10K:10000 100K:100000 1M:1000000 10M:10000000 100M:100000000}"
TABLE_LIST="${TABLE_LIST:-1 2 3 4 5 6}"
MODE_LIST="${MODE_LIST:-hybrid point range inlj hash sortmerge}"
OUT_DIR="${OUT_DIR:-build/log/hybrid_join}"
# OUT_CSV is retained for single-size compatibility. For a multi-size run,
# each size always gets its own file under OUT_DIR to avoid mixing labels.
OUT_CSV="${OUT_CSV:-}"
WORK_DIR="${WORK_DIR:-build/tmp/hybrid_join_sort}"

# Empty means that pgm_hybrid_join detects the inner cardinality from the file.
NUM_KEYS="${NUM_KEYS:-}"
EPSILON="${EPSILON:-16}"
MEMORY_MIB="${MEMORY_MIB:-16}"
# Empty by default: pgm_hybrid_join derives a sort budget matching the hash
# table allocated for the current outer relation. Set SORT_MEMORY_MIB to
# explicitly override the automatic budget.
SORT_MEMORY_MIB="${SORT_MEMORY_MIB:-}"
POLICY="${POLICY:-LRU}"

if [ "${SKIP_GENERATE:-0}" != "1" ]; then
    echo "[*] Generating w1-w6 workloads from inner relation: ${DATASETS_DIRECTORY}/${DATASET}"
    DATASETS_DIRECTORY="$DATASETS_DIRECTORY" \
    DATASET="$DATASET" \
    OUTER_SIZES="$OUTER_SIZES" \
    EPSILON="$EPSILON" \
        bash "$SCRIPT_DIR/run_join_workloads.sh"
fi

mkdir -p "$OUT_DIR"

keys_args=()
if [ -n "$NUM_KEYS" ]; then
    keys_args+=(--keys "$NUM_KEYS")
fi

sort_args=()
if [ -n "$SORT_MEMORY_MIB" ]; then
    sort_args+=(--sort-M "$SORT_MEMORY_MIB")
fi

tag_count=0
for unused_tag in $QUERY_TAG_LIST; do
    tag_count=$((tag_count + 1))
done

for query_tag in $QUERY_TAG_LIST; do
    if [ "$tag_count" -eq 1 ] && [ -n "$OUT_CSV" ]; then
        tag_out_csv="$OUT_CSV"
    else
        tag_out_csv="${OUT_DIR}/${DATASET}_${query_tag}_join_compare.csv"
    fi
    mkdir -p "$(dirname "$tag_out_csv")"

    first=1
    echo "[*] outer size ${query_tag}; results -> $tag_out_csv"
    for table_id in $TABLE_LIST; do
        query_file="${DATASETS_DIRECTORY}/${DATASET}.${query_tag}table${table_id}.bin"
        par_file="${DATASETS_DIRECTORY}/${DATASET}.${query_tag}table${table_id}.par"
        bitmap_file="${DATASETS_DIRECTORY}/${DATASET}.${query_tag}table${table_id}.bitmap"

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

            echo "[*] ${query_tag} ${mode} join table${table_id}"
            "$BINARY" \
                --mode "$mode" \
                --data "${DATASETS_DIRECTORY}/${DATASET}" \
                --queries "$query_file" \
                "${mode_args[@]}" \
                --output "$tag_out_csv" \
                --label "table${table_id}" \
                --epsilon "$EPSILON" \
                --M "$MEMORY_MIB" \
                "${sort_args[@]}" \
                --work-dir "${WORK_DIR}/${query_tag}" \
                "${keys_args[@]}" \
                --policy "$POLICY" \
                "${append_args[@]}"

            first=0
        done
    done
done

echo "[+] Done. Results directory: $OUT_DIR"
