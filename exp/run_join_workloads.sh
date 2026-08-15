#!/usr/bin/env bash
set -euo pipefail

# Generate join workloads and hybrid partitions for six workload mixtures (w1-w6)
# at each configured outer-relation size.
# Output files are placed alongside the dataset:
#   ${DATASET}.10Ktable1.bin/.par/.bitmap
#   ${DATASET}.100Ktable1.bin/.par/.bitmap
#   ${DATASET}.1Mtable1.bin/.par/.bitmap
#   ...
#   ${DATASET}.100Mtable6.bin/.par/.bitmap

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
[ -f config.sh ] && source config.sh

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASETS_DIRECTORY="${DATASETS_DIRECTORY:-$REPO_ROOT/data/datasets/SOSD}"
DATASET="${DATASET:-books_200M_uint64_unique}"
# Space-separated TAG:COUNT entries. Override this variable to generate a subset,
# e.g. OUTER_SIZES="100K:100000 1M:1000000".
OUTER_SIZES="${OUTER_SIZES:-10K:10000 100K:100000 1M:1000000 10M:10000000 50M:50000000 100M:100000000}"
SEED="${SEED:-42}"

N_MIN="${N_MIN:-${WINDOW_SIZE:-1024}}"
K_MAX="${K_MAX:-8192}"
PAGE_SIZE="${PAGE_SIZE:-4096}"
KEY_SIZE="${KEY_SIZE:-8}"
EPSILON="${EPSILON:-16}"
GAMMA="${GAMMA:-0.05}"
PHI="${PHI:-0.0}"

ALPHA="${ALPHA:-1.637e-06}"
BETA="${BETA:-1.719e-06}"
ETA="${ETA:-4.421e-06}"
DELTA="${DELTA:-0.005}"
LAMBDA_POINT="${LAMBDA_POINT:-1.195e-06}"
LAMBDA_RANGE="${LAMBDA_RANGE:-4.669e-07}"

NUM_HOTSPOTS="${NUM_HOTSPOTS:-5}"
HOTSPOT_FRAC="${HOTSPOT_FRAC:-0.01}"
HOTSPOT_ZIPF_A="${HOTSPOT_ZIPF_A:-1.5}"
ZIPF_A="${ZIPF_A:-1.2}"
GENERATOR_BATCH_SIZE="${GENERATOR_BATCH_SIZE:-4000000}"

"$PYTHON_BIN" - \
  "$DATASETS_DIRECTORY" "$DATASET" "$OUTER_SIZES" "$SEED" \
  "$N_MIN" "$K_MAX" "$PAGE_SIZE" "$KEY_SIZE" "$EPSILON" "$GAMMA" "$PHI" \
  "$ALPHA" "$BETA" "$ETA" "$DELTA" "$LAMBDA_POINT" "$LAMBDA_RANGE" \
  "$NUM_HOTSPOTS" "$HOTSPOT_FRAC" "$HOTSPOT_ZIPF_A" "$ZIPF_A" \
  "$GENERATOR_BATCH_SIZE" <<'PY'
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd() / "utils"))
from generate_query import join_partition, sample_mixture_with_replacement

(
    datasets_directory,
    dataset,
    outer_sizes_s,
    seed_s,
    n_min_s,
    k_max_s,
    page_size_s,
    key_size_s,
    epsilon_s,
    gamma_s,
    phi_s,
    alpha_s,
    beta_s,
    eta_s,
    delta_s,
    lambda_point_s,
    lambda_range_s,
    num_hotspots_s,
    hotspot_frac_s,
    hotspot_zipf_a_s,
    zipf_a_s,
    generator_batch_size_s,
) = sys.argv[1:]

data_dir = Path(datasets_directory)
data_path = data_dir / dataset
if not data_path.exists():
    raise FileNotFoundError(f"dataset not found: {data_path}")

seed = int(seed_s)
n_min = int(n_min_s)
k_max = int(k_max_s)
page_size = int(page_size_s)
key_size = int(key_size_s)
epsilon = int(epsilon_s)
gamma = float(gamma_s)
phi = float(phi_s)

alpha = float(alpha_s)
beta = float(beta_s)
eta = float(eta_s)
delta = float(delta_s)
lambda_point = float(lambda_point_s)
lambda_range = float(lambda_range_s)

num_hotspots = int(num_hotspots_s)
hotspot_frac = float(hotspot_frac_s)
hotspot_zipf_a = float(hotspot_zipf_a_s)
zipf_a = float(zipf_a_s)
generator_batch_size = int(generator_batch_size_s)

keys = np.memmap(data_path, dtype=np.uint64, mode="r")
n = len(keys)
if n == 0:
    raise ValueError(f"empty dataset: {data_path}")

workloads = [
    ("w1", 0.0, 0.0, 1.0),
    ("w2", 0.0, 1.0, 0.0),
    ("w3", 1.0, 0.0, 0.0),
    ("w4", 0.4, 0.3, 0.3),
    ("w5", 0.2, 0.2, 0.6),
    ("w6", 0.1, 0.1, 0.8),
]

outer_sizes = []
seen_tags = set()
for item in outer_sizes_s.split():
    try:
        query_tag, count_s = item.split(":", 1)
        num_queries = int(count_s)
    except ValueError as exc:
        raise ValueError(
            f"invalid OUTER_SIZES entry {item!r}; expected TAG:COUNT"
        ) from exc
    if not query_tag or num_queries <= 0:
        raise ValueError(f"invalid OUTER_SIZES entry: {item!r}")
    if query_tag in seen_tags:
        raise ValueError(f"duplicate OUTER_SIZES tag: {query_tag}")
    seen_tags.add(query_tag)
    outer_sizes.append((query_tag, num_queries))

print(f"[*] dataset={data_path} keys={n}")
print(f"[*] outer_sizes={outer_sizes} n_min={n_min} k_max={k_max}")
print(f"[*] sampling with replacement in batches of {generator_batch_size}")
print(
    "[*] model "
    f"alpha={alpha} beta={beta} eta={eta} delta={delta} "
    f"lambda_point={lambda_point} lambda_range={lambda_range}"
)

for size_id, (query_tag, num_queries) in enumerate(outer_sizes):
    print(f"[*] generating outer size {query_tag} ({num_queries} keys)")
    for table_id, (name, hot_ratio, zipf_ratio, uniform_ratio) in enumerate(workloads, start=1):
        query_path = data_dir / f"{dataset}.{query_tag}table{table_id}.bin"
        lengths_file = data_dir / f"{dataset}.{query_tag}table{table_id}.par"
        bitmap_file = data_dir / f"{dataset}.{query_tag}table{table_id}.bitmap"

        print(
            f"[*] {name}: hotspot={hot_ratio:.1f} "
            f"zipf={zipf_ratio:.1f} uniform={uniform_ratio:.1f}"
        )
        queries = sample_mixture_with_replacement(
            keys,
            num_queries,
            seed=seed + size_id * len(workloads) + table_id,
            hotpot_ratio=hot_ratio,
            zipf_ratio=zipf_ratio,
            uniform_ratio=uniform_ratio,
            num_hotpots=num_hotspots,
            hotpot_frac=hotspot_frac,
            hotpot_zipf_a=hotspot_zipf_a,
            zipf_a=zipf_a,
            return_sorted=False,
            batch_size=generator_batch_size,
        )
        queries.tofile(query_path)
        print(f"[+] wrote workload: {query_path}")

        lengths, bitmap = join_partition(
            keys,
            queries,
            alpha=alpha,
            beta=beta,
            eta=eta,
            lambda_point=lambda_point,
            lambda_range=lambda_range,
            delta=delta,
            page_size=page_size,
            key_size=key_size,
            epsilon=epsilon,
            N_min=n_min,
            K_max=k_max,
            gamma=gamma,
            phi=phi,
        )
        np.asarray(lengths, dtype=np.uint64).tofile(lengths_file)
        np.asarray(bitmap, dtype=np.uint8).tofile(bitmap_file)
        print(
            f"[+] wrote partitions: {lengths_file} {bitmap_file} "
            f"partitions={len(lengths)} range_partitions={sum(bitmap)}"
        )

print("[+] done")
PY
