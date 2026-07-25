#!/usr/bin/env bash
set -euo pipefail

# Generate join workloads and hybrid partitions for six workload mixtures (w1-w6).
# Output files are placed alongside the dataset:
#   ${DATASET}.${QUERY_TAG}table1.bin/.par/.bitmap
#   ...
#   ${DATASET}.${QUERY_TAG}table6.bin/.par/.bitmap

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
[ -f config.sh ] && source config.sh

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASETS_DIRECTORY="${DATASETS_DIRECTORY:-$REPO_ROOT/data/datasets/SOSD}"
DATASET="${DATASET:-books_200M_uint64_unique}"
QUERY_TAG="${QUERY_TAG:-1M}"
NUM_QUERIES="${NUM_QUERIES:-1000000}"
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
OVERSAMPLE="${OVERSAMPLE:-500}"
MIN_CANDIDATES="${MIN_CANDIDATES:-1000000}"
STRICT="${STRICT:-true}"

"$PYTHON_BIN" - \
  "$DATASETS_DIRECTORY" "$DATASET" "$QUERY_TAG" "$NUM_QUERIES" "$SEED" \
  "$N_MIN" "$K_MAX" "$PAGE_SIZE" "$KEY_SIZE" "$EPSILON" "$GAMMA" "$PHI" \
  "$ALPHA" "$BETA" "$ETA" "$DELTA" "$LAMBDA_POINT" "$LAMBDA_RANGE" \
  "$NUM_HOTSPOTS" "$HOTSPOT_FRAC" "$HOTSPOT_ZIPF_A" "$ZIPF_A" \
  "$OVERSAMPLE" "$MIN_CANDIDATES" "$STRICT" <<'PY'
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd() / "utils"))
from generate_query import join_partition, sample_unique_mixture

(
    datasets_directory,
    dataset,
    query_tag,
    num_queries_s,
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
    oversample_s,
    min_candidates_s,
    strict_s,
) = sys.argv[1:]

data_dir = Path(datasets_directory)
data_path = data_dir / dataset
if not data_path.exists():
    raise FileNotFoundError(f"dataset not found: {data_path}")

num_queries = int(num_queries_s)
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
oversample = int(oversample_s)
min_candidates = int(min_candidates_s)
strict = strict_s.lower() in {"1", "true", "yes", "y", "on"}

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

print(f"[*] dataset={data_path} keys={n}")
print(f"[*] num_queries={num_queries} n_min={n_min} k_max={k_max}")
print(f"[*] sample_unique_mixture oversample={oversample} min_candidates={min_candidates} strict={strict}")
print(
    "[*] model "
    f"alpha={alpha} beta={beta} eta={eta} delta={delta} "
    f"lambda_point={lambda_point} lambda_range={lambda_range}"
)

for table_id, (name, hot_ratio, zipf_ratio, uniform_ratio) in enumerate(workloads, start=1):
    # if name in ["w3","w2","w4","w5"]:
    #     continue
    query_path = data_dir / f"{dataset}.{query_tag}table{table_id}.bin"
    lengths_file = data_dir / f"{dataset}.{query_tag}table{table_id}.par"
    bitmap_file = data_dir / f"{dataset}.{query_tag}table{table_id}.bitmap"

    print(
        f"[*] {name}: hotspot={hot_ratio:.1f} "
        f"zipf={zipf_ratio:.1f} uniform={uniform_ratio:.1f}"
    )
    queries = sample_unique_mixture(
        keys,
        num_queries,
        seed=seed + table_id,
        hotpot_ratio=hot_ratio,
        zipf_ratio=zipf_ratio,
        uniform_ratio=uniform_ratio,
        num_hotpots=num_hotspots,
        hotpot_frac=hotspot_frac,
        hotpot_zipf_a=hotspot_zipf_a,
        zipf_a=zipf_a,
        oversample=oversample,
        min_candidates=min_candidates,
        return_sorted=False,
        strict=strict,
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
        lengths_file=str(lengths_file),
        bitmap_file=str(bitmap_file),
    )
    print(
        f"[+] wrote partitions: {lengths_file} {bitmap_file} "
        f"partitions={len(lengths)} range_partitions={sum(bitmap)}"
    )

print("[+] done")
PY
