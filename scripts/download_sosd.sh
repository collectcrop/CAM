#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
[ -f config.sh ] && source config.sh

DATASETS_DIRECTORY="${DATASETS_DIRECTORY:-$REPO_ROOT/data/datasets/SOSD}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RAW_DIR="${RAW_DIR:-$DATASETS_DIRECTORY/.sosd_raw}"
FORCE="${FORCE:-0}"
GENERATE_10M="${GENERATE_10M:-1}"
GENERATE_QUERIES="${GENERATE_QUERIES:-1}"
NUM_QUERIES="${NUM_QUERIES:-1000000}"
SEED="${SEED:-42}"
DATASETS="${DATASETS:-all}"

usage() {
  cat <<'EOF'
Usage:
  scripts/download_sosd.sh [all|books|fb|wiki|osm ...]

Environment:
  DATASETS_DIRECTORY  Output directory. Defaults to config.sh.
  RAW_DIR             Cache for decompressed SOSD source files.
  DATASETS            Dataset list when no positional datasets are passed.
  FORCE=1             Recreate output files even if they already exist.
  GENERATE_10M=0      Skip books_10M_uint64_unique/fb_10M_uint64_unique.
  GENERATE_QUERIES=0  Skip default *.query.bin generation.
  NUM_QUERIES=1000000 Number of point queries per generated query file.
  SEED=42             Random seed for generated query files.

Outputs are headerless uint64 files expected by CAM experiment scripts.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if ! command -v zstd >/dev/null 2>&1; then
  echo "[error] zstd is required to decompress SOSD .zst files" >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[error] PYTHON_BIN is not executable or not on PATH: $PYTHON_BIN" >&2
  exit 1
fi

if command -v curl >/dev/null 2>&1; then
  FETCH_CMD=(curl -L --fail --retry 3 --retry-delay 5 --silent --show-error)
elif command -v wget >/dev/null 2>&1; then
  FETCH_CMD=(wget -O -)
else
  echo "[error] curl or wget is required to download SOSD files" >&2
  exit 1
fi

mkdir -p "$DATASETS_DIRECTORY" "$RAW_DIR"

checksum_md5() {
  local path="$1"
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$path" | awk '{print $1}'
  elif command -v md5 >/dev/null 2>&1; then
    md5 -q "$path"
  else
    echo "[error] md5sum or md5 is required for checksum validation" >&2
    exit 1
  fi
}

download_raw_zst() {
  local raw_name="$1"
  local expected_md5="$2"
  local url="$3"
  local raw_path="$RAW_DIR/$raw_name"
  local tmp_path="$raw_path.tmp"

  if [ -f "$raw_path" ] && [ "$FORCE" != "1" ]; then
    local actual_md5
    actual_md5="$(checksum_md5 "$raw_path")"
    if [ "$actual_md5" = "$expected_md5" ]; then
      echo "[raw] reuse $raw_path"
      return 0
    fi
    echo "[warn] checksum mismatch for cached $raw_path; re-downloading" >&2
  fi

  echo "[download] $raw_name"
  rm -f "$tmp_path"
  "${FETCH_CMD[@]}" "$url" | zstd -dc > "$tmp_path"

  local actual_md5
  actual_md5="$(checksum_md5 "$tmp_path")"
  if [ "$actual_md5" != "$expected_md5" ]; then
    rm -f "$tmp_path"
    echo "[error] checksum mismatch for $raw_name" >&2
    echo "        expected: $expected_md5" >&2
    echo "        actual:   $actual_md5" >&2
    exit 1
  fi

  mv "$tmp_path" "$raw_path"
  echo "[raw] verified $raw_path"
}

prepare_payload() {
  local mode="$1"
  local src="$2"
  local dst="$3"
  local step="${4:-1}"
  local expected_keys="${5:-0}"

  if [ -f "$dst" ] && [ "$FORCE" != "1" ]; then
    echo "[data] reuse $dst"
    return 0
  fi

  mkdir -p "$(dirname "$dst")"
  "$PYTHON_BIN" - "$mode" "$src" "$dst" "$step" "$expected_keys" <<'PY'
from pathlib import Path
import os
import sys

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("numpy is required to prepare SOSD datasets") from exc

mode = sys.argv[1]
src = Path(sys.argv[2])
dst = Path(sys.argv[3])
step = int(sys.argv[4])
expected_keys = int(sys.argv[5])

if step < 1:
    raise SystemExit("step must be positive")

raw = np.memmap(src, dtype="<u8", mode="r")
if len(raw) == 0:
    raise SystemExit(f"empty SOSD source file: {src}")

header_count = int(raw[0])
payload = raw[1:] if header_count == len(raw) - 1 else raw

if mode == "strip":
    out = payload
elif mode == "sample":
    out = payload[::step]
else:
    raise SystemExit(f"unknown mode: {mode}")

if expected_keys and len(out) != expected_keys:
    raise SystemExit(
        f"{dst.name}: expected {expected_keys} keys, got {len(out)} from {src.name}"
    )

tmp = dst.with_suffix(dst.suffix + ".tmp")
if tmp.exists():
    tmp.unlink()
out.tofile(tmp)
os.replace(tmp, dst)
print(f"[data] wrote {dst} ({len(out)} keys)")
PY
}

link_alias() {
  local source_name="$1"
  local alias_name="$2"
  local alias_path="$DATASETS_DIRECTORY/$alias_name"

  if [ -e "$alias_path" ] || [ -L "$alias_path" ]; then
    return 0
  fi

  ln -s "$source_name" "$alias_path"
  echo "[data] linked $alias_path -> $source_name"
}

generate_queries() {
  local data_path="$1"
  local query_path="$2"

  if [ "$GENERATE_QUERIES" != "1" ]; then
    return 0
  fi
  if [ -f "$query_path" ] && [ "$FORCE" != "1" ]; then
    echo "[query] reuse $query_path"
    return 0
  fi

  "$PYTHON_BIN" - "$data_path" "$query_path" "$NUM_QUERIES" "$SEED" <<'PY'
from pathlib import Path
import os
import sys

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("numpy is required to generate query files") from exc

data_path = Path(sys.argv[1])
query_path = Path(sys.argv[2])
num_queries = int(sys.argv[3])
seed = int(sys.argv[4])

keys = np.memmap(data_path, dtype="<u8", mode="r")
n = len(keys)
if n == 0:
    raise SystemExit(f"empty dataset: {data_path}")
if num_queries <= 0:
    raise SystemExit("NUM_QUERIES must be positive")

rng = np.random.default_rng(seed)
parts = []

hot_queries = int(num_queries * 0.4)
zipf_queries = int(num_queries * 0.3)
uniform_queries = num_queries - hot_queries - zipf_queries

num_hotspots = 5
hotspot_size = max(1, int(0.01 * n))
for hotspot_id in range(num_hotspots):
    take = hot_queries // num_hotspots
    if hotspot_id < hot_queries % num_hotspots:
        take += 1
    if take <= 0:
        break
    base = int(rng.integers(0, max(1, n - hotspot_size + 1)))
    idx = rng.zipf(1.5, size=take) - 1
    idx = np.clip(idx, 0, hotspot_size - 1)
    parts.append(np.asarray(keys[base + idx], dtype="<u8"))

if zipf_queries > 0:
    idx = rng.zipf(1.2, size=zipf_queries) - 1
    idx = np.clip(idx, 0, n - 1)
    parts.append(np.asarray(keys[idx], dtype="<u8"))

if uniform_queries > 0:
    replace = uniform_queries > n
    idx = rng.choice(n, size=uniform_queries, replace=replace)
    parts.append(np.asarray(keys[idx], dtype="<u8"))

queries = np.concatenate(parts)[:num_queries].astype("<u8", copy=False)
if len(queries) < num_queries:
    idx = rng.integers(0, n, size=num_queries - len(queries))
    queries = np.concatenate([queries, np.asarray(keys[idx], dtype="<u8")])
rng.shuffle(queries)

query_path.parent.mkdir(parents=True, exist_ok=True)
tmp = query_path.with_suffix(query_path.suffix + ".tmp")
if tmp.exists():
    tmp.unlink()
queries.tofile(tmp)
os.replace(tmp, query_path)
print(f"[query] wrote {query_path} ({len(queries)} queries)")
PY
}

download_books() {
  download_raw_zst \
    "books_800M_uint64" \
    "8708eb3e1757640ba18dcd3a0dbb53bc" \
    "https://www.dropbox.com/s/y2u3nbanbnbmg7n/books_800M_uint64.zst?dl=1"

  prepare_payload "sample" "$RAW_DIR/books_800M_uint64" "$DATASETS_DIRECTORY/books_200M_uint64_unique" 4 200000000
  generate_queries "$DATASETS_DIRECTORY/books_200M_uint64_unique" "$DATASETS_DIRECTORY/books_200M_uint64_unique.query.bin"
  if [ "$GENERATE_10M" = "1" ]; then
    prepare_payload "sample" "$RAW_DIR/books_800M_uint64" "$DATASETS_DIRECTORY/books_10M_uint64_unique" 80 10000000
    generate_queries "$DATASETS_DIRECTORY/books_10M_uint64_unique" "$DATASETS_DIRECTORY/books_10M_uint64_unique.query.bin"
  fi
}

download_fb() {
  download_raw_zst \
    "fb_200M_uint64" \
    "3b0f820caa0d62150e87ce94ec989978" \
    "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/EATHF7"

  prepare_payload "strip" "$RAW_DIR/fb_200M_uint64" "$DATASETS_DIRECTORY/fb_200M_uint64_unique" 1 200000000
  generate_queries "$DATASETS_DIRECTORY/fb_200M_uint64_unique" "$DATASETS_DIRECTORY/fb_200M_uint64_unique.query.bin"
  if [ "$GENERATE_10M" = "1" ]; then
    prepare_payload "sample" "$RAW_DIR/fb_200M_uint64" "$DATASETS_DIRECTORY/fb_10M_uint64_unique" 20 10000000
    generate_queries "$DATASETS_DIRECTORY/fb_10M_uint64_unique" "$DATASETS_DIRECTORY/fb_10M_uint64_unique.query.bin"
  fi
}

download_wiki() {
  download_raw_zst \
    "wiki_ts_200M_uint64" \
    "4f1402b1c476d67f77d2da4955432f7d" \
    "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/SVN8PI"

  prepare_payload "strip" "$RAW_DIR/wiki_ts_200M_uint64" "$DATASETS_DIRECTORY/wiki_ts_200M_uint64" 1 200000000
  link_alias "wiki_ts_200M_uint64" "wiki_ts_200M_uint64_unique"
  generate_queries "$DATASETS_DIRECTORY/wiki_ts_200M_uint64" "$DATASETS_DIRECTORY/wiki_ts_200M_uint64.query.bin"
  if [ "$GENERATE_QUERIES" = "1" ]; then
    link_alias "wiki_ts_200M_uint64.query.bin" "wiki_ts_200M_uint64_unique.query.bin"
  fi
}

download_osm() {
  download_raw_zst \
    "osm_cellids_800M_uint64" \
    "70670bf41196b9591e07d0128a281b9a" \
    "https://www.dropbox.com/s/j1d4ufn4fyb4po2/osm_cellids_800M_uint64.zst?dl=1"

  prepare_payload "sample" "$RAW_DIR/osm_cellids_800M_uint64" "$DATASETS_DIRECTORY/osm_cellids_200M_uint64_unique" 4 200000000
  generate_queries "$DATASETS_DIRECTORY/osm_cellids_200M_uint64_unique" "$DATASETS_DIRECTORY/osm_cellids_200M_uint64_unique.query.bin"
}

if [ "$#" -gt 0 ]; then
  requested=("$@")
else
  read -r -a requested <<< "$DATASETS"
fi

expanded=()
for dataset in "${requested[@]}"; do
  case "$dataset" in
    all)
      expanded+=(books fb wiki osm)
      ;;
    books|book|books_200M_uint64_unique|books_10M_uint64_unique)
      expanded+=(books)
      ;;
    fb|facebook|fb_200M_uint64_unique|fb_10M_uint64_unique)
      expanded+=(fb)
      ;;
    wiki|wiki_ts|wiki_ts_200M_uint64|wiki_ts_200M_uint64_unique)
      expanded+=(wiki)
      ;;
    osm|osm_cellids|osm_cellids_200M_uint64_unique)
      expanded+=(osm)
      ;;
    *)
      echo "[error] unknown dataset selector: $dataset" >&2
      usage >&2
      exit 2
      ;;
  esac
done

seen=" "
for dataset in "${expanded[@]}"; do
  case "$seen" in
    *" $dataset "*) continue ;;
  esac
  seen="$seen$dataset "
  "download_$dataset"
done

echo "[done] datasets directory: $DATASETS_DIRECTORY"
