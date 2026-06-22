# CAM: Cache-Aware I/O Cost Modeling for Disk-Based Learned Indexes

CAM is the first cache-aware I/O cost model for disk-based learned indexes that takes practical cache eviction policies (FIFO, LRU, LFU) into consideration. It estimates page access distributions from the structural geometry of learned indexes without full trace replay, and combines them with closed-form cache hit-rate models to estimate effective physical I/Os. CAM is index-agnostic: it instantiates for both error-bounded indexes (PGM-index) and model-routed indexes (RMI). Beyond I/O estimation, CAM enables memory-budgeted index tuning and a hybrid join strategy that adaptively selects point or range probes based on local key density.

**Key results:**
- CAM matches replay accuracy (1.04× Q-error) while reducing estimation time by 17.13×
- CAM-guided PGM tuning improves throughput by 1.17× over multicriteria PGM tuning
- CAM-guided RMI tuning improves throughput by 1.66× over CDFShop
- Hybrid join improves end-to-end performance by up to 8.8× over disk-based INLJ


---

## Prerequisites

### Dataset Path Configuration

**Edit `config.sh`** — this is the single entry point for configuring dataset paths. All experiment scripts read from it.

```bash
# config.sh — edit this line to point to your datasets:
export DATASETS_DIRECTORY="${DATASETS_DIRECTORY:-/mnt/data/Dataset/public/SOSD}"
```

You can also override it inline without editing the file:

```bash
DATASETS_DIRECTORY=/your/data/dir bash exp/run_point_io_exp.sh
```

Datasets should be SOSD-style `uint64` binary files (one key per 8 bytes, little-endian). The default dataset names used by experiments are:
- `books_200M_uint64_unique` / `books_10M_uint64_unique`
- `fb_200M_uint64_unique` / `fb_10M_uint64_unique`
- `wiki_ts_200M_uint64_unique`
- `osm_cellids_200M_uint64_unique` 

### Build

```bash
cmake -S . -B build
cmake --build build -j
```

Logs go to `build/log/...`, figures to `data/outputs/figures/...`.

---

## Experiment Running Examples

### 1. Point Query I/O Estimation

Compares CAM, Replay (trace-driven simulation), and LPM (logical page model) for point query workloads across datasets, workload distributions, epsilon values, and memory budgets. Generates Q-error and estimation time figures.

```bash
# Build the simulator
cmake --build build --target pgm_cache_simulate

# Full run on books and fb datasets, all 6 workloads, 4 memory budgets
# Logs: build/log/exp/  Figures: build/log/exp/figures/
DATASETS="books_200M_uint64_unique fb_200M_uint64_unique" \
WORKLOADS="w1 w2 w3 w4 w5 w6" \
MEMORY_LIST="64 96 128 160" \
bash exp/run_point_io_exp.sh

# Quick smoke run (skip selected stages)
DATASETS="books_200M_uint64_unique" \
SKIP_GENERATE=1 SKIP_ACTUAL=1 bash exp/run_point_io_exp.sh
```

**What it runs:**
- `exp/point_io_exp.py generate` — generates workload query files
- `exp/pgm_cache_simulate` — actual cache simulation (ground truth)
- `exp/point_io_exp.py estimate` — CAM estimation
- `exp/point_io_exp.py summarize` — accuracy summary
- `exp/plot_point_io_exp.py` — plots

---

### 2. Range Query I/O Estimation

Same workflow as point queries but for range queries with configurable range lengths.

```bash
# Build the range simulator
cmake --build build --target pgm_range_cache_simulate

# Full run
DATASETS="books_200M_uint64_unique fb_200M_uint64_unique" \
WORKLOADS="w1 w2 w3 w4 w5 w6" \
RANGE_MIN_LENGTH_KEYS=1 \
RANGE_MAX_LENGTH_KEYS=1024 \
bash exp/run_range_io_exp.sh
# Logs: build/log/range_exp/  Figures: build/log/range_exp/figures/
```

---

### 3. Point Sampling Rate Comparison

Compares actual performance, replayed prefixes, and CAM estimates at different workload sample rates. Evaluates how estimation accuracy degrades with smaller samples.

```bash
# Compare point query estimation at various sample rates on w4
WORKLOAD=w4 SAMPLE_RATES="10 30 50 100" bash exp/run_point_cmp_exp.sh
```

---

### 4. Range Sampling Rate Comparison

```bash
# Compare range query estimation at various sample rates on w6
WORKLOAD=w6 MEMORY_LIST="128" SAMPLE_RATES="10 30 50 100" bash exp/run_range_cmp_exp.sh
```

---

### 5. Epsilon Sweep & Hit-Ratio Analysis

Sweeps PGM epsilon values across cache policies (FIFO, LRU, LFU), estimates I/O cost via CAM, runs actual benchmarks, and produces epsilon-analysis figures (logical IOs vs ε, hit ratio vs ε, estimated I/O throughput vs ε).

```bash
cmake --build build --target pgm_bench

# Books dataset, multiple policies and memory budgets
DATA_FILE=books_200M_uint64_unique \
QUERY_FILE=books_200M_uint64_unique.query.bin \
MEMORY_LIST="64 96 128 160" \
POLICIES="FIFO LRU LFU" \
bash exp/run_epsilon_benchmarks.sh
# Logs: build/log/  Figures: data/outputs/figures/epsilon_analysis/

# Override epsilon sweep range
EPS_LIST="8,10,12,14,16,20,24,32,64,128" \
bash exp/run_epsilon_benchmarks.sh
```

---

### 6. CAM-based PGM Tuning

Compares CAM-selected PGM epsilons with the multicriteria PGM tuner under fixed memory budgets and varying cache splits. Evaluates both throughput (QPS) and tuning time.

```bash
cmake --build build --target pgm_cam_covariance pgm_index_sizes tuner

# Single memory budget on books dataset
python3 exp/run_pgm_tuner_cache_compare.py \
  --data books_200M_uint64_unique \
  --queries books_200M_uint64_unique.query.bin \
  --M 128 \
  --candidate-eps 4-128 \
  --cache-ratios 0.25,0.50,0.75 \
  --cold-start-correction

# Multiple memory budgets
for M in 4 8 16 32; do
  python3 exp/run_pgm_tuner_cache_compare.py \
    --data books_200M_uint64_unique \
    --queries books_200M_uint64_unique.query.bin \
    --M "$M" \
    --candidate-eps 4-128 \
    --cache-ratios 0.25,0.50,0.75 \
    --cold-start-correction
done
```

---

### 7. CAM-based RMI Tuning

Compares CAM-selected RMI branch factor against CDFShop optimizer across memory budgets and cache splits.

**Step 1: Generate RMI headers (one-time per dataset)**
```bash
TRAIN_DATA_PATH=src/rmi/dataset/books_200M_uint64_unique_fixed \
COLLECT_DATA_PATH="$DATASETS_DIRECTORY/books_200M_uint64_unique" \
COLLECT_DATA_HEADER=no \
BF_LIST="256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144" \
bash exp/generate_rmi_headers.sh
```

**Step 2: Run comparison**
```bash
python3 exp/run_rmi_tuner_cache_compare.py \
  --data books_200M_uint64_unique \
  --queries books_200M_uint64_unique.query.bin \
  --M 128 \
  --candidate-bfs 256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144 \
  --rmi-name-prefix books_rmi_linear_spline_linear
```

---

### 8. Hybrid Join Evaluation

Compares four join strategies (hybrid, point-only, range-only, INLJ) on the books and fb datasets.

**Step 1: Generate join workloads and partition metadata**
```bash
# Books dataset — generates .bin (queries), .par (partition lengths), .bitmap (point/range markers)
DATASET=books_200M_uint64_unique bash exp/run_join_workloads.sh

# Facebook dataset
DATASET=fb_200M_uint64_unique bash exp/run_join_workloads.sh
```

**Step 2: Run hybrid join comparison**
```bash
cmake --build build --target pgm_hybrid_join

# Books dataset, all 6 workload tables
# Output: build/log/hybrid_join/${DATASET}_join_compare.csv
DATASET=books_200M_uint64_unique TABLE_LIST="1 2 3 4 5 6" \
bash exp/run_hybrid_join.sh

# Override output path
OUT_CSV=build/log/hybrid_join/fb_compare.csv \
DATASET=fb_200M_uint64_unique bash exp/run_hybrid_join.sh
```

---

### 9. Motivation: Query Latency & I/O Breakdown

Runs buffered external-memory query breakdowns for PGM and RMI on join workloads. Produces stacked latency breakdown plots and I/O size tables.

```bash
# Generate join workloads first (if not done)
DATASET=books_200M_uint64_unique bash exp/run_join_workloads.sh

# Run breakdown with 10 repeats, custom eps/branch-factor
REPEATS=10 BF_LIST="262144" EPS_LIST="24" \
DATASET=books_200M_uint64_unique \
bash exp/run_motivation_breakdown.sh
# Results: build/log/motivation/  Figures: data/outputs/figures/motivation/

# OSM dataset with PGM and RMI direct/buffered modes
DATASET=osm_cellids_200M_uint64_unique \
BASELINES="RMI-DIRECT PGM-DIRECT" \
bash exp/run_motivation_breakdown.sh
```

Useful overrides: `RMI_PREFIX`, `RMI_GENERATE=0`, `TABLE_LIST="1 2 4 6"`, `PLOT=0`.

---

### 10. Join Cost Model Parameter Fitting

Runs point (varying N) and range (varying page spans) calibration sweeps to fit the cost parameters used by the hybrid join partitioning logic.

```bash
cmake --build build --target pgm_join_fit

# Full sweep (point + range)
DATASET=books_10M_uint64_unique NUM_KEYS=10000000 bash exp/run_join_fit.sh all

# Point-only sweep
bash exp/run_join_fit.sh point

# Range-only sweep
bash exp/run_join_fit.sh range
```

---

## Quick Start: Reproduce Core Paper Results

Run the following sequence to reproduce the main paper experiments:

```bash
# 1. Build everything
cmake -S . -B build && cmake --build build -j

# 2. Point I/O estimation
DATASETS="books_200M_uint64_unique" bash exp/run_point_io_exp.sh

# 3. Range I/O estimation
DATASETS="books_200M_uint64_unique" bash exp/run_range_io_exp.sh

# 4. PGM tuning comparison
python3 exp/run_pgm_tuner_cache_compare.py \
  --data books_200M_uint64_unique \
  --queries books_200M_uint64_unique.query.bin \
  --M 128 --cold-start-correction

# 5. Hybrid join — requires join workloads first
DATASET=books_200M_uint64_unique bash exp/run_join_workloads.sh
DATASET=books_200M_uint64_unique bash exp/run_hybrid_join.sh

# 6. Motivation breakdown
DATASET=books_200M_uint64_unique bash exp/run_motivation_breakdown.sh
```
