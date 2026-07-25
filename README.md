# CAM: Cache-Aware I/O Cost Modeling for Disk-Based Learned Indexes

CAM is the first cache-aware I/O cost model for disk-based learned indexes that takes practical cache eviction policies (FIFO, LRU, LFU) into consideration. It estimates page access distributions from the structural geometry of learned indexes without full trace replay, and combines them with closed-form cache hit-rate models to estimate effective physical I/Os. CAM is index-agnostic: it instantiates for both error-bounded indexes (PGM-index) and model-routed indexes (RMI). Beyond I/O estimation, CAM enables memory-budgeted index tuning and a hybrid join strategy that adaptively selects point or range probes based on local key density.

## Prerequisites

### Dataset Path Configuration

`config.sh` is the single entry point for configuring dataset paths. All experiment scripts read from it. By default, datasets are stored inside this repository under `data/datasets/SOSD`:

```bash
source config.sh
echo "$DATASETS_DIRECTORY"
echo "$PYTHON_BIN"
```

Local environment variables centralized in `config.sh`:
- `DATASETS_DIRECTORY`: SOSD dataset directory.
- `PYTHON_BIN`: Python interpreter used by experiment scripts.
- `CAM_LOG_DIRECTORY`: default log directory for legacy utility modules.
- `CAM_REAL_DIRECTORY`: default real-measurement directory for legacy utility modules.
- `MPLBACKEND`: Matplotlib backend; defaults to `Agg` for headless plotting.
- `MPLCONFIGDIR` / `XDG_CACHE_HOME`: Matplotlib/font cache locations under `build/` by default.
- `CAM_PLOT_USETEX`: set to `1` to use LaTeX text rendering in plots. Defaults to `0`.

You can also override it inline without editing the file:

```bash
DATASETS_DIRECTORY=/your/data/dir PYTHON_BIN=/path/to/python bash exp/run_point_io_exp.sh
```

To download and prepare the SOSD datasets used by the experiments:

```bash
# Downloads books/fb/wiki/osm into $DATASETS_DIRECTORY.
bash scripts/download_sosd.sh
```

### Build

```bash
cmake -S . -B build
cmake --build build -j
```

Logs go to `build/log/...`, figures to `data/outputs/figures/...`.

---

## Experiment Running Examples

### 1. Point Query I/O Estimation

Runs point-query I/O estimation experiments across datasets, workload distributions, epsilon values, and memory budgets. The workflow generates point-query workloads, runs the cache simulator to collect actual I/O measurements, runs CAM estimates, then summarizes estimation accuracy and time.

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
CAM timing warms the position-cache preparation once per dataset by default to avoid assigning cold memmap/page-cache cost to the first sample rate. Set `CAM_WARMUP_POSITION_CACHE=0` to include cold-cache setup in the first timed rate.

```bash
# Compare point query estimation at various sample rates on w4
WORKLOAD=w4 SAMPLE_RATES="10 30 50 100" bash exp/run_point_cmp_exp.sh
```

---

### 4. Range Sampling Rate Comparison

Compares actual range-query performance, replayed prefixes, and CAM estimates at different workload sample rates.
CAM timing also warms the range position-cache preparation once per dataset by default to avoid assigning cold memmap/page-cache cost to the first sample rate. Set `CAM_WARMUP_POSITION_CACHE=0` to include cold-cache setup in the first timed rate.

```bash
# Compare range query estimation at various sample rates on w6
WORKLOAD=w6 MEMORY_LIST="128" SAMPLE_RATES="10 30 50 100" bash exp/run_range_cmp_exp.sh
```

---

### 5. Epsilon Sweep & Hit-Ratio Analysis

Sweeps PGM epsilon values across cache policies (FIFO, LRU, LFU), estimates I/O cost via CAM, runs actual benchmarks, and produces the logical IOs vs ε figure. `run_epsilon_benchmarks.sh` only writes `*_logical_ios_vs_epsilon.pdf` during the plotting stage by default; set `EPSILON_PLOT_ONLY_LOGICAL_IOS=0` to restore the full plotting script output.

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

Compares CAM-selected PGM epsilons with the multicriteria PGM tuner under fixed memory budgets and varying cache splits. Evaluates both throughput (QPS) and tuning time. CAM uses `--cam-size-mode powerlaw` by default: it measures anchor index sizes, fits `S(eps)=a*eps^-b+c`, and uses that model during epsilon selection. Pass `--cam-size-mode estimated` to use the faster analytic size estimate.

```bash
cmake --build build --target pgm_cam_covariance pgm_index_sizes tuner

# Single memory budget on books dataset
python3 exp/run_pgm_tuner_cache_compare.py \
  --data books_200M_uint64_unique \
  --queries books_200M_uint64_unique.query.bin \
  --M 4 \
  --candidate-eps 4-128 \
  --cache-ratios 0.25,0.50,0.75 \
  --cold-start-correction
```

---

### 7. CAM-based RMI Tuning

Compares CAM-selected RMI branch factor against CDFShop optimizer across memory budgets and cache splits.

**Step 1: Generate RMI headers (one-time per dataset)**
```bash
# TRAIN_DATA_PATH is generated automatically from COLLECT_DATA_PATH by adding
# the uint64 count header required by the RMI trainer.
TRAIN_DATA_PATH=src/rmi/dataset/books_200M_uint64_unique_fixed \
COLLECT_DATA_PATH=data/datasets/SOSD/books_200M_uint64_unique \
COLLECT_DATA_HEADER=no \
BF_LIST="256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144" \
bash exp/generate_rmi_headers.sh
```

**Step 2: Run comparison**
```bash
python3 exp/run_rmi_tuner_cache_compare.py \
    --data books_200M_uint64_unique \
    --queries books_200M_uint64_unique.query.bin \
    --M 4 \
    --candidate-bfs 256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144 \
    --rmi-name-prefix books_rmi_linear_spline_linear \
    --header no
```

---

### 8. Join Cost Model Parameter Fitting

Runs point (varying N) and range (varying page spans) calibration sweeps to fit the cost parameters used by the hybrid join partitioning logic.

```bash
cmake --build build --target pgm_join_fit

# Full sweep (point + range) and fit ALPHA/BETA/ETA/DELTA/LAMBDA_*.
DATASET=books_10M_uint64_unique NUM_KEYS=10000000 bash exp/run_join_fit.sh all

# Point-only sweep
bash exp/run_join_fit.sh point

# Range-only sweep
bash exp/run_join_fit.sh range
```

The benchmark CSVs and fitted parameters are written under `build/log/join_fit/`. Point probes fit `alpha`, `delta`, and `lambda_point`; range probes fit `beta`, `eta`, and `lambda_range`. The `.env` output can be sourced before generating hybrid join workloads:

```bash
source build/log/join_fit/books_10M_uint64_unique_join_cost_params.env
DATASET=books_200M_uint64_unique bash exp/run_join_workloads.sh
```

### 9. Hybrid Join Evaluation

Compares four join strategies (hybrid, point-only, range-only, INLJ) on the books and fb datasets.

**Step 1: Generate join workloads and partition metadata**
```bash
# Books dataset — generates .bin (queries), .par (partition lengths), .bitmap (point/range markers)
source build/log/join_fit/books_10M_uint64_unique_join_cost_params.env 
DATASET=books_200M_uint64_unique bash exp/run_join_workloads.sh
```

**Step 2: Run hybrid join comparison**
```bash
cmake --build build --target pgm_hybrid_join

# Books dataset, all 6 workload tables
# Output: build/log/hybrid_join/${DATASET}_join_compare.csv
DATASET=books_200M_uint64_unique TABLE_LIST="1 2 3 4 5 6" \
bash exp/run_hybrid_join.sh
```
