# CAM

## Experiments

Most experiment entry points live under `exp/`. The scripts assume SOSD-style `uint64` datasets are available under `/mnt/data/Dataset/public/SOSD` unless `DATASETS_DIRECTORY` is overridden. Build the C++ binaries first when a script does not build them automatically:

```bash
cmake -S . -B build
cmake --build build -j
```

Generated logs usually go to `build/log/...`, while figures are written under `data/outputs/figures/...` or the script-specific `FIGURE_DIR`.

### Point I/O

Description: generates point-query workloads, runs `pgm_cache_simulate`, estimates cache/I/O behavior with CAM, summarizes accuracy, and plots the point I/O estimation results.

Main scripts: `exp/run_point_io_exp.sh`, `exp/point_io_exp.py`, `exp/plot_point_io_exp.py`.

Example:

```bash
cmake --build build --target pgm_cache_simulate
DATASETS="books_200M_uint64_unique fb_200M_uint64_unique" \
WORKLOADS="w1 w2 w3 w4 w5 w6" \
MEMORY_LIST="64 96 128 160" \
bash exp/run_point_io_exp.sh
```

For a quick smoke run, skip selected stages with `SKIP_GENERATE=1`, `SKIP_ACTUAL=1`, `SKIP_ESTIMATE=1`, `SKIP_SUMMARIZE=1`, or `SKIP_PLOT=1`.

### Range I/O

Description: mirrors the point I/O workflow for range-query workloads, including range generation, actual cache simulation, CAM estimation, summary generation, and plotting.

Main scripts: `exp/run_range_io_exp.sh`, `exp/range_io_exp.py`, `exp/plot_point_io_exp.py`.

Example:

```bash
cmake --build build --target pgm_range_cache_simulate
DATASETS="books_200M_uint64_unique fb_200M_uint64_unique" \
WORKLOADS="w1 w2 w3 w4 w5 w6" \
RANGE_MIN_LENGTH_KEYS=1 \
RANGE_MAX_LENGTH_KEYS=1024 \
bash exp/run_range_io_exp.sh
```

### Point/Range Sampling Comparison

Description: compares actual performance, replayed prefixes, and CAM estimates at different workload sample rates for point or range workloads.

Main scripts: `exp/run_point_cmp_exp.sh`, `exp/run_range_cmp_exp.sh`, `exp/point_cmp_exp.py`, `exp/range_cmp_exp.py`, `exp/extract_cmp_summary_table.py`.

Example:

```bash
WORKLOAD=w1 SAMPLE_RATES="10 30 50 100" bash exp/run_point_cmp_exp.sh
WORKLOAD=w6 MEMORY_LIST="128" SAMPLE_RATES="10 30 50 100" bash exp/run_range_cmp_exp.sh
```

### Epsilon Benchmarks

Description: sweeps PGM epsilon values, compares CAM-estimated choices against measured benchmark results, and produces epsilon-analysis figures.

Main scripts: `exp/run_books_epsilon_benchmarks.sh`, `exp/pgm_bench.cpp`.

Example:

```bash
cmake --build build --target pgm_bench
DATA_FILE=books_200M_uint64_unique \
QUERY_FILE=books_200M_uint64_unique.query.bin \
MEMORY_LIST="64 96 128 160" \
POLICIES="FIFO LRU LFU" \
bash exp/run_books_epsilon_benchmarks.sh
```

### PGM Tuner

Description: compares CAM-selected PGM epsilons with PGM tuner baselines under the same memory budget and fixed cache splits.

Main script: `exp/run_pgm_tuner_cache_compare.py`.

Example:

```bash
cmake --build build --target pgm_cam_covariance pgm_index_sizes tuner
python3 exp/run_pgm_tuner_cache_compare.py \
  --data books_200M_uint64_unique \
  --queries books_200M_uint64_unique.query.bin \
  --M 128 \
  --candidate-eps 4-128 \
  --cache-ratios 0.25,0.50,0.75 \
  --cold-start-correction
```

### RMI Tuner

Description: compares a CDFShop optimizer-selected RMI branch factor with an optimalBF-selected branch factor under the same memory budget. The workflow uses generated RMI artifacts in `src/rmi/rmi_data`, `src/rmi/rmi_eval/generated`, and `src/rmi/rmi_eval/results`.

Main scripts: `exp/generate_rmi_headers.sh`, `exp/run_rmi_tuner_cache_compare.py`.

Example:

```bash
TRAIN_DATA_PATH=src/rmi/dataset/books_200M_uint64_unique_fixed \
COLLECT_DATA_PATH=/mnt/data/Dataset/public/SOSD/books_200M_uint64_unique \
COLLECT_DATA_HEADER=no \
BF_LIST="256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144" \
bash exp/generate_rmi_headers.sh

python3 exp/run_rmi_tuner_cache_compare.py \
  --data books_200M_uint64_unique \
  --queries books_200M_uint64_unique.query.bin \
  --M 128 \
  --candidate-bfs 256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144 \
  --rmi-name-prefix books_rmi_linear_spline_linear
```

### Hybrid Join

Description: generates table-style join workloads with `.par` and `.bitmap` partition metadata, then compares hybrid, point, range, and INLJ execution modes.

Main scripts: `exp/run_books200m_join_workloads.sh`, `exp/run_books200m_hybrid_join.sh`, `exp/pgm_hybrid_join.cpp`.

Example:

```bash
DATASET=books_200M_uint64_unique bash exp/run_books200m_join_workloads.sh
cmake --build build --target pgm_hybrid_join
DATASET=books_200M_uint64_unique TABLE_LIST="1 2 3 4 5 6" \
bash exp/run_books200m_hybrid_join.sh
```

For the Facebook dataset, override `DATASET=fb_200M_uint64_unique` in both commands.

### Motivation Query Breakdown

Description: runs no-cache external-memory query breakdowns for PGM and RMI on join workloads, averages repeated runs, exports RMI leaf-error histograms, and plots the stacked latency breakdown. When the selected RMI artifacts are missing, the script calls `exp/generate_rmi_headers.sh` automatically and prepares a headered RMI training file if needed.

Main scripts: `exp/run_books200m_motivation_breakdown.sh`, `exp/motivation_query_breakdown.cpp`, `visualize/plot_motivation_query_breakdown.py`.

Example:

```bash
DATASET=fb_200M_uint64_unique bash exp/run_books200m_join_workloads.sh
REPEATS=10 BF_LIST="262144" EPS_LIST="24" \
bash exp/run_books200m_motivation_breakdown.sh
```

Useful overrides include `RMI_PREFIX`, `RMI_TRAIN_DATA_PATH`, `RMI_GENERATE=0`, `BASELINES="RMI PGM"`, `TABLE_LIST="1 2 4 6"`, and `PLOT=0`. Non-books RMI artifacts generated here are consumed by `motivation_query_breakdown` through metadata files; `rmi_bench` remains wired to its compiled RMI namespaces.

### Join Fitting

Description: runs point and range calibration sweeps for fitting join-cost parameters used by the workload partitioning logic.

Main scripts: `exp/run_join_fit.sh`, `exp/pgm_join_fit.cpp`.

Example:

```bash
cmake --build build --target pgm_join_fit
DATASET=books_10M_uint64_unique NUM_KEYS=10000000 bash exp/run_join_fit.sh all
```
