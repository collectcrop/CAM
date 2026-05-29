# ./build/pgm_bench \
#   --data /mnt/data/Dataset/public/SOSD/books_10M_uint64_unique \
#   --queries /mnt/data/Dataset/public/SOSD/books_10M_uint64_unique.query.bin \
#   --keys 10000000 \
#   --M 60 \

# mkdir -p build/log

# for M in 10 20 40 60; do
#   echo "[pgm_range_bench] running M=${M} MiB" >&2
#   ./build/pgm_range_bench \
#     --data /mnt/data/Dataset/public/SOSD/fb_10M_uint64_unique \
#     --queries /mnt/data/Dataset/public/SOSD/fb_10M_uint64_unique.range.bin \
#     --keys 10000000 \
#     --M "${M}" \
#     --policies FIFO,LRU,LFU \
#     > "build/log/fb_10M_M${M}_range_bench.csv"
# done


# ./build/pgm_cam_covariance \
#   --data /mnt/data/Dataset/public/SOSD/books_10M_uint64_unique \
#   --queries /mnt/data/Dataset/public/SOSD/books_10M_uint64_unique.query.bin \
#   --keys 10000000 \
#   --M 10 \
#   --epsilons 4,8,10,12,14,16,20,24,32,64,128 \
#   --policies FIFO,LRU,LFU \
#   --budget-mode raw \
#   --summary-out build/log/cmp/books_10M_M60_summary_real.csv
#   # --detail-out build/log/books_10M_M60_cam_cov_detail.csv


# python visualize/plot_epsilon_benchmarks.py \
#   --estimate-paths build/log/books_10M_uint64_unique_FIFO.log build/log/books_10M_uint64_unique_LFU.log build/log/books_10M_uint64_unique_LRU.log \
#   --bench-paths build/log/books_10M_M10_bench.csv build/log/books_10M_M20_bench.csv build/log/books_10M_M40_bench.csv build/log/books_10M_M60_bench.csv \
#   --fitcam-root build/log/fitcam_q30 \
#   --output-dir data/outputs/figures/epsilon_analysis

# for M in 8; do
#   ./build/rmi_bench \
#     --data src/rmi/dataset/books_10M_uint64_unique_fixed \
#     --queries /mnt/data/Dataset/public/SOSD/books_10M_uint64_unique.query.bin \
#     --rmi-data-dir src/rmi/rmi_data \
#     --branch-factors all \
#     --header yes \
#     --M "$M" \
#     > "build/log/rmi/books_10M_M${M}_rmi_bench.csv"
# done


# python utils/plot_rmi_io_compare.py \
#   --estimate-log build/log/rmi/books_10M_M8_rmi_optimalBF_summary.log \
#   --bench-csv build/log/rmi/books_10M_M8_rmi_bench.csv \
#   --output build/log/rmi/books_10M_M8_rmi_io_compare.pdf

# python utils/plot_rmi_io_compare.py \
#   --estimate-log \
#     build/log/rmi/books_10M_M8_rmi_optimalBF_summary.log \
#     build/log/rmi/books_10M_M16_rmi_optimalBF_summary.log \
#     build/log/rmi/books_10M_M32_rmi_optimalBF_summary.log \
#     build/log/rmi/books_10M_M64_rmi_optimalBF_summary.log \
#   --bench-csv \
#     build/log/rmi/books_10M_M8_rmi_bench.csv \
#     build/log/rmi/books_10M_M16_rmi_bench.csv \
#     build/log/rmi/books_10M_M32_rmi_bench.csv \
#     build/log/rmi/books_10M_M64_rmi_bench.csv \
#   --m-values 8 16 32 64 \
#   --output build/log/rmi/books_10M_rmi_io_compare.pdf


# python utils/plot_rmi_fitrmi_compare.py \
  # --comparison-csv build/log/fitrmi_q30/books_10M/fit_output/LRU/books_10M_LRU_q30_fitrmi_corrected_vs_real.csv \
  # --output-dir data/outputs/figures/rmi_fitrmi/books_10M \
  # --dataset-tag books_10M \
  # --policies LRU \
  # --m-values 16 32 64 \
  # --min-bf 64


# ~/miniconda3/bin/python utils/plot_epsilon_benchmarks.py \
#   --estimate-paths build/log/fb_10M_uint64_unique_FIFO.log build/log/fb_10M_uint64_unique_LFU.log build/log/fb_10M_uint64_unique_LRU.log \
#   --bench-paths build/log/fb_10M_M10_range_bench.csv build/log/fb_10M_M20_range_bench.csv build/log/fb_10M_M40_range_bench.csv build/log/fb_10M_M60_range_bench.csv \
#   --dataset-filter fb_10M \
#   --m-values 10 20 40 60 \
#   --fitcam-root build/log/fitcam_q30 \
#   --output-dir data/outputs/figures/epsilon_analysis


# python utils/plot_epsilon_error_benchmarks.py \
#   --estimate-paths \
#     build/log/books_10M_uint64_unique_FIFO.log \
#     build/log/books_10M_uint64_unique_LRU.log \
#     build/log/books_10M_uint64_unique_LFU.log \
#   --bench-paths \
#     build/log/books_10M_M10_bench.csv \
#     build/log/books_10M_M20_bench.csv \
#     build/log/books_10M_M40_bench.csv \
#     build/log/books_10M_M60_bench.csv \
#   --dataset-filter books_10M

# ~/miniconda3/bin/python utils/plot_memory_hit_ratio_compare.py \
#   --cmp-dir build/log/cmp \
#   --sim-csv build/log/cmp/books_10M_eps32_memory_sweep_sim_merged.csv \
#   --dataset-tag books_10MB \
#   --memory-list 10 20 40 60


# ./build/pgm_cache_simulate \
#   --data /mnt/data/Dataset/public/SOSD/books_10M_uint64_unique \
#   --queries /mnt/data/Dataset/public/SOSD/books_10M_uint64_unique.8Mquery.bin \
#   --keys 10000000 \
#   --M 10 \
#   --policies FIFO,LRU,LFU \
#   --strategies all_in_once \
#   --budget-mode estimated \
#   --summary-out build/log/cmp/books_10M_M10_8Mquery_summary_sim.csv


# ./build/sort_cache_tradeoff \
#   --data /mnt/data/Dataset/public/SOSD/books_200M_uint64_unique \
#   --queries /mnt/data/Dataset/public/SOSD/books_200M_uint64_unique.1Mtable1.bin \
#   --keys 200000000 \
#   --M 64 \
#   --sort-mibs 0,8,16,32 \
#   --policies LRU \
#   --header-mode no \
#   --summary-out build/log/sort_cache_tradeoff.csv


# ./build/rmi_bench \
#   --data src/rmi/dataset/books_10M_uint64_unique_fixed \
#   --queries /mnt/data/Dataset/public/SOSD/books_10M_uint64_unique.query.bin \
#   --rmi-data-dir src/rmi/rmi_data \
#   --branch-factors 64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152 \
#   --header yes \
#   --M 16 \
#   > build/log/rmi/books_10M_M16_rmi_bench.csv

# ./exp/run_pgm_tuner_cache_compare.py \
#   --data books_200M_uint64_unique \
#   --queries books_200M_uint64_unique.query.bin \
#   --keys 200000000 \
#   --M 8 \
#   --candidate-eps 4-128 \
#   --cache-ratios 0.25,0.50,0.75 \
#   --tuner-bin ./build/tuner \
#   --cam-bin ./build/pgm_cam_covariance

# python3 exp/run_rmi_tuner_cache_compare.py \
#   --data books_200M_uint64_unique \
#   --queries books_200M_uint64_unique.query.bin \
#   --optimizer-data books_200M_uint64 \
#   --keys 200000000 \
#   --M 4 \
#   --header no \
#   --dataset-tag books_200M \
#   --policies FIFO,LRU,LFU \
#   --strategies all_in_once \
#   --tuning-policy LRU \
#   --optimizer-threads 8 \
#   --output-dir build/log/rmi_tuner_cache_compare \
#   --candidate-bfs 256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152 \
#   --force-optimizer

# python3 exp/extract_cmp_summary_table.py \
#   --root build/log/point_cmp \
#   --kind point \
#   --workloads w1 w2 w4 w6 \
#   --datasets books fb osm wiki \
#   --rates 10 30 50 100 \
#   --M 128 \
#   --methods CAM replay lpm \
#   --metric q_error

  python3 exp/extract_cmp_summary_table.py \
  --root build/log/range_cmp \
  --kind range \
  --workloads w1 w2 w4 w6 \
  --datasets books fb osm wiki \
  --rates 10 30 50 100 \
  --M 128 \
  --methods CAM replay lpm \
  --metric q_error