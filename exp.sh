# ./build/pgm_bench \
#   --data /mnt/backup_disk/Dataset/public/SOSD/books_10M_uint64_unique \
#   --queries /mnt/backup_disk/Dataset/public/SOSD/books_10M_uint64_unique.query.bin \
#   --keys 10000000 \
#   --M 60 \

# mkdir -p build/log

# for M in 10 20 40 60; do
#   echo "[pgm_range_bench] running M=${M} MiB" >&2
#   ./build/pgm_range_bench \
#     --data /mnt/backup_disk/Dataset/public/SOSD/fb_10M_uint64_unique \
#     --queries /mnt/backup_disk/Dataset/public/SOSD/fb_10M_uint64_unique.range.bin \
#     --keys 10000000 \
#     --M "${M}" \
#     --policies FIFO,LRU,LFU \
#     > "build/log/fb_10M_M${M}_range_bench.csv"
# done


# ./build/pgm_cam_covariance \
#   --data /mnt/backup_disk/Dataset/public/SOSD/books_10M_uint64_unique \
#   --queries /mnt/backup_disk/Dataset/public/SOSD/books_10M_uint64_unique.query.bin \
#   --keys 10000000 \
#   --M 10 \
#   --epsilons 4,8,10,12,14,16,20,24,32,64,128 \
#   --policies FIFO,LRU,LFU \
#   --budget-mode raw \
#   --summary-out build/log/cmp/books_10M_M60_summary_real.csv
#   # --detail-out build/log/books_10M_M60_cam_cov_detail.csv


# python utils/plot_epsilon_benchmarks.py \
#   --estimate-paths build/log/books_10M_uint64_unique_FIFO.log build/log/books_10M_uint64_unique_LFU.log build/log/books_10M_uint64_unique_LRU.log \
#   --bench-paths build/log/books_10M_M10_bench.csv build/log/books_10M_M20_bench.csv build/log/books_10M_M40_bench.csv build/log/books_10M_M60_bench.csv \
#   --fitcam-root build/log/fitcam_q30 \
#   --output-dir data/outputs/figures/epsilon_analysis

for M in 8; do
  ./build/rmi_bench \
    --data src/rmi/dataset/books_10M_uint64_unique_fixed \
    --queries /mnt/backup_disk/Dataset/public/SOSD/books_10M_uint64_unique.query.bin \
    --rmi-data-dir src/rmi/rmi_data \
    --branch-factors all \
    --header yes \
    --M "$M" \
    > "build/log/rmi/books_10M_M${M}_rmi_bench.csv"
done


# python utils/plot_rmi_io_compare.py \
#   --estimate-log build/log/rmi/books_10M_M32_rmi_optimalBF_summary.log \
#   --bench-csv build/log/rmi/books_10M_M32_rmi_bench.csv \
#   --output build/log/rmi/books_10M_M32_rmi_io_compare.pdf

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
#   --data /mnt/backup_disk/Dataset/public/SOSD/books_10M_uint64_unique \
#   --queries /mnt/backup_disk/Dataset/public/SOSD/books_10M_uint64_unique.8Mquery.bin \
#   --keys 10000000 \
#   --M 10 \
#   --policies FIFO,LRU,LFU \
#   --strategies all_in_once \
#   --budget-mode estimated \
#   --summary-out build/log/cmp/books_10M_M10_8Mquery_summary_sim.csv


# ./build/sort_cache_tradeoff \
#   --data /mnt/backup_disk/Dataset/public/SOSD/books_200M_uint64_unique \
#   --queries /mnt/backup_disk/Dataset/public/SOSD/books_200M_uint64_unique.1Mtable1.bin \
#   --keys 200000000 \
#   --M 64 \
#   --sort-mibs 0,8,16,32 \
#   --policies LRU \
#   --header-mode no \
#   --summary-out build/log/sort_cache_tradeoff.csv


# ./build/rmi_bench \
#   --data src/rmi/dataset/books_10M_uint64_unique_fixed \
#   --queries /mnt/backup_disk/Dataset/public/SOSD/books_10M_uint64_unique.query.bin \
#   --rmi-data-dir src/rmi/rmi_data \
#   --branch-factors 64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152 \
#   --header yes \
#   --M 16 \
#   > build/log/rmi/books_10M_M16_rmi_bench.csv
