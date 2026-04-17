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
#   --output-dir data/outputs/figures/epsilon_analysis

./build/pgm_cache_simulate \
  --data /mnt/backup_disk/Dataset/public/SOSD/books_100M_uint64_unique \
  --queries /mnt/backup_disk/Dataset/public/SOSD/books_100M_uint64_unique.query.bin \
  --keys 100000000 \
  --M 10 \
  --policies FIFO,LRU,LFU \
  --strategies all_in_once \
  --budget-mode estimated \
  --summary-out build/log/cmp/books_100M_M10_summary_sim.csv


# ./build/sort_cache_tradeoff \
#   --data /mnt/backup_disk/Dataset/public/SOSD/books_200M_uint64_unique \
#   --queries /mnt/backup_disk/Dataset/public/SOSD/books_200M_uint64_unique.1Mtable1.bin \
#   --keys 200000000 \
#   --M 64 \
#   --sort-mibs 0,8,16,32 \
#   --policies LRU \
#   --header-mode no \
#   --summary-out build/log/sort_cache_tradeoff.csv
