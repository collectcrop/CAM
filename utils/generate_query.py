import numpy as np
import random
import math
DATASETS_DIRECTORY = "/mnt/data/Dataset/public/SOSD/"

def generate_realistic_queries_from_data(keys, num_queries=100000, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    n = len(keys)
    queries = []

    # 1. hotpot query
    hotpot_ratio = 0.4
    hotpot_queries = int(num_queries * hotpot_ratio)
    num_hotpots = 5
    hotpot_size = int(0.01 * n)  # each hotpot 1% key
    for _ in range(num_hotpots):
        base = random.randint(0, n - hotpot_size)
        hotpot_indices = np.random.zipf(1.5, hotpot_queries // num_hotpots)
        hotpot_indices = np.clip(hotpot_indices, 0, hotpot_size - 1)
        queries.extend(keys[base + hotpot_indices])

    # 2. zipf query
    zipf_ratio = 0.3
    zipf_queries = int(num_queries * zipf_ratio)
    zipf_indices = np.random.zipf(1.2, zipf_queries)
    zipf_indices = np.clip(zipf_indices, 0, n - 1)
    queries.extend(keys[zipf_indices])

    # 3. Uniform query
    uniform_ratio = 0.3
    uniform_queries = int(num_queries * uniform_ratio)
    queries.extend(np.random.choice(keys, size=uniform_queries, replace=False))

    queries = np.array(queries[:num_queries], dtype=np.uint64)
    np.random.shuffle(queries)
    return queries

def generate_range_queries(num_queries, key_space_size,
                           start_dist='uniform',
                           length_dist='exponential',
                           max_length=1000000,
                           exp_scale=100):
    """
    generate range query sample
    - num_queries: query numbers
    - key_space_size: keys range
    - start_dist: distribution type of starting point, support 'uniform' or 'normal'
    - length_dist: distribution type of len, support 'uniform' or 'exponential'
    - max_length: len limit
    - exp_scale: exponential distribution factor
    """

    # generate starting point
    if start_dist == 'uniform':
        starts = np.random.randint(0, key_space_size, size=num_queries)
    elif start_dist == 'normal':
        mean = key_space_size // 2
        std = key_space_size // 6
        starts = np.random.normal(loc=mean, scale=std, size=num_queries).astype(int)
        starts = np.clip(starts, 0, key_space_size-1)
    else:
        raise ValueError("Unsupported start_dist")

    # generate length
    if length_dist == 'uniform':
        lengths = np.random.randint(1, max_length+1, size=num_queries)
    elif length_dist == 'exponential':
        lengths = np.random.exponential(scale=exp_scale, size=num_queries).astype(int) + 1
        lengths = np.clip(lengths, 1, max_length)
    else:
        raise ValueError("Unsupported length_dist")

    # calc end point, below key_space_size - 1
    ends = starts + lengths - 1
    ends = np.clip(ends, 0, key_space_size-1)

    # return [(lo, hi), ...] list
    queries = list(zip(starts, ends))
    queries = np.array(queries[:num_queries], dtype=np.uint64)
    return queries

def generate_range_queries_from_data(keys, num_queries,
                                     start_dist='uniform',
                                     length_dist='exponential',
                                     max_length_keys=100000,
                                     exp_scale=100,
                                     seed=42):
    np.random.seed(seed)
    keys = np.asarray(keys, dtype=np.uint64)
    n = len(keys)
    if n == 0:
        raise ValueError("keys is empty")

    # ---- 1. generate start position `start_idx` ----
    if start_dist == 'uniform':
        start_idx = np.random.randint(0, n, size=num_queries)
    elif start_dist == 'normal':
        mean = n // 2
        std = n // 6
        start_idx = np.random.normal(loc=mean, scale=std, size=num_queries).astype(int)
        start_idx = np.clip(start_idx, 0, n - 1)
    else:
        raise ValueError(f"Unsupported start_dist: {start_dist}")

    # ---- 2. generate length ----
    if length_dist == 'uniform':
        lengths = np.random.randint(1, max_length_keys + 1, size=num_queries)
    elif length_dist == 'exponential':
        lengths = np.random.exponential(scale=exp_scale, size=num_queries).astype(int) + 1
        lengths = np.clip(lengths, 1, max_length_keys)
    else:
        raise ValueError(f"Unsupported length_dist: {length_dist}")

    # ---- 3. calculate end_idx ----
    end_idx = start_idx + lengths - 1
    end_idx = np.clip(end_idx, 0, n - 1)

    lo_idx = np.minimum(start_idx, end_idx)
    hi_idx = np.maximum(start_idx, end_idx)

    # ---- 4. cast to keys ----
    lo_keys = keys[lo_idx]
    hi_keys = keys[hi_idx]

    queries = np.stack([lo_keys, hi_keys], axis=1).astype(np.uint64)
    return queries


def sample_unique_mixture(
    keys, k, seed=42,
    hotpot_ratio=0.4, zipf_ratio=0.3, uniform_ratio=0.3,
    num_hotpots=5, hotpot_frac=0.01,
    hotpot_zipf_a=1.5, zipf_a=1.2,
    oversample=20,                
    min_candidates=1_000_000,     
    return_sorted=True,
    strict=True,                  
):
    keys = np.asarray(keys, dtype=np.uint64)
    n = len(keys)
    if k > n:
        raise ValueError(f"k={k} > n={n}, cannot sample unique keys")

    rng = np.random.default_rng(seed)
    random.seed(seed)

    m = max(int(k * oversample), min_candidates)

    # ---- generate candidates----
    cand_parts = []

    # 1) hotspot
    m_hot = int(m * hotpot_ratio)
    if m_hot > 0:
        hotpot_size = max(1, int(hotpot_frac * n))
        per = int(np.ceil(m_hot / num_hotpots))
        for _ in range(num_hotpots):
            base = random.randint(0, max(0, n - hotpot_size))
            idx = rng.zipf(hotpot_zipf_a, size=per) - 1
            idx = np.clip(idx, 0, hotpot_size - 1)
            cand_parts.append(keys[base + idx])

    # 2) zipf over full space
    m_zipf = int(m * zipf_ratio)
    if m_zipf > 0:
        idx = rng.zipf(zipf_a, size=m_zipf) - 1
        idx = np.clip(idx, 0, n - 1)
        cand_parts.append(keys[idx])

    # 3) uniform
    m_uni = m - m_hot - m_zipf
    if m_uni > 0:
        idx = rng.integers(0, n, size=m_uni, endpoint=False)
        cand_parts.append(keys[idx])

    if not cand_parts:
        raise ValueError("No candidates generated; check ratios")

    cand = np.concatenate(cand_parts).astype(np.uint64, copy=False)
    rng.shuffle(cand)

    chosen = np.empty(k, dtype=np.uint64)
    seen = set()
    cnt = 0
    for x in cand:
        xi = int(x)
        if xi in seen:
            continue
        seen.add(xi)
        chosen[cnt] = x
        cnt += 1
        if cnt >= k:
            break

    if cnt < k:
        if strict:
            raise RuntimeError(
                f"Not enough unique keys in one-shot oversample: got {cnt}, need {k}. "
                f"Try larger oversample (e.g., 30/50) or reduce hotspot_ratio/Zipf skew."
            )
        # fallback
        remain = k - cnt
        extra = []
        while len(extra) < remain:
            idx = int(rng.integers(0, n))
            v = int(keys[idx])
            if v not in seen:
                seen.add(v)
                extra.append(keys[idx])
        chosen[cnt:] = np.array(extra, dtype=np.uint64)
        cnt = k

    if return_sorted:
        chosen.sort()
    else:
        rng.shuffle(chosen)

    return chosen

def _compute_page_intervals(keys, queries, epsilon, ipp):
    idx = np.searchsorted(keys, queries, side="left")
    idx = np.clip(idx, 0, len(keys) - 1).astype(np.int64)
    l_page = (np.maximum(0, idx - epsilon) // ipp).astype(np.int64)
    r_page = (np.minimum(len(keys) - 1, idx + epsilon) // ipp).astype(np.int64)
    return l_page, r_page


def _save_partitions(lengths, bitmap, lengths_file, bitmap_file):
    if lengths_file:
        np.array(lengths, dtype=np.int64).tofile(lengths_file)
    if bitmap_file:
        np.array(bitmap, dtype=np.int8).tofile(bitmap_file)
    if lengths_file or bitmap_file:
        print("[+] save partitions to", lengths_file, bitmap_file)


def join_partition(
    keys: np.ndarray,
    queries: np.ndarray,
    alpha: float,
    beta: float,
    eta: float,
    lambda_point: float,
    lambda_range: float = None,
    delta: float = 0.0,
    page_size: int = 4096,
    key_size: int = 8,
    epsilon: int = 16,
    N_min: int = 4096,
    K_max: int = 1024,
    gamma: float = 0.05,      # hysteresis margin: require range <= (1-gamma)*point
    phi: float = 0.0,         # absolute gain threshold in seconds (optional)
    lengths_file: str = "",
    bitmap_file: str = "",
):
    """
    Partition by immediate threshold crossing:
      - cut immediately when range becomes clearly better than point (after N_min)
      - or cut when K reaches K_max
    bitmap: 0=point, 1=range
    lengths: #queries per partition
    """

    if lambda_range is None:
        lambda_range = lambda_point

    ipp = page_size // key_size
    assert ipp > 0

    if np.any(queries[1:] < queries[:-1]):
        queries = np.sort(queries)

    Q = len(queries)
    if Q == 0:
        _save_partitions([], [], lengths_file, bitmap_file)
        return [], []

    l_page, r_page = _compute_page_intervals(keys, queries, epsilon, ipp)

    lengths, bitmap = [], []

    i = 0

    while i < Q:
        N = 0
        pmin = int(l_page[i])
        pmax = int(r_page[i])

        # union length of page intervals for point (distinct pages)
        union_len = 0
        curL = None
        curR = None

        j = i
        cut_reason = None

        while j < Q:
            # extend window
            N += 1
            lj = int(l_page[j]); rj = int(r_page[j])

            if lj < pmin: pmin = lj
            if rj > pmax: pmax = rj
            K = pmax - pmin + 1

            # union merge
            if curL is None:
                curL, curR = lj, rj
                union_len = (curR - curL + 1)
            else:
                if lj > curR:
                    union_len += (rj - lj + 1)
                    curL, curR = lj, rj
                else:
                    if rj > curR:
                        union_len += (rj - curR)
                        curR = rj

            eligible = N >= N_min

            # hard cut by K_max once eligible (or always, if you prefer)
            if eligible and K >= K_max:
                cut_reason = "Kmax"
                break

            if eligible:
                cost_point = delta + alpha * N + lambda_point * union_len
                cost_range = beta * K + eta + lambda_range * K
                gain = cost_point - cost_range

                # immediate trigger
                if (gain > phi) and (cost_range <= (1.0 - gamma) * cost_point):
                    cut_reason = "threshold"
                    break

            j += 1

        # decide partition end
        if j >= Q:
            j = Q - 1

        part_len = j - i + 1

        # decide strategy for this partition (evaluate at final window [i..j])
        # recompute final K and union_len is already for [i..j] in the loop
        # if loop ended because j == Q-1 without running eligible checks, we still decide here.
        # For correctness, re-evaluate with final K and union_len.
        # (K depends on pmin/pmax which are final values in loop.)
        K_final = pmax - pmin + 1
        cost_point_final = delta + alpha * part_len + lambda_point * union_len
        cost_range_final = beta * K_final + eta + lambda_range * K_final
        gain_final = cost_point_final - cost_range_final

        use_range = (part_len >= N_min) and (gain_final > phi) and (cost_range_final <= (1.0 - gamma) * cost_point_final)

        bitmap.append(1 if use_range else 0)
        lengths.append(part_len)

        i = j + 1

    _save_partitions(lengths, bitmap, lengths_file, bitmap_file)
    return lengths, bitmap

# def join_partition_fixed_window(
#     keys: np.ndarray,
#     queries: np.ndarray,
#     window_size: int,
#     alpha: float,
#     beta: float,
#     eta: float,
#     lambda_point: float,
#     lambda_range: float = None,
#     delta: float = 0.0,
#     page_size: int = 4096,
#     key_size: int = 8,
#     epsilon: int = 16,
#     gamma: float = 0.0,
#     phi: float = 0.0,
#     lengths_file: str = "",
#     bitmap_file: str = "",
# ):
#     """
#     Partition sorted queries into fixed-size windows, then choose point or range
#     per window using the same cost model as join_partition.

#     For each window of window_size queries:
#         N = number of queries in window
#         K = page span [pmin, pmax]
#         d = distinct pages touched by point probes (union of per-key page intervals)

#         cost_point = delta + alpha * N + lambda_point * d
#         cost_range = beta * K + eta + lambda_range * K

#     Chooses range if range is cheaper by margin (gain > phi and
#     cost_range <= (1-gamma)*cost_point).

#     bitmap: 0=point, 1=range
#     lengths: #queries per window (always window_size except possibly last)
#     """
#     if lambda_range is None:
#         lambda_range = lambda_point

#     ipp = page_size // key_size
#     assert ipp > 0

#     if np.any(queries[1:] < queries[:-1]):
#         queries = np.sort(queries)

#     Q = len(queries)
#     if Q == 0:
#         _save_partitions([], [], lengths_file, bitmap_file)
#         return [], []

#     if window_size <= 0:
#         raise ValueError(f"window_size must be > 0, got {window_size}")

#     l_page, r_page = _compute_page_intervals(keys, queries, epsilon, ipp)

#     lengths, bitmap = [], []

#     i = 0
#     while i < Q:
#         end = min(i + window_size, Q)
#         N = end - i

#         pmin = int(l_page[i])
#         pmax = int(r_page[i])

#         curL = int(l_page[i])
#         curR = int(r_page[i])
#         union_len = curR - curL + 1

#         for k in range(i + 1, end):
#             lk = int(l_page[k])
#             rk = int(r_page[k])

#             if lk < pmin:
#                 pmin = lk
#             if rk > pmax:
#                 pmax = rk

#             if lk > curR:
#                 union_len += (rk - lk + 1)
#                 curL, curR = lk, rk
#             elif rk > curR:
#                 union_len += (rk - curR)
#                 curR = rk

#         K = pmax - pmin + 1

#         cost_point = delta + alpha * N + lambda_point * union_len
#         cost_range = eta + (beta + lambda_range) * K
#         gain = cost_point - cost_range

#         use_range = (
#             (gain > phi)
#             and (cost_range <= (1.0 - gamma) * cost_point)
#         )

#         bitmap.append(1 if use_range else 0)
#         lengths.append(N)

#         i = end

#     _save_partitions(lengths, bitmap, lengths_file, bitmap_file)
#     return lengths, bitmap


def main():
    
    # sizeList = [1e7,2e7,3e7,5e7,7e7,9e7,1e8,2e8]
    # datasets = ["fb","books","osm_cellids","wiki_ts"]
    # sizeList = [1e7]
    # datasets = ["books"]
    # """ point """
    # num_queries = 8000000       #1000000
    # for dataset in datasets:
    #     for size in sizeList:
    #         print(f"[*] Generate queries for {dataset}_{int(size/1e6)}M_uint64_unique")
    #         raw = np.fromfile(f"{DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique", dtype=np.uint64)
    #         keys = raw
    #         print(f"[*] Loaded {len(keys)} keys.")
    #         queries = generate_realistic_queries_from_data(keys,num_queries)
    #         queries.tofile(f"{DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique.8Mquery.bin")
    #         print(f"[+] save queries to {DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique.8Mquery.bin successfully!")
    
    """ range """
    # num_queries = 4000000
    # queries = generate_range_queries(num_queries, 8000000000,
    #                              start_dist='uniform',
    #                              length_dist='uniform',
    #                              max_length=5000000,
    #                              exp_scale=10
    #                              )
    # queries.tofile(f"{DATASETS_DIRECTORY}range_query_{int(num_queries/1e6)}M_uu.bin")
    # print(f"[+] save queries to {DATASETS_DIRECTORY}range_query_{int(num_queries/1e6)}M_uu.bin successfully!")
    
    # datasets = ["fb"]
    # sizeList = [1e7,2e7,5e7,7e7,1e8]
    # num_queries = 4000000
    # for dataset in datasets:
    #     for size in sizeList:
    #         raw = np.fromfile(f"{DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique",
    #                         dtype=np.uint64)
    #         keys = raw        

    #         queries = generate_range_queries_from_data(
    #             keys,
    #             num_queries,
    #             start_dist='uniform',     
    #             length_dist='uniform', 
    #             max_length_keys=5000,      
    #             exp_scale=100,
    #             seed=42
    #         )

    #         out_path = f"{DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique.{int(num_queries/1e6)}Mrange.bin"
    #         queries.tofile(out_path)
    #         print(f"[+] save range queries to {out_path} successfully!")
    
    """ join """
    # sizeList = [2e8]
    # datasets = ["books"]
    # num_queries = 1000000
    # for dataset in datasets:
    #     for size in sizeList:
    #         print(f"[*] Generate queries for {dataset}_{int(size/1e6)}M_uint64_unique")
    #         raw = np.fromfile(f"{DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique", dtype=np.uint64)
    #         keys = raw
    #         print(f"[*] Loaded {len(keys)} keys.")
    #         # queries = sample_unique_mixture(keys,num_queries)
    #         queries = sample_unique_mixture(keys,num_queries,hotpot_ratio=0.4,zipf_ratio=0.3,oversample=100,return_sorted=False)
    #         print(f"[*] Loaded {len(queries)} queries.")
    #         queries.tofile(f"{DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique.{int(num_queries/1e6)}Mtable1.bin")
    #         print(f"[+] save queries to {DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique.{int(num_queries/1e6)}Mtable1.bin successfully!")
    
    # datasets = ["books"]
    # num_queries = 100000
    # for dataset in datasets:
    #     for size in sizeList:
    #         print(f"[*] Generate queries for {dataset}_{int(size/1e6)}M_uint64_unique")
    #         raw = np.fromfile(f"{DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique", dtype=np.uint64)
    #         keys = raw
    #         print(f"[*] Loaded {len(keys)} keys.")
    #         queries = generate_join_table_from_data(keys,num_queries,num_segments=10,active_segments=5)
    #         queries.tofile(f"{DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique.{int(num_queries/1e3)}Ktable2.bin")
    #         print(f"[+] save queries to {DATASETS_DIRECTORY}{dataset}_{int(size/1e6)}M_uint64_unique.{int(num_queries/1e3)}Ktable2.bin successfully!")

    
    """ partition join"""
    # alpha = 1.637e-06
    # beta = 1.719e-06
    # lambda_point = 1.195e-06
    # lambda_range = 4.669e-07
    # delta = 0.005
    # eta = 4.421e-06
    # page_size = 4096
    # epsilon = 16
    # queryfile = "books_200M_uint64_unique.1Mtable2.bin"
    # dataset = "books_200M_uint64_unique"
    # raw = np.fromfile(f"{DATASETS_DIRECTORY}{dataset}", dtype=np.uint64)
    # keys = raw
    # queries = np.fromfile(f"{DATASETS_DIRECTORY}{queryfile}", dtype=np.uint64)
    # lengths_file=f"{DATASETS_DIRECTORY}{queryfile}.par".replace(".bin","")
    # bitmap_file=f"{DATASETS_DIRECTORY}{queryfile}.bitmap".replace(".bin","")
    # join_partition(keys,queries,alpha=alpha,beta=beta,eta=eta,lambda_point=lambda_point,lambda_range=lambda_range,delta=delta,
    #                page_size=page_size,key_size=8,epsilon=epsilon,N_min=4096,K_max=8192
    #                ,lengths_file=lengths_file,bitmap_file=bitmap_file)
            
if __name__ == '__main__':
    main()
