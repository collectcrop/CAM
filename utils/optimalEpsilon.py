import math
import numpy as np
import time
import os, sys
from scipy.optimize import brentq  
from scipy.special import zeta     
from collections import Counter
from scipy.signal import fftconvolve

alpha = 1
DATASETS_DIRECTORY = "/mnt/backup_disk/Dataset/public/SOSD/"
LOG_DIRECTORY = "build/log/"
BUDGET_MODE = "RAW"
def build_uniform_box_kernel(epsilon):
    L = 2 * epsilon + 1
    return np.ones(L, dtype=np.float64) / L

def triangular_kernel_from_box(epsilon):
    L = 2 * epsilon + 1
    # discrete triangular: [1,2,3,...,L-1,L,L-1,...,1] normalized by L^2
    up = np.arange(1, L+1, dtype=np.float64)
    down = up[-2::-1]  # L-1 down to 1
    tri = np.concatenate([up, down])
    return tri / (L * L)  # normalization: sum(tri) = L^2 / L^2 = 1

def prepare_query_histogram(query_file, data):
    """
    Precompute query position histogram H once:
      H[p] = number of queries whose predecessor position is p.
    This can be reused across all epsilons.
    """
    if isinstance(query_file, str):
        queries = np.fromfile(query_file, dtype=np.uint64)
    else:
        queries = np.asarray(query_file, dtype=np.uint64)

    Q = len(queries)
    N = len(data)
    pos = np.searchsorted(data, queries, side='right') - 1
    pos = np.clip(pos, 0, N - 1).astype(np.int64)
    H = np.bincount(pos, minlength=N).astype(np.float64)
    return H, Q

def prepare_query_positions(query_file, data):
    """
    Precompute predecessor positions once:
      pos[i] = predecessor position of query i in sorted data.
    """
    if isinstance(query_file, str):
        queries = np.fromfile(query_file, dtype=np.uint64)
    else:
        queries = np.asarray(query_file, dtype=np.uint64)

    N = len(data)
    pos = np.searchsorted(data, queries, side='right') - 1
    pos = np.clip(pos, 0, N - 1).astype(np.int64)
    return pos, len(pos)

def _page_prob_table_for_epsilon(epsilon, ipp):
    """
    Build exact page-level probability table:
      probs[d][s] = P(page q+d is accessed | true position = q*ipp + s)
    where s in [0, ipp-1], and d is relative page offset.

    Returns:
      d_vals: np.array of relative page offsets
      prob_table: np.array shape [len(d_vals), ipp]
    """
    eps = int(epsilon)
    ipp = int(ipp)
    denom = float(2 * eps + 1)

    s = np.arange(ipp, dtype=np.int64)

    # Global possible relative-page range
    # Need [s-2eps, s+2eps] intersects page d => d candidates are very few.
    d_min = (0 - 2 * eps) // ipp
    d_max = (ipp - 1 + 2 * eps) // ipp
    d_vals = np.arange(d_min, d_max + 1, dtype=np.int64)

    table = np.zeros((len(d_vals), ipp), dtype=np.float64)

    for i, d in enumerate(d_vals):
        # Closed form:
        # L = max(-eps, d*ipp - s - eps)
        # U = min( eps, (d+1)*ipp - 1 - s + eps)
        L = np.maximum(-eps, d * ipp - s - eps)
        U = np.minimum( eps, (d + 1) * ipp - 1 - s + eps)
        table[i] = np.maximum(0, U - L + 1) / denom

    return d_vals, table

def estimate_page_counts_from_queryfile(query_file, data, epsilon, ipp, use_fft=False, H=None, Q=None):
    """
    Optimized exact page-level estimator for point queries.

    Supported modes:
      1) H is None:
         - directly compute query predecessor positions and aggregate page counts
      2) H is a dense histogram over true positions (len(H) == len(data)):
         - convert only the nonzero support of H into sparse (page, offset, count)
      3) H is a dict produced by prepare_query_positions/prepare_sparse_histogram-like cache:
         - H = {"kind": "positions", "pos": pos}
         - H = {"kind": "sparse_hist", "nz": nz, "cnt": cnt}

    Args:
      query_file: binary query file path or numpy array
      data: sorted data keys (np.ndarray)
      epsilon: int
      ipp: items per page
      use_fft: kept only for interface compatibility; ignored
      H: optional cache object / dense histogram
      Q: optional total number of queries

    Returns:
      page_counts: expected per-page reference counts
      T_pos: None (kept for interface compatibility)
      Q: number of queries
    """
    assert isinstance(data, np.ndarray)
    N = len(data)
    P = math.ceil(N / ipp)

    # Precompute exact per-offset page probabilities
    d_vals, prob_table = _page_prob_table_for_epsilon(epsilon, ipp)

    page_counts = np.zeros(P, dtype=np.float64)

    # ------------------------------------------------------------
    # Case A: direct positions path (best when sweeping is not cached)
    # ------------------------------------------------------------
    if H is None:
        pos, Q_local = prepare_query_positions(query_file, data)
        pages = pos // ipp
        offsets = pos % ipp

        for row, d in enumerate(d_vals):
            tgt = pages + d
            valid = (tgt >= 0) & (tgt < P)
            if not np.any(valid):
                continue

            w = prob_table[row, offsets[valid]]
            # bincount accumulates all references to target pages in C
            page_counts += np.bincount(tgt[valid], weights=w, minlength=P)

        return page_counts, None, Q_local

    # ------------------------------------------------------------
    # Case B: cached positions
    # ------------------------------------------------------------
    if isinstance(H, dict) and H.get("kind") == "positions":
        pos = np.asarray(H["pos"], dtype=np.int64)
        Q_local = len(pos) if Q is None else Q
        pages = pos // ipp
        offsets = pos % ipp

        for row, d in enumerate(d_vals):
            tgt = pages + d
            valid = (tgt >= 0) & (tgt < P)
            if not np.any(valid):
                continue

            w = prob_table[row, offsets[valid]]
            page_counts += np.bincount(tgt[valid], weights=w, minlength=P)

        return page_counts, None, Q_local

    # ------------------------------------------------------------
    # Case C: cached sparse histogram
    # ------------------------------------------------------------
    if isinstance(H, dict) and H.get("kind") == "sparse_hist":
        nz = np.asarray(H["nz"], dtype=np.int64)
        cnt = np.asarray(H["cnt"], dtype=np.float64)
        Q_local = int(round(cnt.sum())) if Q is None else Q

        pages = nz // ipp
        offsets = nz % ipp

        for row, d in enumerate(d_vals):
            tgt = pages + d
            valid = (tgt >= 0) & (tgt < P)
            if not np.any(valid):
                continue

            w = cnt[valid] * prob_table[row, offsets[valid]]
            page_counts += np.bincount(tgt[valid], weights=w, minlength=P)

        return page_counts, None, Q_local

    # ------------------------------------------------------------
    # Case D: dense histogram H[r]
    # ------------------------------------------------------------
    H = np.asarray(H)
    if H.shape[0] != N:
        raise ValueError(f"len(H)={H.shape[0]} != len(data)={N}")

    nz = np.flatnonzero(H)
    if nz.size == 0:
        return page_counts, None, 0 if Q is None else Q

    cnt = H[nz].astype(np.float64, copy=False)
    Q_local = int(round(cnt.sum())) if Q is None else Q

    pages = nz // ipp
    offsets = nz % ipp

    for row, d in enumerate(d_vals):
        tgt = pages + d
        valid = (tgt >= 0) & (tgt < P)
        if not np.any(valid):
            continue

        w = cnt[valid] * prob_table[row, offsets[valid]]
        page_counts += np.bincount(tgt[valid], weights=w, minlength=P)

    return page_counts, None, Q_local

# def estimate_page_counts_from_queryfile(query_file, data, epsilon, ipp, use_fft=False, H=None, Q=None):
#     """
#     args:
#       query_file: binary file path or numpy array of query keys (uint64)
#       data: sorted data keys (np.array dtype uint64)
#       epsilon: int
#       ipp: items per page
#     returns:
#       page_counts: np.array length num_pages, expected counts per page (sum ~= Q * (2eps+1))
#       T_pos: np.array length N, expected counts per position
#       Q: queries length
#     """    
#     assert isinstance(data, np.ndarray)
#     N = len(data)
#     if H is None:
#         H, Q = prepare_query_histogram(query_file, data)
#     else:
#         H = np.asarray(H, dtype=np.float64)
#         if H.shape[0] != N:
#             raise ValueError(f"len(H)={H.shape[0]} != len(data)={N}")
#         if Q is None:
#             Q = int(round(float(H.sum())))

#     # 2) construct k = g * h. assume g = uniform box, h = uniform box => k = triangular
#     k = triangular_kernel_from_box(epsilon)  # length K = 4*eps + 1, sums to 1

#     # 3) convolution
#     if use_fft:
#         T = fftconvolve(H, k, mode='same')
#     else:
#         T = np.convolve(H, k, mode='same')

#     # 4) page aggregation
#     num_pages = math.ceil(N / ipp)
#     pad_len = num_pages * ipp - N
#     if pad_len > 0:
#         T_padded = np.concatenate([T, np.zeros(pad_len, dtype=T.dtype)])
#     else:
#         T_padded = T
#     page_counts = T_padded.reshape(num_pages, ipp).sum(axis=1)  # expected counts per page

#     return page_counts, T, Q

def extract_data_gap_distribution(data_file):
    """
    input:
        data_file: binary file path of sorted data keys (uint64)
    output:
        miu: float, mean of gaps
        sigma: float, stddev of gaps
    """
    data = np.fromfile(data_file, dtype=np.uint64)[1:]
    gaps = np.diff(data)
    miu = np.mean(gaps)
    sigma = np.std(gaps)
    return miu, sigma

def zipf_popularity(N, alpha):
    norm_const = sum(1 / (i ** alpha) for i in range(1, N + 1))
    return np.array([1 / (i ** alpha) / norm_const for i in range(1, N + 1)])

# def che_characteristic_time(qs, C, Q):
#     # qs: array of popularity q(i)
#     print("[*] starting solve characteristic time")
#     m = int(np.sum(qs > 0))
#     def f(t):
#         return np.sum(1 - np.exp(-qs * t)) - C
#     # root-finding to solve C = Σ(1 - e^{-q_i t})
#     print(C,m)
#     return brentq(f, 1e-6, 1e6)

def che_characteristic_time(qs, C, t0=1e-9, grow=10.0, max_iter=60):
    """
    Solve for t_C in:  C = sum_i (1 - exp(-q_i * t_C))
    - qs: array-like of per-object request rates or counts (q_i >= 0)
    - C : cache capacity measured in "number of objects" (or equivalently the unit that matches 1 per object)
    - Q : (optional) total queries in the window; not used in solving, here just for signature compatibility

    Returns:
        t_C (float): finite positive solution; 0.0 when C<=0; np.inf when C >= m (no finite solution).
    """
    qs = np.asarray(qs, dtype=float)
    if np.any(~np.isfinite(qs)) or np.any(qs < 0):
        raise ValueError("qs must be finite and >= 0")
    m = int(np.sum(qs > 0))

    # Boundary/degenerate cases
    if C <= 0:
        return 0.0
    if C >= m:
        # No finite solution; in practice "all objects fit" => t_C = +inf
        return np.inf

    # Scale to improve conditioning
    qpos = qs[qs > 0]
    qbar = float(np.mean(qpos)) if qpos.size > 0 else 1.0
    r = np.where(qs > 0, qs / qbar, 0.0)

    def f_tau(tau):
        return np.sum(1.0 - np.exp(-r * tau)) - C

    # Bracket automatically on (t0, +inf)
    a, b = t0, t0
    fa = f_tau(a)
    for _ in range(max_iter):
        b *= grow
        fb = f_tau(b)
        if fa * fb <= 0:
            break
    else:
        # Extremely pathological scaling; fall back to a wide static bracket
        a, b = 1e-12, 1e12

    tau = brentq(f_tau, a, b)
    return tau / qbar


def che_hit_rates(qs, t_C):
    return 1 - np.exp(-qs * t_C)

def predict_height_segments(epsilon,epsilon_i,n,k):
    h = 1
    num = math.ceil(n/(k*epsilon**2))
    segments = num
    while (num>1):
        h = h+1
        num = math.ceil(num/(k*epsilon_i**2))
        segments += num
    return h,segments            # 1+math.ceil(math.log(n/(k*epsilon**2),k*epsilon_i**2))


# def expected_DAC(epsilon, ipp):
#     dac = 0
#     for k in range(ipp + 1):
#         term = 1 + math.ceil((epsilon - k) / ipp) + math.ceil((epsilon - ipp + k) / ipp)
#         dac += term
#     return dac / ipp

def expected_DAC(epsilon, ipp ,s="all_in_once"):
    if s == "all_in_once":
        return 1 + (2*epsilon/ipp)
    elif s == "one_by_one":
        return 1 + (epsilon/ipp)

def expected_IAC(epsilon, ipp):
    return 1 + (2*epsilon/ipp)

def uniform_ratio(C,N):
    return C / N

def zipf_ratio(C,N,alpha):
    qs = zipf_popularity(N, alpha)
    t_C = che_characteristic_time(qs, C)
    hit_rates = che_hit_rates(qs, t_C)
    return np.sum(qs * hit_rates)

# def sample_ratio(C,N,qs,Q):
#     t_C = che_characteristic_time(qs, C, Q)
#     print("[+] successfully solved characteristic_time")
#     hit_rates = che_hit_rates(qs, t_C)
#     return np.sum(qs * hit_rates)

def lru_hit_ratio(qs, C, Q=0):
    """
    Che's approximation for LRU.
    qs must be a probability vector summing to 1.
    """
    m = int(np.sum(qs > 0))
    if C >= m:
        return 1.0

    t_C = che_characteristic_time(qs, C)
    if np.isinf(t_C):
        return 1.0

    hit_rates = 1.0 - np.exp(-qs * t_C)
    return float(np.sum(qs * hit_rates))

def lfu_hit_ratio(qs, C):
    """
    Exact LFU hit ratio under IRM.
    qs must be a probability vector summing to 1.
    """
    if C <= 0:
        return 0.0

    qs = np.asarray(qs, dtype=np.float64)
    qs = qs[qs > 0]
    if qs.size == 0:
        return 0.0

    m = qs.size
    if C >= m:
        return 1.0

    qs_sorted = np.sort(qs)[::-1]
    return float(np.sum(qs_sorted[:int(C)]))


def fifo_random_characteristic_time(qs, C, t0=1e-12, grow=10.0, max_iter=80):
    """
    Solve tau_C from:
        C = sum_i (q_i * tau) / (1 - q_i + q_i * tau)
    qs should sum to 1.
    """
    qs = np.asarray(qs, dtype=np.float64)
    if np.any(qs < 0) or np.any(~np.isfinite(qs)):
        raise ValueError("qs must be finite and non-negative")
    qs = qs[qs > 0]
    m = qs.size
    if C <= 0:
        return 0.0
    if C >= m:
        return np.inf
    def f(tau):
        vals = (qs * tau) / (1.0 - qs + qs * tau)
        return np.sum(vals) - C
    a = t0
    fa = f(a)
    b = a
    for _ in range(max_iter):
        b *= grow
        fb = f(b)
        if fa * fb <= 0:
            return brentq(f, a, b)
    return np.inf

def fifo_random_hit_rates(qs, tau_C):
    """
    Per-object hit probability under FIFO/RANDOM approximation.
    """
    qs = np.asarray(qs, dtype=np.float64)
    if np.isinf(tau_C):
        return np.ones_like(qs)
    return (qs * tau_C) / (1.0 - qs + qs * tau_C)

def fifo_random_hit_ratio(qs, C):
    """
    Approximate FIFO/RANDOM hit ratio under IRM.
    """
    qs = np.asarray(qs, dtype=np.float64)
    qs = qs[qs > 0]
    if qs.size == 0 or C <= 0:
        return 0.0

    m = qs.size
    if C >= m:
        return 1.0

    tau_C = fifo_random_characteristic_time(qs, C)
    if np.isinf(tau_C):
        return 1.0

    h_i = fifo_random_hit_rates(qs, tau_C)
    return float(np.sum(qs * h_i))

def validate_ratio(ratio):
    if ratio >= 1.0:
        h = 1.0
    elif ratio <= 0:
        h = 0.0
    else:
        h = ratio
    return h

def model_cost_given_capacity(
    epsilon,
    n,
    seg_size,
    ipp,
    ps,
    C_pages,
    type="sample",
    data_file="",
    query_file="",
    s="all_in_once",
):
    # index size
    M_index = n * seg_size / (2 * epsilon)

    # buffer size
    M_buffer = C_pages * ps

    M_eff = M_index + M_buffer
    cost, h = cost_function(
        epsilon,
        n,
        seg_size,
        M_eff,
        ipp,
        ps,
        query_file=query_file,
        data_file=data_file,
        s=s,
    )
    return cost

def join_cost_function(
    epsilon,
    n,
    seg_size,
    M,
    ipp,
    ps,
    data_file="",
    join_file="",
    par_file="",
    bitmap_file="",
    assume_sorted=True,
    return_detail=False,
):
    """
    Sorted-order hit-rate model:
      - misses = #distinct pages touched (union of page intervals)
      - hit rate = (n_ref - N_distinct) / n_ref
      - cost (avg physical IO per join key) = N_distinct / Q

    Partition rule:
      - bitmap=0 (point region): execute as point probes per key with window [pos-eps, pos+eps]
      - bitmap=1 (range region): execute as a range scan covering the whole segment window
    """

    data = np.fromfile(data_file, dtype=np.uint64)[1:]
    queries = np.fromfile(join_file, dtype=np.uint64)
    Q = int(len(queries))
    N = int(len(data))
    if Q == 0 or N == 0:
        if return_detail:
            return 0.0, 0.0, {"Q": Q, "n_refs": 0, "N_distinct": 0}
        return 0.0, 0.0

    # sorted precondition (optional safeguard)
    if assume_sorted and np.any(queries[1:] < queries[:-1]):
        queries = np.sort(queries)

    lengths = np.fromfile(par_file, dtype=np.int64)
    bitmap  = np.fromfile(bitmap_file, dtype=np.int8)
    if lengths.size != bitmap.size:
        raise ValueError(f"lengths.size({lengths.size}) != bitmap.size({bitmap.size})")
    if int(lengths.sum()) != Q:
        raise ValueError(f"sum(lengths)={int(lengths.sum())} != Q={Q}")

    M_index  = n * seg_size / (2 * epsilon)
    M_buffer = M - M_index
    C_pages  = M_buffer / ps

    C_delta  = 1 + int(math.ceil((2.0 * epsilon) / ipp))
    union_len = 0
    curL = None
    curR = None
    def push_interval(l, r):
        nonlocal union_len, curL, curR
        if l > r:
            return
        if curL is None:
            curL, curR = l, r
            union_len += (curR - curL + 1)
            return
        if l > curR + 1:
            curL, curR = l, r
            union_len += (curR - curL + 1)
        else:
            if r > curR:
                union_len += (r - curR)
                curR = r

    n_refs = 0

    off = 0
    for L, b in zip(lengths, bitmap):
        L = int(L)
        seg = queries[off:off+L]
        off += L

        if b == 0:
            # point: per key interval [pos-eps, pos+eps] -> pages [l,r]
            pos = np.searchsorted(data, seg, side="right") - 1
            pos = np.clip(pos, 0, N-1).astype(np.int64)

            start_pos = np.maximum(0, pos - int(epsilon))
            end_pos   = np.minimum(N-1, pos + int(epsilon))
            l_pages   = (start_pos // ipp).astype(np.int64)
            r_pages   = (end_pos   // ipp).astype(np.int64)

            # logical refs add
            n_refs += int(np.sum(r_pages - l_pages + 1))

            # union add (intervals are monotone in sorted order)
            for l, r in zip(l_pages, r_pages):
                push_interval(int(l), int(r))

        else:
            # range: one scan interval for the whole partition
            # use first/last key to bound, then expand by epsilon
            lo = int(seg[0])
            hi = int(seg[-1])

            pos_lo = int(np.searchsorted(data, lo, side="right") - 1)
            pos_hi = int(np.searchsorted(data, hi, side="right") - 1)
            pos_lo = max(0, min(N-1, pos_lo))
            pos_hi = max(0, min(N-1, pos_hi))

            start_pos = max(0, pos_lo - int(epsilon))
            end_pos   = min(N-1, pos_hi + int(epsilon))
            l = start_pos // ipp
            r = end_pos   // ipp

            # logical refs: scan each page once
            n_refs += int(r - l + 1)

            # union add
            push_interval(int(l), int(r))

    if n_refs <= 0:
        if return_detail:
            return 0.0, 0.0, {"Q": Q, "n_refs": 0, "N_distinct": int(union_len)}
        return 0.0, 0.0

    # sorted-order hit estimate
    h = (n_refs - union_len) / float(n_refs)
    if h < 0: h = 0.0
    if h > 1: h = 1.0

    # avg physical IO per join key = distinct pages / Q
    cost = union_len / float(Q)

    detail = {
        "Q": Q,
        "n_refs": int(n_refs),
        "N_distinct": int(union_len),
        "h_sorted": float(h),
        "C_pages": float(C_pages),
        "C_delta": int(C_delta),
        "may_be_optimistic": bool(C_pages < C_delta),
    }
    return (cost, h, detail) if return_detail else (cost, h)


def get_RDAC(rlo, rhi, epsilon, ipp):
    """
    Exact E[RDAC] under conditioned-uniform (u,v) in F_delta:
      u,v in [-eps, eps], v >= u - delta, delta=rhi-rlo.
    Works for scalar or numpy arrays rlo/rhi. Returns numpy array.
    Complexity: O(eps) per query.
    """
    eps = int(epsilon)
    C = int(ipp)
    u_vals = np.arange(-eps, eps + 1, dtype=np.int64)  # size = 2eps+1

    rlo = np.asarray(rlo, dtype=np.int64)
    rhi = np.asarray(rhi, dtype=np.int64)
    # ensure broadcastable
    rlo_flat = rlo.reshape(-1)
    rhi_flat = rhi.reshape(-1)

    out = np.empty_like(rlo_flat, dtype=np.float64)

    for i in range(rlo_flat.size):
        lo = int(rlo_flat[i])
        hi = int(rhi_flat[i])
        if hi < lo:
            print("[Error] rhi < rlo Detected.")
        delta = hi - lo

        # L(u) = max(-eps, u - delta)
        L = np.maximum(-eps, u_vals - delta)  # int
        w = (eps - L + 1).astype(np.int64)    # number of feasible v per u
        F = int(w.sum())                       # |F_delta|

        # S(u) = floor((rlo + u - eps)/C); clamp start_pos >= 0 for robustness
        start_pos = np.maximum(0, lo + u_vals - eps)
        Su = (start_pos // C).astype(np.int64)

        # E(v) = floor((rhi + v + eps)/C)
        v_vals = u_vals
        Ev = ((hi + v_vals + eps) // C).astype(np.int64)

        # Prefix sums of Ev over v in [-eps..eps]
        # PE[k] = sum_{v=-eps}^{v_vals[k]} Ev(v)
        PE = np.cumsum(Ev, dtype=np.int64)
        total_E = int(PE[-1])

        # sum_{u} sum_{v=L(u)}^{eps} Ev(v)
        # map L(u) to index in v_vals: idx = L(u) - (-eps) = L(u) + eps
        idx = (L + eps).astype(np.int64)
        # sum_{v=L}^{eps} Ev(v) = total_E - sum_{v=-eps}^{L-1} Ev(v)
        # prefix up to L-1 corresponds to PE[idx-1], if idx>0 else 0
        prefix_before = np.where(idx > 0, PE[idx - 1], 0)
        inner_sum_E = (total_E - prefix_before).sum(dtype=np.int64)

        # sum_{u} w(u) * S(u)
        sum_wS = (w * Su).sum(dtype=np.int64)

        out[i] = 1.0 + (inner_sum_E - sum_wS) / F

    return out.reshape(rlo.shape)

def cache_hit_ratio(policy, C, qs, Q=0):
    """
    policy: "LRU" | "LFU" | "FIFO" | "RANDOM"
    C: cache capacity in pages
    qs: page popularity distribution (sum(qs)=1)
    Q: optional trace length, only for degenerate handling
    """
    qs = np.asarray(qs, dtype=np.float64)
    if qs.size == 0 or C <= 0:
        return 0.0

    # only positive-probability objects matter
    qs = qs[qs > 0]
    if qs.size == 0:
        return 0.0

    # normalize for safety
    s = qs.sum()
    if s <= 0:
        return 0.0
    qs = qs / s

    if policy.upper() == "LRU":
        return lru_hit_ratio(qs, C, Q)
    elif policy.upper() == "LFU":
        return lfu_hit_ratio(qs, C)
    elif policy.upper() in ("FIFO", "RANDOM"):
        return fifo_random_hit_ratio(qs, C)
    else:
        raise ValueError(f"Unknown policy: {policy}")
    
    
def estimate_page_counts_from_range_queryfile(rlos, rhis, epsilon, ipp, N):
    """
    Expected per-page reference counts under conditioned-uniform error pairs:
      u,v in [-eps,eps], v >= u - delta, delta=rhi-rlo
    and all-at-once fetch interval [S(u), E(v)] inclusive, where
      S(u)=floor((rlo+u-eps)/ipp), E(v)=floor((rhi+v+eps)/ipp).
    We exploit that pages in [plo, phi] are always referenced with prob 1,
    and only a small number of boundary-adjacent pages have fractional probs.
    """
    eps = int(epsilon)
    C = int(ipp)
    u_vals = np.arange(-eps, eps + 1, dtype=np.int64)  # [-eps..eps]
    count = Counter()

    # true positions
    # lo_pos = np.searchsorted(data, lo_keys, side='right') - 1
    # hi_pos = np.searchsorted(data, hi_keys, side='right') - 1
    # lo_pos = np.clip(lo_pos, 0, N - 1).astype(np.int64)
    # hi_pos = np.clip(hi_pos, 0, N - 1).astype(np.int64)

    # ensure lo_pos <= hi_pos (robust)
    swap = rlos > rhis
    if np.any(swap):
        print("[Error] rhi < rlo Detected.")
        sys.exit(1)

    for i in range(len(rlos)):
        rlo = int(rlos[i])
        rhi = int(rhis[i])
        delta = rhi - rlo

        plo = rlo // C
        phi = rhi // C

        # --- feasible-set weights ---
        # L(u)=max(-eps, u-delta), number of feasible v for each u: w(u)=eps-L(u)+1
        L = np.maximum(-eps, u_vals - delta)
        w = (eps - L + 1).astype(np.int64)
        F = float(w.sum())  # |F_delta|

        # --- core pages: always referenced with prob 1 ---
        for p in range(int(plo), int(phi) + 1):
            count[p] += 1.0

        # --- compute S(u) pages (start page) for left boundary ---
        # clamp start_pos >= 0 for robustness near beginning
        start_pos = np.maximum(0, rlo + u_vals - eps)
        Su = (start_pos // C).astype(np.int64)

        # Left boundary pages range: from min_start_page to plo-1
        min_start_pos = max(0, rlo - 2 * eps)
        min_start_page = min_start_pos // C

        for p in range(int(min_start_page), int(plo)):
            # prob = sum_{u: Su<=p} w(u) / F
            prob = float(w[Su <= p].sum()) / F
            if prob > 0:
                count[p] += prob

        # --- right boundary pages ---
        # Rightmost possible end position: rhi + 2eps (clamp to N-1)
        max_end_pos = min(N - 1, rhi + 2 * eps)
        max_end_page = max_end_pos // C

        # For a page p>phi, need E(v) >= p.
        # E(v)=floor((rhi+v+eps)/C) >= p  <=>  rhi+v+eps >= p*C  <=> v >= p*C-(rhi+eps)
        for p in range(int(phi) + 1, int(max_end_page) + 1):
            v0 = p * C - (rhi + eps)          # minimal v (may be < -eps)
            Vp = max(-eps, v0)                # clamp to [-eps, ...]
            if Vp > eps:
                continue  # impossible to reach this page

            # for each u, feasible v lower bound is max(L(u), Vp)
            lb = np.maximum(L, Vp)
            cnt_v = np.maximum(0, eps - lb + 1)  # number of v satisfying both constraints
            prob = float(cnt_v.sum()) / F
            if prob > 0:
                count[p] += prob

    return count

def range_cost_function(epsilon, n, seg_size, M, ipp, ps, 
                        query_file="", data_file="",cache_policy="LRU"):
    M_index = n * seg_size / (2 * epsilon)
    M_buffer = M - M_index
    C = M_buffer/ps
    total_pages = math.ceil(n / ipp)

    data = np.fromfile(data_file, dtype=np.uint64)
    queries = np.fromfile(query_file, dtype=np.uint64).reshape(-1, 2)
    lo_keys,hi_keys = queries[:,0],queries[:,1]
    rlo = np.searchsorted(data, lo_keys, side='right') - 1
    rhi = np.searchsorted(data, hi_keys, side='right') - 1
    rlo = np.clip(rlo, 0, n - 1).astype(np.int64)
    rhi = np.clip(rhi, 0, n - 1).astype(np.int64)
    # keys = rhi - rlo
    # RDAC = rhi/ipp - rlo/ipp + 1 + 2*epsilon/ipp
    # old_RDAC = rhi/ipp - rlo/ipp + 1 + 2*epsilon/ipp
    RDAC = get_RDAC(rlo,rhi,epsilon,ipp)
    print("RDAC mean:", RDAC.mean(), "max:", RDAC.max())
    # print("old RDAC mean:", old_RDAC.mean(), "max:", old_RDAC.max())
    page_counts = estimate_page_counts_from_range_queryfile(rlo, rhi, epsilon, ipp, n)
    total = sum(page_counts.values())
    q = np.array([f / total for f in page_counts.values()], dtype=np.float64)
    q = np.sort(q)[::-1]
    
    # bug here!!!!!
    buffer_ratio = cache_hit_ratio(cache_policy, C, total_pages, q)
    # print(buffer_ratio)
    h = validate_ratio(buffer_ratio)
    print((1-h)*RDAC.sum()/len(queries))
    return (1-h)*RDAC.sum()/len(queries), h
    
def cost_function(epsilon, n, seg_size, M, ipp, ps,
                  query_file="", data_file="", s="all_in_once", cache_policy="LRU",
                  data_arr=None, H=None, Q=None):
    M_index = n * seg_size / (2 * epsilon)
    M_buffer = M - M_index if BUDGET_MODE!="RAW" else M
    C = M_buffer/ps
    total_pages = math.ceil(n / ipp)
    if (C==0.0):
        h = 0.0
    else:
        data = data_arr if data_arr is not None else np.fromfile(data_file,dtype=np.uint64)[1:]
        page_counts, Tpos, Q_local = estimate_page_counts_from_queryfile(
            query_file, data, epsilon, ipp, H=H, Q=Q
        )
        total_page_requests = page_counts.sum()
        if total_page_requests <= 0:
            h = 0.0
        else:
            q = page_counts / total_page_requests
            h = cache_hit_ratio(cache_policy, int(C), q, Q_local)
        # q = page_counts / total_page_requests
        # q = np.sort(q)[::-1]
        # buffer_ratio = sample_ratio(C, total_pages, q, Q)
        
        
        h = validate_ratio(h)
    return (1 - h) * expected_DAC(epsilon, ipp, s), h

def getExpectedRangeCostPerEpsilon(ipp, seg_size, M, n, ps,
                                   data_file="",query_file="",cache_policy="LRU"):
    data = f"{DATASETS_DIRECTORY}{data_file}"
    query = f"{DATASETS_DIRECTORY}{query_file}"
    eps_list = []
    cost_list = []
    h_list = []
    time_list = []
    least_eps = math.ceil(n*seg_size/(2*M))
    r = range(least_eps, 65, 2)
    # r = range(8,17)
    for eps in r:
        t1 = time.time()
        cost,h = range_cost_function(eps, n, seg_size, M, ipp, ps, query, data, cache_policy)
        eps_list.append(eps)
        cost_list.append(cost)
        h_list.append(h)
        t2 = time.time()
        print(f"eps: {eps}, cost: {cost}, ratio: {h}, time: {t2-t1}")
        time_list.append(t2-t1)
    print(eps_list)
    print("cost:",cost_list)
    print("ratio:",h_list)
    
    print("group avg time:", sum(time_list)/len(time_list))
    
    log_filename = f"{query_file}.log".replace(".bin","")
    with open(LOG_DIRECTORY+log_filename,'a') as f:
        f.write("M,epsilon,cost,ratio\n")
        for i in range(len(cost_list)):
            f.write(f"{M>>20},{eps_list[i]},{cost_list[i]},{h_list[i]}\n")
    return eps_list, cost_list

def getExpectedCostPerEpsilon(ipp, seg_size, M, n, ps,
                              data_file="",query_file="",s="all_in_once",cache_policy="LRU",log_path=""):
    data = f"{DATASETS_DIRECTORY}{data_file}"
    query = f"{DATASETS_DIRECTORY}{query_file}"
    data_arr = np.fromfile(data, dtype=np.uint64)[1:]
    H, Q = prepare_query_histogram(query, data_arr)
    eps_list = []
    cost_list = []
    h_list = []
    time_list = []
    least_eps = math.ceil(n*seg_size/(2*M))
    for eps in range(least_eps if (least_eps%2==0) else least_eps+1, 129, 2):     
        t1 = time.time()
        cost,h = cost_function(
            eps, n, seg_size, M, ipp, ps,
            query, data, s, cache_policy,
            data_arr=data_arr, H=H, Q=Q
        )
        eps_list.append(eps)
        cost_list.append(cost)
        h_list.append(h)
        t2 = time.time()
        print(f"eps: {eps}, cost: {cost}, ratio: {h}, time: {t2-t1}")
        time_list.append(t2-t1)
    print(eps_list)
    print("cost:",cost_list)
    print("ratio:",h_list)
    groups = {
        "8-16":   (8, 16),
        "16-32":  (16, 32),
        "32-64":  (32, 64),
        "64-128": (64, 128),
    }

    group_times = {name: [] for name in groups}

    for eps, t in zip(eps_list, time_list):
        for name, (lo, hi) in groups.items():
            if lo <= eps < hi:
                group_times[name].append(t)
                break

    group_avg_time = {}
    for name, ts in group_times.items():
        if ts:
            group_avg_time[name] = sum(ts) / len(ts)
        else:
            group_avg_time[name] = 0.0  
    print("group avg time:", group_avg_time)
    
    
    log_filename = f"{query_file}_{cache_policy}.log".replace(".query.bin","")
    log_path = log_path or LOG_DIRECTORY+log_filename
    with open(log_path,'a+') as f:
        f.seek(0)          # 先跳到文件开头
        content = f.read() # 读取已有内容

        if not content:
            f.write("M,epsilon,cost,ratio,time\n")
            
        for i in range(len(cost_list)):
            f.write(f"{M>>20},{eps_list[i]},{cost_list[i]},{h_list[i]},{time_list[i]}\n")
        
    return eps_list, cost_list
    

def main():
    for m in [10]:
        M = m*1024*1024
        data_file = f"books_10M_uint64_unique"
        query_file = f"books_10M_uint64_unique.query.bin"
        for s in ["LRU","LFU","FIFO"]:
            eps_list,cost_list = getExpectedCostPerEpsilon(ipp=512,seg_size=16,M=M,n=int(1e7),ps=4096,
                                            data_file=data_file,query_file=query_file,s="all_in_once",
                                            cache_policy=s,log_path=f"{LOG_DIRECTORY}cmp/books_10M_M{M>>20}_summary_{s}.log")
    
    # data_file = f"fb_100M_uint64_unique"
    # query_file = f"fb_100M_uint64_unique.4Mrange.bin"
    # data_file = f"books_20M_uint64_unique"
    # query_file = f"books_20M_uint64_unique.range.bin"
    # getExpectedRangeCostPerEpsilon(n=int(1e8),seg_size=16,M=M,ipp=512,ps=4096,
    #                                query_file=query_file,data_file=data_file,fraction=0.3)
    
    # data_file = f"books_200M_uint64_unique"
    # query_file = f"books_200M_uint64_unique.4Mtable2.bin"
    # getExpectedJoinCostPerEpsilon(ipp=512,seg_size=16,M=M,n=int(2e7),ps=4096,data_file=data_file,query_file=query_file,
    #                                )
    
if __name__ == "__main__":
    main()
