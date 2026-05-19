import math
import csv
import re
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, Sequence

import numpy as np

try:
    from .cache_hit_models import (
        cache_hit_ratio,
        validate_ratio,
        che_characteristic_time,
        fifo_random_characteristic_time,
    )
except ImportError:
    from cache_hit_models import (
        cache_hit_ratio,
        validate_ratio,
        che_characteristic_time,
        fifo_random_characteristic_time,
    )


def expected_DAC(epsilon: int, ipp: int, strategy: str = "all_in_once") -> float:
    epsilon = int(epsilon)
    ipp = int(ipp)
    if strategy == "all_in_once":
        return 1.0 + (2.0 * epsilon / ipp)
    if strategy == "one_by_one":
        return 1.0 + (1.0 * epsilon / ipp)
    raise ValueError(f"Unknown strategy: {strategy}")


def load_rmi_query_records(path: str, delimiter: str = ",") -> Dict[str, np.ndarray]:
    path = str(path)
    with open(path, "r", newline="") as f:
        filtered_rows = (
            line for line in f
            if line.strip() and not line.lstrip().startswith("#")
        )
        reader = csv.DictReader(filtered_rows, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {path}")
        fields = {name.strip().lower(): name for name in reader.fieldnames}

        required = ["true_pos", "leaf_id", "err"]
        missing = [c for c in required if c not in fields]
        if missing:
            raise ValueError(f"Missing required columns {missing} in {path}")

        pos_list = []
        leaf_list = []
        err_list = []
        for row in reader:
            pos_list.append(int(row[fields["true_pos"]]))
            leaf_list.append(int(row[fields["leaf_id"]]))
            err_list.append(int(row[fields["err"]]))

    return {
        "pos": np.asarray(pos_list, dtype=np.int64),
        "leaf": np.asarray(leaf_list, dtype=np.int64),
        "err": np.asarray(err_list, dtype=np.int64),
    }


def load_rmi_result_meta(path: str, delimiter: str = ",") -> Dict[str, str]:
    meta: Dict[str, str] = {}
    with open(path, "r", newline="") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if not line.startswith("#"):
                break
            payload = line[1:]
            parts = payload.split(delimiter, 1)
            if len(parts) != 2:
                continue
            key, value = parts[0].strip().lower(), parts[1].strip()
            if key:
                meta[key] = value
    return meta


def infer_branch_factor(meta: Dict[str, str], path: str) -> Optional[int]:
    candidates = []
    if "name" in meta:
        candidates.append(meta["name"])
    candidates.append(Path(path).stem)

    for item in candidates:
        match = re.search(r"_(\d+)$", item)
        if match:
            return int(match.group(1))
    return None


def parse_policy_list(value: str) -> list[str]:
    policies = []
    for token in value.split(","):
        policy = token.strip().upper()
        if not policy:
            continue
        if policy not in {"FIFO", "LRU", "LFU", "RANDOM"}:
            raise ValueError(f"Unknown policy: {policy}")
        if policy not in policies:
            policies.append(policy)
    if not policies:
        raise ValueError("empty policy list")
    return policies


# ---------------------------------------------------------------------------
# Leaf-level aggregation
# ---------------------------------------------------------------------------

def aggregate_leaf_stats(
    leaf_ids: np.ndarray,
    errs: np.ndarray,
    num_leaf: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    leaf_ids = np.asarray(leaf_ids, dtype=np.int64)
    errs = np.asarray(errs, dtype=np.int64)
    if leaf_ids.shape != errs.shape:
        raise ValueError("leaf_ids and errs must have the same shape")
    if leaf_ids.size == 0:
        raise ValueError("empty inputs")

    if num_leaf is None:
        num_leaf = int(leaf_ids.max()) + 1
    num_leaf = int(num_leaf)

    cnt_j = np.bincount(leaf_ids, minlength=num_leaf).astype(np.int64)
    total = int(cnt_j.sum())
    if total <= 0:
        raise ValueError("invalid total query count")

    w_j = cnt_j.astype(np.float64) / total
    eps_j = np.zeros(num_leaf, dtype=np.int64)
    np.maximum.at(eps_j, leaf_ids, errs)
    used_j = cnt_j > 0
    return w_j, eps_j, cnt_j, used_j


# ---------------------------------------------------------------------------
# EDAC estimation
# ---------------------------------------------------------------------------

def estimate_rmi_edac_from_leaf_stats(
    w_j: np.ndarray,
    eps_j: np.ndarray,
    strategy: str,
    ipp: int,
) -> float:
    w_j = np.asarray(w_j, dtype=np.float64)
    eps_j = np.asarray(eps_j, dtype=np.int64)
    if w_j.shape != eps_j.shape:
        raise ValueError("w_j and eps_j must have the same shape")

    lam = 2.0 if strategy == "all_in_once" else 1.0 if strategy == "one_by_one" else None
    if lam is None:
        raise ValueError(f"Unknown strategy: {strategy}")
    return float(np.sum(w_j * (1.0 + lam * eps_j / float(ipp))))


def estimate_rmi_edac(
    leaf_ids: np.ndarray,
    errs: np.ndarray,
    ipp: int,
    strategy: str = "all_in_once",
    num_leaf: Optional[int] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w_j, eps_j, cnt_j, used_j = aggregate_leaf_stats(leaf_ids, errs, num_leaf=num_leaf)
    edac = estimate_rmi_edac_from_leaf_stats(w_j, eps_j, strategy=strategy, ipp=ipp)
    return edac, w_j, eps_j, cnt_j, used_j


# ---------------------------------------------------------------------------
# Page-reference estimation
# ---------------------------------------------------------------------------

def _page_prob_table_for_epsilon(epsilon: int, ipp: int) -> Tuple[np.ndarray, np.ndarray]:
    eps = int(epsilon)
    ipp = int(ipp)
    denom = float(2 * eps + 1)

    s = np.arange(ipp, dtype=np.int64)
    d_min = (0 - 2 * eps) // ipp
    d_max = (ipp - 1 + 2 * eps) // ipp
    d_vals = np.arange(d_min, d_max + 1, dtype=np.int64)

    table = np.zeros((len(d_vals), ipp), dtype=np.float64)
    for i, d in enumerate(d_vals):
        L = np.maximum(-eps, d * ipp - s - eps)
        U = np.minimum(eps, (d + 1) * ipp - 1 - s + eps)
        table[i] = np.maximum(0, U - L + 1) / denom
    return d_vals, table


def estimate_page_counts_from_positions(
    pos: np.ndarray,
    epsilon: int,
    ipp: int,
    n_records: int,
    prob_cache: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.int64)
    if pos.size == 0:
        return np.zeros(math.ceil(n_records / ipp), dtype=np.float64)

    num_pages = math.ceil(n_records / ipp)
    if prob_cache is not None and int(epsilon) in prob_cache:
        d_vals, prob_table = prob_cache[int(epsilon)]
    else:
        d_vals, prob_table = _page_prob_table_for_epsilon(int(epsilon), int(ipp))
        if prob_cache is not None:
            prob_cache[int(epsilon)] = (d_vals, prob_table)

    pages = pos // ipp
    offsets = pos % ipp
    page_counts = np.zeros(num_pages, dtype=np.float64)

    for row, d in enumerate(d_vals):
        tgt = pages + d
        valid = (tgt >= 0) & (tgt < num_pages)
        if not np.any(valid):
            continue
        w = prob_table[row, offsets[valid]]
        page_counts += np.bincount(tgt[valid], weights=w, minlength=num_pages)

    return page_counts


def estimate_page_counts_from_rmi_records(
    pos: np.ndarray,
    leaf_ids: np.ndarray,
    eps_j: np.ndarray,
    ipp: int,
    n_records: int,
) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.int64)
    leaf_ids = np.asarray(leaf_ids, dtype=np.int64)
    eps_j = np.asarray(eps_j, dtype=np.int64)

    if pos.shape != leaf_ids.shape:
        raise ValueError("pos and leaf_ids must have the same shape")

    num_pages = math.ceil(n_records / ipp)
    page_counts = np.zeros(num_pages, dtype=np.float64)
    prob_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    eps_per_query = eps_j[leaf_ids]
    for eps in np.unique(eps_per_query):
        mask = eps_per_query == eps
        sub_pos = pos[mask]
        if sub_pos.size == 0:
            continue
        page_counts += estimate_page_counts_from_positions(
            sub_pos,
            int(eps),
            ipp,
            n_records,
            prob_cache=prob_cache,
        )

    return page_counts


def estimate_page_counts_by_leaf(
    pos: np.ndarray,
    leaf_ids: np.ndarray,
    eps_j: np.ndarray,
    ipp: int,
    n_records: int,
) -> Dict[int, np.ndarray]:
    pos = np.asarray(pos, dtype=np.int64)
    leaf_ids = np.asarray(leaf_ids, dtype=np.int64)
    eps_j = np.asarray(eps_j, dtype=np.int64)

    if pos.shape != leaf_ids.shape:
        raise ValueError("pos and leaf_ids must have the same shape")

    prob_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    leaf_page_counts: Dict[int, np.ndarray] = {}
    used_leaf = np.unique(leaf_ids)

    for j in used_leaf:
        mask = (leaf_ids == j)
        sub_pos = pos[mask]
        eps = int(eps_j[j])
        leaf_page_counts[int(j)] = estimate_page_counts_from_positions(
            sub_pos,
            eps,
            ipp,
            n_records,
            prob_cache=prob_cache,
        )

    return leaf_page_counts


def estimate_page_probs_from_rmi_records(
    pos: np.ndarray,
    leaf_ids: np.ndarray,
    eps_j: np.ndarray,
    ipp: int,
    n_records: int,
) -> Tuple[np.ndarray, np.ndarray]:
    page_counts = estimate_page_counts_from_rmi_records(pos, leaf_ids, eps_j, ipp, n_records)
    total = float(page_counts.sum())
    if total <= 0:
        return page_counts, np.zeros_like(page_counts)
    return page_counts, page_counts / total


# ---------------------------------------------------------------------------
# Per-page hit probabilities under a shared global cache
# ---------------------------------------------------------------------------

def lru_page_hit_probs(qs: np.ndarray, C: int) -> np.ndarray:
    qs = np.asarray(qs, dtype=np.float64)
    hp = np.zeros_like(qs, dtype=np.float64)
    if qs.size == 0 or C <= 0:
        return hp

    s = qs.sum()
    if s <= 0:
        return hp
    qs = qs / s
    pos = np.flatnonzero(qs > 0)
    if pos.size == 0:
        return hp
    if C >= pos.size:
        hp[pos] = 1.0
        return hp

    t_C = che_characteristic_time(qs, C)
    if np.isinf(t_C):
        hp[pos] = 1.0
        return hp

    hp = 1.0 - np.exp(-qs * t_C)
    hp[qs <= 0] = 0.0
    return hp


def lfu_page_hit_probs(qs: np.ndarray, C: int) -> np.ndarray:
    qs = np.asarray(qs, dtype=np.float64)
    hp = np.zeros_like(qs, dtype=np.float64)
    if qs.size == 0 or C <= 0:
        return hp

    s = qs.sum()
    if s <= 0:
        return hp
    qs = qs / s

    pos = np.flatnonzero(qs > 0)
    if pos.size == 0:
        return hp
    if C >= pos.size:
        hp[pos] = 1.0
        return hp

    order = np.argsort(qs)[::-1]
    hp[order[:int(C)]] = 1.0
    return hp


def fifo_random_hit_probs(qs: np.ndarray, C: int) -> np.ndarray:
    qs = np.asarray(qs, dtype=np.float64)
    hp = np.zeros_like(qs, dtype=np.float64)
    if qs.size == 0 or C <= 0:
        return hp

    s = qs.sum()
    if s <= 0:
        return hp
    qs = qs / s

    pos = np.flatnonzero(qs > 0)
    if pos.size == 0:
        return hp
    if C >= pos.size:
        hp[pos] = 1.0
        return hp

    tau_C = fifo_random_characteristic_time(qs, C)
    if np.isinf(tau_C):
        hp[pos] = 1.0
        return hp

    hp[pos] = (qs[pos] * tau_C) / (1.0 - qs[pos] + qs[pos] * tau_C)
    return hp


def cache_page_hit_probs(policy: str, qs: np.ndarray, C: int) -> np.ndarray:
    policy = policy.upper()
    if policy == "LRU":
        return lru_page_hit_probs(qs, C)
    if policy == "LFU":
        return lfu_page_hit_probs(qs, C)
    if policy in ("FIFO", "RANDOM"):
        return fifo_random_hit_probs(qs, C)
    raise ValueError(f"Unknown policy: {policy}")


def estimate_hit_ratios_from_page_probs(
    page_probs: np.ndarray,
    cache_pages: int,
    total_queries: int,
    policies: Sequence[str],
) -> Dict[str, float]:
    page_probs = np.asarray(page_probs, dtype=np.float64)
    ratios: Dict[str, float] = {}
    for policy in policies:
        h = cache_hit_ratio(policy, int(cache_pages), page_probs, total_queries)
        ratios[policy] = float(validate_ratio(h))
    return ratios


# ---------------------------------------------------------------------------
# Original global evaluator
# ---------------------------------------------------------------------------

def evaluate_rmi_configuration(
    records: Dict[str, np.ndarray],
    n_records: int,
    ipp: int,
    strategy: str = "all_in_once",
    num_leaf: Optional[int] = None,
    cache_pages: Optional[int] = None,
    policies: Sequence[str] = ("FIFO", "LRU", "LFU"),
    eps_transform_mode: str = "none",
    eps_transform_q: float = 0.99,
    eps_transform_alpha: float = 0.5,
) -> Dict[str, object]:
    pos = np.asarray(records["pos"], dtype=np.int64)
    leaf_ids = np.asarray(records["leaf"], dtype=np.int64)
    errs = np.asarray(records["err"], dtype=np.int64)

    _edac_unused, w_j, eps_j, cnt_j, used_j = estimate_rmi_edac(
        leaf_ids=leaf_ids,
        errs=errs,
        ipp=ipp,
        strategy=strategy,
        num_leaf=num_leaf,
    )

    lam = 2.0 if strategy == "all_in_once" else 1.0 if strategy == "one_by_one" else None
    if lam is None:
        raise ValueError(f"Unknown strategy: {strategy}")

    eps_eff_j, eps_cap = transform_eps_j(
        eps_j=eps_j,
        w_j=w_j,
        mode=eps_transform_mode,
        q=eps_transform_q,
        alpha=eps_transform_alpha,
    )
    edac_j = 1.0 + lam * eps_eff_j / float(ipp)
    edac = float(np.sum(w_j * edac_j))
    
    page_counts, page_probs = estimate_page_probs_from_rmi_records(
        pos=pos,
        leaf_ids=leaf_ids,
        eps_j=eps_j,
        ipp=ipp,
        n_records=n_records,
    )

    result: Dict[str, object] = {
        "edac": edac,
        "w_j": w_j,
        "eps_j": eps_j,
        "eps_eff_j": eps_eff_j,
        "cnt_j": cnt_j,
        "used_j": used_j,
        "page_counts": page_counts,
        "page_probs": page_probs,
    }
    if eps_cap is not None:
        result["eps_cap"] = eps_cap
    if cache_pages is not None:
        hit_ratio_by_policy = estimate_hit_ratios_from_page_probs(
            page_probs=page_probs,
            cache_pages=int(cache_pages),
            total_queries=int(pos.size),
            policies=policies,
        )
        cost_by_policy = {
            policy: float((1.0 - hit_ratio_by_policy[policy]) * edac)
            for policy in hit_ratio_by_policy
        }
        result["hit_ratio_by_policy"] = hit_ratio_by_policy
        result["cost_by_policy"] = cost_by_policy
    return result


# ---------------------------------------------------------------------------
# New leafwise evaluator
# ---------------------------------------------------------------------------

def evaluate_rmi_configuration_leafwise(
    records: Dict[str, np.ndarray],
    n_records: int,
    ipp: int,
    strategy: str = "all_in_once",
    cache_pages: Optional[int] = None,
    policies: Sequence[str] = ("FIFO", "LRU", "LFU"),
    num_leaf: Optional[int] = None,
) -> Dict[str, object]:
    pos = np.asarray(records["pos"], dtype=np.int64)
    leaf_ids = np.asarray(records["leaf"], dtype=np.int64)
    errs = np.asarray(records["err"], dtype=np.int64)

    w_j, eps_j, cnt_j, used_j = aggregate_leaf_stats(
        leaf_ids=leaf_ids,
        errs=errs,
        num_leaf=num_leaf,
    )

    lam = 2.0 if strategy == "all_in_once" else 1.0 if strategy == "one_by_one" else None
    if lam is None:
        raise ValueError(f"Unknown strategy: {strategy}")

    edac_j = 1.0 + lam * eps_j / float(ipp)
    edac_global = float(np.sum(w_j * edac_j))
    

    leaf_page_counts = estimate_page_counts_by_leaf(
        pos=pos,
        leaf_ids=leaf_ids,
        eps_j=eps_j,
        ipp=ipp,
        n_records=n_records,
    )

    num_pages = math.ceil(n_records / ipp)
    global_page_counts = np.zeros(num_pages, dtype=np.float64)
    for arr in leaf_page_counts.values():
        global_page_counts += arr

    total_refs = float(global_page_counts.sum())
    if total_refs <= 0:
        global_page_probs = np.zeros_like(global_page_counts)
    else:
        global_page_probs = global_page_counts / total_refs

    result: Dict[str, object] = {
        "edac": edac_global,
        "edac_j": edac_j,
        "w_j": w_j,
        "eps_j": eps_j,
        "cnt_j": cnt_j,
        "used_j": used_j,
        "page_counts": global_page_counts,
        "page_probs": global_page_probs,
    }
        
    if cache_pages is not None:
        hit_ratio_by_policy: Dict[str, float] = {}
        cost_by_policy: Dict[str, float] = {}
        leaf_hit_ratio_by_policy: Dict[str, np.ndarray] = {}
        page_hit_prob_by_policy: Dict[str, np.ndarray] = {}

        for policy in policies:
            hp = cache_page_hit_probs(policy, global_page_probs, int(cache_pages))
            h_j = np.zeros_like(w_j, dtype=np.float64)

            for j, arr in leaf_page_counts.items():
                s = float(arr.sum())
                if s <= 0:
                    h_j[j] = 0.0
                else:
                    q_leaf = arr / s
                    h_j[j] = float(np.sum(q_leaf * hp))

            total_hit = float(np.sum(w_j * h_j))
            total_cost = float(np.sum(w_j * (1.0 - h_j) * edac_j))

            hit_ratio_by_policy[policy] = float(validate_ratio(total_hit))
            cost_by_policy[policy] = total_cost
            leaf_hit_ratio_by_policy[policy] = h_j
            page_hit_prob_by_policy[policy] = hp

        result["hit_ratio_by_policy"] = hit_ratio_by_policy
        result["cost_by_policy"] = cost_by_policy
        result["leaf_hit_ratio_by_policy"] = leaf_hit_ratio_by_policy
        result["page_hit_prob_by_policy"] = page_hit_prob_by_policy

    return result


def append_summary_rows(
    log_path: str,
    memory_mib: float,
    epsilon_or_branch: int,
    ratios: Dict[str, float],
    costs: Dict[str, float],
    estimated_total_ios: Dict[str, float],
    elapsed_s: float,
    header_mode: str = "epsilon",
) -> None:
    out_path = Path(log_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    fieldnames = (
        ["M", "epsilon", "policy", "cost", "ratio", "estimated_total_ios", "time"]
        if header_mode == "epsilon"
        else ["M", "branch_factor", "policy", "cost", "ratio", "estimated_total_ios", "time"]
    )
    with open(out_path, "a", newline="") as f:
        if write_header:
            if header_mode == "epsilon":
                f.write(",".join(fieldnames) + "\n")
            else:
                f.write(",".join(fieldnames) + "\n")
        for policy in ratios:
            if header_mode == "epsilon":
                f.write(
                    f"{memory_mib:g},{epsilon_or_branch},{policy},"
                    f"{costs[policy]},{ratios[policy]},{estimated_total_ios[policy]},{elapsed_s}\n"
                )
            else:
                f.write(
                    f"{memory_mib:g},{epsilon_or_branch},{policy},"
                    f"{costs[policy]},{ratios[policy]},{estimated_total_ios[policy]},{elapsed_s}\n"
                )

    if header_mode != "branch_factor":
        return

    policy_rank = {"FIFO": 0, "LRU": 1, "LFU": 2, "RANDOM": 3}
    with open(out_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    rows.sort(
        key=lambda row: (
            policy_rank.get(str(row["policy"]).upper(), 99),
            int(float(row["branch_factor"])),
        )
    )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def weighted_quantile(values, weights, q):
    """
    Weighted quantile in [0,1].
    values, weights: 1D arrays
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    mask = weights > 0
    values = values[mask]
    weights = weights[mask]
    if values.size == 0:
        return 0.0

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]
    idx = np.searchsorted(cdf, q, side="left")
    idx = min(idx, len(values) - 1)
    return float(values[idx])

def transform_eps_j(
    eps_j: np.ndarray,
    w_j: np.ndarray,
    mode: str = "none",
    q: float = 0.99,
    alpha: float = 0.5,
):
    """
    Transform per-leaf eps_j before computing EDAC.

    mode:
      - none
      - cap        : eps <- min(eps, weighted_quantile(eps, w, q))
      - logcap     : eps <- T * log(1 + eps / T)
      - power      : eps <- eps ** alpha
    """
    eps_j = np.asarray(eps_j, dtype=np.float64)
    w_j = np.asarray(w_j, dtype=np.float64)

    if mode == "none":
        return eps_j, None

    T = weighted_quantile(eps_j, w_j, q)

    if mode == "cap":
        return np.minimum(eps_j, T), T

    if mode == "logcap":
        if T <= 0:
            return eps_j, T
        return T * np.log1p(eps_j / T), T

    if mode == "power":
        return np.power(eps_j, alpha), None

    raise ValueError(f"Unknown eps transform mode: {mode}")
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Estimate RMI EDAC and page reference counts from query-level dump.")
    parser.add_argument("records_csv", type=str, help="CSV with columns: true_pos, leaf_id, err")
    parser.add_argument("n_records", type=int, help="Number of records in the sorted data array")
    parser.add_argument("--ipp", type=int, required=True, help="Items per page")
    parser.add_argument("--strategy", type=str, default="all_in_once", choices=["all_in_once", "one_by_one"])
    parser.add_argument("--memory-mib", type=float, default=None, help="Total memory budget in MiB. Used to derive cache pages as floor((M - rmi_size) / page_size).")
    parser.add_argument("--cache-bytes", type=int, default=None, help="Explicit cache budget in bytes. Overrides the derived cache size if provided.")
    parser.add_argument("--page-size", type=int, default=4096, help="Page size in bytes, default 4096.")
    parser.add_argument("--policies", type=str, default="FIFO,LRU,LFU", help="Comma-separated cache policies.")
    parser.add_argument("--mode", type=str, default="leafwise", choices=["global", "leafwise"], help="global: (1-h_global)*EDAC_global; leafwise: sum_j w_j(1-h_j)EDAC_j")
    parser.add_argument("--log-path", type=str, default="", help="Optional CSV log path.")
    parser.add_argument("--out", type=str, default="", help="Optional .npz output path")
    parser.add_argument("--header-mode", type=str, default="epsilon", choices=["epsilon", "branch_factor"], help="CSV header style.")
    parser.add_argument("--eps-transform", type=str, default="none",choices=["none", "cap", "logcap", "power"])
    parser.add_argument("--eps-transform-q", type=float, default=0.99)
    parser.add_argument("--eps-transform-alpha", type=float, default=0.5)
    args = parser.parse_args()

    if args.memory_mib is not None and args.cache_bytes is not None:
        raise ValueError("Use only one of --memory-mib or --cache-bytes")

    meta = load_rmi_result_meta(args.records_csv)
    policies = parse_policy_list(args.policies)
    branch_factor = infer_branch_factor(meta, args.records_csv)
    rmi_size = int(meta.get("rmi_size", "0"))

    cache_bytes: Optional[int] = None
    memory_mib: Optional[float] = args.memory_mib
    if args.cache_bytes is not None:
        cache_bytes = max(0, int(args.cache_bytes))
        if memory_mib is None:
            memory_mib = (cache_bytes + rmi_size) / float(1 << 20)
    elif args.memory_mib is not None:
        total_budget_bytes = int(round(args.memory_mib * (1 << 20)))
        cache_bytes = max(0, total_budget_bytes - rmi_size)

    cache_pages = None if cache_bytes is None else cache_bytes // int(args.page_size)

    rec = load_rmi_query_records(args.records_csv)
    t0 = time.perf_counter()
    if args.mode == "leafwise":
        result = evaluate_rmi_configuration_leafwise(
            rec,
            n_records=args.n_records,
            ipp=args.ipp,
            strategy=args.strategy,
            cache_pages=cache_pages,
            policies=policies,
        )
    else:
        result = evaluate_rmi_configuration(
            rec,
            n_records=args.n_records,
            ipp=args.ipp,
            strategy=args.strategy,
            cache_pages=cache_pages,
            policies=policies,
            eps_transform_mode=args.eps_transform,
            eps_transform_q=args.eps_transform_q,
            eps_transform_alpha=args.eps_transform_alpha,
        )
    elapsed_s = time.perf_counter() - t0

    print(f"MODE,{args.mode}")
    print(f"EDAC,{result['edac']:.10f}")
    print(f"USED_LEAF,{int(np.sum(result['used_j']))}")
    print(f"TOTAL_PAGE_REF,{float(np.sum(result['page_counts'])):.10f}")
    if "name" in meta:
        print(f"NAME,{meta['name']}")
    if branch_factor is not None:
        print(f"BRANCH_FACTOR,{branch_factor}")
    if rmi_size > 0:
        print(f"RMI_SIZE,{rmi_size}")
    if cache_bytes is not None:
        print(f"CACHE_BYTES,{cache_bytes}")
        print(f"CACHE_PAGES,{cache_pages}")

    if "hit_ratio_by_policy" in result and "cost_by_policy" in result:
        hit_ratio_by_policy = result["hit_ratio_by_policy"]
        cost_by_policy = result["cost_by_policy"]
        queries = int(rec["pos"].size)
        estimated_total_ios = {
            policy: float(cost_by_policy[policy] * queries)
            for policy in policies
        }

        if args.header_mode == "epsilon":
            print("M,epsilon,policy,cost,ratio,estimated_total_ios,time")
            for policy in policies:
                print(
                    f"{memory_mib:g},{branch_factor if branch_factor is not None else -1},"
                    f"{policy},{cost_by_policy[policy]},{hit_ratio_by_policy[policy]},"
                    f"{estimated_total_ios[policy]},{elapsed_s}"
                )
        else:
            print("M,branch_factor,policy,cost,ratio,estimated_total_ios,time")
            for policy in policies:
                print(
                    f"{memory_mib:g},{branch_factor if branch_factor is not None else -1},"
                    f"{policy},{cost_by_policy[policy]},{hit_ratio_by_policy[policy]},"
                    f"{estimated_total_ios[policy]},{elapsed_s}"
                )

        if args.log_path:
            if memory_mib is None:
                raise ValueError("--log-path requires either --memory-mib or --cache-bytes")
            append_summary_rows(
                log_path=args.log_path,
                memory_mib=memory_mib,
                epsilon_or_branch=branch_factor if branch_factor is not None else -1,
                ratios=hit_ratio_by_policy,
                costs=cost_by_policy,
                estimated_total_ios=estimated_total_ios,
                elapsed_s=elapsed_s,
                header_mode=args.header_mode,
            )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        npz_payload = dict(result)

        if "hit_ratio_by_policy" in npz_payload:
            ratios = npz_payload.pop("hit_ratio_by_policy")
            for policy, value in ratios.items():
                npz_payload[f"hit_ratio_{policy}"] = np.asarray(value, dtype=np.float64)

        if "cost_by_policy" in npz_payload:
            costs = npz_payload.pop("cost_by_policy")
            for policy, value in costs.items():
                npz_payload[f"cost_{policy}"] = np.asarray(value, dtype=np.float64)

        if "leaf_hit_ratio_by_policy" in npz_payload:
            ratios = npz_payload.pop("leaf_hit_ratio_by_policy")
            for policy, value in ratios.items():
                npz_payload[f"leaf_hit_ratio_{policy}"] = np.asarray(value, dtype=np.float64)

        if "page_hit_prob_by_policy" in npz_payload:
            probs = npz_payload.pop("page_hit_prob_by_policy")
            for policy, value in probs.items():
                npz_payload[f"page_hit_prob_{policy}"] = np.asarray(value, dtype=np.float64)

        np.savez_compressed(out, **npz_payload)


if __name__ == "__main__":
    _main()
