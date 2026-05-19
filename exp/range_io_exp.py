#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import optimalEpsilon  # noqa: E402


WORKLOAD_RATIOS: dict[str, tuple[float, float, float]] = {
    "w1": (0.0, 0.0, 1.0),
    "w2": (0.0, 1.0, 0.0),
    "w3": (1.0, 0.0, 0.0),
    "w4": (0.4, 0.3, 0.3),
    "w5": (0.2, 0.2, 0.6),
    "w6": (0.1, 0.1, 0.8),
}


def parse_epsilons(text: str) -> list[int]:
    out = [int(tok.strip()) for tok in text.replace(" ", ",").split(",") if tok.strip()]
    if not out:
        raise ValueError("empty epsilon list")
    return list(dict.fromkeys(out))


def dataset_label(dataset: str) -> str:
    if dataset.startswith("fb_"):
        return "fb"
    if dataset.startswith("wiki_ts_"):
        return "wiki"
    if dataset.startswith("osm_cellids_"):
        return "osm"
    if dataset.startswith("books_"):
        return "books"
    return dataset.replace("_uint64_unique", "")


def stable_dataset_seed(dataset: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(dataset))


def dataset_path(datasets_directory: Path, dataset: str) -> Path:
    path = Path(dataset)
    if path.is_absolute():
        return path
    return datasets_directory / dataset


def workload_path(workload_dir: Path, dataset: str, workload: str) -> Path:
    return workload_dir / dataset / f"{dataset}.{workload}.range.bin"


def actual_path(actual_dir: Path, dataset: str, workload: str, m_mib: int, policy: str) -> Path:
    return actual_dir / dataset / f"{dataset}_{workload}_M{m_mib}_{policy.upper()}_range_actual.csv"


def estimate_path(estimate_dir: Path, dataset: str, workload: str, policy: str) -> Path:
    return estimate_dir / dataset / f"{dataset}_{workload}_{policy.upper()}_range_estimate.csv"


def sample_point_like_start_positions(
    n: int,
    *,
    num_queries: int,
    seed: int,
    hot_ratio: float,
    zipf_ratio: float,
    uniform_ratio: float,
    num_hotspots: int,
    hotspot_frac: float,
    hotspot_zipf_a: float,
    zipf_a: float,
) -> tuple[np.ndarray, dict[str, int]]:
    rng = np.random.default_rng(seed)
    if n <= 0:
        raise ValueError("empty key array")

    parts: list[np.ndarray] = []
    m_hot = int(num_queries * hot_ratio)
    m_zipf = int(num_queries * zipf_ratio)
    m_uniform = num_queries - m_hot - m_zipf

    if m_hot > 0:
        hotspot_size = max(1, int(hotspot_frac * n))
        hotspot_size = min(hotspot_size, n)
        per_hotspot = int(math.ceil(m_hot / max(1, num_hotspots)))
        hot_parts: list[np.ndarray] = []
        for _ in range(max(1, num_hotspots)):
            hi_base = max(0, n - hotspot_size)
            base = int(rng.integers(0, hi_base + 1))
            idx = rng.zipf(hotspot_zipf_a, size=per_hotspot) - 1
            idx = np.clip(idx, 0, hotspot_size - 1)
            hot_parts.append((base + idx).astype(np.int64, copy=False))
        parts.append(np.concatenate(hot_parts)[:m_hot])

    if m_zipf > 0:
        idx = rng.zipf(zipf_a, size=m_zipf) - 1
        parts.append(np.clip(idx, 0, n - 1).astype(np.int64, copy=False))

    if m_uniform > 0:
        parts.append(rng.integers(0, n, size=m_uniform, endpoint=False, dtype=np.int64))

    if not parts:
        raise ValueError("workload ratios generated no range starts")

    start_idx = np.concatenate(parts).astype(np.int64, copy=False)
    if start_idx.size != num_queries:
        start_idx = start_idx[:num_queries]
    rng.shuffle(start_idx)
    counts = {
        "hotspot_queries": m_hot,
        "zipf_queries": m_zipf,
        "uniform_queries": m_uniform,
    }
    return start_idx, counts


def sample_range_workload(
    keys: np.ndarray,
    *,
    num_queries: int,
    seed: int,
    hot_ratio: float,
    zipf_ratio: float,
    uniform_ratio: float,
    num_hotspots: int,
    hotspot_frac: float,
    hotspot_zipf_a: float,
    zipf_a: float,
    min_length_keys: int,
    max_length_keys: int,
) -> tuple[np.ndarray, dict[str, int]]:
    rng = np.random.default_rng(seed + 1_000_003)
    n = int(len(keys))
    start_idx, counts = sample_point_like_start_positions(
        n,
        num_queries=num_queries,
        seed=seed,
        hot_ratio=hot_ratio,
        zipf_ratio=zipf_ratio,
        uniform_ratio=uniform_ratio,
        num_hotspots=num_hotspots,
        hotspot_frac=hotspot_frac,
        hotspot_zipf_a=hotspot_zipf_a,
        zipf_a=zipf_a,
    )

    min_len = max(1, int(min_length_keys))
    max_len = max(min_len, min(int(max_length_keys), n))
    lengths = rng.integers(min_len, max_len + 1, size=num_queries, dtype=np.int64)

    end_idx = np.clip(start_idx + lengths - 1, 0, n - 1)
    lo_idx = np.minimum(start_idx, end_idx)
    hi_idx = np.maximum(start_idx, end_idx)
    queries = np.stack([keys[lo_idx], keys[hi_idx]], axis=1).astype(np.uint64, copy=False)
    counts["min_length_keys"] = min_len
    counts["max_length_keys"] = max_len
    return queries, counts


def cmd_generate(args: argparse.Namespace) -> None:
    rows: list[dict[str, object]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        data_path_ = dataset_path(args.datasets_directory, dataset)
        if not data_path_.exists():
            raise FileNotFoundError(f"missing dataset: {data_path_}")
        keys = np.memmap(data_path_, dtype=np.uint64, mode="r")

        for workload in args.workloads:
            if workload not in WORKLOAD_RATIOS:
                raise ValueError(f"unknown workload {workload}; known: {sorted(WORKLOAD_RATIOS)}")
            hot, zipf, uniform = WORKLOAD_RATIOS[workload]
            out = workload_path(args.output_dir, dataset, workload)
            out.parent.mkdir(parents=True, exist_ok=True)
            sampling_counts = {
                "hotspot_queries": int(args.num_queries * hot),
                "zipf_queries": int(args.num_queries * zipf),
                "uniform_queries": args.num_queries - int(args.num_queries * hot) - int(args.num_queries * zipf),
                "min_length_keys": max(1, int(args.min_length_keys)),
                "max_length_keys": max(
                    max(1, int(args.min_length_keys)),
                    min(int(args.max_length_keys), int(len(keys))),
                ),
            }

            if out.exists() and not args.force:
                query_count = out.stat().st_size // (2 * np.dtype(np.uint64).itemsize)
                queries = np.fromfile(out, dtype=np.uint64).reshape(-1, 2)
                sampling_counts.update(
                    {
                        "hotspot_queries": int(query_count * hot),
                        "zipf_queries": int(query_count * zipf),
                        "uniform_queries": query_count - int(query_count * hot) - int(query_count * zipf),
                    }
                )
                print(f"[generate][skip] {out} ({query_count} ranges)")
            else:
                workload_id = int(workload[1:]) if workload.startswith("w") else 0
                seed = args.seed + workload_id + stable_dataset_seed(dataset)
                queries, sampling_counts = sample_range_workload(
                    keys,
                    num_queries=args.num_queries,
                    seed=seed,
                    hot_ratio=hot,
                    zipf_ratio=zipf,
                    uniform_ratio=uniform,
                    num_hotspots=args.num_hotspots,
                    hotspot_frac=args.hotspot_frac,
                    hotspot_zipf_a=args.hotspot_zipf_a,
                    zipf_a=args.zipf_a,
                    min_length_keys=args.min_length_keys,
                    max_length_keys=args.max_length_keys,
                )
                queries.tofile(out)
                query_count = int(queries.shape[0])
                print(f"[generate] {dataset} {workload} -> {out} ({query_count} ranges)")

            if queries.size:
                lo_pos = np.searchsorted(keys, queries[:, 0], side="right") - 1
                hi_pos = np.searchsorted(keys, queries[:, 1], side="right") - 1
                lengths = np.maximum(0, hi_pos - lo_pos + 1)
            else:
                lengths = np.asarray([], dtype=np.int64)

            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": dataset_label(dataset),
                    "workload": workload,
                    "query_path": str(out),
                    "ranges": query_count,
                    "hotspot_ratio": hot,
                    "zipf_ratio": zipf,
                    "uniform_ratio": uniform,
                    "hotspot_queries": sampling_counts["hotspot_queries"],
                    "zipf_queries": sampling_counts["zipf_queries"],
                    "uniform_queries": sampling_counts["uniform_queries"],
                    "num_hotspots": args.num_hotspots,
                    "hotspot_frac": args.hotspot_frac,
                    "hotspot_zipf_a": args.hotspot_zipf_a,
                    "zipf_a": args.zipf_a,
                    "length_dist": "uniform",
                    "min_length_keys": sampling_counts["min_length_keys"],
                    "max_length_keys": sampling_counts["max_length_keys"],
                    "mean_range_keys": float(np.mean(lengths)) if lengths.size else 0.0,
                    "p50_range_keys": float(np.percentile(lengths, 50)) if lengths.size else 0.0,
                    "p95_range_keys": float(np.percentile(lengths, 95)) if lengths.size else 0.0,
                }
            )

    manifest = args.output_dir / "range_workload_manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[generate] manifest -> {manifest}")


def prepare_range_prefix(
    query_path_: Path,
    query_limit: int,
    learning_fraction: float,
) -> tuple[np.ndarray, int, int]:
    raw = np.fromfile(query_path_, dtype=np.uint64)
    if raw.size % 2 != 0:
        raise ValueError(f"range query file has odd uint64 count: {query_path_}")
    queries = raw.reshape(-1, 2)
    if query_limit > 0:
        queries = queries[:query_limit]
    total_queries = int(queries.shape[0])
    if total_queries <= 0:
        raise ValueError(f"empty range query file: {query_path_}")
    if learning_fraction <= 0:
        raise ValueError("--learning-fraction must be positive")
    keep = total_queries if learning_fraction >= 1.0 else max(1, int(total_queries * learning_fraction))
    return queries[:keep], total_queries, keep


def estimate_range_cost(
    *,
    epsilon: int,
    n_keys: int,
    seg_size: int,
    memory_bytes: int,
    ipp: int,
    page_size: int,
    policy: str,
    data: np.ndarray,
    queries: np.ndarray,
    first_touch_scale: float,
    conservative: bool,
    cold_start_correction: bool,
) -> tuple[float, float, dict[str, float]]:
    index_bytes = float(n_keys * seg_size / (2 * epsilon))
    buffer_bytes = max(0.0, float(memory_bytes) - index_bytes)
    cache_pages = max(0, int(buffer_bytes / page_size))
    total_pages = math.ceil(n_keys / ipp)

    page_counts, total_refs, q = optimalEpsilon.estimate_page_counts_from_range_queryfile(
        lo_keys=queries[:, 0],
        hi_keys=queries[:, 1],
        data=data,
        epsilon=epsilon,
        ipp=ipp,
        conservative=conservative,
    )
    avg_rdac = float(total_refs) / float(max(1, queries.shape[0]))
    scaled_total_refs = float(total_refs) * max(0.0, float(first_touch_scale))
    expected_distinct_pages = float(np.count_nonzero(page_counts > 0))

    if cache_pages <= 0 or total_refs <= 0:
        h = 0.0
        steady_hit_ratio = 0.0
        cold_miss_ratio = 0.0
    else:
        h = optimalEpsilon.shared_cache_hit_ratio(policy, cache_pages, q, Q=scaled_total_refs)
        steady_hit_ratio = float(optimalEpsilon.shared_validate_ratio(h))
        cold_miss_ratio = 0.0
        if cold_start_correction and scaled_total_refs > 0:
            cold_miss_ratio = float(
                optimalEpsilon.shared_validate_ratio(expected_distinct_pages / scaled_total_refs)
            )
            h = min(h, 1.0 - cold_miss_ratio)
        h = float(optimalEpsilon.shared_validate_ratio(h))

    estimated_avg_io = (1.0 - h) * avg_rdac
    detail = {
        "index_bytes": index_bytes,
        "buffer_bytes": buffer_bytes,
        "cache_pages": float(cache_pages),
        "total_pages": float(total_pages),
        "total_page_requests": scaled_total_refs,
        "learning_page_requests": float(total_refs),
        "expected_distinct_pages": expected_distinct_pages,
        "cold_miss_ratio": cold_miss_ratio,
        "steady_hit_ratio": steady_hit_ratio,
        "avg_rdac": avg_rdac,
    }
    return estimated_avg_io, h, detail


def cmd_estimate(args: argparse.Namespace) -> None:
    epsilons = parse_epsilons(args.epsilons)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        data_path_ = dataset_path(args.datasets_directory, dataset)
        if not data_path_.exists():
            raise FileNotFoundError(f"missing dataset: {data_path_}")
        data = np.memmap(data_path_, dtype=np.uint64, mode="r")
        n_keys = int(len(data))

        for workload in args.workloads:
            q_path = workload_path(args.workload_dir, dataset, workload)
            if not q_path.exists():
                raise FileNotFoundError(f"missing workload file: {q_path}")

            preprocess_t0 = time.perf_counter()
            queries, total_queries, estimate_queries = prepare_range_prefix(
                q_path,
                query_limit=args.query_limit,
                learning_fraction=args.learning_fraction,
            )
            preprocess_time_s = time.perf_counter() - preprocess_t0
            first_touch_scale = float(total_queries) / float(max(1, estimate_queries))

            out = estimate_path(args.output_dir, dataset, workload, args.policy)
            out.parent.mkdir(parents=True, exist_ok=True)
            rows: list[dict[str, object]] = []

            for m_mib in args.memory_list:
                memory_bytes = int(m_mib) * 1024 * 1024
                for eps in epsilons:
                    t0 = time.perf_counter()
                    estimated_avg_io, hit_ratio, detail = estimate_range_cost(
                        epsilon=int(eps),
                        n_keys=n_keys,
                        seg_size=args.seg_size,
                        memory_bytes=memory_bytes,
                        ipp=args.ipp,
                        page_size=args.page_size,
                        policy=args.policy,
                        data=data,
                        queries=queries,
                        first_touch_scale=first_touch_scale,
                        conservative=not args.non_conservative,
                        cold_start_correction=args.cold_start_correction,
                    )
                    estimate_time_s = time.perf_counter() - t0
                    rows.append(
                        {
                            "dataset": dataset,
                            "dataset_label": dataset_label(dataset),
                            "workload": workload,
                            "M": int(m_mib),
                            "epsilon": int(eps),
                            "policy": args.policy.upper(),
                            "strategy": args.strategy,
                            "cost": float(estimated_avg_io),
                            "ratio": float(hit_ratio),
                            "estimated_avg_io": float(estimated_avg_io),
                            "estimated_total_ios": float(estimated_avg_io) * float(total_queries),
                            "estimated_avg_rdac": float(detail["avg_rdac"]),
                            "estimate_time_s": float(estimate_time_s),
                            "preprocess_time_s": float(preprocess_time_s),
                            "estimate_queries": estimate_queries,
                            "total_queries": total_queries,
                            "budget_mode": "estimated",
                            "cold_start_correction": int(args.cold_start_correction),
                            "conservative": int(not args.non_conservative),
                            "steady_hit_ratio": float(detail["steady_hit_ratio"]),
                            "cold_miss_ratio": float(detail["cold_miss_ratio"]),
                            "expected_distinct_pages": float(detail["expected_distinct_pages"]),
                            "estimated_total_page_requests": float(detail["total_page_requests"]),
                            "estimated_learning_page_requests": float(detail["learning_page_requests"]),
                            "estimated_cache_pages": float(detail["cache_pages"]),
                            "estimated_index_bytes": float(detail["index_bytes"]),
                            "estimated_buffer_bytes": float(detail["buffer_bytes"]),
                            "first_touch_scale": first_touch_scale,
                        }
                    )

            pd.DataFrame(rows).to_csv(out, index=False)
            print(
                f"[estimate] {dataset} {workload} {args.policy.upper()} -> {out} "
                f"({len(rows)} rows, preprocess={preprocess_time_s:.6f}s)"
            )


def load_actual_csv(path: Path, dataset: str, workload: str, policy: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "epsilon",
        "policy",
        "strategy",
        "budget_mode",
        "memory_budget_bytes",
        "queries",
        "total_cache_misses",
        "avg_cam_io",
    }
    if not required.issubset(df.columns):
        raise ValueError(f"unexpected actual CSV format: {path}")
    df = df.copy()
    df["dataset"] = dataset
    df["dataset_label"] = dataset_label(dataset)
    df["workload"] = workload
    df["policy"] = df["policy"].astype(str).str.upper()
    df = df[df["policy"] == policy.upper()].copy()
    df = df[df["budget_mode"].astype(str).str.lower() == "estimated"].copy()
    df["M"] = (pd.to_numeric(df["memory_budget_bytes"], errors="coerce") / (1 << 20)).round().astype(int)
    df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce").astype(int)
    df["actual_total_ios"] = pd.to_numeric(df["total_cache_misses"], errors="coerce")
    df["actual_avg_io"] = pd.to_numeric(df["avg_cam_io"], errors="coerce")
    return df.loc[
        :,
        [
            "dataset",
            "dataset_label",
            "workload",
            "M",
            "epsilon",
            "policy",
            "strategy",
            "queries",
            "actual_total_ios",
            "actual_avg_io",
            "simulate_wall_ns",
            "index_build_ns",
        ],
    ]


def cmd_summarize(args: argparse.Namespace) -> None:
    frames_actual: list[pd.DataFrame] = []
    frames_est: list[pd.DataFrame] = []

    for dataset in args.datasets:
        for workload in args.workloads:
            est = estimate_path(args.estimate_dir, dataset, workload, args.policy)
            if not est.exists():
                raise FileNotFoundError(f"missing estimate CSV: {est}")
            est_df = pd.read_csv(est)
            est_df["dataset"] = dataset
            est_df["dataset_label"] = dataset_label(dataset)
            est_df["workload"] = workload
            frames_est.append(est_df)

            for m_mib in args.memory_list:
                act = actual_path(args.actual_dir, dataset, workload, m_mib, args.policy)
                if not act.exists():
                    raise FileNotFoundError(f"missing actual CSV: {act}")
                frames_actual.append(load_actual_csv(act, dataset, workload, args.policy))

    actual = pd.concat(frames_actual, ignore_index=True)
    estimate = pd.concat(frames_est, ignore_index=True)
    estimate["policy"] = estimate["policy"].astype(str).str.upper()
    estimate["M"] = pd.to_numeric(estimate["M"], errors="coerce").astype(int)
    estimate["epsilon"] = pd.to_numeric(estimate["epsilon"], errors="coerce").astype(int)
    for col in (
        "steady_hit_ratio",
        "cold_miss_ratio",
        "expected_distinct_pages",
        "estimated_cache_pages",
        "estimated_avg_rdac",
    ):
        if col not in estimate.columns:
            estimate[col] = np.nan

    keys = ["dataset", "dataset_label", "workload", "M", "epsilon", "policy", "strategy"]
    merged = actual.merge(estimate, on=keys, how="inner", suffixes=("_actual", "_estimate"))
    if merged.empty:
        raise ValueError("actual and estimate CSVs have no overlapping rows")

    eps_set = set(parse_epsilons(args.epsilons))
    merged = merged[merged["epsilon"].isin(eps_set)].copy()
    if merged.empty:
        raise ValueError("no rows remain after epsilon filtering")

    actual_ios = pd.to_numeric(merged["actual_total_ios"], errors="coerce")
    estimated_ios = pd.to_numeric(merged["estimated_total_ios"], errors="coerce")
    merged["absolute_error"] = (estimated_ios - actual_ios).abs()
    merged["relative_error"] = np.where(actual_ios > 0, merged["absolute_error"] / actual_ios, np.nan)
    merged["accuracy"] = np.where(actual_ios > 0, 1.0 - merged["relative_error"], np.nan)
    merged["accuracy"] = merged["accuracy"].clip(lower=0.0, upper=1.0)

    group_keys = ["dataset", "dataset_label", "workload", "M", "policy", "strategy"]
    summary = (
        merged.groupby(group_keys, as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_relative_error=("relative_error", "mean"),
            epsilon_points=("epsilon", "count"),
            mean_estimate_time_s=("estimate_time_s", "mean"),
            total_estimate_time_s=("estimate_time_s", "sum"),
            mean_actual_total_ios=("actual_total_ios", "mean"),
            mean_estimated_total_ios=("estimated_total_ios", "mean"),
            actual_queries=("queries", "max"),
            estimate_queries=("estimate_queries", "max"),
            mean_actual_avg_io=("actual_avg_io", "mean"),
            mean_estimated_avg_io=("estimated_avg_io", "mean"),
            mean_estimated_avg_rdac=("estimated_avg_rdac", "mean"),
            mean_steady_hit_ratio=("steady_hit_ratio", "mean"),
            mean_cold_miss_ratio=("cold_miss_ratio", "mean"),
            mean_expected_distinct_pages=("expected_distinct_pages", "mean"),
            mean_estimated_cache_pages=("estimated_cache_pages", "mean"),
        )
        .sort_values(["dataset_label", "M", "workload"])
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = args.output_dir / "range_io_merged.csv"
    summary_path = args.output_dir / "range_io_accuracy_summary.csv"
    merged.to_csv(merged_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"[summarize] merged -> {merged_path}")
    print(f"[summarize] summary -> {summary_path}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--datasets-directory", type=Path, default=Path("/mnt/data/Dataset/public/SOSD"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "books_200M_uint64_unique",
            "fb_200M_uint64_unique",
            "wiki_ts_200M_uint64_unique",
            "osm_cellids_200M_uint64_unique",
        ],
    )
    parser.add_argument("--workloads", nargs="+", default=["w1", "w2", "w3", "w4", "w5", "w6"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Range-query IO estimation experiment helper.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate", help="Generate w1-w6 range-query workloads.")
    add_common_args(p_gen)
    p_gen.add_argument("--output-dir", type=Path, required=True)
    p_gen.add_argument("--num-queries", type=int, default=1_000_000)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument("--num-hotspots", type=int, default=5)
    p_gen.add_argument("--hotspot-frac", type=float, default=0.01)
    p_gen.add_argument("--hotspot-zipf-a", type=float, default=1.5)
    p_gen.add_argument("--zipf-a", type=float, default=1.2)
    p_gen.add_argument("--min-length-keys", type=int, default=1)
    p_gen.add_argument("--max-length-keys", type=int, default=1024)
    p_gen.add_argument("--force", action="store_true")
    p_gen.set_defaults(func=cmd_generate)

    p_est = sub.add_parser("estimate", help="Run CAM range-query IO estimates.")
    add_common_args(p_est)
    p_est.add_argument("--workload-dir", type=Path, required=True)
    p_est.add_argument("--output-dir", type=Path, required=True)
    p_est.add_argument("--memory-list", nargs="+", type=int, required=True)
    p_est.add_argument("--epsilons", required=True)
    p_est.add_argument("--policy", default="LRU")
    p_est.add_argument("--strategy", default="all_in_once")
    p_est.add_argument("--query-limit", type=int, default=0)
    p_est.add_argument("--learning-fraction", type=float, default=0.3)
    p_est.add_argument("--seg-size", type=int, default=16)
    p_est.add_argument("--ipp", type=int, default=512)
    p_est.add_argument("--page-size", type=int, default=4096)
    p_est.add_argument(
        "--non-conservative",
        action="store_true",
        help="Use true range pages instead of CAM's conservative [lo-2eps, hi+2eps] interval.",
    )
    p_est.add_argument(
        "--cold-start-correction",
        action="store_true",
        help="Bound the estimated hit ratio by expected compulsory first-reference misses.",
    )
    p_est.set_defaults(func=cmd_estimate)

    p_sum = sub.add_parser("summarize", help="Merge actual and estimated range IO logs.")
    add_common_args(p_sum)
    p_sum.add_argument("--actual-dir", type=Path, required=True)
    p_sum.add_argument("--estimate-dir", type=Path, required=True)
    p_sum.add_argument("--output-dir", type=Path, required=True)
    p_sum.add_argument("--memory-list", nargs="+", type=int, required=True)
    p_sum.add_argument("--epsilons", required=True)
    p_sum.add_argument("--policy", default="LRU")
    p_sum.set_defaults(func=cmd_summarize)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
