#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
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
    return workload_dir / dataset / f"{dataset}.{workload}.bin"


def actual_path(actual_dir: Path, dataset: str, workload: str, m_mib: int, policy: str) -> Path:
    return actual_dir / dataset / f"{dataset}_{workload}_M{m_mib}_{policy.upper()}_actual.csv"


def estimate_path(estimate_dir: Path, dataset: str, workload: str, policy: str) -> Path:
    return estimate_dir / dataset / f"{dataset}_{workload}_{policy.upper()}_estimate.csv"


def sample_point_mixture(
    keys: np.ndarray,
    num_queries: int,
    seed: int,
    hot_ratio: float,
    zipf_ratio: float,
    uniform_ratio: float,
    num_hotspots: int,
    hotspot_frac: float,
    hotspot_zipf_a: float,
    zipf_a: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    n = int(len(keys))
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
            hot_parts.append(np.asarray(keys[base + idx], dtype=np.uint64))
        parts.append(np.concatenate(hot_parts)[:m_hot])

    if m_zipf > 0:
        idx = rng.zipf(zipf_a, size=m_zipf) - 1
        idx = np.clip(idx, 0, n - 1)
        parts.append(np.asarray(keys[idx], dtype=np.uint64))

    if m_uniform > 0:
        idx = rng.integers(0, n, size=m_uniform, endpoint=False)
        parts.append(np.asarray(keys[idx], dtype=np.uint64))

    if not parts:
        raise ValueError("workload ratios generated no queries")

    queries = np.concatenate(parts).astype(np.uint64, copy=False)
    if queries.size != num_queries:
        queries = queries[:num_queries]
    rng.shuffle(queries)
    return queries


def cmd_generate(args: argparse.Namespace) -> None:
    rows: list[dict[str, object]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        data_path = dataset_path(args.datasets_directory, dataset)
        if not data_path.exists():
            raise FileNotFoundError(f"missing dataset: {data_path}")
        keys = np.memmap(data_path, dtype=np.uint64, mode="r")

        for workload in args.workloads:
            if workload not in WORKLOAD_RATIOS:
                raise ValueError(f"unknown workload {workload}; known: {sorted(WORKLOAD_RATIOS)}")
            out = workload_path(args.output_dir, dataset, workload)
            out.parent.mkdir(parents=True, exist_ok=True)
            hot, zipf, uniform = WORKLOAD_RATIOS[workload]

            if out.exists() and not args.force:
                query_count = out.stat().st_size // np.dtype(np.uint64).itemsize
                print(f"[generate][skip] {out} ({query_count} queries)")
            else:
                workload_id = int(workload[1:]) if workload.startswith("w") else 0
                seed = args.seed + workload_id + stable_dataset_seed(dataset)
                queries = sample_point_mixture(
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
                )
                queries.tofile(out)
                query_count = int(queries.size)
                print(f"[generate] {dataset} {workload} -> {out} ({query_count} queries)")

            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": dataset_label(dataset),
                    "workload": workload,
                    "query_path": str(out),
                    "queries": query_count,
                    "hotspot_ratio": hot,
                    "zipf_ratio": zipf,
                    "uniform_ratio": uniform,
                }
            )

    manifest = args.output_dir / "workload_manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[generate] manifest -> {manifest}")


def prepare_position_cache(
    query_path_: Path,
    data: np.ndarray,
    query_limit: int,
    learning_fraction: float,
) -> tuple[dict[str, np.ndarray], int, int]:
    queries = np.fromfile(query_path_, dtype=np.uint64)
    if query_limit > 0:
        queries = queries[:query_limit]
    total_queries = int(queries.size)
    if total_queries <= 0:
        raise ValueError(f"empty query file: {query_path_}")

    if learning_fraction <= 0:
        raise ValueError("--learning-fraction must be positive")
    keep = total_queries if learning_fraction >= 1.0 else max(1, int(total_queries * learning_fraction))
    queries = queries[:keep]

    pos = np.searchsorted(data, queries, side="right") - 1
    pos = np.clip(pos, 0, len(data) - 1).astype(np.int64)
    return {"kind": "positions", "pos": pos}, total_queries, int(pos.size)


def cmd_estimate(args: argparse.Namespace) -> None:
    optimalEpsilon.BUDGET_MODE = "ESTIMATED"
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
            H, total_queries, estimate_queries = prepare_position_cache(
                q_path,
                data=data,
                query_limit=args.query_limit,
                learning_fraction=args.learning_fraction,
            )
            preprocess_time_s = time.perf_counter() - preprocess_t0
            first_touch_scale = float(total_queries) / float(max(1, estimate_queries))

            out = estimate_path(args.output_dir, dataset, workload, args.policy)
            out.parent.mkdir(parents=True, exist_ok=True)
            rows: list[dict[str, object]] = []

            for m_mib in args.memory_list:
                for eps in epsilons:
                    t0 = time.perf_counter()
                    estimated_avg_io, hit_ratio, detail = optimalEpsilon.cost_function(
                        eps,
                        n=n_keys,
                        seg_size=args.seg_size,
                        M=m_mib * 1024 * 1024,
                        ipp=args.ipp,
                        ps=args.page_size,
                        query_file=str(q_path),
                        data_file=str(data_path_),
                        s=args.strategy,
                        cache_policy=args.policy,
                        data_arr=data,
                        H=H,
                        Q=estimate_queries,
                        cold_start_correction=args.cold_start_correction,
                        first_touch_scale=first_touch_scale,
                        return_detail=True,
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
                            "estimate_time_s": float(estimate_time_s),
                            "preprocess_time_s": float(preprocess_time_s),
                            "estimate_queries": estimate_queries,
                            "total_queries": total_queries,
                            "budget_mode": "estimated",
                            "cold_start_correction": int(args.cold_start_correction),
                            "steady_hit_ratio": float(detail.get("steady_hit_ratio", hit_ratio)),
                            "cold_miss_ratio": float(detail.get("cold_miss_ratio", 0.0)),
                            "expected_distinct_pages": float(detail.get("expected_distinct_pages", 0.0)),
                            "estimated_total_page_requests": float(detail.get("total_page_requests", 0.0)),
                            "estimated_cache_pages": float(detail.get("cache_pages", 0.0)),
                            "estimated_index_bytes": float(detail.get("index_bytes", 0.0)),
                            "estimated_buffer_bytes": float(detail.get("buffer_bytes", 0.0)),
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
            mean_steady_hit_ratio=("steady_hit_ratio", "mean"),
            mean_cold_miss_ratio=("cold_miss_ratio", "mean"),
            mean_expected_distinct_pages=("expected_distinct_pages", "mean"),
            mean_estimated_cache_pages=("estimated_cache_pages", "mean"),
        )
        .sort_values(["dataset_label", "M", "workload"])
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = args.output_dir / "point_io_merged.csv"
    summary_path = args.output_dir / "point_io_accuracy_summary.csv"
    merged.to_csv(merged_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"[summarize] merged -> {merged_path}")
    print(f"[summarize] summary -> {summary_path}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--datasets-directory", type=Path, default=Path("/mnt/data/Dataset/public/SOSD"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["books_200M_uint64_unique", "fb_200M_uint64_unique", "wiki_ts_200M_uint64_unique", "osm_cellids_200M_uint64_unique"],
    )
    parser.add_argument("--workloads", nargs="+", default=["w1", "w2", "w3", "w4", "w5", "w6"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Point-query IO estimation experiment helper.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate", help="Generate w1-w6 point-query workloads.")
    add_common_args(p_gen)
    p_gen.add_argument("--output-dir", type=Path, required=True)
    p_gen.add_argument("--num-queries", type=int, default=1_000_000)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument("--num-hotspots", type=int, default=5)
    p_gen.add_argument("--hotspot-frac", type=float, default=0.01)
    p_gen.add_argument("--hotspot-zipf-a", type=float, default=1.5)
    p_gen.add_argument("--zipf-a", type=float, default=1.2)
    p_gen.add_argument("--force", action="store_true")
    p_gen.set_defaults(func=cmd_generate)

    p_est = sub.add_parser("estimate", help="Run CAM point-query IO estimates.")
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
        "--cold-start-correction",
        action="store_true",
        help="Bound the estimated hit ratio by expected compulsory first-reference misses.",
    )
    p_est.set_defaults(func=cmd_estimate)

    p_sum = sub.add_parser("summarize", help="Merge actual and estimated IO logs.")
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
