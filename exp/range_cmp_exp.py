#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import point_cmp_exp as point_cmp  # noqa: E402
import range_io_exp as range_io  # noqa: E402


DEFAULT_DATASETS = [
    "books_200M_uint64_unique",
    "fb_200M_uint64_unique",
    range_io.RAW_WIKI_DATASET,
    "osm_cellids_200M_uint64_unique",
]
DEFAULT_DATASETS_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "datasets" / "SOSD"


def split_tokens(values: list[str] | str) -> list[str]:
    return point_cmp.split_tokens(values)


def parse_epsilons(text: str) -> list[int]:
    return point_cmp.parse_epsilons(text)


def parse_sample_rates(values: list[str] | str) -> list[dict[str, float | str]]:
    return point_cmp.parse_sample_rates(values)


def dataset_label(dataset: str) -> str:
    return range_io.dataset_label(dataset)


def dataset_path(datasets_directory: Path, dataset: str) -> Path:
    return range_io.dataset_path(datasets_directory, dataset)


def workload_path(workload_dir: Path, dataset: str, workload: str) -> Path:
    return range_io.workload_path(workload_dir, dataset, workload)


def prefix_path(prefix_dir: Path, dataset: str, workload: str, label: str) -> Path:
    return prefix_dir / dataset / f"{dataset}.{workload}.p{label}.range.bin"


def actual_path(actual_dir: Path, dataset: str, workload: str, m_mib: int, policy: str) -> Path:
    return range_io.actual_path(actual_dir, dataset, workload, m_mib, policy)


def replay_path(replay_dir: Path, dataset: str, workload: str, label: str, m_mib: int, policy: str) -> Path:
    return replay_dir / dataset / f"{dataset}_{workload}_p{label}_M{m_mib}_{policy.upper()}_range_replay.csv"


def sample_size(total_queries: int, fraction: float) -> int:
    return point_cmp.sample_size(total_queries, fraction)


def require_columns(df: pd.DataFrame, path: Path, columns: set[str]) -> None:
    point_cmp.require_columns(df, path, columns)


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_limited_ranges(query_path: Path, query_limit: int) -> np.ndarray:
    raw = np.fromfile(query_path, dtype=np.uint64)
    if raw.size % 2 != 0:
        raise ValueError(f"range query file has odd uint64 count: {query_path}")
    ranges = raw.reshape(-1, 2)
    if query_limit > 0:
        ranges = ranges[:query_limit]
    if ranges.shape[0] == 0:
        raise ValueError(f"empty range query file: {query_path}")
    return ranges


def cmd_make_prefixes(args: argparse.Namespace) -> None:
    rates = parse_sample_rates(args.sample_rates)
    rows: list[dict[str, object]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        q_path = workload_path(args.workload_dir, dataset, args.workload)
        if not q_path.exists():
            raise FileNotFoundError(f"missing range workload file: {q_path}")
        ranges = load_limited_ranges(q_path, args.query_limit)
        total_queries = int(ranges.shape[0])

        for rate in rates:
            label = str(rate["sample_label"])
            fraction = float(rate["sample_fraction"])
            keep = sample_size(total_queries, fraction)
            out = prefix_path(args.output_dir, dataset, args.workload, label)
            out.parent.mkdir(parents=True, exist_ok=True)

            existing_queries = out.stat().st_size // (2 * np.dtype(np.uint64).itemsize) if out.exists() else -1
            if args.force or existing_queries != keep:
                np.asarray(ranges[:keep], dtype=np.uint64).tofile(out)
                action = "write"
            else:
                action = "skip"
            print(f"[prefix][range][{action}] {dataset} {args.workload} p{label} -> {out} ({keep}/{total_queries})")

            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": dataset_label(dataset),
                    "workload": args.workload,
                    "sample_rate_percent": float(rate["sample_rate_percent"]),
                    "sample_fraction": fraction,
                    "sample_label": label,
                    "query_path": str(out),
                    "total_queries": total_queries,
                    "sample_queries": keep,
                }
            )

    manifest = args.output_dir / "prefix_manifest.csv"
    write_rows_csv(manifest, rows)
    print(f"[prefix][range] manifest -> {manifest}")


def cmd_cam_estimate(args: argparse.Namespace) -> None:
    range_io.optimalEpsilon.BUDGET_MODE = "ESTIMATED"
    range_io.validate_timing_args(args)
    epsilons = parse_epsilons(args.epsilons)
    rates = parse_sample_rates(args.sample_rates)
    rows: list[dict[str, object]] = []

    for dataset in args.datasets:
        data_path_ = dataset_path(args.datasets_directory, dataset)
        if not data_path_.exists():
            raise FileNotFoundError(f"missing dataset: {data_path_}")
        data = range_io.open_key_array(data_path_)
        n_keys = int(len(data))

        q_path = workload_path(args.workload_dir, dataset, args.workload)
        if not q_path.exists():
            raise FileNotFoundError(f"missing range workload file: {q_path}")
        full_ranges = load_limited_ranges(q_path, args.query_limit)
        total_queries = int(full_ranges.shape[0])
        warmup_time_s = 0.0
        if args.warmup_position_cache:
            max_keep = max(sample_size(total_queries, float(rate["sample_fraction"])) for rate in rates)
            warmup_t0 = time.perf_counter()
            range_io.prepare_range_position_cache(data, full_ranges[:max_keep])
            warmup_time_s = time.perf_counter() - warmup_t0
            print(
                f"[cam][range][warmup] {dataset} {args.workload} "
                f"({max_keep}/{total_queries}, setup={warmup_time_s:.6f}s)"
            )

        for rate in rates:
            label = str(rate["sample_label"])
            fraction = float(rate["sample_fraction"])
            percent = float(rate["sample_rate_percent"])

            setup_t0 = time.perf_counter()
            keep = sample_size(total_queries, fraction)
            sample_ranges = np.asarray(full_ranges[:keep], dtype=np.uint64)
            range_position_cache = range_io.prepare_range_position_cache(data, sample_ranges)
            setup_time_s = time.perf_counter() - setup_t0
            first_touch_scale = float(total_queries) / float(keep)

            for m_mib in args.memory_list:
                memory_bytes = int(m_mib) * 1024 * 1024
                for eps in epsilons:
                    estimate_kwargs = {
                        "epsilon": int(eps),
                        "n_keys": n_keys,
                        "seg_size": args.seg_size,
                        "memory_bytes": memory_bytes,
                        "ipp": args.ipp,
                        "page_size": args.page_size,
                        "policy": args.policy,
                        "data": data,
                        "queries": sample_ranges,
                        "range_position_cache": range_position_cache,
                        "first_touch_scale": first_touch_scale,
                        "conservative": not args.non_conservative,
                        "cold_start_correction": args.cold_start_correction,
                    }
                    for _ in range(args.warmup_repeats):
                        range_io.estimate_range_cost(**estimate_kwargs)

                    timings: list[float] = []
                    estimated_avg_io = 0.0
                    hit_ratio = 0.0
                    detail: dict[str, float] = {}
                    for _ in range(args.timing_repeats):
                        t0 = time.perf_counter()
                        estimated_avg_io, hit_ratio, detail = range_io.estimate_range_cost(**estimate_kwargs)
                        timings.append(time.perf_counter() - t0)
                    core_time_s = float(np.median(timings))
                    rows.append(
                        {
                            "method": "CAM",
                            "dataset": dataset,
                            "dataset_label": dataset_label(dataset),
                            "workload": args.workload,
                            "sample_rate_percent": percent,
                            "sample_fraction": fraction,
                            "sample_label": label,
                            "M": int(m_mib),
                            "epsilon": int(eps),
                            "policy": args.policy.upper(),
                            "strategy": args.strategy,
                            "estimated_avg_io": float(estimated_avg_io),
                            "estimated_total_ios": float(estimated_avg_io) * float(total_queries),
                            "estimated_avg_rdac": float(detail["avg_rdac"]),
                            "estimate_core_time_s": core_time_s,
                            "estimate_setup_time_s": float(setup_time_s),
                            "estimate_time_s": float(core_time_s + setup_time_s),
                            "estimate_time_mean_s": float(np.mean(timings)),
                            "estimate_time_min_s": float(np.min(timings)),
                            "estimate_time_max_s": float(np.max(timings)),
                            "estimate_time_std_s": float(np.std(timings, ddof=0)),
                            "estimate_timing_repeats": int(args.timing_repeats),
                            "estimate_warmup_repeats": int(args.warmup_repeats),
                            "setup_time_shared": 1,
                            "total_queries": total_queries,
                            "sample_queries": keep,
                            "ratio": float(hit_ratio),
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
                            "warmup_position_cache": int(args.warmup_position_cache),
                            "warmup_position_cache_time_s": float(warmup_time_s),
                        }
                    )

            print(
                f"[cam][range] {dataset} {args.workload} p{label} "
                f"({keep}/{total_queries}, setup={setup_time_s:.6f}s)"
            )

    out = args.output_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[cam][range] estimates -> {out} ({len(rows)} rows)")


def load_actual_csv(path: Path, dataset: str, workload: str, policy: str, strategy: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(
        df,
        path,
        {
            "epsilon",
            "policy",
            "strategy",
            "budget_mode",
            "memory_budget_bytes",
            "queries",
            "total_dac",
            "total_cache_hits",
            "total_cache_misses",
            "global_hit_ratio",
            "avg_dac",
            "avg_cam_io",
            "simulate_wall_ns",
            "index_build_ns",
        },
    )
    out = df.copy()
    out["dataset"] = dataset
    out["dataset_label"] = dataset_label(dataset)
    out["workload"] = workload
    out["policy"] = out["policy"].astype(str).str.upper()
    out["strategy"] = out["strategy"].astype(str)
    out = out[out["policy"] == policy.upper()].copy()
    out = out[out["strategy"] == strategy].copy()
    out = out[out["budget_mode"].astype(str).str.lower() == "estimated"].copy()
    out["M"] = (pd.to_numeric(out["memory_budget_bytes"], errors="coerce") / (1 << 20)).round().astype(int)
    out["epsilon"] = pd.to_numeric(out["epsilon"], errors="coerce").astype(int)
    out["actual_queries"] = pd.to_numeric(out["queries"], errors="coerce").astype(int)
    out["actual_total_ios"] = pd.to_numeric(out["total_cache_misses"], errors="coerce")
    out["actual_avg_io"] = pd.to_numeric(out["avg_cam_io"], errors="coerce")
    out["actual_total_dac"] = pd.to_numeric(out["total_dac"], errors="coerce")
    out["actual_avg_dac"] = pd.to_numeric(out["avg_dac"], errors="coerce")
    out["actual_cache_hits"] = pd.to_numeric(out["total_cache_hits"], errors="coerce")
    out["actual_hit_ratio"] = pd.to_numeric(out["global_hit_ratio"], errors="coerce")
    out["actual_simulate_time_s"] = pd.to_numeric(out["simulate_wall_ns"], errors="coerce") / 1e9
    out["actual_index_build_time_s"] = pd.to_numeric(out["index_build_ns"], errors="coerce") / 1e9
    return out[
        [
            "dataset",
            "dataset_label",
            "workload",
            "M",
            "epsilon",
            "policy",
            "strategy",
            "actual_queries",
            "actual_total_ios",
            "actual_avg_io",
            "actual_total_dac",
            "actual_avg_dac",
            "actual_cache_hits",
            "actual_hit_ratio",
            "actual_simulate_time_s",
            "actual_index_build_time_s",
        ]
    ]


def load_replay_csv(
    path: Path,
    dataset: str,
    workload: str,
    rate: dict[str, float | str],
    m_mib: int,
    policy: str,
    strategy: str,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(
        df,
        path,
        {
            "epsilon",
            "policy",
            "strategy",
            "budget_mode",
            "queries",
            "total_dac",
            "total_cache_hits",
            "total_cache_misses",
            "global_hit_ratio",
            "avg_dac",
            "avg_cam_io",
            "simulate_wall_ns",
            "index_build_ns",
        },
    )
    out = df.copy()
    out["policy"] = out["policy"].astype(str).str.upper()
    out["strategy"] = out["strategy"].astype(str)
    out = out[out["policy"] == policy.upper()].copy()
    out = out[out["strategy"] == strategy].copy()
    out = out[out["budget_mode"].astype(str).str.lower() == "estimated"].copy()
    out["method"] = "replay"
    out["dataset"] = dataset
    out["dataset_label"] = dataset_label(dataset)
    out["workload"] = workload
    out["sample_rate_percent"] = float(rate["sample_rate_percent"])
    out["sample_fraction"] = float(rate["sample_fraction"])
    out["sample_label"] = str(rate["sample_label"])
    out["M"] = int(m_mib)
    out["epsilon"] = pd.to_numeric(out["epsilon"], errors="coerce").astype(int)
    out["sample_queries"] = pd.to_numeric(out["queries"], errors="coerce").astype(int)
    out["sample_total_ios"] = pd.to_numeric(out["total_cache_misses"], errors="coerce")
    out["estimated_avg_io"] = pd.to_numeric(out["avg_cam_io"], errors="coerce")
    out["estimated_avg_rdac"] = pd.to_numeric(out["avg_dac"], errors="coerce")
    out["ratio"] = pd.to_numeric(out["global_hit_ratio"], errors="coerce")
    out["estimated_total_page_requests"] = pd.to_numeric(out["total_dac"], errors="coerce")
    if "cache_pages" in out.columns:
        out["estimated_cache_pages"] = pd.to_numeric(out["cache_pages"], errors="coerce")
    else:
        out["estimated_cache_pages"] = np.nan
    out["estimate_core_time_s"] = pd.to_numeric(out["simulate_wall_ns"], errors="coerce") / 1e9
    out["estimate_setup_time_s"] = pd.to_numeric(out["index_build_ns"], errors="coerce") / 1e9
    out["estimate_time_s"] = out["estimate_core_time_s"] + out["estimate_setup_time_s"]
    out["setup_time_shared"] = 0
    return out[
        [
            "method",
            "dataset",
            "dataset_label",
            "workload",
            "sample_rate_percent",
            "sample_fraction",
            "sample_label",
            "M",
            "epsilon",
            "policy",
            "strategy",
            "sample_queries",
            "sample_total_ios",
            "estimated_avg_io",
            "estimated_avg_rdac",
            "ratio",
            "estimated_total_page_requests",
            "estimated_cache_pages",
            "estimate_core_time_s",
            "estimate_setup_time_s",
            "estimate_time_s",
            "setup_time_shared",
        ]
    ]


def load_cam_csv(path: Path, policy: str, strategy: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(
        df,
        path,
        {
            "method",
            "dataset",
            "dataset_label",
            "workload",
            "sample_rate_percent",
            "sample_fraction",
            "sample_label",
            "M",
            "epsilon",
            "policy",
            "strategy",
            "sample_queries",
            "estimated_avg_io",
            "estimate_core_time_s",
            "estimate_setup_time_s",
            "estimate_time_s",
            "setup_time_shared",
        },
    )
    out = df.copy()
    out["policy"] = out["policy"].astype(str).str.upper()
    out["strategy"] = out["strategy"].astype(str)
    out = out[out["policy"] == policy.upper()].copy()
    out = out[out["strategy"] == strategy].copy()
    return out


def summarize_group(df: pd.DataFrame) -> dict[str, object]:
    first = df.iloc[0]
    method = str(first["method"])
    setup_time = (
        pd.to_numeric(df["estimate_setup_time_s"], errors="coerce").max()
        if method == "CAM"
        else pd.to_numeric(df["estimate_setup_time_s"], errors="coerce").sum()
    )
    core_time = pd.to_numeric(df["estimate_core_time_s"], errors="coerce").sum()
    row = {
        "method": method,
        "dataset": first["dataset"],
        "dataset_label": first["dataset_label"],
        "workload": first["workload"],
        "sample_rate_percent": first["sample_rate_percent"],
        "sample_fraction": first["sample_fraction"],
        "sample_label": first["sample_label"],
        "M": int(first["M"]),
        "policy": first["policy"],
        "strategy": first["strategy"],
        "epsilon_points": int(df["epsilon"].nunique()),
        "mean_accuracy": float(pd.to_numeric(df["accuracy"], errors="coerce").mean()),
        "mean_absolute_error": float(pd.to_numeric(df["absolute_error"], errors="coerce").mean()),
        "mean_relative_error": float(pd.to_numeric(df["relative_error"], errors="coerce").mean()),
        "mean_signed_relative_error": float(pd.to_numeric(df["signed_relative_error"], errors="coerce").mean()),
        "mean_actual_total_ios": float(pd.to_numeric(df["actual_total_ios"], errors="coerce").mean()),
        "mean_estimated_total_ios": float(pd.to_numeric(df["estimated_total_ios"], errors="coerce").mean()),
        "actual_queries": int(pd.to_numeric(df["actual_queries"], errors="coerce").max()),
        "sample_queries": int(pd.to_numeric(df["sample_queries"], errors="coerce").max()),
        "total_estimate_core_time_s": float(core_time),
        "total_estimate_setup_time_s": float(setup_time),
        "total_estimate_time_s": float(core_time + setup_time),
        "mean_estimate_time_s": float(pd.to_numeric(df["estimate_time_s"], errors="coerce").mean()),
        "mean_actual_avg_io": float(pd.to_numeric(df["actual_avg_io"], errors="coerce").mean()),
        "mean_estimated_avg_io": float(pd.to_numeric(df["estimated_avg_io"], errors="coerce").mean()),
    }

    optional_means = {
        "mean_actual_avg_dac": "actual_avg_dac",
        "mean_actual_hit_ratio": "actual_hit_ratio",
        "mean_estimated_avg_rdac": "estimated_avg_rdac",
        "mean_estimated_hit_ratio": "ratio",
        "mean_steady_hit_ratio": "steady_hit_ratio",
        "mean_cold_miss_ratio": "cold_miss_ratio",
        "mean_expected_distinct_pages": "expected_distinct_pages",
        "mean_estimated_cache_pages": "estimated_cache_pages",
        "mean_estimated_total_page_requests": "estimated_total_page_requests",
    }
    for out_col, src_col in optional_means.items():
        if src_col in df.columns:
            row[out_col] = float(pd.to_numeric(df[src_col], errors="coerce").mean())
        else:
            row[out_col] = math.nan
    return row


def cmd_sample_label(args: argparse.Namespace) -> None:
    labels = [str(rate["sample_label"]) for rate in parse_sample_rates(args.sample_rates)]
    print(" ".join(labels))


def cmd_summarize(args: argparse.Namespace) -> None:
    eps_set = set(parse_epsilons(args.epsilons))
    rates = parse_sample_rates(args.sample_rates)
    rate_labels = {str(rate["sample_label"]) for rate in rates}
    memory_set = {int(m_mib) for m_mib in args.memory_list}

    actual_frames: list[pd.DataFrame] = []
    replay_frames: list[pd.DataFrame] = []

    for dataset in args.datasets:
        for m_mib in args.memory_list:
            act = actual_path(args.actual_dir, dataset, args.workload, m_mib, args.policy)
            if not act.exists():
                raise FileNotFoundError(f"missing actual CSV: {act}")
            actual_frames.append(load_actual_csv(act, dataset, args.workload, args.policy, args.strategy))

            for rate in rates:
                label = str(rate["sample_label"])
                rep = replay_path(args.replay_dir, dataset, args.workload, label, m_mib, args.policy)
                if not rep.exists():
                    raise FileNotFoundError(f"missing replay CSV: {rep}")
                replay_frames.append(load_replay_csv(rep, dataset, args.workload, rate, m_mib, args.policy, args.strategy))

    actual = pd.concat(actual_frames, ignore_index=True)
    actual = actual[actual["epsilon"].isin(eps_set)].copy()
    if actual.empty:
        raise ValueError("no actual rows remain after epsilon filtering")

    replay = pd.concat(replay_frames, ignore_index=True)
    cam = load_cam_csv(args.cam_csv, args.policy, args.strategy)
    cam = cam[cam["sample_label"].astype(str).isin(rate_labels)].copy()
    cam["M"] = pd.to_numeric(cam["M"], errors="coerce").astype(int)
    cam = cam[cam["M"].isin(memory_set)].copy()
    estimate = pd.concat([replay, cam], ignore_index=True, sort=False)
    estimate["epsilon"] = pd.to_numeric(estimate["epsilon"], errors="coerce").astype(int)
    estimate["M"] = pd.to_numeric(estimate["M"], errors="coerce").astype(int)
    estimate = estimate[estimate["epsilon"].isin(eps_set)].copy()
    if estimate.empty:
        raise ValueError("no estimate rows remain after epsilon filtering")

    keys = ["dataset", "dataset_label", "workload", "M", "epsilon", "policy", "strategy"]
    merged = estimate.merge(actual, on=keys, how="inner")
    if merged.empty:
        raise ValueError("actual and estimate CSVs have no overlapping rows")

    merged["estimated_total_ios"] = pd.to_numeric(merged["estimated_avg_io"], errors="coerce") * pd.to_numeric(
        merged["actual_queries"], errors="coerce"
    )
    actual_ios = pd.to_numeric(merged["actual_total_ios"], errors="coerce")
    estimated_ios = pd.to_numeric(merged["estimated_total_ios"], errors="coerce")
    merged["absolute_error"] = (estimated_ios - actual_ios).abs()
    merged["signed_relative_error"] = np.where(actual_ios > 0, (estimated_ios - actual_ios) / actual_ios, np.nan)
    merged["relative_error"] = merged["signed_relative_error"].abs()
    merged["accuracy"] = np.where(actual_ios > 0, 1.0 - merged["relative_error"], np.nan)
    merged["accuracy"] = merged["accuracy"].clip(lower=0.0, upper=1.0)

    group_keys = [
        "method",
        "dataset",
        "dataset_label",
        "workload",
        "sample_rate_percent",
        "sample_fraction",
        "sample_label",
        "M",
        "policy",
        "strategy",
    ]
    summary_rows = [
        summarize_group(group_df.sort_values("epsilon"))
        for _, group_df in merged.groupby(group_keys, sort=True, dropna=False)
    ]
    summary = pd.DataFrame(summary_rows).sort_values(
        ["dataset_label", "M", "sample_rate_percent", "method"], kind="stable"
    )
    merged = merged.sort_values(
        ["dataset_label", "M", "epsilon", "sample_rate_percent", "method"], kind="stable"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "range_cmp_detail.csv"
    summary_path = args.output_dir / "range_cmp_summary.csv"
    merged.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"[summarize][range] detail -> {detail_path}")
    print(f"[summarize][range] summary -> {summary_path}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--datasets-directory", type=Path, default=Path(os.environ.get("DATASETS_DIRECTORY", str(DEFAULT_DATASETS_DIRECTORY))))
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--workload", default="w4")


def add_sample_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-rates", nargs="+", default=["20", "50", "80", "100"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare range-query replay and CAM estimates by sample rate.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prefix = sub.add_parser("make-prefixes", help="Write first-x-percent range workload prefixes.")
    add_common_args(p_prefix)
    add_sample_args(p_prefix)
    p_prefix.add_argument("--workload-dir", type=Path, required=True)
    p_prefix.add_argument("--output-dir", type=Path, required=True)
    p_prefix.add_argument("--query-limit", type=int, default=0)
    p_prefix.add_argument("--force", action="store_true")
    p_prefix.set_defaults(func=cmd_make_prefixes)

    p_cam = sub.add_parser("cam-estimate", help="Run CAM range estimates through utils/optimalEpsilon.py.")
    add_common_args(p_cam)
    add_sample_args(p_cam)
    p_cam.add_argument("--workload-dir", type=Path, required=True)
    p_cam.add_argument("--output-csv", type=Path, required=True)
    p_cam.add_argument("--memory-list", nargs="+", type=int, required=True)
    p_cam.add_argument("--epsilons", required=True)
    p_cam.add_argument("--policy", default="LRU")
    p_cam.add_argument("--strategy", default="all_in_once")
    p_cam.add_argument("--query-limit", type=int, default=0)
    p_cam.add_argument("--seg-size", type=int, default=16)
    p_cam.add_argument("--ipp", type=int, default=512)
    p_cam.add_argument("--page-size", type=int, default=4096)
    p_cam.add_argument("--warmup-repeats", type=int, default=0)
    p_cam.add_argument("--timing-repeats", type=int, default=1)
    p_cam.add_argument("--non-conservative", action="store_true")
    p_cam.add_argument("--cold-start-correction", action="store_true")
    p_cam.add_argument(
        "--no-warmup-position-cache",
        dest="warmup_position_cache",
        action="store_false",
        help="Include cold memmap/page-cache effects in the first timed sample-rate setup.",
    )
    p_cam.set_defaults(warmup_position_cache=True)
    p_cam.set_defaults(func=cmd_cam_estimate)

    p_sum = sub.add_parser("summarize", help="Merge actual, replay, and CAM range estimate CSVs.")
    add_common_args(p_sum)
    add_sample_args(p_sum)
    p_sum.add_argument("--actual-dir", type=Path, required=True)
    p_sum.add_argument("--replay-dir", type=Path, required=True)
    p_sum.add_argument("--cam-csv", type=Path, required=True)
    p_sum.add_argument("--output-dir", type=Path, required=True)
    p_sum.add_argument("--memory-list", nargs="+", type=int, required=True)
    p_sum.add_argument("--epsilons", required=True)
    p_sum.add_argument("--policy", default="LRU")
    p_sum.add_argument("--strategy", default="all_in_once")
    p_sum.set_defaults(func=cmd_summarize)

    p_label = sub.add_parser("sample-label", help="Print normalized sample-rate filename labels.")
    p_label.add_argument("sample_rates", nargs="+")
    p_label.set_defaults(func=cmd_sample_label)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
