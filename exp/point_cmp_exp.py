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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import optimalEpsilon  # noqa: E402


DEFAULT_DATASETS = [
    "books_200M_uint64_unique",
    "fb_200M_uint64_unique",
    "wiki_ts_200M_uint64_unique",
    "osm_cellids_200M_uint64_unique",
]
DEFAULT_DATASETS_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "datasets" / "SOSD"


def split_tokens(values: list[str] | str) -> list[str]:
    if isinstance(values, str):
        values = [values]
    out: list[str] = []
    for value in values:
        for token in value.replace(",", " ").split():
            token = token.strip()
            if token:
                out.append(token)
    return out


def parse_epsilons(text: str) -> list[int]:
    epsilons = [int(token) for token in split_tokens(text)]
    if not epsilons:
        raise ValueError("empty epsilon list")
    return list(dict.fromkeys(epsilons))


def sample_label(percent: float) -> str:
    if math.isclose(percent, round(percent), rel_tol=0.0, abs_tol=1e-9):
        return str(int(round(percent)))
    return f"{percent:.6g}".replace(".", "_")


def parse_sample_rates(values: list[str] | str) -> list[dict[str, float | str]]:
    rates: list[dict[str, float | str]] = []
    seen: set[str] = set()
    for token in split_tokens(values):
        cleaned = token[:-1] if token.endswith("%") else token
        percent = float(cleaned)
        if percent <= 0:
            raise ValueError(f"sample rate must be positive: {token}")
        if percent > 100.0:
            raise ValueError(f"sample rate must be <= 100%: {token}")
        fraction = percent / 100.0
        label = sample_label(percent)
        if label in seen:
            continue
        seen.add(label)
        rates.append(
            {
                "sample_fraction": float(fraction),
                "sample_rate_percent": float(percent),
                "sample_label": label,
            }
        )
    if not rates:
        raise ValueError("empty sample-rate list")
    return rates


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


def dataset_path(datasets_directory: Path, dataset: str) -> Path:
    path = Path(dataset)
    if path.is_absolute():
        return path
    return datasets_directory / dataset


def open_key_array(path: Path) -> np.ndarray:
    mm = np.memmap(path, dtype=np.uint64, mode="r")
    if len(mm) > 0 and int(mm[0]) == len(mm) - 1:
        return mm[1:]
    return mm


def workload_path(workload_dir: Path, dataset: str, workload: str) -> Path:
    return workload_dir / dataset / f"{dataset}.{workload}.bin"


def prefix_path(prefix_dir: Path, dataset: str, workload: str, label: str) -> Path:
    return prefix_dir / dataset / f"{dataset}.{workload}.p{label}.bin"


def actual_path(actual_dir: Path, dataset: str, workload: str, m_mib: int, policy: str) -> Path:
    return actual_dir / dataset / f"{dataset}_{workload}_M{m_mib}_{policy.upper()}_actual.csv"


def replay_path(replay_dir: Path, dataset: str, workload: str, label: str, m_mib: int, policy: str) -> Path:
    return replay_dir / dataset / f"{dataset}_{workload}_p{label}_M{m_mib}_{policy.upper()}_replay.csv"


def lpm_path(lpm_dir: Path, dataset: str, workload: str, m_mib: int) -> Path:
    return lpm_dir / dataset / f"{dataset}_{workload}_M{m_mib}_NONE_actual.csv"


def sample_size(total_queries: int, fraction: float) -> int:
    if total_queries <= 0:
        raise ValueError("empty workload")
    if fraction >= 1.0:
        return total_queries
    return max(1, int(total_queries * fraction))


def load_limited_queries(query_path: Path, query_limit: int) -> np.ndarray:
    queries = np.fromfile(query_path, dtype=np.uint64)
    if query_limit > 0:
        queries = queries[:query_limit]
    if queries.size == 0:
        raise ValueError(f"empty query file: {query_path}")
    return queries


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cmd_make_prefixes(args: argparse.Namespace) -> None:
    rates = parse_sample_rates(args.sample_rates)
    rows: list[dict[str, object]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        q_path = workload_path(args.workload_dir, dataset, args.workload)
        if not q_path.exists():
            raise FileNotFoundError(f"missing workload file: {q_path}")
        queries = load_limited_queries(q_path, args.query_limit)
        total_queries = int(queries.size)

        for rate in rates:
            label = str(rate["sample_label"])
            fraction = float(rate["sample_fraction"])
            keep = sample_size(total_queries, fraction)
            out = prefix_path(args.output_dir, dataset, args.workload, label)
            out.parent.mkdir(parents=True, exist_ok=True)

            existing_queries = out.stat().st_size // np.dtype(np.uint64).itemsize if out.exists() else -1
            if args.force or existing_queries != keep:
                np.asarray(queries[:keep], dtype=np.uint64).tofile(out)
                action = "write"
            else:
                action = "skip"
            print(f"[prefix][{action}] {dataset} {args.workload} p{label} -> {out} ({keep}/{total_queries})")

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
    print(f"[prefix] manifest -> {manifest}")


def prepare_position_cache(data: np.ndarray, queries: np.ndarray) -> dict[str, np.ndarray]:
    pos = np.searchsorted(data, queries, side="right") - 1
    pos = np.clip(pos, 0, len(data) - 1).astype(np.int64)
    return {"kind": "positions", "pos": pos}


def cmd_cam_estimate(args: argparse.Namespace) -> None:
    optimalEpsilon.BUDGET_MODE = "ESTIMATED"
    epsilons = parse_epsilons(args.epsilons)
    rates = parse_sample_rates(args.sample_rates)
    rows: list[dict[str, object]] = []

    for dataset in args.datasets:
        data_path_ = dataset_path(args.datasets_directory, dataset)
        if not data_path_.exists():
            raise FileNotFoundError(f"missing dataset: {data_path_}")
        data = open_key_array(data_path_)
        n_keys = int(len(data))

        q_path = workload_path(args.workload_dir, dataset, args.workload)
        if not q_path.exists():
            raise FileNotFoundError(f"missing workload file: {q_path}")
        full_queries = load_limited_queries(q_path, args.query_limit)
        total_queries = int(full_queries.size)
        warmup_time_s = 0.0
        if args.warmup_position_cache:
            max_keep = max(sample_size(total_queries, float(rate["sample_fraction"])) for rate in rates)
            warmup_t0 = time.perf_counter()
            prepare_position_cache(data, full_queries[:max_keep])
            warmup_time_s = time.perf_counter() - warmup_t0
            print(
                f"[cam][warmup] {dataset} {args.workload} "
                f"({max_keep}/{total_queries}, setup={warmup_time_s:.6f}s)"
            )

        for rate in rates:
            label = str(rate["sample_label"])
            fraction = float(rate["sample_fraction"])
            percent = float(rate["sample_rate_percent"])
            keep = sample_size(total_queries, fraction)
            sample_queries = full_queries[:keep]

            setup_t0 = time.perf_counter()
            h_cache = prepare_position_cache(data, sample_queries)
            setup_time_s = time.perf_counter() - setup_t0
            first_touch_scale = float(total_queries) / float(keep)

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
                        H=h_cache,
                        Q=keep,
                        cold_start_correction=args.cold_start_correction,
                        first_touch_scale=first_touch_scale,
                        return_detail=True,
                    )
                    core_time_s = time.perf_counter() - t0
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
                            "estimate_core_time_s": float(core_time_s),
                            "estimate_setup_time_s": float(setup_time_s),
                            "estimate_time_s": float(core_time_s + setup_time_s),
                            "setup_time_shared": 1,
                            "total_queries": total_queries,
                            "sample_queries": keep,
                            "ratio": float(hit_ratio),
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
                            "warmup_position_cache": int(args.warmup_position_cache),
                            "warmup_position_cache_time_s": float(warmup_time_s),
                        }
                    )

            print(
                f"[cam] {dataset} {args.workload} p{label} "
                f"({keep}/{total_queries}, setup={setup_time_s:.6f}s)"
            )

    out = args.output_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[cam] estimates -> {out} ({len(rows)} rows)")


def require_columns(df: pd.DataFrame, path: Path, columns: set[str]) -> None:
    missing = columns.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


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
            "total_cache_misses",
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
            "total_cache_misses",
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


def load_lpm_csv(
    path: Path,
    dataset: str,
    workload: str,
    m_mib: int,
    policy: str,
    strategy: str,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(
        df,
        path,
        {"epsilon", "policy", "strategy", "queries", "total_cache_misses", "simulate_wall_ns", "index_build_ns"},
    )
    out = df.copy()
    out = out[out["policy"].astype(str).str.upper() == "NONE"].copy()
    out = out[out["strategy"].astype(str) == strategy].copy()
    out["method"] = "LPM"
    out["dataset"] = dataset
    out["dataset_label"] = dataset_label(dataset)
    out["workload"] = workload
    out["sample_rate_percent"] = 100.0
    out["sample_fraction"] = 1.0
    out["sample_label"] = "full"
    out["M"] = int(m_mib)
    out["epsilon"] = pd.to_numeric(out["epsilon"], errors="coerce").astype(int)
    out["policy"] = policy.upper()
    out["strategy"] = strategy
    out["sample_queries"] = pd.to_numeric(out["queries"], errors="coerce").astype(int)
    out["sample_total_ios"] = pd.to_numeric(out["total_cache_misses"], errors="coerce")
    out["estimated_avg_io"] = out["sample_total_ios"] / out["sample_queries"]
    out["estimate_core_time_s"] = pd.to_numeric(out["simulate_wall_ns"], errors="coerce") / 1e9
    out["estimate_setup_time_s"] = pd.to_numeric(out["index_build_ns"], errors="coerce") / 1e9
    out["estimate_time_s"] = out["estimate_core_time_s"] + out["estimate_setup_time_s"]
    out["setup_time_shared"] = 0
    return out[
        [
            "method", "dataset", "dataset_label", "workload", "sample_rate_percent",
            "sample_fraction", "sample_label", "M", "epsilon", "policy", "strategy",
            "sample_queries", "sample_total_ios", "estimated_avg_io", "estimate_core_time_s",
            "estimate_setup_time_s", "estimate_time_s", "setup_time_shared",
        ]
    ]


def make_simple_lpm_rows(
    actual: pd.DataFrame,
    epsilons: set[int],
    items_per_page: int,
) -> pd.DataFrame:
    if items_per_page <= 0:
        raise ValueError("simple-LPM items per page must be positive")
    rows: list[dict[str, object]] = []
    for _, actual_row in actual.iterrows():
        epsilon = int(actual_row["epsilon"])
        if epsilon not in epsilons:
            continue
        started = time.perf_counter_ns()
        estimated_avg_io = 1.0 + 2.0 * epsilon / items_per_page
        estimated_total_ios = estimated_avg_io * int(actual_row["actual_queries"])
        elapsed_s = (time.perf_counter_ns() - started) / 1e9
        rows.append(
            {
                "method": "simple-LPM",
                "dataset": actual_row["dataset"],
                "dataset_label": actual_row["dataset_label"],
                "workload": actual_row["workload"],
                "sample_rate_percent": 100.0,
                "sample_fraction": 1.0,
                "sample_label": "formula",
                "M": int(actual_row["M"]),
                "epsilon": epsilon,
                "policy": actual_row["policy"],
                "strategy": actual_row["strategy"],
                "sample_queries": int(actual_row["actual_queries"]),
                "sample_total_ios": estimated_total_ios,
                "estimated_avg_io": estimated_avg_io,
                "estimate_core_time_s": elapsed_s,
                "estimate_setup_time_s": 0.0,
                "estimate_time_s": elapsed_s,
                "setup_time_shared": 0,
            }
        )
    return pd.DataFrame(rows)


def summarize_group(df: pd.DataFrame) -> dict[str, object]:
    first = df.iloc[0]
    method = str(first["method"])
    setup_time = (
        pd.to_numeric(df["estimate_setup_time_s"], errors="coerce").max()
        if method == "CAM"
        else pd.to_numeric(df["estimate_setup_time_s"], errors="coerce").sum()
    )
    core_time = pd.to_numeric(df["estimate_core_time_s"], errors="coerce").sum()
    return {
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
    }


def cmd_sample_label(args: argparse.Namespace) -> None:
    labels = [str(rate["sample_label"]) for rate in parse_sample_rates(args.sample_rates)]
    print(" ".join(labels))


def cmd_summarize(args: argparse.Namespace) -> None:
    eps_set = set(parse_epsilons(args.epsilons))
    rates = parse_sample_rates(args.sample_rates)

    actual_frames: list[pd.DataFrame] = []
    replay_frames: list[pd.DataFrame] = []
    lpm_frames: list[pd.DataFrame] = []

    for dataset in args.datasets:
        for m_mib in args.memory_list:
            act = actual_path(args.actual_dir, dataset, args.workload, m_mib, args.policy)
            if not act.exists():
                raise FileNotFoundError(f"missing actual CSV: {act}")
            actual_frames.append(load_actual_csv(act, dataset, args.workload, args.policy, args.strategy))

            lpm = lpm_path(args.lpm_dir, dataset, args.workload, m_mib)
            if not lpm.exists():
                raise FileNotFoundError(f"missing LPM wocache CSV: {lpm}")
            lpm_frames.append(load_lpm_csv(lpm, dataset, args.workload, m_mib, args.policy, args.strategy))

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
    lpm = pd.concat(lpm_frames, ignore_index=True)
    simple_lpm = make_simple_lpm_rows(actual, eps_set, args.simple_lpm_items_per_page)
    estimate = pd.concat([replay, cam, lpm, simple_lpm], ignore_index=True, sort=False)
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
    replay100 = merged[
        (merged["method"] == "replay")
        & np.isclose(pd.to_numeric(merged["sample_rate_percent"], errors="coerce"), 100.0)
    ][keys + ["estimated_total_ios"]].rename(columns={"estimated_total_ios": "replay100_total_ios"})
    if replay100.duplicated(keys).any():
        raise ValueError("multiple Replay-100 truth rows found for the same configuration")
    merged = merged.merge(replay100, on=keys, how="left")
    baseline_mask = merged["method"].isin(["LPM", "simple-LPM"])
    if merged.loc[baseline_mask, "replay100_total_ios"].isna().any():
        raise ValueError("missing Replay-100 truth for LPM/simple-LPM rows; include sample rate 100")
    merged.loc[baseline_mask, "actual_total_ios"] = merged.loc[baseline_mask, "replay100_total_ios"]
    merged.loc[baseline_mask, "actual_avg_io"] = (
        merged.loc[baseline_mask, "actual_total_ios"] / merged.loc[baseline_mask, "actual_queries"]
    )
    actual_ios = pd.to_numeric(merged["actual_total_ios"], errors="coerce")
    estimated_ios = pd.to_numeric(merged["estimated_total_ios"], errors="coerce")
    merged["absolute_error"] = (estimated_ios - actual_ios).abs()
    merged["signed_relative_error"] = np.where(actual_ios > 0, (estimated_ios - actual_ios) / actual_ios, np.nan)
    merged["relative_error"] = merged["signed_relative_error"].abs()
    merged["q_error"] = np.where(
        (actual_ios > 0) & (estimated_ios > 0),
        np.maximum(estimated_ios / actual_ios, actual_ios / estimated_ios),
        np.nan,
    )
    merged["accuracy"] = np.where(actual_ios > 0, 1.0 - merged["relative_error"], np.nan)
    merged["accuracy"] = merged["accuracy"].clip(lower=0.0, upper=1.0)

    simple_results = merged[merged["method"] == "simple-LPM"].sort_values(
        ["dataset_label", "M", "epsilon"], kind="stable"
    )
    for _, row in simple_results.iterrows():
        print(
            "[simple-LPM] "
            f"dataset={row['dataset']} M={int(row['M'])}MiB epsilon={int(row['epsilon'])} "
            f"avg_dac={float(row['estimated_avg_io']):.10f} "
            f"estimated_ios={float(row['estimated_total_ios']):.6f} "
            f"replay100_ios={float(row['actual_total_ios']):.6f} "
            f"relative_error={float(row['relative_error']):.10f} "
            f"q_error={float(row['q_error']):.10f} "
            f"time_s={float(row['estimate_time_s']):.9f}"
        )

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
    detail_path = args.output_dir / "point_cmp_detail.csv"
    summary_path = args.output_dir / "point_cmp_summary.csv"
    merged.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"[summarize] detail -> {detail_path}")
    print(f"[summarize] summary -> {summary_path}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--datasets-directory", type=Path, default=Path(os.environ.get("DATASETS_DIRECTORY", str(DEFAULT_DATASETS_DIRECTORY))))
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--workload", default="w4")


def add_sample_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-rates", nargs="+", default=["20", "50", "80", "100"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare point-query replay and CAM estimates by sample rate.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prefix = sub.add_parser("make-prefixes", help="Write first-x-percent point workload prefixes.")
    add_common_args(p_prefix)
    add_sample_args(p_prefix)
    p_prefix.add_argument("--workload-dir", type=Path, required=True)
    p_prefix.add_argument("--output-dir", type=Path, required=True)
    p_prefix.add_argument("--query-limit", type=int, default=0)
    p_prefix.add_argument("--force", action="store_true")
    p_prefix.set_defaults(func=cmd_make_prefixes)

    p_cam = sub.add_parser("cam-estimate", help="Run CAM estimates through utils/optimalEpsilon.py.")
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
    p_cam.add_argument("--cold-start-correction", action="store_true")
    p_cam.add_argument(
        "--no-warmup-position-cache",
        dest="warmup_position_cache",
        action="store_false",
        help="Include cold memmap/page-cache effects in the first timed sample-rate setup.",
    )
    p_cam.set_defaults(warmup_position_cache=True)
    p_cam.set_defaults(func=cmd_cam_estimate)

    p_sum = sub.add_parser("summarize", help="Merge actual, replay, and CAM estimate CSVs.")
    add_common_args(p_sum)
    add_sample_args(p_sum)
    p_sum.add_argument("--actual-dir", type=Path, required=True)
    p_sum.add_argument("--lpm-dir", type=Path, required=True)
    p_sum.add_argument("--replay-dir", type=Path, required=True)
    p_sum.add_argument("--cam-csv", type=Path, required=True)
    p_sum.add_argument("--output-dir", type=Path, required=True)
    p_sum.add_argument("--memory-list", nargs="+", type=int, required=True)
    p_sum.add_argument("--epsilons", required=True)
    p_sum.add_argument("--policy", default="LRU")
    p_sum.add_argument("--strategy", default="all_in_once")
    p_sum.add_argument("--simple-lpm-items-per-page", type=int, default=512)
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
