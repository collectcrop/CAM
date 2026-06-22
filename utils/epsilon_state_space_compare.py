#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import time
from pathlib import Path

DEFAULT_DATASETS_DIRECTORY = os.environ.get("DATASETS_DIRECTORY", "/mnt/data/Dataset/public/SOSD")


def parse_policy_list(raw: str) -> list[str]:
    tokens = [token.strip().upper() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError("--policies must not be empty")
    allowed = {"FIFO", "LRU", "LFU", "NONE"}
    unique: list[str] = []
    for token in tokens:
        if token not in allowed:
            raise ValueError(f"unsupported policy: {token} (allowed: FIFO,LRU,LFU,NONE)")
        if token not in unique:
            unique.append(token)
    return unique


def parse_epsilons(
    epsilons_raw: str,
    epsilon_start: int,
    epsilon_end: int,
    epsilon_step: int,
) -> list[int]:
    if epsilons_raw:
        out = sorted({int(token.strip()) for token in epsilons_raw.split(",") if token.strip()})
    else:
        out = list(range(epsilon_start, epsilon_end + 1, epsilon_step))

    if not out:
        raise ValueError("epsilon set is empty")
    for eps in out:
        if eps <= 0:
            raise ValueError(f"epsilon={eps} is invalid; epsilon must be positive")
    return out


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_dataset_path(raw: str) -> str:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return str(p)
    if p.exists():
        return str(p.resolve())
    dataset_candidate = (Path(DEFAULT_DATASETS_DIRECTORY) / p).expanduser()
    return str(dataset_candidate)


def normalize_dataset_key(data_path: str) -> str:
    name = Path(data_path).name
    for suffix in ("_uint64_unique", "_uint64_sorted", "_uint64", "_unique", "_sorted"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def run_estimator(
    data_path: str,
    query_path: str,
    n_keys: int,
    m_mib: int,
    epsilons: list[int],
    policies: list[str],
    seg_size: int,
    ipp: int,
    page_size: int,
    strategy: str,
    budget_mode: str,
    index_size_bin: Path,
    index_size_csv: Path,
    cold_start_correction: bool,
) -> list[dict[str, float | int | str]]:
    try:
        import optimalEpsilon
        from optimalEpsilon import cost_function
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to import optimalEpsilon dependencies. "
            "Please run this script with an environment that has numpy/scipy installed "
            "(for example: ~/miniconda3/bin/python)."
        ) from exc

    rows: list[dict[str, float | int | str]] = []
    m_bytes = m_mib << 20
    measured_index_sizes: dict[int, int] = {}
    if budget_mode == "measured":
        measured_index_sizes = run_index_size_tool(
            index_size_bin=index_size_bin,
            data_path=data_path,
            n_keys=n_keys,
            epsilons=epsilons,
            output_csv=index_size_csv,
        )

    original_budget_mode = optimalEpsilon.BUDGET_MODE
    optimalEpsilon.BUDGET_MODE = "MEASURED" if budget_mode == "measured" else "ESTIMATED"
    for policy in policies:
        for eps in epsilons:
            measured_index_bytes = measured_index_sizes.get(eps, 0)
            if budget_mode == "measured":
                estimator_memory = max(0, m_bytes - measured_index_bytes)
            else:
                estimator_memory = m_bytes

            t0 = time.perf_counter()
            est_avg_io, est_hit_ratio, detail = cost_function(
                epsilon=eps,
                n=n_keys,
                seg_size=seg_size,
                M=m_bytes,
                ipp=ipp,
                ps=page_size,
                query_file=query_path,
                data_file=data_path,
                s=strategy,
                cache_policy=policy,
                measured_index_bytes=measured_index_bytes if budget_mode == "measured" else None,
                cold_start_correction=cold_start_correction,
                return_detail=True,
            )
            dt = time.perf_counter() - t0
            rows.append(
                {
                    "epsilon": eps,
                    "policy": policy,
                    "estimated_avg_logical_ios": float(est_avg_io),
                    "estimated_hit_ratio": float(est_hit_ratio),
                    "estimate_time_sec": float(dt),
                    "estimated_budget_mode": budget_mode,
                    "measured_index_bytes": measured_index_bytes,
                    "estimated_cache_bytes": estimator_memory
                    if budget_mode == "measured"
                    else max(0, m_bytes - int(n_keys * seg_size / (2 * eps))),
                    "cold_start_correction": int(cold_start_correction),
                    "steady_hit_ratio": float(detail.get("steady_hit_ratio", est_hit_ratio)),
                    "cold_miss_ratio": float(detail.get("cold_miss_ratio", 0.0)),
                    "expected_distinct_pages": float(detail.get("expected_distinct_pages", 0.0)),
                }
            )
    optimalEpsilon.BUDGET_MODE = original_budget_mode
    return rows


def run_index_size_tool(
    index_size_bin: Path,
    data_path: str,
    n_keys: int,
    epsilons: list[int],
    output_csv: Path,
) -> dict[int, int]:
    ensure_parent(output_csv)
    cmd = [
        str(index_size_bin),
        "--data",
        data_path,
        "--keys",
        str(n_keys),
        "--epsilons",
        ",".join(str(eps) for eps in epsilons),
        "--output",
        str(output_csv),
    ]
    subprocess.run(cmd, check=True)

    out: dict[int, int] = {}
    with output_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"epsilon", "measured_index_bytes"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"unexpected index-size CSV header in {output_csv}")
        for row in reader:
            out[int(row["epsilon"])] = int(row["measured_index_bytes"])
    return out


def run_simulator(
    sim_bin: Path,
    data_path: str,
    query_path: str,
    n_keys: int,
    m_mib: int,
    epsilons: list[int],
    policies: list[str],
    strategy: str,
    budget_mode: str,
    summary_out: Path,
    query_limit: int,
) -> float:
    ensure_parent(summary_out)
    cmd = [
        str(sim_bin),
        "--data",
        data_path,
        "--queries",
        query_path,
        "--keys",
        str(n_keys),
        "--M",
        str(m_mib),
        "--epsilons",
        ",".join(str(eps) for eps in epsilons),
        "--policies",
        ",".join(policies),
        "--strategies",
        strategy,
        "--budget-mode",
        budget_mode,
        "--summary-out",
        str(summary_out),
    ]
    if query_limit > 0:
        cmd.extend(["--query-limit", str(query_limit)])

    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - t0


def parse_simulation_summary(summary_path: Path, strategy: str) -> dict[tuple[int, str], dict[str, float]]:
    rows: dict[tuple[int, str], dict[str, float]] = {}
    with summary_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "epsilon",
            "policy",
            "strategy",
            "queries",
            "global_hit_ratio",
            "mean_cam_io",
            "wall_ns",
            "throughput_qps",
        }
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"unexpected summary CSV header in {summary_path}")

        for row in reader:
            if row["strategy"].strip().lower() != strategy.lower():
                continue
            key = (int(row["epsilon"]), row["policy"].strip().upper())
            rows[key] = {
                "queries": float(row["queries"]),
                "actual_hit_ratio": float(row["global_hit_ratio"]),
                "actual_avg_logical_ios": float(row["mean_cam_io"]),
                "actual_query_wall_sec": float(row["wall_ns"]) / 1e9,
                "actual_throughput_qps": float(row["throughput_qps"]),
            }
    return rows


def safe_rel_error(delta: float, baseline: float) -> float:
    return float("nan") if baseline == 0.0 else abs(delta) / abs(baseline)


def merge_rows(
    dataset_key: str,
    m_mib: int,
    estimate_rows: list[dict[str, float | int | str]],
    actual_rows: dict[tuple[int, str], dict[str, float]],
    sim_summary_path: Path,
) -> list[dict[str, float | int | str]]:
    merged: list[dict[str, float | int | str]] = []
    for row in estimate_rows:
        eps = int(row["epsilon"])
        policy = str(row["policy"])
        actual = actual_rows.get((eps, policy))
        if actual is None:
            continue

        est_avg_io = float(row["estimated_avg_logical_ios"])
        est_hit_ratio = float(row["estimated_hit_ratio"])
        est_time = float(row["estimate_time_sec"])
        estimated_budget_mode = str(row.get("estimated_budget_mode", "estimated"))
        measured_index_bytes = int(row.get("measured_index_bytes", 0))
        estimated_cache_bytes = int(row.get("estimated_cache_bytes", 0))
        cold_start_correction = int(row.get("cold_start_correction", 0))
        steady_hit_ratio = float(row.get("steady_hit_ratio", est_hit_ratio))
        cold_miss_ratio = float(row.get("cold_miss_ratio", 0.0))
        expected_distinct_pages = float(row.get("expected_distinct_pages", 0.0))

        act_avg_io = float(actual["actual_avg_logical_ios"])
        act_hit_ratio = float(actual["actual_hit_ratio"])
        act_time = float(actual["actual_query_wall_sec"])

        avg_io_err = est_avg_io - act_avg_io
        hit_err = est_hit_ratio - act_hit_ratio
        time_err = est_time - act_time

        merged.append(
            {
                "dataset_key": dataset_key,
                "policy": policy,
                "M": m_mib,
                "epsilon": eps,
                "estimated_avg_logical_ios": est_avg_io,
                "estimated_hit_ratio": est_hit_ratio,
                "estimate_time_sec": est_time,
                "estimated_budget_mode": estimated_budget_mode,
                "measured_index_bytes": measured_index_bytes,
                "estimated_cache_bytes": estimated_cache_bytes,
                "cold_start_correction": cold_start_correction,
                "steady_hit_ratio": steady_hit_ratio,
                "cold_miss_ratio": cold_miss_ratio,
                "expected_distinct_pages": expected_distinct_pages,
                "actual_queries": int(actual["queries"]),
                "actual_avg_logical_ios": act_avg_io,
                "actual_hit_ratio": act_hit_ratio,
                "actual_query_wall_sec": act_time,
                "actual_throughput_qps": float(actual["actual_throughput_qps"]),
                "avg_logical_ios_error": avg_io_err,
                "avg_logical_ios_abs_pct_error": safe_rel_error(avg_io_err, act_avg_io),
                "hit_ratio_error": hit_err,
                "hit_ratio_abs_error": abs(hit_err),
                "hit_ratio_abs_pct_error": safe_rel_error(hit_err, act_hit_ratio),
                "estimate_time_error_sec": time_err,
                "estimate_time_abs_pct_error": safe_rel_error(time_err, act_time),
                "sim_summary_source": str(summary_path_to_absolute(sim_summary_path)),
            }
        )
    return merged


def summary_path_to_absolute(path: Path) -> Path:
    return path if path.is_absolute() else path.resolve()


def write_csv(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    ensure_parent(output_path)
    if not rows:
        raise ValueError("no merged rows to write")

    fieldnames = [
        "dataset_key",
        "policy",
        "M",
        "epsilon",
        "estimated_avg_logical_ios",
        "estimated_hit_ratio",
        "estimate_time_sec",
        "estimated_budget_mode",
        "measured_index_bytes",
        "estimated_cache_bytes",
        "cold_start_correction",
        "steady_hit_ratio",
        "cold_miss_ratio",
        "expected_distinct_pages",
        "actual_queries",
        "actual_avg_logical_ios",
        "actual_hit_ratio",
        "actual_query_wall_sec",
        "actual_throughput_qps",
        "avg_logical_ios_error",
        "avg_logical_ios_abs_pct_error",
        "hit_ratio_error",
        "hit_ratio_abs_error",
        "hit_ratio_abs_pct_error",
        "estimate_time_error_sec",
        "estimate_time_abs_pct_error",
        "sim_summary_source",
    ]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Traverse epsilon state-space, compare optimalEpsilon.py estimates "
            "against build/pgm_cam_covariance simulation outputs."
        )
    )
    parser.add_argument("--data", required=True, help="dataset key file path")
    parser.add_argument("--queries", required=True, help="query file path")
    parser.add_argument("--keys", required=True, type=int, help="number of indexed keys")
    parser.add_argument("--M", required=True, type=int, help="memory budget in MiB")
    parser.add_argument("--seg-size", type=int, default=16, help="estimated segment bytes")
    parser.add_argument("--ipp", type=int, default=512, help="items per page")
    parser.add_argument("--page-size", type=int, default=4096, help="page size in bytes")
    parser.add_argument(
        "--strategy",
        default="all_in_once",
        choices=["all_in_once", "one_by_one"],
        help="query strategy used by both estimator and simulator",
    )
    parser.add_argument(
        "--policies",
        default="FIFO,LRU,LFU",
        help="comma-separated cache policies (FIFO,LRU,LFU,NONE)",
    )
    parser.add_argument(
        "--epsilons",
        default="",
        help="optional explicit epsilon list, e.g. 8,10,12 (overrides start/end/step)",
    )
    parser.add_argument("--epsilon-start", type=int, default=None, help="epsilon sweep start (inclusive)")
    parser.add_argument("--epsilon-end", type=int, default=128, help="epsilon sweep end (inclusive)")
    parser.add_argument("--epsilon-step", type=int, default=2, help="epsilon sweep step")
    parser.add_argument(
        "--sim-bin",
        type=Path,
        default=Path("build/pgm_cam_covariance"),
        help="path to pgm_cam_covariance binary",
    )
    parser.add_argument(
        "--index-size-bin",
        type=Path,
        default=Path("build/pgm_index_sizes"),
        help="path to pgm_index_sizes binary used by measured CAM estimates",
    )
    parser.add_argument(
        "--budget-mode",
        default="estimated",
        choices=["estimated", "measured"],
        help="memory reservation mode for simulator",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=0,
        help="optional query cap passed to simulator (0 means all)",
    )
    parser.add_argument(
        "--sim-summary-out",
        type=Path,
        default=None,
        help="output CSV path from pgm_cam_covariance",
    )
    parser.add_argument(
        "--index-size-out",
        type=Path,
        default=None,
        help="output CSV path for measured PGM index sizes",
    )
    parser.add_argument(
        "--merged-out",
        type=Path,
        default=None,
        help="merged comparison CSV output path",
    )
    parser.add_argument(
        "--reuse-sim-summary",
        action="store_true",
        help="skip running simulator and reuse --sim-summary-out",
    )
    parser.add_argument(
        "--cold-start-correction",
        action="store_true",
        help="Enable cold-start compulsory-miss correction in point-query CAM estimates.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    policies = parse_policy_list(args.policies)
    m_bytes = args.M << 20
    least_eps = math.ceil(args.keys * args.seg_size / (2 * m_bytes))
    least_even_eps = max(2, least_eps if least_eps % 2 == 0 else least_eps + 1)
    epsilon_start = least_even_eps if args.epsilon_start is None else args.epsilon_start

    epsilons = parse_epsilons(args.epsilons, epsilon_start, args.epsilon_end, args.epsilon_step)

    data_path = resolve_dataset_path(args.data)
    query_path = resolve_dataset_path(args.queries)
    dataset_key = normalize_dataset_key(data_path)

    sim_summary_out = (
        args.sim_summary_out
        if args.sim_summary_out is not None
        else Path(f"build/log/{dataset_key}_M{args.M}_statespace_summary.csv")
    )
    index_size_out = (
        args.index_size_out
        if args.index_size_out is not None
        else Path(f"build/log/{dataset_key}_M{args.M}_statespace_index_sizes.csv")
    )
    merged_out = (
        args.merged_out
        if args.merged_out is not None
        else Path(f"data/outputs/figures/epsilon_analysis/{dataset_key}_M{args.M}_statespace_compare.csv")
    )

    print(f"[info] epsilon count={len(epsilons)}, first={epsilons[0]}, last={epsilons[-1]}")
    print(f"[info] policies={policies}")

    estimate_rows = run_estimator(
        data_path=data_path,
        query_path=query_path,
        n_keys=args.keys,
        m_mib=args.M,
        epsilons=epsilons,
        policies=policies,
        seg_size=args.seg_size,
        ipp=args.ipp,
        page_size=args.page_size,
        strategy=args.strategy,
        budget_mode=args.budget_mode,
        index_size_bin=args.index_size_bin,
        index_size_csv=index_size_out,
        cold_start_correction=args.cold_start_correction,
    )

    if not args.reuse_sim_summary:
        if not args.sim_bin.exists():
            raise FileNotFoundError(f"simulator binary not found: {args.sim_bin}")
        sim_total_sec = run_simulator(
            sim_bin=args.sim_bin,
            data_path=data_path,
            query_path=query_path,
            n_keys=args.keys,
            m_mib=args.M,
            epsilons=epsilons,
            policies=policies,
            strategy=args.strategy,
            budget_mode=args.budget_mode,
            summary_out=sim_summary_out,
            query_limit=args.query_limit,
        )
        print(f"[info] simulator completed in {sim_total_sec:.3f}s, summary={sim_summary_out}")
    elif not sim_summary_out.exists():
        raise FileNotFoundError(f"--reuse-sim-summary is set but file does not exist: {sim_summary_out}")

    actual_rows = parse_simulation_summary(sim_summary_out, args.strategy)
    merged_rows = merge_rows(
        dataset_key=dataset_key,
        m_mib=args.M,
        estimate_rows=estimate_rows,
        actual_rows=actual_rows,
        sim_summary_path=sim_summary_out,
    )

    expected = len(epsilons) * len(policies)
    if len(merged_rows) != expected:
        print(
            f"[warn] merged rows={len(merged_rows)} but expected={expected}. "
            "Some epsilon/policy combinations are missing in simulator output."
        )

    write_csv(merged_rows, merged_out)
    print(f"[done] merged comparison written to: {merged_out}")


if __name__ == "__main__":
    main()
