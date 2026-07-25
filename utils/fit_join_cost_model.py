#!/usr/bin/env python3
"""Fit hybrid join cost-model parameters from join-fit benchmark CSVs."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit point/range join cost-model parameters from CSV outputs."
    )
    parser.add_argument("--data-dir", default="build/log/join_fit")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--epsilon", type=int, default=16)
    parser.add_argument("--mode", choices=("all", "point", "range"), default="all")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-env", default="")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], name: str) -> float:
    try:
        return float(row[name])
    except KeyError as exc:
        raise KeyError(f"missing column {name!r}") from exc


def maybe_float(row: dict[str, str], name: str) -> float | None:
    value = row.get(name)
    if value is None or value == "":
        return None
    return float(value)


def detect_io_col(row: dict[str, str]) -> str:
    for name in ("IOs", "avg_IOs", "physical_ios"):
        if name in row:
            return name
    raise KeyError("no I/O count column found (expected IOs, avg_IOs, or physical_ios)")


def detect_range_span_col(row: dict[str, str]) -> str:
    for name in ("range_pages", "DAC", "logical_pages_read"):
        if name in row:
            return name
    return detect_io_col(row)


def extract_n(path: Path) -> int | None:
    match = re.search(r"(\d+)Mquery", path.name)
    if not match:
        return None
    return int(match.group(1)) * 1_000_000


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute quantile of empty values")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def iqr_filter(values: list[float]) -> tuple[list[float], float, float]:
    q1 = quantile(values, 0.25)
    q3 = quantile(values, 0.75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return [v for v in values if lo <= v <= hi], lo, hi


def linear_fit(xs: list[float], ys: list[float], x_name: str) -> tuple[float, float]:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError(f"need at least two rows to fit against {x_name}")
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0.0:
        raise ValueError(f"need at least two distinct {x_name} values")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    intercept = mean_y - slope * mean_x
    return slope, intercept


def finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"non-finite fitted value: {value}")
    return value


def point_files(data_dir: Path, dataset: str) -> list[Path]:
    pattern = f"{dataset}_*Mquery_join.point.csv" if dataset else "*_*Mquery_join.point.csv"
    return [Path(p) for p in sorted(glob.glob(str(data_dir / pattern)))]


def range_files(data_dir: Path, dataset: str) -> list[Path]:
    if dataset:
        preferred = data_dir / f"{dataset}_query_join.range.csv"
        if preferred.exists():
            return [preferred]
    return [Path(p) for p in sorted(glob.glob(str(data_dir / "*query_join*.range.csv")))]


def fit_point(data_dir: Path, dataset: str, epsilon: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    inputs = point_files(data_dir, dataset)
    for path in inputs:
        for row in read_csv_rows(path):
            if int(float(row.get("epsilon", "nan"))) != epsilon:
                continue
            n_value = maybe_float(row, "num_queries")
            if n_value is None:
                n = extract_n(path)
                if n is None:
                    raise ValueError(f"cannot infer num_queries from {path}")
                n_value = float(n)
            rows.append({"path": path, "row": row, "N": n_value})

    if not rows:
        raise FileNotFoundError(f"no point data found in {data_dir}")

    io_col = detect_io_col(rows[0]["row"])
    latency_values = [
        as_float(item["row"], "IO_time_s") / as_float(item["row"], io_col)
        for item in rows
        if as_float(item["row"], io_col) > 0.0
    ]
    if not latency_values:
        raise ValueError("no point rows with positive I/O counts")
    filtered_latency, latency_lo, latency_hi = iqr_filter(latency_values)
    if not filtered_latency:
        raise ValueError("no point rows left after latency outlier filtering")

    lambda_point = finite(statistics.median(filtered_latency))
    xs = [float(item["N"]) for item in rows]
    ys = [
        as_float(item["row"], "total_wall_time_s") - as_float(item["row"], "IO_time_s")
        for item in rows
    ]
    alpha, delta = linear_fit(xs, ys, "num_queries")
    return {
        "lambda_point_s_per_page": finite(lambda_point),
        "lambda_point_us_per_page": finite(lambda_point * 1e6),
        "alpha_s_per_key": finite(alpha),
        "delta_s": finite(delta),
        "rows": len(rows),
        "io_rows": len(latency_values),
        "filtered_io_rows": len(filtered_latency),
        "latency_iqr_low_s": finite(latency_lo),
        "latency_iqr_high_s": finite(latency_hi),
        "input_files": sorted({str(item["path"]) for item in rows}),
        "io_column": io_col,
    }


def fit_range(data_dir: Path, dataset: str, epsilon: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    inputs = range_files(data_dir, dataset)
    for path in inputs:
        for row in read_csv_rows(path):
            if int(float(row.get("epsilon", "nan"))) == epsilon:
                rows.append({"path": path, "row": row})

    if not rows:
        raise FileNotFoundError(f"no range data found in {data_dir}")

    io_col = detect_io_col(rows[0]["row"])
    k_col = detect_range_span_col(rows[0]["row"])
    rows = [item for item in rows if as_float(item["row"], k_col) > 0.0]
    if not rows:
        raise ValueError(f"no range rows with positive {k_col}")

    latency_values = [
        as_float(item["row"], "IO_time_s") / as_float(item["row"], io_col)
        for item in rows
        if as_float(item["row"], io_col) > 0.0
    ]
    if not latency_values:
        raise ValueError("no range rows with positive I/O counts")
    filtered_latency, latency_lo, latency_hi = iqr_filter(latency_values)
    if not filtered_latency:
        raise ValueError("no range rows left after latency outlier filtering")

    lambda_range = finite(statistics.median(filtered_latency))
    xs = [as_float(item["row"], k_col) for item in rows]
    ys = [
        as_float(item["row"], "total_wall_time_s") - as_float(item["row"], "IO_time_s")
        for item in rows
    ]
    beta, eta = linear_fit(xs, ys, k_col)
    return {
        "lambda_range_s_per_page": finite(lambda_range),
        "lambda_range_us_per_page": finite(lambda_range * 1e6),
        "beta_s_per_page_scan": finite(beta),
        "eta_s": finite(eta),
        "rows": len(rows),
        "io_rows": len(latency_values),
        "filtered_io_rows": len(filtered_latency),
        "latency_iqr_low_s": finite(latency_lo),
        "latency_iqr_high_s": finite(latency_hi),
        "input_files": sorted({str(item["path"]) for item in rows}),
        "io_column": io_col,
        "range_span_column": k_col,
    }


def default_output_path(data_dir: Path, dataset: str, suffix: str) -> Path:
    stem = f"{dataset}_join_cost_params" if dataset else "join_cost_params"
    return data_dir / f"{stem}.{suffix}"


def partition_env(result: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    point = result.get("point")
    if point:
        out["ALPHA"] = point["alpha_s_per_key"]
        out["DELTA"] = point["delta_s"]
        out["LAMBDA_POINT"] = point["lambda_point_s_per_page"]
    range_part = result.get("range")
    if range_part:
        out["BETA"] = range_part["beta_s_per_page_scan"]
        out["ETA"] = range_part["eta_s"]
        out["LAMBDA_RANGE"] = range_part["lambda_range_s_per_page"]
    return out


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, float, str, str]] = []
    point = payload.get("point")
    if point:
        rows.extend(
            [
                ("lambda_point", point["lambda_point_s_per_page"], "s/page", "point"),
                ("alpha", point["alpha_s_per_key"], "s/key", "point"),
                ("delta", point["delta_s"], "s", "point"),
            ]
        )
    range_part = payload.get("range")
    if range_part:
        rows.extend(
            [
                ("lambda_range", range_part["lambda_range_s_per_page"], "s/page", "range"),
                ("beta", range_part["beta_s_per_page_scan"], "s/page_scan", "range"),
                ("eta", range_part["eta_s"], "s", "range"),
            ]
        )

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value", "unit", "source"])
        writer.writerows(rows)


def write_env(path: Path, values: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Source this file before exp/run_join_workloads.sh.",
    ]
    for key in sorted(values):
        lines.append(f"export {key}={values[key]:.17g}")
    path.write_text("\n".join(lines) + "\n")


def print_summary(payload: dict[str, Any], env_values: dict[str, float]) -> None:
    point = payload.get("point")
    if point:
        print("=== Fitted point cost model ===")
        print(
            "lambda_point (s/page)  = "
            f"{point['lambda_point_s_per_page']:.17g} "
            f"= {point['lambda_point_us_per_page']:.6g} us/page"
        )
        print(f"alpha     (s/key)       = {point['alpha_s_per_key']:.17g}")
        print(f"delta     (s, intercept)= {point['delta_s']:.17g}")
    range_part = payload.get("range")
    if range_part:
        if point:
            print()
        print("=== Fitted range cost model ===")
        print(
            "lambda_range (s/page)  = "
            f"{range_part['lambda_range_s_per_page']:.17g} "
            f"= {range_part['lambda_range_us_per_page']:.6g} us/page"
        )
        print(f"beta  (s/page_scan)     = {range_part['beta_s_per_page_scan']:.17g}")
        print(f"eta   (s, fixed overhead)= {range_part['eta_s']:.17g}")
    if env_values:
        print()
        print("=== Environment values ===")
        for key in sorted(env_values):
            print(f"{key}={env_values[key]:.17g}")


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    result: dict[str, Any] = {
        "epsilon": args.epsilon,
        "dataset": args.dataset,
        "data_dir": str(data_dir),
        "mode": args.mode,
    }

    if args.mode in ("all", "point"):
        result["point"] = fit_point(data_dir, args.dataset, args.epsilon)
    if args.mode in ("all", "range"):
        result["range"] = fit_range(data_dir, args.dataset, args.epsilon)

    env_values = partition_env(result)
    result["partition_env"] = env_values

    output_json = Path(args.output_json) if args.output_json else default_output_path(data_dir, args.dataset, "json")
    output_csv = Path(args.output_csv) if args.output_csv else default_output_path(data_dir, args.dataset, "csv")
    output_env = Path(args.output_env) if args.output_env else default_output_path(data_dir, args.dataset, "env")

    write_json(output_json, result)
    write_csv(output_csv, result)
    write_env(output_env, env_values)

    print_summary(result, env_values)
    print()
    print(f"[fit] json -> {output_json}")
    print(f"[fit] csv  -> {output_csv}")
    print(f"[fit] env  -> {output_env}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
