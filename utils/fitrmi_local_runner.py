#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ACTUAL_IO_COLUMN_CANDIDATES = [
    "avg_logical_ios",
    "avg_physical_ios",
    "logical_ios",
    "physical_ios",
    "cache_misses",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit an additive RMI residual model from real RMI benchmark CSVs "
            "and optimalBF estimate logs."
        )
    )
    parser.add_argument("--bench-dir", type=Path, required=True)
    parser.add_argument(
        "--bench-pattern",
        default="{dataset_tag}_M{M}_rmi_q30_bench.csv",
        help="Filename pattern under bench-dir. Supports {dataset_tag} and {M}.",
    )
    parser.add_argument("--estimate-dir", type=Path, required=True)
    parser.add_argument(
        "--estimate-pattern",
        default="{dataset_tag}_M{M}_rmi_q30_optimalBF_summary.log",
        help="Filename pattern under estimate-dir. Supports {dataset_tag} and {M}.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-tag", default="books_10M")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--strategy", default="all_in_once")
    parser.add_argument("--train-m", nargs="+", type=int, required=True)
    parser.add_argument(
        "--output-m",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional M values to emit comparison/revised outputs for. "
            "Defaults to --train-m."
        ),
    )
    parser.add_argument(
        "--actual-column",
        default="avg_logical_ios",
        help="Actual I/O column from rmi_bench CSV. Defaults to per-query logical I/O.",
    )
    parser.add_argument("--min-bf", type=int, default=None)
    parser.add_argument("--max-bf", type=int, default=None)
    parser.add_argument("--ridge-lambda", type=float, default=1e-6)
    parser.add_argument(
        "--comparison-csv-name",
        default="fitrmi_corrected_vs_real.csv",
        help="Filename for merged comparison output under output-dir.",
    )
    parser.add_argument(
        "--coef-name",
        default="fitrmi_coef.txt",
        help="Filename for fitted coefficient text output under output-dir.",
    )
    return parser.parse_args()


def build_residual_features_bf_reciprocal_M(
    branch_factor: np.ndarray,
    M_mib: np.ndarray,
) -> np.ndarray:
    bf = np.asarray(branch_factor, dtype=np.float64)
    M = np.asarray(M_mib, dtype=np.float64)
    if np.any(bf <= 0):
        raise ValueError("branch_factor must be positive for the 1/BF feature")
    return np.column_stack([bf, 1.0 / bf, M, np.ones_like(bf)])


def fit_ridge_with_standardization(
    X: np.ndarray,
    y: np.ndarray,
    ridge_lambda: float,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError("expected X with columns [BF, 1/BF, M, 1]")

    features = X[:, :3]
    intercept = X[:, 3:]
    mu = features.mean(axis=0)
    sigma = features.std(axis=0)
    sigma = np.where(sigma > 0.0, sigma, 1.0)
    Xs = np.column_stack([(features - mu) / sigma, intercept])

    if ridge_lambda > 0.0:
        penalty = np.eye(Xs.shape[1], dtype=np.float64)
        penalty[-1, -1] = 0.0
        coef_std = np.linalg.solve(
            Xs.T @ Xs + float(ridge_lambda) * penalty,
            Xs.T @ y,
        )
    else:
        coef_std, *_ = np.linalg.lstsq(Xs, y, rcond=None)

    coef = np.zeros_like(coef_std)
    coef[:3] = coef_std[:3] / sigma
    coef[3] = coef_std[3] - np.sum(coef_std[:3] * mu / sigma)
    return coef


def predict_residual(
    branch_factor: np.ndarray,
    M_mib: np.ndarray,
    coef: np.ndarray,
) -> np.ndarray:
    X = build_residual_features_bf_reciprocal_M(branch_factor, M_mib)
    return X @ np.asarray(coef, dtype=np.float64)


def resolve_pattern(root: Path, pattern: str, dataset_tag: str, m_value: int) -> Path:
    return root / pattern.format(dataset_tag=dataset_tag, M=m_value)


def filter_bf_range(
    df: pd.DataFrame,
    *,
    min_bf: int | None,
    max_bf: int | None,
) -> pd.DataFrame:
    out = df.copy()
    if min_bf is not None:
        out = out[out["branch_factor"] >= int(min_bf)].copy()
    if max_bf is not None:
        out = out[out["branch_factor"] <= int(max_bf)].copy()
    return out


def find_actual_column(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested
    for column in ACTUAL_IO_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(
        "Unable to locate an actual I/O column. Expected one of: "
        + ", ".join([requested] + ACTUAL_IO_COLUMN_CANDIDATES)
    )


def load_bench_part(
    path: Path,
    *,
    policy: str,
    strategy: str,
    actual_column: str,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"branch_factor", "policy"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")

    if "strategy" in df.columns:
        df = df[df["strategy"].astype(str) == strategy]
    df = df[df["policy"].astype(str).str.upper() == policy.upper()]
    actual_col = find_actual_column(df, actual_column)

    keep = ["branch_factor", "policy", actual_col]
    optional = [
        c
        for c in ["queries", "logical_ios", "physical_ios", "cache_misses"]
        if c in df.columns and c not in keep
    ]
    out = df.loc[:, keep + optional].copy()
    out["branch_factor"] = pd.to_numeric(out["branch_factor"], errors="coerce")
    out["policy"] = out["policy"].astype(str).str.upper()
    for column in optional:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    actual_values = pd.to_numeric(out[actual_col], errors="coerce")
    if actual_col in {"logical_ios", "physical_ios", "cache_misses"}:
        if "queries" not in out.columns:
            raise ValueError(f"{path} needs queries to normalize total column {actual_col}")
        queries = pd.to_numeric(out["queries"], errors="coerce")
        actual_values = actual_values / queries
    out["actual_avg_ios"] = actual_values

    agg: dict[str, str] = {"actual_avg_ios": "mean"}
    for column in optional:
        agg[column] = "mean"
    grouped = (
        out.dropna(subset=["branch_factor", "actual_avg_ios"])
        .groupby(["branch_factor", "policy"], as_index=False)
        .agg(agg)
    )
    grouped["branch_factor"] = grouped["branch_factor"].astype(int)
    return grouped


def load_estimate_part(path: Path, *, m_value: int, policy: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"branch_factor", "policy", "cost"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")

    out = df.copy()
    out["policy"] = out["policy"].astype(str).str.upper()
    out = out[out["policy"] == policy.upper()]
    if "M" in out.columns:
        out = out[pd.to_numeric(out["M"], errors="coerce") == float(m_value)]

    keep = ["branch_factor", "policy", "cost"]
    for column in ["ratio", "estimated_total_ios", "time"]:
        if column in out.columns:
            keep.append(column)

    out = out.loc[:, keep].copy()
    out["branch_factor"] = pd.to_numeric(out["branch_factor"], errors="coerce")
    out["estimated_cost"] = pd.to_numeric(out["cost"], errors="coerce")
    for column in ["ratio", "estimated_total_ios", "time"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.dropna(subset=["branch_factor", "estimated_cost"])
    out["branch_factor"] = out["branch_factor"].astype(int)
    return out.drop(columns=["cost"])


def load_filtered_frame(args: argparse.Namespace, m_values: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for m_value in m_values:
        bench_path = resolve_pattern(args.bench_dir, args.bench_pattern, args.dataset_tag, m_value)
        estimate_path = resolve_pattern(args.estimate_dir, args.estimate_pattern, args.dataset_tag, m_value)
        bench = load_bench_part(
            bench_path,
            policy=args.policy,
            strategy=args.strategy,
            actual_column=args.actual_column,
        )
        estimate = load_estimate_part(estimate_path, m_value=m_value, policy=args.policy)
        bench = filter_bf_range(bench, min_bf=args.min_bf, max_bf=args.max_bf)
        estimate = filter_bf_range(estimate, min_bf=args.min_bf, max_bf=args.max_bf)
        merged = bench.merge(estimate, on=["branch_factor", "policy"], how="inner")
        merged.insert(0, "M", int(m_value))
        frames.append(merged)

    if not frames:
        raise ValueError("no training files were loaded")
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise ValueError("no overlapping RMI rows after policy/strategy/BF filters")
    return df.sort_values(["M", "branch_factor"]).reset_index(drop=True)


def add_predictions(df: pd.DataFrame, coef: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["actual_residual"] = out["actual_avg_ios"] - out["estimated_cost"]
    out["predicted_residual"] = predict_residual(
        out["branch_factor"].to_numpy(dtype=np.float64),
        out["M"].to_numpy(dtype=np.float64),
        coef,
    )
    out["corrected_cost"] = out["estimated_cost"] + out["predicted_residual"]
    out["error_before"] = out["estimated_cost"] - out["actual_avg_ios"]
    out["error_after"] = out["corrected_cost"] - out["actual_avg_ios"]
    denom = np.maximum(np.abs(out["actual_avg_ios"].to_numpy(dtype=np.float64)), 1e-12)
    out["abs_rel_error_before"] = np.abs(out["error_before"]) / denom
    out["abs_rel_error_after"] = np.abs(out["error_after"]) / denom
    if "queries" in out.columns:
        out["corrected_total_ios"] = out["corrected_cost"] * out["queries"]
    return out


def metric_summary(df: pd.DataFrame) -> dict[str, float]:
    return {
        "rows": float(len(df)),
        "mae_before": float(np.mean(np.abs(df["error_before"]))),
        "mae_after": float(np.mean(np.abs(df["error_after"]))),
        "rmse_before": float(np.sqrt(np.mean(np.square(df["error_before"])))),
        "rmse_after": float(np.sqrt(np.mean(np.square(df["error_after"])))),
        "mean_abs_rel_before": float(np.mean(df["abs_rel_error_before"])),
        "mean_abs_rel_after": float(np.mean(df["abs_rel_error_after"])),
    }


def write_coef_file(
    path: Path,
    *,
    coef: np.ndarray,
    args: argparse.Namespace,
    train_metrics: dict[str, float],
    output_metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "model,residual = k1*BF + k2/BF + k3*M + k4",
        f"policy,{args.policy.upper()}",
        f"strategy,{args.strategy}",
        f"train_m,{' '.join(str(m) for m in args.train_m)}",
        f"output_m,{' '.join(str(m) for m in (args.output_m or args.train_m))}",
        f"min_bf,{'' if args.min_bf is None else int(args.min_bf)}",
        f"max_bf,{'' if args.max_bf is None else int(args.max_bf)}",
        f"ridge_lambda,{args.ridge_lambda:g}",
        "k1,k2,k3,k4",
        ",".join(f"{value:.10g}" for value in np.asarray(coef, dtype=np.float64)),
    ]
    for key, value in train_metrics.items():
        lines.append(f"train_{key},{value:.10g}")
    for key, value in output_metrics.items():
        lines.append(f"output_{key},{value:.10g}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def revise_estimate_log(
    input_path: Path,
    output_path: Path,
    *,
    coef: np.ndarray,
    policy: str,
    min_bf: int | None,
    max_bf: int | None,
) -> None:
    df = pd.read_csv(input_path)
    required = {"M", "branch_factor", "policy", "cost"}
    if not required.issubset(df.columns):
        raise ValueError(f"{input_path} must contain columns {sorted(required)}")

    out = df.copy()
    out["original_cost"] = pd.to_numeric(df["cost"], errors="coerce")
    if "estimated_total_ios" in out.columns:
        out["original_estimated_total_ios"] = pd.to_numeric(
            df["estimated_total_ios"], errors="coerce"
        )
    out["predicted_residual"] = np.nan

    mask = out["policy"].astype(str).str.upper() == policy.upper()
    branch_factor = pd.to_numeric(out["branch_factor"], errors="coerce")
    if min_bf is not None:
        mask &= branch_factor >= int(min_bf)
    if max_bf is not None:
        mask &= branch_factor <= int(max_bf)
    bf = pd.to_numeric(out.loc[mask, "branch_factor"], errors="coerce").to_numpy(dtype=np.float64)
    M = pd.to_numeric(out.loc[mask, "M"], errors="coerce").to_numpy(dtype=np.float64)
    cost = pd.to_numeric(out.loc[mask, "cost"], errors="coerce").to_numpy(dtype=np.float64)

    residual = predict_residual(bf, M, coef)
    corrected = cost + residual
    out.loc[mask, "predicted_residual"] = residual
    out.loc[mask, "cost"] = corrected

    if "estimated_total_ios" in out.columns:
        original_total = pd.to_numeric(
            out.loc[mask, "estimated_total_ios"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        queries = np.divide(
            original_total,
            cost,
            out=np.zeros_like(original_total, dtype=np.float64),
            where=np.abs(cost) > 1e-15,
        )
        out.loc[mask, "estimated_total_ios"] = corrected * queries

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, float_format="%.10g")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_m_values = args.output_m or args.train_m
    training = load_filtered_frame(args, args.train_m)
    residual = training["actual_avg_ios"].to_numpy(dtype=np.float64) - training["estimated_cost"].to_numpy(dtype=np.float64)
    X = build_residual_features_bf_reciprocal_M(
        training["branch_factor"].to_numpy(dtype=np.float64),
        training["M"].to_numpy(dtype=np.float64),
    )
    coef = fit_ridge_with_standardization(X, residual, ridge_lambda=args.ridge_lambda)

    train_comparison = add_predictions(training, coef)
    output_frame = load_filtered_frame(args, output_m_values)
    comparison = add_predictions(output_frame, coef)
    train_metrics = metric_summary(train_comparison)
    output_metrics = metric_summary(comparison)

    coef_path = output_dir / args.coef_name
    write_coef_file(
        coef_path,
        coef=coef,
        args=args,
        train_metrics=train_metrics,
        output_metrics=output_metrics,
    )

    comparison_path = output_dir / args.comparison_csv_name
    comparison.to_csv(comparison_path, index=False, float_format="%.10g")

    revised_dir = output_dir / "revised"
    for m_value in output_m_values:
        estimate_path = resolve_pattern(args.estimate_dir, args.estimate_pattern, args.dataset_tag, m_value)
        revised_path = revised_dir / f"{estimate_path.stem}_{args.policy.upper()}_fitrmi_revised.csv"
        revise_estimate_log(
            estimate_path.resolve(),
            revised_path,
            coef=coef,
            policy=args.policy,
            min_bf=args.min_bf,
            max_bf=args.max_bf,
        )
        print(revised_path)

    print(f"[fitRMI] coef={np.asarray(coef)}")
    print(coef_path)
    print(comparison_path)


if __name__ == "__main__":
    main()
