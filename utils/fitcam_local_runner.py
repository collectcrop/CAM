#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fitCAM  # noqa: E402


IO_COLUMN_CANDIDATES = [
    "avg_IOs",
    "mean_cam_io",
    "avg_logical_ios",
    "actual_avg_logical_ios",
]
FIT_POLICY = "LRU"


def compatible_compute_model_costs_per_query(
    M_mib: float,
    eps_list: np.ndarray,
    *,
    n: int,
    seg_size: int,
    ipp: int,
    ps: int,
    type: str,
    data_file: str,
    query_file: str,
    fetch_strategy: str = "all_in_once",
    mode: str = "point",
    cold_start_correction: bool = False,
) -> np.ndarray:
    M_bytes = float(M_mib) * 1024.0 * 1024.0
    data_path = f"{fitCAM.base.DATASETS_DIRECTORY}{data_file}"
    query_path = f"{fitCAM.base.DATASETS_DIRECTORY}{query_file}"

    eps_arr = np.asarray(eps_list, dtype=np.float64)
    cost_hat = np.zeros_like(eps_arr, dtype=np.float64)

    for idx, eps in enumerate(eps_arr):
        if mode == "point":
            cost_value, _ = fitCAM.base.cost_function(
                eps,
                n,
                seg_size,
                M_bytes,
                ipp,
                ps,
                query_file=query_path,
                data_file=data_path,
                s=fetch_strategy,
                cache_policy=FIT_POLICY,
                cold_start_correction=cold_start_correction,
            )
        elif mode == "range":
            cost_value, _ = fitCAM.base.range_cost_function(
                eps,
                n,
                seg_size,
                M_bytes,
                ipp,
                ps,
                query_file=query_path,
                data_file=data_path,
                policy=FIT_POLICY,
                cold_start_correction=cold_start_correction,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        cost_hat[idx] = float(cost_value)

    return cost_hat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit fitCAM residual parameters from local CAM summary CSVs and "
            "write corrected estimated logs."
        )
    )
    parser.add_argument("--datasets-directory", type=Path, required=True)
    parser.add_argument("--real-summary-dir", type=Path, required=True)
    parser.add_argument("--real-summary-pattern", required=True)
    parser.add_argument("--estimate-log", type=Path, required=True)
    parser.add_argument(
        "--apply-estimate-log",
        type=Path,
        default=None,
        help="Optional extra estimated log to revise with the fitted coef.",
    )
    parser.add_argument(
        "--apply-output-log",
        type=Path,
        default=None,
        help="Output path for the revised version of --apply-estimate-log.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--policy", default="LRU")
    parser.add_argument("--strategy", default="all_in_once")
    parser.add_argument("--train-m", nargs="+", type=int, required=True)
    parser.add_argument("--holdout-m", type=int, default=None)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seg-size", type=int, default=16)
    parser.add_argument("--ipp", type=int, default=512)
    parser.add_argument("--ps", type=int, default=4096)
    parser.add_argument("--type", default="sample")
    parser.add_argument("--mode", choices=["point", "range"], default="point")
    parser.add_argument("--fetch-strategy", default="all_in_once")
    parser.add_argument("--max-eps", type=int, default=64)
    parser.add_argument("--eps0", type=float, default=0.0)
    parser.add_argument("--ridge-lambda", type=float, default=1e-6)
    parser.add_argument(
        "--cold-start-correction",
        action="store_true",
        help="Enable cold-start compulsory-miss correction in point-query CAM estimates.",
    )
    parser.add_argument(
        "--comparison-csv-name",
        default="fitcam_corrected_vs_real.csv",
        help="Filename for merged comparison output under output-dir.",
    )
    parser.add_argument(
        "--coef-name",
        default="fitcam_coef.txt",
        help="Filename for fitted coefficient text output under output-dir.",
    )
    return parser.parse_args()


def find_real_io_column(df: pd.DataFrame) -> str:
    for column in IO_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(
        "Unable to locate a real-I/O column. Expected one of: "
        + ", ".join(IO_COLUMN_CANDIDATES)
    )


def summary_path(summary_dir: Path, pattern: str, dataset_tag: str, m_value: int) -> Path:
    return summary_dir / pattern.format(dataset_tag=dataset_tag, M=m_value)


def normalize_summary_to_fitcam_real(
    input_path: Path,
    output_path: Path,
    *,
    policy: str,
    strategy: str,
    max_eps: int,
) -> Path:
    df = pd.read_csv(input_path)
    if "policy" in df.columns:
        df = df[df["policy"].astype(str).str.upper() == policy.upper()]
    if "strategy" in df.columns:
        df = df[df["strategy"].astype(str) == strategy]
    if "epsilon" not in df.columns:
        raise ValueError(f"{input_path} is missing epsilon")

    io_column = find_real_io_column(df)
    out = (
        df.loc[df["epsilon"] <= max_eps, ["epsilon", io_column]]
        .rename(columns={io_column: "avg_IOs"})
        .groupby("epsilon", as_index=False)["avg_IOs"]
        .mean()
        .sort_values("epsilon")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, float_format="%.10g")
    return output_path


def extract_policy_rows(
    summary_csv: Path,
    *,
    policy: str,
    strategy: str,
    max_eps: int,
) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    if "policy" in df.columns:
        df = df[df["policy"].astype(str).str.upper() == policy.upper()]
    if "strategy" in df.columns:
        df = df[df["strategy"].astype(str) == strategy]
    io_column = find_real_io_column(df)
    return (
        df.loc[df["epsilon"] <= max_eps, ["epsilon", io_column]]
        .rename(columns={io_column: "actual_avg_ios"})
        .groupby("epsilon", as_index=False)["actual_avg_ios"]
        .mean()
        .sort_values("epsilon")
    )


def build_comparison_frame(
    estimate_log: Path,
    revised_log: Path,
    real_summary_dir: Path,
    real_summary_pattern: str,
    *,
    dataset_tag: str,
    policy: str,
    strategy: str,
    max_eps: int,
    m_values: list[int],
) -> pd.DataFrame:
    est_df = pd.read_csv(estimate_log)
    rev_df = pd.read_csv(revised_log).rename(
        columns={
            "cost": "corrected_cost",
            "ratio": "predicted_residual",
        }
    )
    est_df = est_df.rename(columns={"cost": "estimated_cost"})
    merged_frames: list[pd.DataFrame] = []

    for m_value in m_values:
        real_summary = summary_path(real_summary_dir, real_summary_pattern, dataset_tag, m_value)
        real_df = extract_policy_rows(
            real_summary,
            policy=policy,
            strategy=strategy,
            max_eps=max_eps,
        )
        est_part = est_df[(est_df["M"] == m_value) & (est_df["epsilon"] <= max_eps)][["M", "epsilon", "estimated_cost"]]
        rev_part = rev_df[(rev_df["M"] == m_value) & (rev_df["epsilon"] <= max_eps)][["M", "epsilon", "corrected_cost", "predicted_residual"]]
        part = real_df.merge(est_part, on="epsilon", how="inner").merge(rev_part, on=["M", "epsilon"], how="inner")
        part.insert(0, "policy", policy.upper())
        merged_frames.append(part)

    if not merged_frames:
        return pd.DataFrame(columns=["policy", "M", "epsilon", "actual_avg_ios", "estimated_cost", "corrected_cost", "predicted_residual"])
    return pd.concat(merged_frames, ignore_index=True)


def main() -> None:
    global FIT_POLICY
    args = parse_args()
    datasets_directory = args.datasets_directory.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    FIT_POLICY = args.policy.upper()
    fitCAM.base.DATASETS_DIRECTORY = str(datasets_directory) + "/"
    fitCAM.base.compute_model_costs_per_query = compatible_compute_model_costs_per_query

    normalized_real_dir = output_dir / "real_fitcam"
    real_csv_files: list[str] = []
    for m_value in args.train_m:
        in_path = summary_path(args.real_summary_dir.resolve(), args.real_summary_pattern, args.dataset_tag, m_value)
        out_path = normalized_real_dir / f"{args.dataset_tag}_M{m_value}_{args.policy.lower()}_fitcam_real.csv"
        normalized = normalize_summary_to_fitcam_real(
            in_path,
            out_path,
            policy=args.policy,
            strategy=args.strategy,
            max_eps=args.max_eps,
        )
        real_csv_files.append(str(normalized))

    coef = fitCAM.fit_residual_model_global_eps_reciprocal_M(
        Ms_mib=args.train_m,
        real_csv_files=real_csv_files,
        n=args.n,
        seg_size=args.seg_size,
        ipp=args.ipp,
        ps=args.ps,
        type=args.type,
        data_file=args.data_file,
        query_file=args.query_file,
        fetch_strategy=args.fetch_strategy,
        mode=args.mode,
        max_eps=args.max_eps,
        eps0=args.eps0,
        ridge_lambda=args.ridge_lambda,
        cold_start_correction=args.cold_start_correction,
        real_is_total=False,
        num_queries=0,
    )

    coef_path = output_dir / args.coef_name
    coef_path.write_text(
        "B,k1,k2,k3\n" + ",".join(f"{value:.10g}" for value in np.asarray(coef, dtype=np.float64)) + "\n",
        encoding="utf-8",
    )
    print(f"[fitCAM] coef={np.asarray(coef)}")
    print(coef_path)

    if args.holdout_m is not None:
        holdout_summary = summary_path(
            args.real_summary_dir.resolve(),
            args.real_summary_pattern,
            args.dataset_tag,
            args.holdout_m,
        )
        holdout_real_csv = normalized_real_dir / f"{args.dataset_tag}_M{args.holdout_m}_{args.policy.lower()}_fitcam_real.csv"
        normalize_summary_to_fitcam_real(
            holdout_summary,
            holdout_real_csv,
            policy=args.policy,
            strategy=args.strategy,
            max_eps=args.max_eps,
        )
        fitCAM.evaluate_holdout_memory_budget_eps_reciprocal_M(
            M_holdout_mib=args.holdout_m,
            real_csv_path=str(holdout_real_csv),
            coef=coef,
            n=args.n,
            seg_size=args.seg_size,
            ipp=args.ipp,
            ps=args.ps,
            type=args.type,
            data_file=args.data_file,
            query_file=args.query_file,
            fetch_strategy=args.fetch_strategy,
            mode=args.mode,
            max_eps=args.max_eps,
            eps0=args.eps0,
            cold_start_correction=args.cold_start_correction,
            real_is_total=False,
            num_queries=0,
        )

    revised_dir = output_dir / "revised"
    revised_dir.mkdir(parents=True, exist_ok=True)
    revised_log = revised_dir / f"{args.estimate_log.stem}_fitcam_revised.csv"
    fitCAM.apply_residual_correction_to_estimated_log_eps_reciprocal_M(
        input_log_path=str(args.estimate_log.resolve()),
        output_log_path=str(revised_log),
        coef=coef,
        eps0=args.eps0,
        residual_clip=None,
        estimated_is_total=False,
        num_queries=0,
        assume_M_unit="MiB",
        ratio_field="residual",
    )

    comparison_m_values = list(dict.fromkeys(args.train_m + ([args.holdout_m] if args.holdout_m is not None else [])))
    comparison_df = build_comparison_frame(
        estimate_log=args.estimate_log.resolve(),
        revised_log=revised_log,
        real_summary_dir=args.real_summary_dir.resolve(),
        real_summary_pattern=args.real_summary_pattern,
        dataset_tag=args.dataset_tag,
        policy=args.policy,
        strategy=args.strategy,
        max_eps=args.max_eps,
        m_values=comparison_m_values,
    )
    comparison_path = output_dir / args.comparison_csv_name
    comparison_df.to_csv(comparison_path, index=False, float_format="%.10g")

    if args.apply_estimate_log is not None:
        if args.apply_output_log is not None:
            applied_revised_log = args.apply_output_log.resolve()
        else:
            applied_revised_log = revised_dir / f"{args.apply_estimate_log.stem}_fitcam_revised.csv"
        applied_revised_log.parent.mkdir(parents=True, exist_ok=True)
        fitCAM.apply_residual_correction_to_estimated_log_eps_reciprocal_M(
            input_log_path=str(args.apply_estimate_log.resolve()),
            output_log_path=str(applied_revised_log),
            coef=coef,
            eps0=args.eps0,
            residual_clip=None,
            estimated_is_total=False,
            num_queries=0,
            assume_M_unit="MiB",
            ratio_field="residual",
        )
        print(applied_revised_log)

    print(revised_log)
    print(comparison_path)


if __name__ == "__main__":
    main()
