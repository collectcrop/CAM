#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

TMP_MPLCONFIGDIR = Path("/tmp/matplotlib")
TMP_XDG_CACHE_HOME = Path("/tmp/xdg-cache")
TMP_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
TMP_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_XDG_CACHE_HOME))

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover - runtime environment guard
    raise SystemExit(
        "This script requires matplotlib and pandas. "
        "Run it with ~/miniconda3/bin/python."
    ) from exc

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times", "Computer Modern Roman"],
        "axes.unicode_minus": False,
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)

TITLE_FONTSIZE = 25
XLABEL_FONTSIZE = 25
YLABEL_FONTSIZE = 25
TICK_FONTSIZE = 20
TICK_FONT = {"labelsize": TICK_FONTSIZE}
MARKER_SIZE_PRIMARY = 8
MARKER_SIZE_ERROR = 9
MIN_BRANCH_FACTOR = 64
CURVE_COLOR_MAP = {
    "real": "black",
    "estimate": "blue",
    "calibrated": "red",
    "error_before": "blue",
    "error_after": "red",
}
REQUIRED_COLUMNS = {
    "M",
    "branch_factor",
    "policy",
    "actual_avg_ios",
    "estimated_cost",
    "corrected_cost",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot RMI actual, estimated, and calibrated IO curves from "
            "fitrmi_local_runner.py comparison CSVs."
        )
    )
    parser.add_argument(
        "--comparison-csv",
        nargs="+",
        type=Path,
        required=True,
        help=(
            "One or more *_fitrmi_corrected_vs_real.csv files. "
            "Each file may contain one or more policies."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/outputs/figures/rmi_fitrmi"),
        help="Directory for PDF figures, legends, and merged plotting CSVs.",
    )
    parser.add_argument(
        "--dataset-tag",
        default="books_10M",
        help="Prefix used in output filenames.",
    )
    parser.add_argument(
        "--policies",
        nargs="*",
        default=None,
        help="Optional policy allowlist, for example: FIFO LRU LFU.",
    )
    parser.add_argument(
        "--m-values",
        nargs="*",
        type=int,
        default=None,
        help="Optional M allowlist, for example: 16 32 64.",
    )
    parser.add_argument(
        "--min-bf",
        type=int,
        default=MIN_BRANCH_FACTOR,
        help="Minimum branch factor to plot.",
    )
    parser.add_argument(
        "--max-bf",
        type=int,
        default=None,
        help="Maximum branch factor to plot, for example 524288 for 2^19.",
    )
    parser.add_argument(
        "--per-query",
        action="store_true",
        help="Plot average IO per query instead of total IO.",
    )
    parser.add_argument(
        "--no-error-axis",
        action="store_true",
        help="Only draw actual/estimate/calibrated curves; omit right-axis error markers.",
    )
    return parser.parse_args()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(column).strip() for column in out.columns]
    return out


def load_comparison_csvs(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = normalize_columns(pd.read_csv(path))
        if not REQUIRED_COLUMNS.issubset(df.columns):
            raise ValueError(f"{path} must contain columns {sorted(REQUIRED_COLUMNS)}")

        out = df.copy()
        out["source"] = str(path.resolve())
        out["policy"] = out["policy"].astype(str).str.upper()
        for column in ["M", "branch_factor", "actual_avg_ios", "estimated_cost", "corrected_cost"]:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        if "queries" in out.columns:
            out["queries"] = pd.to_numeric(out["queries"], errors="coerce")
        else:
            out["queries"] = np.nan

        out = out.dropna(
            subset=["M", "branch_factor", "actual_avg_ios", "estimated_cost", "corrected_cost"]
        ).copy()
        out["M"] = out["M"].astype(int)
        out["branch_factor"] = out["branch_factor"].astype(int)
        frames.append(out)

    if not frames:
        raise ValueError("No comparison rows were loaded.")
    return pd.concat(frames, ignore_index=True)


def prepare_plot_frame(
    df: pd.DataFrame,
    *,
    policies: set[str] | None,
    m_values: set[int] | None,
    min_bf: int,
    max_bf: int | None,
    per_query: bool,
) -> pd.DataFrame:
    out = df.copy()
    if policies is not None:
        out = out[out["policy"].isin(policies)].copy()
    if m_values is not None:
        out = out[out["M"].isin(m_values)].copy()
    out = out[out["branch_factor"] >= int(min_bf)].copy()
    if max_bf is not None:
        out = out[out["branch_factor"] <= int(max_bf)].copy()
    if out.empty:
        raise ValueError("No rows remain after policy/M/branch-factor filters.")

    if per_query:
        out["actual_ios"] = out["actual_avg_ios"]
        out["estimated_ios"] = out["estimated_cost"]
        out["corrected_ios"] = out["corrected_cost"]
        out["io_unit"] = "Avg I/Os per Query"
        out["error_unit"] = "Error (Avg I/Os per Query)"
    else:
        if out["queries"].isna().any():
            missing = out[out["queries"].isna()][["policy", "M", "branch_factor", "source"]]
            raise ValueError(
                "Total-IO plots require a queries column in the comparison CSV. "
                f"Missing examples: {missing.head(5).to_dict('records')}"
            )
        if "actual_total_ios" in out.columns:
            out["actual_ios"] = pd.to_numeric(out["actual_total_ios"], errors="coerce")
        elif "logical_ios" in out.columns:
            out["actual_ios"] = pd.to_numeric(out["logical_ios"], errors="coerce")
        else:
            out["actual_ios"] = out["actual_avg_ios"] * out["queries"]
        if "estimated_total_ios" in out.columns:
            out["estimated_ios"] = pd.to_numeric(out["estimated_total_ios"], errors="coerce")
        else:
            out["estimated_ios"] = out["estimated_cost"] * out["queries"]
        if "corrected_total_ios" in out.columns:
            out["corrected_ios"] = pd.to_numeric(out["corrected_total_ios"], errors="coerce")
        else:
            out["corrected_ios"] = out["corrected_cost"] * out["queries"]
        out["io_unit"] = "Total I/Os"
        out["error_unit"] = "Error (Total I/Os)"

    out["error_before"] = out["estimated_ios"] - out["actual_ios"]
    out["error_after"] = out["corrected_ios"] - out["actual_ios"]
    out = out.dropna(subset=["actual_ios", "estimated_ios", "corrected_ios"]).copy()
    if out.empty:
        raise ValueError("No complete actual/estimated/corrected rows remain.")
    return out.sort_values(["policy", "M", "branch_factor"]).reset_index(drop=True)


def subplot_layout(count: int) -> tuple[int, int]:
    if count <= 1:
        return 1, 1
    return int(np.ceil(count / 2)), 2


def make_axes(m_values: list[int], width_single: float = 6.0, height_single: float = 4.0):
    nrows, ncols = subplot_layout(len(m_values))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(width_single * ncols, height_single * nrows),
        squeeze=False,
    )
    axes_list = list(axes.flat)
    for axis in axes_list[len(m_values):]:
        axis.remove()
    return fig, axes_list, nrows, ncols


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_legend_figure(handles: list, labels: list[str], output_path: Path, ncol: int = 3) -> None:
    if not handles or not labels:
        return
    fig = plt.figure(figsize=(min(14.0, max(6.0, 1.2 * len(labels))), 1.2))
    legend = fig.legend(
        handles,
        labels,
        loc="center",
        ncol=min(ncol, len(labels)),
        frameon=False,
    )
    for text in legend.get_texts():
        text.set_fontsize(TICK_FONT["labelsize"])
    save_figure(fig, output_path)


def apply_tick_font(axis) -> None:
    axis.tick_params(axis="both", **TICK_FONT)
    tick_size = TICK_FONT["labelsize"]
    axis.xaxis.get_offset_text().set_fontsize(tick_size)
    axis.yaxis.get_offset_text().set_fontsize(tick_size)


def apply_scientific_y(axis) -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    axis.yaxis.set_major_formatter(formatter)


def apply_outer_labels(axis, idx: int, nrows: int, ncols: int, xlabel: str, ylabel: str) -> None:
    row = idx // ncols
    col = idx % ncols
    axis.set_xlabel(xlabel if row == nrows - 1 else "", fontsize=XLABEL_FONTSIZE)
    axis.set_ylabel(ylabel if col == 0 else "", fontsize=YLABEL_FONTSIZE)


def apply_outer_right_ylabel(axis_right, idx: int, ncols: int, ylabel: str) -> None:
    col = idx % ncols
    axis_right.set_ylabel(ylabel if col == ncols - 1 else "", fontsize=YLABEL_FONTSIZE)


def series_rows(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    return (
        df.dropna(subset=["branch_factor", value_column])
        .sort_values("branch_factor")
        .copy()
    )


def set_branch_factor_ticks(axis, part: pd.DataFrame) -> None:
    bf_values = sorted(part["branch_factor"].dropna().astype(int).unique().tolist())
    if not bf_values:
        return
    bf_exp = [int(round(np.log2(value))) for value in bf_values]
    axis.set_xticks(bf_exp)
    axis.set_xticklabels([fr"$2^{{{exp}}}$" for exp in bf_exp], rotation=45)


def add_bf_exp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bf_exp"] = np.log2(out["branch_factor"]).round().astype(int)
    return out


def plot_policy_curves(
    policy_df: pd.DataFrame,
    output_path: Path,
    legend_output_path: Path,
    *,
    show_error_axis: bool,
) -> None:
    policy_df = add_bf_exp(policy_df)
    m_values = sorted(policy_df["M"].unique().tolist())
    y_label = str(policy_df["io_unit"].iloc[0])
    err_label = str(policy_df["error_unit"].iloc[0])
    fig, axes, nrows, ncols = make_axes(m_values)
    legend_map: dict[str, object] = {}

    for idx, (axis, m_value) in enumerate(zip(axes, m_values)):
        axis_right = axis.twinx() if show_error_axis else None
        part = policy_df[policy_df["M"] == m_value].sort_values("branch_factor")
        if part.empty:
            continue

        actual_part = series_rows(part, "actual_ios")
        estimated_part = series_rows(part, "estimated_ios")
        calibrated_part = series_rows(part, "corrected_ios")

        if not actual_part.empty:
            actual_line, = axis.plot(
                actual_part["bf_exp"],
                actual_part["actual_ios"],
                color=CURVE_COLOR_MAP["real"],
                marker="o",
                linestyle="-",
                linewidth=2,
                markersize=MARKER_SIZE_PRIMARY,
                label="real",
            )
            legend_map.setdefault("real", actual_line)
        if not estimated_part.empty:
            estimate_line, = axis.plot(
                estimated_part["bf_exp"],
                estimated_part["estimated_ios"],
                color=CURVE_COLOR_MAP["estimate"],
                marker=None,
                linestyle="-",
                linewidth=2,
                solid_capstyle="round",
                solid_joinstyle="round",
                label="estimate",
            )
            legend_map.setdefault("estimate", estimate_line)
        if not calibrated_part.empty:
            calibrated_line, = axis.plot(
                calibrated_part["bf_exp"],
                calibrated_part["corrected_ios"],
                color=CURVE_COLOR_MAP["calibrated"],
                marker=None,
                linestyle="-",
                linewidth=2,
                solid_capstyle="round",
                solid_joinstyle="round",
                label="calibrated estimate",
            )
            legend_map.setdefault("calibrated estimate", calibrated_line)

        if axis_right is not None:
            err_before_part = series_rows(part, "error_before")
            err_after_part = series_rows(part, "error_after")
            if not err_before_part.empty:
                err_before_line, = axis_right.plot(
                    err_before_part["bf_exp"],
                    err_before_part["error_before"],
                    color=CURVE_COLOR_MAP["error_before"],
                    marker="x",
                    linestyle="None",
                    markersize=MARKER_SIZE_ERROR,
                    label="error before calibration",
                )
                legend_map.setdefault("error before calibration", err_before_line)
            if not err_after_part.empty:
                err_after_line, = axis_right.plot(
                    err_after_part["bf_exp"],
                    err_after_part["error_after"],
                    color=CURVE_COLOR_MAP["error_after"],
                    marker="x",
                    linestyle="None",
                    markersize=MARKER_SIZE_ERROR,
                    label="error after calibration",
                )
                legend_map.setdefault("error after calibration", err_after_line)
            axis_right.axhline(0.0, color="tab:gray", linestyle=":", linewidth=1.2, alpha=0.8)
            apply_outer_right_ylabel(axis_right, idx, ncols, err_label)
            apply_scientific_y(axis_right)
            apply_tick_font(axis_right)

        set_branch_factor_ticks(axis, part)
        axis.set_title(f"M = {m_value}MB", fontsize=TITLE_FONTSIZE)
        apply_outer_labels(axis, idx, nrows, ncols, "Branch Factor", y_label)
        axis.grid(alpha=0.3)
        apply_scientific_y(axis)
        apply_tick_font(axis)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_path)
    save_legend_figure(list(legend_map.values()), list(legend_map.keys()), legend_output_path, ncol=3)


def write_policy_outputs(
    df: pd.DataFrame,
    output_dir: Path,
    dataset_tag: str,
    *,
    show_error_axis: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: list[Path] = []
    for policy in sorted(df["policy"].dropna().astype(str).unique().tolist()):
        policy_df = df[df["policy"] == policy].copy()
        if policy_df.empty:
            continue
        policy_name = policy.lower()
        merged_path = output_dir / f"{dataset_tag}_{policy_name}_fitrmi_curve_metrics.csv"
        figure_path = output_dir / f"{dataset_tag}_{policy_name}_fitrmi_curve_vs_bf.pdf"
        legend_path = output_dir / f"{dataset_tag}_{policy_name}_fitrmi_curve_vs_bf_legend.pdf"
        policy_df.to_csv(merged_path, index=False, float_format="%.10g")
        plot_policy_curves(
            policy_df,
            figure_path,
            legend_path,
            show_error_axis=show_error_axis,
        )
        artifact_paths.extend([merged_path, figure_path, legend_path])
    return artifact_paths


def main() -> None:
    args = parse_args()
    policies = {policy.upper() for policy in args.policies} if args.policies else None
    m_values = set(args.m_values) if args.m_values else None

    raw_df = load_comparison_csvs([path.resolve() for path in args.comparison_csv])
    plot_df = prepare_plot_frame(
        raw_df,
        policies=policies,
        m_values=m_values,
        min_bf=args.min_bf,
        max_bf=args.max_bf,
        per_query=args.per_query,
    )
    artifacts = write_policy_outputs(
        plot_df,
        args.output_dir.resolve(),
        args.dataset_tag,
        show_error_axis=not args.no_error_axis,
    )
    for path in artifacts:
        print(path)


if __name__ == "__main__":
    main()
