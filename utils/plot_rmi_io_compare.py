#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times", "Computer Modern Roman"],
        "axes.unicode_minus": False,
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)

XLABEL_FONTSIZE = 30
YLABEL_FONTSIZE = 30
TICK_FONTSIZE = 25
LEGEND_FONTSIZE = 20
TICK_FONT = {"labelsize": TICK_FONTSIZE}
ERROR_MARKER_SIZE = 9

MIN_BRANCH_FACTOR = 512
POLICIES = ("FIFO", "LRU", "LFU")
POLICY_COLORS = {
    "FIFO": "tab:blue",
    "LRU": "tab:orange",
    "LFU": "tab:green",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot estimated vs actual total IOs for RMI branch factors."
    )
    parser.add_argument(
        "--estimate-log",
        type=Path,
        required=True,
        help="optimalBF summary log with branch_factor/policy columns.",
    )
    parser.add_argument(
        "--bench-csv",
        type=Path,
        required=True,
        help="rmi_bench CSV with actual experiment rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output figure path.",
    )
    parser.add_argument(
        "--legend-output",
        type=Path,
        default=None,
        help="Optional separate legend figure path. Default: <output-stem>_legend<output-suffix>.",
    )
    parser.add_argument(
        "--actual-column",
        type=str,
        default="logical_ios",
        choices=["logical_ios", "physical_ios", "cache_misses"],
        help="Which actual IO metric to compare against.",
    )
    return parser.parse_args()


def default_legend_output_path(output: Path) -> Path:
    suffix = output.suffix or ".pdf"
    return output.with_name(f"{output.stem}_legend{suffix}")


def apply_scientific_y(axis, *, offset_side: str) -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    axis.yaxis.set_major_formatter(formatter)
    axis.yaxis.set_offset_position(offset_side)


def place_y_offset_text(axis, *, x: float, ha: str) -> None:
    offset_text = axis.yaxis.get_offset_text()
    offset_text.set_fontsize(TICK_FONTSIZE)
    offset_text.set_x(x)
    offset_text.set_y(1.01)
    offset_text.set_ha(ha)
    offset_text.set_va("bottom")


def save_legend_figure(handles: list, labels: list[str], output: Path) -> None:
    if not handles or not labels:
        return
    fig = plt.figure(figsize=(min(16.0, max(7.0, 1.35 * len(labels))), 1.35))
    legend = fig.legend(
        handles,
        labels,
        loc="center",
        ncol=min(3, len(labels)),
        frameon=False,
    )
    for text in legend.get_texts():
        text.set_fontsize(LEGEND_FONTSIZE)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_estimate_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"branch_factor", "policy"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Unexpected columns in {path}; need at least {sorted(required)}"
        )

    out = df.copy()
    out["branch_factor"] = pd.to_numeric(out["branch_factor"], errors="coerce")
    out["policy"] = out["policy"].astype(str).str.upper()

    if "estimated_total_ios" in out.columns:
        out["estimated_total_ios"] = pd.to_numeric(
            out["estimated_total_ios"], errors="coerce"
        )
    elif {"cost"}.issubset(out.columns):
        raise ValueError(
            f"{path} is missing estimated_total_ios. Regenerate the summary log with the updated optimalBF.py."
        )
    else:
        raise ValueError(
            f"Unexpected columns in {path}; need estimated_total_ios."
        )

    out = out.dropna(subset=["branch_factor", "policy", "estimated_total_ios"])
    out["branch_factor"] = out["branch_factor"].astype(int)
    return out.loc[:, ["branch_factor", "policy", "estimated_total_ios"]]


def load_bench_csv(path: Path, actual_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"branch_factor", "policy", actual_column}
    if not required.issubset(df.columns):
        raise ValueError(f"Unexpected columns in {path}; need {sorted(required)}")

    out = df.loc[:, ["branch_factor", "policy", actual_column]].copy()
    out["branch_factor"] = pd.to_numeric(out["branch_factor"], errors="coerce")
    out["policy"] = out["policy"].astype(str).str.upper()
    out["actual_total_ios"] = pd.to_numeric(out[actual_column], errors="coerce")
    out = out.dropna(subset=["branch_factor", "policy", "actual_total_ios"])
    out["branch_factor"] = out["branch_factor"].astype(int)
    return out.loc[:, ["branch_factor", "policy", "actual_total_ios"]]


def plot_compare(
    estimate_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    output: Path,
    legend_output: Path,
) -> None:
    estimate_df = estimate_df[estimate_df["branch_factor"] >= MIN_BRANCH_FACTOR].copy()
    bench_df = bench_df[bench_df["branch_factor"] >= MIN_BRANCH_FACTOR].copy()
    key_columns = ["branch_factor", "policy"]
    key_diff = estimate_df[key_columns].merge(
        bench_df[key_columns],
        how="outer",
        on=key_columns,
        indicator=True,
    )
    only_estimate = key_diff[key_diff["_merge"] == "left_only"]
    only_bench = key_diff[key_diff["_merge"] == "right_only"]
    if not only_estimate.empty:
        missing = (
            only_estimate.groupby("policy")["branch_factor"]
            .apply(lambda values: sorted(values.astype(int).tolist()))
            .to_dict()
        )
        print(f"[merge] estimate rows missing in bench CSV: {missing}", file=sys.stderr)
    if not only_bench.empty:
        missing = (
            only_bench.groupby("policy")["branch_factor"]
            .apply(lambda values: sorted(values.astype(int).tolist()))
            .to_dict()
        )
        print(f"[merge] bench rows missing in estimate log: {missing}", file=sys.stderr)

    merged = estimate_df.merge(
        bench_df, on=key_columns, how="inner"
    ).sort_values(["policy", "branch_factor"])

    if merged.empty:
        raise ValueError(
            "No overlapping branch_factor/policy rows >= 512 between estimate log and bench CSV."
        )

    merged["bf_exp"] = np.log2(merged["branch_factor"]).round().astype(int)
    merged["io_error"] = merged["estimated_total_ios"] - merged["actual_total_ios"]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax_right = ax.twinx()

    for policy in POLICIES:
        part = merged[merged["policy"] == policy]
        if part.empty:
            continue

        color = POLICY_COLORS[policy]
        ax.plot(
            part["bf_exp"],
            part["estimated_total_ios"],
            color=color,
            marker="o",
            linewidth=2,
            label=f"{policy} estimate",
        )
        ax.plot(
            part["bf_exp"],
            part["actual_total_ios"],
            color=color,
            marker="s",
            linestyle="--",
            linewidth=2,
            label=f"{policy} actual",
        )
        ax_right.plot(
            part["bf_exp"],
            part["io_error"],
            color=color,
            marker="x",
            linestyle="None",
            markersize=ERROR_MARKER_SIZE,
            markeredgewidth=2,
            label=f"{policy} error",
        )

    xticks = sorted(merged["bf_exp"].unique())
    ax.set_xticks(xticks)
    ax.set_xticklabels([fr"$2^{{{exp}}}$" for exp in xticks])
    ax.set_xlabel("Branch Factor", fontsize=XLABEL_FONTSIZE)
    ax.set_ylabel("Total IOs", fontsize=YLABEL_FONTSIZE)
    ax_right.set_ylabel("Estimate Error (Total IOs)", fontsize=YLABEL_FONTSIZE)
    ax_right.axhline(0.0, color="tab:gray", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    right_handles, right_labels = ax_right.get_legend_handles_labels()
    save_legend_figure(handles + right_handles, labels + right_labels, legend_output)
    apply_scientific_y(ax, offset_side="left")
    apply_scientific_y(ax_right, offset_side="right")
    ax.tick_params(axis="x", rotation=45, **TICK_FONT)
    ax.tick_params(axis="y", **TICK_FONT)
    ax_right.tick_params(axis="y", **TICK_FONT)
    fig.canvas.draw()
    place_y_offset_text(ax, x=0.0, ha="left")
    place_y_offset_text(ax_right, x=1.0, ha="right")
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    estimate_df = load_estimate_log(args.estimate_log.resolve())
    bench_df = load_bench_csv(args.bench_csv.resolve(), actual_column=args.actual_column)
    output = args.output.resolve()
    legend_output = (
        args.legend_output.resolve()
        if args.legend_output is not None
        else default_legend_output_path(output)
    )
    plot_compare(estimate_df, bench_df, output, legend_output)
    print(output)
    print(legend_output)


if __name__ == "__main__":
    main()
