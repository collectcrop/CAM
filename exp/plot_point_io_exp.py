#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

USE_TEX = os.environ.get("CAM_PLOT_USETEX", "0").lower() in {"1", "true", "yes", "on"}

plot_rc = {
    "text.usetex": USE_TEX,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Computer Modern Roman"],
    "axes.unicode_minus": False,
}
if USE_TEX:
    plot_rc["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.rcParams.update(plot_rc)

DATASET_ORDER = ("books","fb", "wiki", "osm")
WORKLOAD_ORDER = ("w1", "w2", "w3", "w4", "w5", "w6")
WORKLOAD_COLORS = {
    "w1": "tab:blue",
    "w2": "tab:orange",
    "w3": "tab:green",
    "w4": "tab:red",
    "w5": "tab:purple",
    "w6": "tab:brown",
}
DATASET_TITLES = {
    "books": "Books",
    "fb": "FB",
    "wiki": "Wiki",
    "osm": "OSM",
}

XLABEL_FONTSIZE = 24
YLABEL_FONTSIZE = 24
TITLE_FONTSIZE = 24
TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot point-query CAM estimation accuracy and estimate time."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("build/log/exp/summary/point_io_accuracy_summary.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("build/log/exp/figures"))
    parser.add_argument("--accuracy-output", type=Path, default=None)
    parser.add_argument("--time-output", type=Path, default=None)
    parser.add_argument("--legend-output", type=Path, default=None)
    parser.add_argument("--formats", nargs="+", default=["pdf"])
    return parser.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"dataset_label", "workload", "M", "mean_accuracy", "total_estimate_time_s"}
    if not required.issubset(df.columns):
        raise ValueError(f"Unexpected summary CSV columns in {path}; need {sorted(required)}")
    df = df.copy()
    df["dataset_label"] = df["dataset_label"].astype(str)
    df["workload"] = df["workload"].astype(str)
    df["M"] = pd.to_numeric(df["M"], errors="coerce").astype(int)
    df["mean_accuracy"] = pd.to_numeric(df["mean_accuracy"], errors="coerce")
    df["total_estimate_time_s"] = pd.to_numeric(df["total_estimate_time_s"], errors="coerce")
    df = df.dropna(subset=["mean_accuracy", "total_estimate_time_s"])
    return df


def ordered_values(found: list[str], order: tuple[str, ...]) -> list[str]:
    out = [v for v in order if v in found]
    out.extend(v for v in sorted(found) if v not in out)
    return out


def bar_value(
    df: pd.DataFrame,
    dataset_label: str,
    memory_mib: int,
    workload: str,
    metric: str,
) -> float:
    part = df[
        (df["dataset_label"] == dataset_label)
        & (df["M"] == memory_mib)
        & (df["workload"] == workload)
    ]
    if part.empty:
        return np.nan
    if len(part) > 1:
        part = part.sort_values(["dataset_label", "M", "workload"])
    return float(part[metric].iloc[0])


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_stem: Path,
    formats: list[str],
    percent: bool = False,
) -> None:
    datasets = ordered_values(df["dataset_label"].dropna().unique().tolist(), DATASET_ORDER)
    workloads = ordered_values(df["workload"].dropna().unique().tolist(), WORKLOAD_ORDER)
    memories = sorted(df["M"].dropna().astype(int).unique().tolist())
    if not datasets or not workloads or not memories:
        raise ValueError("No data available to plot.")

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets), 4.2), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    x = np.arange(len(memories), dtype=float)
    bar_width = min(0.16, 0.82 / max(1, len(workloads)))

    for ax, dataset in zip(axes, datasets):
        for idx, workload in enumerate(workloads):
            offset = (idx - (len(workloads) - 1) / 2.0) * bar_width
            values = [
                bar_value(df, dataset_label=dataset, memory_mib=m, workload=workload, metric=metric)
                for m in memories
            ]
            ax.bar(
                x + offset,
                values,
                width=bar_width,
                color=WORKLOAD_COLORS.get(workload, "tab:gray"),
                edgecolor="black",
                linewidth=0.7,
                label=workload.upper(),
            )

        ax.set_title(DATASET_TITLES.get(dataset, dataset), fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("Memory Budget (MB)", fontsize=XLABEL_FONTSIZE)
        ax.set_xticks(x)
        ax.set_xticklabels([str(m) for m in memories], fontsize=TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        ax.grid(axis="y", alpha=0.3)
        ax.margins(x=0.04)
        if percent:
            ax.set_ylim(0.0, 1.0)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    axes[0].set_ylabel(ylabel, fontsize=YLABEL_FONTSIZE)
    fig.tight_layout(w_pad=1.0)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(path)
    plt.close(fig)


def save_legend(output_stem: Path, workloads: list[str], formats: list[str]) -> None:
    handles = [
        Patch(
            facecolor=WORKLOAD_COLORS.get(workload, "tab:gray"),
            edgecolor="black",
            linewidth=0.7,
            label=workload.upper(),
        )
        for workload in workloads
    ]
    fig = plt.figure(figsize=(8, 1.2))
    fig.legend(
        handles=handles,
        labels=[w.upper() for w in workloads],
        ncol=len(workloads),
        loc="center",
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_summary(args.summary_csv)
    formats = [fmt.lstrip(".") for fmt in args.formats]
    output_dir = args.output_dir.resolve()
    accuracy_stem = (args.accuracy_output or output_dir / "point_io_estimation_accuracy").resolve()
    time_stem = (args.time_output or output_dir / "point_io_estimation_time").resolve()
    legend_stem = (args.legend_output or output_dir / "point_io_workload_legend").resolve()

    plot_metric(
        df,
        metric="mean_accuracy",
        ylabel="Mean Accuracy",
        output_stem=accuracy_stem,
        formats=formats,
        percent=True,
    )
    plot_metric(
        df,
        metric="total_estimate_time_s",
        ylabel="Estimation Time (s)",
        output_stem=time_stem,
        formats=formats,
        percent=False,
    )
    workloads = ordered_values(df["workload"].dropna().unique().tolist(), WORKLOAD_ORDER)
    save_legend(legend_stem, workloads=workloads, formats=formats)


if __name__ == "__main__":
    main()
