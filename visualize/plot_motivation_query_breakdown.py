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

BASELINE_GROUPS = [
    ("buffered", ["PGM", "RMI"]),
    ("direct", ["PGM-DIRECT", "RMI-DIRECT"]),
]
COMPONENTS = [
    ("avg_lastmile_search_ns", "Last-mile search", "#72B7B2"),
    ("avg_index_traversal_ns", "Index traversal", "#F58518"),
    ("avg_cache_ns", "Cache", "#54A24B"),
    ("avg_io_ns", "I/O", "#4C78A8"),
]

DEFAULT_LABELS = ["w1", "w2", "w3", "w4", "w5", "w6"]
DEFAULT_UNIT = "us"
DEFAULT_FORMATS = ["pdf", "png"]

FIGURE_WIDTH_SINGLE = 3.3
FIGURE_WIDTH_PER_LABEL = 1.35
FIGURE_HEIGHT = 3.6
LEGEND_WIDTH = 6.4
LEGEND_HEIGHT = 0.8
WORKLOAD_STEP = 0.72
GROUP_WIDTH = 0.58
BAR_WIDTH_SCALE = 0.88
SPLIT_FIGURE_WIDTH_PER_LABEL = 1
SPLIT_WORKLOAD_STEP = 0.72
SPLIT_GROUP_WIDTH = 0.58
SPLIT_BAR_WIDTH_SCALE = 0.80
BAR_EDGE_WIDTH = 0.45
GRID_ALPHA = 0.22
OUTPUT_DPI = 220
Y_HEADROOM = 1.18

AXIS_LABEL_FONTSIZE = 25
TICK_FONTSIZE = 20
BASELINE_LABEL_FONTSIZE = 15
LEGEND_FONTSIZE = 15
BASELINE_LABEL_ROTATION = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot stacked query-breakdown bars for motivation experiments."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("build/log/motivation/osm_cellids_200M_query_breakdown.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/outputs/figures/motivation/osm_cellids_200M_query_breakdown_stacked"),
        help="Output stem without extension; _buffered/_legend suffixes are added.",
    )
    parser.add_argument("--formats", nargs="+", default=DEFAULT_FORMATS)
    parser.add_argument(
        "--unit",
        choices=["ns", "us"],
        default=DEFAULT_UNIT,
        help="Y-axis time unit for per-query breakdown.",
    )
    parser.add_argument(
        "--workload",
        default=None,
        help="Optional single workload to plot, e.g. w4 or table4.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional workload labels to plot; overrides --workload.",
    )
    parser.add_argument(
        "--split-baselines",
        action="store_true",
        help="Plot PGM and RMI as separate stacked figures instead of grouped bars.",
    )
    return parser.parse_args()


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "label",
        "baseline",
        "avg_index_traversal_ns",
        "avg_io_ns",
        "avg_lastmile_search_ns",
        "avg_wall_ns",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    df = df.copy()
    if "avg_cache_ns" not in df.columns:
        df["avg_cache_ns"] = 0.0
    numeric_cols = [name for name, _, _ in COMPONENTS] + ["avg_wall_ns"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["label"] = df["label"].astype(str)
    df["baseline"] = df["baseline"].astype(str)
    return df.dropna(subset=[name for name, _, _ in COMPONENTS])


def normalize_workload_label(value: str) -> str:
    value = str(value).strip()
    lower = value.lower()
    if lower.startswith("w") and lower[1:].isdigit():
        return f"table{lower[1:]}"
    return value


def display_workload_label(label: str) -> str:
    if label.startswith("table") and label[5:].isdigit():
        return f"w{label[5:]}"
    return label


def ordered_labels(labels: list[str]) -> list[str]:
    def key(label: str) -> tuple[int, str]:
        if label.startswith("table") and label[5:].isdigit():
            return (int(label[5:]), label)
        return (10**9, label)

    return sorted(labels, key=key)


def pick_row(df: pd.DataFrame, label: str, baseline: str) -> pd.Series | None:
    part = df[(df["label"] == label) & (df["baseline"] == baseline)]
    if part.empty:
        return None
    if len(part) > 1:
        sort_cols = ["index_bytes", "model"] if "model" in part.columns else ["baseline"]
        part = part.sort_values(sort_cols)
    return part.iloc[0]


def y_scale(unit: str) -> tuple[float, str]:
    if unit == "ns":
        return 1.0, "Latency (ns)"
    return 1000.0, r"Latency ($\mathrm{\mu s}$)"


def output_stem(base_output: Path, suffix: str) -> Path:
    return base_output.with_name(f"{base_output.name}_{suffix}")


def baseline_suffix(baseline: str) -> str:
    return baseline.lower().replace("-", "_")


def save_legend(output: Path, formats: list[str]) -> None:
    handles = [
        Patch(facecolor=color, edgecolor="black", linewidth=BAR_EDGE_WIDTH, label=name)
        for _col, name, color in COMPONENTS
    ]
    fig, ax = plt.subplots(figsize=(LEGEND_WIDTH, LEGEND_HEIGHT))
    ax.axis("off")
    ax.legend(
        handles=handles,
        labels=[name for _col, name, _color in COMPONENTS],
        ncol=len(COMPONENTS),
        loc="center",
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
    )
    stem = output_stem(output, "legend")
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = stem.with_suffix(f".{fmt.lstrip('.')}")
        fig.savefig(path, dpi=OUTPUT_DPI, bbox_inches="tight")
        print(path)
    plt.close(fig)


def plot_group(
    df: pd.DataFrame,
    output: Path,
    formats: list[str],
    unit: str,
    suffix: str,
    baselines: list[str],
    split_layout: bool = False,
    log_y: bool = False,
) -> None:
    labels = ordered_labels(df["label"].dropna().unique().tolist())
    if not labels:
        raise ValueError("no workload labels to plot")

    scale, ylabel = y_scale(unit)
    width_per_label = SPLIT_FIGURE_WIDTH_PER_LABEL if split_layout else FIGURE_WIDTH_PER_LABEL
    workload_step = SPLIT_WORKLOAD_STEP if split_layout else WORKLOAD_STEP
    group_width = SPLIT_GROUP_WIDTH if split_layout else GROUP_WIDTH
    bar_width_scale = SPLIT_BAR_WIDTH_SCALE if split_layout else BAR_WIDTH_SCALE
    fig_width = max(
        FIGURE_WIDTH_SINGLE,
        width_per_label * (1 + (len(labels) - 1) * workload_step) + 0.5,
    )
    fig, ax = plt.subplots(figsize=(fig_width, FIGURE_HEIGHT))

    bar_width = group_width / len(baselines)
    x = np.arange(len(labels), dtype=float) * workload_step
    ymax = 0.0
    positive_values: list[float] = []
    bar_labels: list[tuple[float, float, str]] = []

    for b_idx, baseline in enumerate(baselines):
        xpos = x - group_width / 2.0 + bar_width * (b_idx + 0.5)
        bottoms = np.zeros(len(labels), dtype=float)
        for col, _name, color in COMPONENTS:
            values = []
            for label in labels:
                row = pick_row(df, label, baseline)
                values.append(np.nan if row is None else float(row[col]) / scale)
            values_arr = np.asarray(values, dtype=float)
            positive_values.extend(
                values_arr[np.isfinite(values_arr) & (values_arr > 0)].tolist()
            )
            ax.bar(
                xpos,
                values_arr,
                width=bar_width * bar_width_scale,
                bottom=bottoms,
                color=color,
                edgecolor="black",
                linewidth=BAR_EDGE_WIDTH,
            )
            bottoms = np.where(np.isnan(values_arr), bottoms, bottoms + np.nan_to_num(values_arr))
        if len(bottoms):
            ymax = max(ymax, float(np.nanmax(bottoms)))
            for xi, height in zip(xpos, bottoms):
                if not split_layout and np.isfinite(height) and height > 0:
                    bar_labels.append((float(xi), float(height), baseline.replace("-DIRECT", "-D")))

    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlabel("Workload", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("")
    ax.set_xticks(x)
    ax.set_xticklabels([display_workload_label(label) for label in labels], fontsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.margins(x=0.08)
    if log_y:
        ax.set_yscale("log", base=10)
        if positive_values and ymax > 0:
            ymin = min(positive_values) * 0.5
            ymax_limit = ymax * Y_HEADROOM
            if ymin >= ymax_limit:
                ymin = ymax_limit / 10.0
            ax.set_ylim(max(ymin, np.nextafter(0.0, 1.0)), ymax_limit)
        else:
            ax.set_ylim(1.0, 10.0)
    else:
        ax.set_ylim(0.0, ymax * Y_HEADROOM if ymax > 0 else 1.0)

    label_offset = ymax * 0.025 if ymax > 0 else 0.02
    for xi, height, baseline_label in bar_labels:
        ax.text(
            xi,
            height + label_offset,
            baseline_label,
            ha="center",
            va="bottom",
            rotation=BASELINE_LABEL_ROTATION,
            fontsize=BASELINE_LABEL_FONTSIZE,
            clip_on=False,
        )

    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    stem = output_stem(output, suffix)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = stem.with_suffix(f".{fmt.lstrip('.')}")
        fig.savefig(path, dpi=OUTPUT_DPI, bbox_inches="tight")
        print(path)
    plt.close(fig)


def plot(
    df: pd.DataFrame,
    output: Path,
    formats: list[str],
    unit: str,
    split_baselines: bool,
) -> None:
    plotted = False
    for suffix, baselines in BASELINE_GROUPS:
        part = df[df["baseline"].isin(baselines)].copy()
        if part.empty:
            continue
        split_group = split_baselines or suffix == "direct"
        log_y = suffix == "direct"
        if split_group:
            for baseline in baselines:
                baseline_part = part[part["baseline"] == baseline].copy()
                if baseline_part.empty:
                    continue
                plot_group(
                    baseline_part,
                    output,
                    formats,
                    unit,
                    f"{suffix}_{baseline_suffix(baseline)}",
                    [baseline],
                    split_layout=True,
                    log_y=log_y,
                )
                plotted = True
        else:
            plot_group(part, output, formats, unit, suffix, baselines, log_y=log_y)
            plotted = True
    if not plotted:
        raise ValueError("no rows to plot for the requested baseline groups")
    save_legend(output, formats)


def main() -> None:
    args = parse_args()
    df = normalize_schema(pd.read_csv(args.summary_csv))
    if args.labels:
        requested_labels = args.labels
    elif args.workload:
        requested_labels = [args.workload]
    else:
        requested_labels = DEFAULT_LABELS
    keep = {normalize_workload_label(label) for label in requested_labels}
    df = df[df["label"].isin(keep)].copy()
    formats = [fmt.lstrip(".") for fmt in args.formats]
    plot(df, args.output, formats, args.unit, args.split_baselines)


if __name__ == "__main__":
    main()
