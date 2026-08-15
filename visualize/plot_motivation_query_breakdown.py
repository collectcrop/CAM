#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

USE_TEX = os.environ.get("CAM_PLOT_USETEX", "0").lower() in {
    "1", "true", "yes", "on"
}
plot_rc = {
    "text.usetex": USE_TEX,
    "font.family": "serif",
    "font.serif": [
        "Times New Roman", "Times", "DejaVu Serif", "Computer Modern Roman"
    ],
    "axes.unicode_minus": False,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
if USE_TEX:
    plot_rc["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.rcParams.update(plot_rc)


@dataclass(frozen=True)
class Component:
    column: str
    label: str
    short_label: str
    color: str


COMPONENTS = [
    Component("avg_index_traversal_ns", "Index traversal", "Index", "#ECA82C"),
    Component("avg_cache_ns", "Cache", "Cache", "#59A14F"),
    Component("avg_io_ns", "I/O", "I/O", "#4E79A7"),
    Component("avg_lastmile_search_ns", "Last-mile search", "Search", "#76B7B2"),
    Component("avg_other_ns", "Other", "Other", "#BAB0AC"),
]
# Put the dominant I/O segment on the common zero baseline in the composition plot.
COMPOSITION_COMPONENTS = [
    COMPONENTS[2], COMPONENTS[1], COMPONENTS[3], COMPONENTS[0], COMPONENTS[4]
]
METHOD_STYLES = [("#4E79A7", ""), ("#F28E2B", "///")]
DEFAULT_LABELS = ["w1", "w2", "w3", "w4", "w5", "w6"]
DEFAULT_BASELINES = ["PGM-DIRECT", "RMI-DIRECT"]
DEFAULT_FORMATS = ["pdf", "png"]
PAPER_WIDTH = 7.15
COMPOSITION_FIGURE_SIZE = (PAPER_WIDTH, 2.65)
OUTPUT_DPI = 300
BAR_EDGE_WIDTH = 0.45
GRID_ALPHA = 0.22


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot an absolute grouped latency breakdown and a normalized composition "
            "breakdown for PGM and RMI."
        )
    )
    parser.add_argument(
        "--summary-csv", type=Path,
        default=Path("build/log/motivation/osm_cellids_200M_query_breakdown.csv"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/outputs/figures/motivation/osm_cellids_200M_query_breakdown"),
        help="Output stem; _absolute and _composition suffixes are added.",
    )
    parser.add_argument("--formats", nargs="+", default=DEFAULT_FORMATS)
    parser.add_argument(
        "--unit", choices=["ns", "us"], default="us",
        help="Time unit used in the absolute-latency figure.",
    )
    parser.add_argument(
        "--baselines", nargs=2, metavar=("FIRST", "SECOND"),
        default=DEFAULT_BASELINES,
        help="The two baseline rows to compare (default: PGM-DIRECT RMI-DIRECT).",
    )
    parser.add_argument(
        "--workload", default=None,
        help="Optional single workload to plot, e.g. w4 or table4.",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Optional workload labels to plot; overrides --workload.",
    )
    parser.add_argument(
        "--no-ratio-annotations", action="store_true",
        help="Do not show the second/first I/O ratio in absolute-panel titles.",
    )
    return parser.parse_args()


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "label", "baseline", "avg_index_traversal_ns", "avg_io_ns",
        "avg_lastmile_search_ns", "avg_wall_ns",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    df = df.copy()
    if "avg_cache_ns" not in df.columns:
        df["avg_cache_ns"] = 0.0
    numeric_candidates = {
        component.column for component in COMPONENTS
        if component.column != "avg_other_ns"
    }
    numeric_candidates.update({"avg_wall_ns", "avg_other_ns", "other_ns", "queries"})
    for column in numeric_candidates.intersection(df.columns):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    core_columns = [component.column for component in COMPONENTS[:-1]]
    required_numeric = core_columns + ["avg_wall_ns"]
    bad_rows = df[required_numeric].isna().any(axis=1)
    if bad_rows.any():
        raise ValueError(
            f"non-numeric required latency values in rows: {df.index[bad_rows].tolist()}"
        )

    if "avg_other_ns" not in df.columns:
        if {"other_ns", "queries"}.issubset(df.columns):
            invalid_queries = df["queries"].isna() | (df["queries"] <= 0)
            if invalid_queries.any():
                raise ValueError(
                    f"invalid query counts in rows: {df.index[invalid_queries].tolist()}"
                )
            df["avg_other_ns"] = df["other_ns"] / df["queries"]
        else:
            # Fallback for older logs that retain only the wall-clock total.
            df["avg_other_ns"] = df["avg_wall_ns"] - df[core_columns].sum(axis=1)

    negative_other = df["avg_other_ns"] < -1e-6
    if negative_other.any():
        raise ValueError(
            f"negative residual/other latency in rows: {df.index[negative_other].tolist()}"
        )
    df["avg_other_ns"] = df["avg_other_ns"].clip(lower=0.0)

    component_columns = [component.column for component in COMPONENTS]
    component_sum = df[component_columns].sum(axis=1)
    tolerance = np.maximum(1.0, df["avg_wall_ns"].abs() * 1e-5)
    mismatch = (component_sum - df["avg_wall_ns"]).abs() > tolerance
    if mismatch.any():
        details = [
            {
                "row": int(index),
                "component_sum_ns": float(component_sum.loc[index]),
                "avg_wall_ns": float(df.loc[index, "avg_wall_ns"]),
            }
            for index in df.index[mismatch]
        ]
        raise ValueError(f"component latencies do not sum to avg_wall_ns: {details}")

    df["label"] = df["label"].astype(str)
    df["baseline"] = df["baseline"].astype(str)
    return df


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


def display_baseline_name(baseline: str) -> str:
    return baseline.removesuffix("-DIRECT")


def prepare_rows(
    df: pd.DataFrame, labels: list[str], baselines: list[str]
) -> pd.DataFrame:
    selected = df[
        df["label"].isin(labels) & df["baseline"].isin(baselines)
    ].copy()
    counts = selected.groupby(["label", "baseline"], sort=False).size()
    duplicates = counts[counts > 1]
    if not duplicates.empty:
        raise ValueError(
            "multiple rows found for workload/baseline pairs; filter the CSV first: "
            f"{duplicates.to_dict()}"
        )
    present = set(zip(selected["label"], selected["baseline"]))
    missing = [
        (display_workload_label(label), baseline)
        for label in labels for baseline in baselines
        if (label, baseline) not in present
    ]
    if missing:
        raise ValueError(f"missing workload/baseline rows: {missing}")
    return selected.set_index(["label", "baseline"]).sort_index()


def y_scale(unit: str) -> tuple[float, str]:
    if unit == "ns":
        return 1.0, "Latency per query (ns, log scale)"
    return 1000.0, r"Latency per query ($\mathrm{\mu s}$, log scale)"


def save_figure(fig: plt.Figure, stem: Path, formats: list[str]) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = stem.with_suffix(f".{fmt.lstrip('.')}")
        fig.savefig(path, dpi=OUTPUT_DPI, bbox_inches="tight", pad_inches=0.03)
        print(path)


def plot_absolute(
    rows: pd.DataFrame,
    labels: list[str],
    baselines: list[str],
    output: Path,
    formats: list[str],
    unit: str,
    annotate_ratios: bool,
) -> None:
    scale, ylabel = y_scale(unit)
    categories = COMPONENTS + [
        Component("avg_wall_ns", "Overall", "Overall", "#000000")
    ]
    all_values = np.asarray([
        float(rows.loc[(label, baseline), component.column]) / scale
        for label in labels
        for baseline in baselines
        for component in categories
    ])
    positive = all_values[np.isfinite(all_values) & (all_values > 0)]
    if not len(positive):
        raise ValueError("absolute plot has no positive latency values")
    lower = 10 ** np.floor(np.log10(float(positive.min())))
    upper = 10 ** np.ceil(np.log10(float(positive.max())))
    if lower >= upper:
        upper = lower * 10.0

    ncols = min(3, len(labels))
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(PAPER_WIDTH, 2.15 * nrows + 0.15),
        sharex=True, sharey=True, squeeze=False,
    )
    axes_flat = axes.ravel()
    x = np.arange(len(categories), dtype=float)
    group_width = 0.76
    bar_width = group_width / len(baselines)

    for panel_index, label in enumerate(labels):
        ax = axes_flat[panel_index]
        for baseline_index, baseline in enumerate(baselines):
            values = np.asarray([
                float(rows.loc[(label, baseline), component.column]) / scale
                for component in categories
            ])
            xpos = x - group_width / 2.0 + bar_width * (baseline_index + 0.5)
            color, hatch = METHOD_STYLES[baseline_index % len(METHOD_STYLES)]
            # Every bar starts at the same visible bound and ends at its absolute
            # value. This preserves direct comparison on a log axis; stacked log
            # segments do not have that property.
            ax.bar(
                xpos, values - lower, bottom=lower,
                width=bar_width * 0.88, color=color, hatch=hatch,
                edgecolor="black", linewidth=BAR_EDGE_WIDTH,
                label=display_baseline_name(baseline), zorder=3,
            )

        title = display_workload_label(label)
        if annotate_ratios:
            first_io = float(rows.loc[(label, baselines[0]), "avg_io_ns"])
            second_io = float(rows.loc[(label, baselines[1]), "avg_io_ns"])
            ratio = second_io / first_io if first_io > 0 else np.nan
            ratio_name = (
                f"{display_baseline_name(baselines[1])}/"
                f"{display_baseline_name(baselines[0])}"
            )
            title += f"  ({ratio_name} I/O: {ratio:.2f}$\\times$)"
        ax.set_title(title, fontsize=9.0, pad=3.0)
        ax.set_yscale("log", base=10)
        ax.set_ylim(lower, upper)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [component.short_label for component in categories],
            rotation=24, ha="right", rotation_mode="anchor", fontsize=7.2,
        )
        ax.tick_params(axis="y", which="both", labelsize=7.5, width=0.6)
        ax.tick_params(axis="x", width=0.6, pad=1.5)
        ax.grid(axis="y", which="major", alpha=GRID_ALPHA, linewidth=0.6, zorder=0)
        ax.grid(axis="y", which="minor", alpha=0.08, linewidth=0.4, zorder=0)
        ax.margins(x=0.03)

    for ax in axes_flat[len(labels):]:
        ax.set_visible(False)

    fig.supylabel(ylabel, fontsize=9.5, x=0.006)
    handles = [
        Patch(
            facecolor=METHOD_STYLES[index][0],
            hatch=METHOD_STYLES[index][1], edgecolor="black",
            linewidth=BAR_EDGE_WIDTH, label=display_baseline_name(baseline),
        )
        for index, baseline in enumerate(baselines)
    ]
    fig.legend(
        handles=handles, ncol=len(handles), loc="upper center",
        bbox_to_anchor=(0.5, 1.005), frameon=False, fontsize=8.5,
        handlelength=1.8, columnspacing=1.6,
    )
    fig.subplots_adjust(
        left=0.075, right=0.995, bottom=0.09, top=0.91,
        wspace=0.12, hspace=0.42,
    )
    save_figure(fig, output.with_name(f"{output.name}_absolute"), formats)
    plt.close(fig)


def text_color_for_component(component: Component) -> str:
    if component.column in {"avg_io_ns", "avg_cache_ns"}:
        return "white"
    return "black"


def plot_composition(
    rows: pd.DataFrame,
    labels: list[str],
    baselines: list[str],
    output: Path,
    formats: list[str],
) -> None:
    fig, axes = plt.subplots(
        1, len(baselines), figsize=COMPOSITION_FIGURE_SIZE,
        sharex=True, sharey=True, squeeze=False,
    )
    x = np.arange(len(labels), dtype=float)

    for baseline_index, baseline in enumerate(baselines):
        ax = axes[0, baseline_index]
        totals = np.asarray([
            float(rows.loc[(label, baseline), "avg_wall_ns"]) for label in labels
        ])
        bottoms = np.zeros(len(labels), dtype=float)
        for component in COMPOSITION_COMPONENTS:
            values = np.asarray([
                float(rows.loc[(label, baseline), component.column])
                for label in labels
            ])
            percentages = 100.0 * values / totals
            bars = ax.bar(
                x, percentages, bottom=bottoms, width=0.72,
                color=component.color, edgecolor="black",
                linewidth=BAR_EDGE_WIDTH, label=component.label, zorder=3,
            )
            for bar, percentage in zip(bars, percentages):
                if percentage >= 7.0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_y() + bar.get_height() / 2.0,
                        f"{percentage:.0f}%", ha="center", va="center",
                        color=text_color_for_component(component),
                        fontsize=6.8, fontweight="semibold", clip_on=True,
                    )
            bottoms += percentages

        ax.set_title(display_baseline_name(baseline), fontsize=10.0, pad=3.0)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [display_workload_label(label) for label in labels], fontsize=8.0
        )
        ax.set_xlabel("Workload", fontsize=9.0)
        ax.set_ylim(0.0, 100.0)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="y", labelsize=8.0, width=0.6)
        ax.tick_params(axis="x", width=0.6)
        ax.grid(axis="y", alpha=GRID_ALPHA, linewidth=0.6, zorder=0)
        ax.margins(x=0.04)

    axes[0, 0].set_ylabel("Share of per-query latency (%)", fontsize=9.5)
    handles = [
        Patch(
            facecolor=component.color, edgecolor="black",
            linewidth=BAR_EDGE_WIDTH, label=component.label,
        )
        for component in COMPOSITION_COMPONENTS
    ]
    fig.legend(
        handles=handles, ncol=len(handles), loc="lower center",
        bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=8.0,
        handlelength=1.5, columnspacing=1.1,
    )
    fig.subplots_adjust(
        left=0.075, right=0.995, bottom=0.25, top=0.89, wspace=0.08
    )
    save_figure(fig, output.with_name(f"{output.name}_composition"), formats)
    plt.close(fig)


def print_numeric_summary(
    rows: pd.DataFrame,
    labels: list[str],
    baselines: list[str],
    unit: str,
) -> None:
    scale, _ylabel = y_scale(unit)
    print(
        f"I/O and overall latency per query ({unit}); "
        f"ratio={display_baseline_name(baselines[1])}/"
        f"{display_baseline_name(baselines[0])}"
    )
    print(
        "workload,first_io,second_io,io_ratio,"
        "first_overall,second_overall,overall_ratio"
    )
    for label in labels:
        first_io = float(rows.loc[(label, baselines[0]), "avg_io_ns"]) / scale
        second_io = float(rows.loc[(label, baselines[1]), "avg_io_ns"]) / scale
        first_wall = float(rows.loc[(label, baselines[0]), "avg_wall_ns"]) / scale
        second_wall = float(rows.loc[(label, baselines[1]), "avg_wall_ns"]) / scale
        print(
            f"{display_workload_label(label)},{first_io:.3f},{second_io:.3f},"
            f"{second_io / first_io:.3f},{first_wall:.3f},{second_wall:.3f},"
            f"{second_wall / first_wall:.3f}"
        )


def main() -> None:
    args = parse_args()
    df = normalize_schema(pd.read_csv(args.summary_csv))
    if args.labels:
        requested_labels = args.labels
    elif args.workload:
        requested_labels = [args.workload]
    else:
        requested_labels = DEFAULT_LABELS
    labels = ordered_labels(list(dict.fromkeys(
        normalize_workload_label(label) for label in requested_labels
    )))
    baselines = list(args.baselines)
    rows = prepare_rows(df, labels, baselines)
    formats = [fmt.lstrip(".") for fmt in args.formats]

    print_numeric_summary(rows, labels, baselines, args.unit)
    plot_absolute(
        rows, labels, baselines, args.output, formats, args.unit,
        annotate_ratios=not args.no_ratio_annotations,
    )
    plot_composition(rows, labels, baselines, args.output, formats)


if __name__ == "__main__":
    main()
