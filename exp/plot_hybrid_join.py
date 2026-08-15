#!/usr/bin/env python3
"""Plot hybrid-join strategies over all outer relation sizes in one PDF."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable

for name, value in {
    "MPLCONFIGDIR": "/tmp/cam-hybrid-join-matplotlib",
    "XDG_CACHE_HOME": "/tmp/cam-hybrid-join-xdg-cache",
}.items():
    Path(value).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(name, value)

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires matplotlib, numpy, and pandas. "
        "Run it with the configured plotting Python environment."
    ) from exc


DEFAULT_INPUT_DIR = Path("build/log/hybrid_join")
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times", "Computer Modern Roman"],
        "axes.unicode_minus": False,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

DEFAULT_OUTPUT = Path("data/outputs/figures/hybrid_join/hybrid_join_comparison.pdf")
DEFAULT_MODES = ["hybrid", "point", "range", "inlj", "hash", "sortmerge"]
MODE_ALIASES = {
    "hybrid": "hybrid",
    "point": "point",
    "range": "range",
    "inlj": "inlj",
    "hash": "hash",
    "hashjoin": "hash",
    "hash_join": "hash",
    "sortmerge": "sortmerge",
    "sortmergejoin": "sortmerge",
    "sort_merge": "sortmerge",
    "sort_merge_join": "sortmerge",
    "smj": "sortmerge",
}
MODE_LABELS = {
    "hybrid": "Hybrid Join",
    "point": "Point-only",
    "range": "Range-only",
    "inlj": "INLJ",
    "hash": "Hash Join",
    "sortmerge": "Sort-Merge Join",
}
MODE_STYLES = {
    "hybrid": ("#D62728", "o", "-"),
    "point": ("#1F77B4", "s", "--"),
    "range": ("#2CA02C", "^", "-."),
    "inlj": ("#9467BD", "D", ":"),
    "hash": ("#8C564B", "P", "--"),
    "sortmerge": ("#FF7F0E", "X", "-"),
}
METRICS = {
    "throughput_qps": (1.0, "Throughput (tuples/s)", True),
    "wall_ns": (1.0e9, "End-to-end latency (s)", True),
    "query_wall_ns": (1.0e9, "Query latency (s)", True),
    "physical_ios": (1.0, "Physical I/Os", True),
    "avg_physical_ios": (1.0, "Physical I/Os per tuple", True),
}
REQUIRED_COLUMNS = {"label", "mode", "queries"}
XLABEL_FONTSIZE = 25
YLABEL_FONTSIZE = 25
TICK_FONTSIZE = 20
LEGEND_FONTSIZE = 20
SUBPLOT_TITLE_FONTSIZE = 25
DPI = 300
NUMERIC_COLUMNS = set(METRICS).union({"queries"})
FILE_PATTERN = re.compile(
    r"^(?P<dataset>.+)_(?P<tag>\d+(?:\.\d+)?[KMG]?)_join_compare\.csv$",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw workloads as subplots, outer relation sizes on the x-axis, "
            "and join strategies as lines in one PDF."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--inputs", nargs="+", type=Path, help="Explicit CSVs; overrides --input-dir."
    )
    parser.add_argument("--dataset-filter", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--report", type=Path, help="Markdown report path; defaults next to the PDF."
    )
    parser.add_argument("--metric", choices=sorted(METRICS), default="throughput_qps")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--linear-y", action="store_true")
    parser.add_argument(
        "--strict-completeness",
        action="store_true",
        help="Fail if any workload/size/strategy combination is missing.",
    )
    return parser.parse_args()


def normalize_token(value: object) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower()).strip("_")


def normalize_mode(value: object) -> str:
    token = normalize_token(value)
    return MODE_ALIASES.get(token, token)


def workload_sort_key(label: str) -> tuple[int, str]:
    match = re.fullmatch(r"(?:table|w)(\d+)", label, flags=re.IGNORECASE)
    return (int(match.group(1)), label) if match else (10**9, label)


def display_workload(label: str) -> str:
    match = re.fullmatch(r"(?:table|w)(\d+)", label, flags=re.IGNORECASE)
    return f"w{match.group(1)}" if match else label


def format_size(value: int | float) -> str:
    size = float(value)
    for divisor, suffix in ((1.0e9, "G"), (1.0e6, "M"), (1.0e3, "K")):
        if size >= divisor and size % divisor == 0:
            return f"{size / divisor:g}{suffix}"
    return f"{size:g}"


def discover_inputs(args: argparse.Namespace) -> list[Path]:
    if args.inputs:
        candidates = [path.expanduser() for path in args.inputs]
    else:
        if not args.input_dir.is_dir():
            raise FileNotFoundError(f"input directory does not exist: {args.input_dir}")
        candidates = sorted(args.input_dir.glob("*_join_compare.csv"))
    needle = args.dataset_filter.lower() if args.dataset_filter else None
    paths = []
    for path in candidates:
        if not path.is_file():
            raise FileNotFoundError(f"input CSV does not exist: {path}")
        if needle is None or needle in path.name.lower():
            paths.append(path.resolve())
    if not paths:
        raise FileNotFoundError("no matching *_join_compare.csv files found")
    return paths


def normalize_modes(values: Iterable[str]) -> list[str]:
    modes = []
    for value in values:
        mode = normalize_mode(value)
        if mode not in MODE_LABELS:
            raise ValueError(f"unknown mode {value!r}; choose from {DEFAULT_MODES}")
        if mode not in modes:
            modes.append(mode)
    return modes


def load_results(
    paths: list[Path], metric: str
) -> tuple[pd.DataFrame, list[Path], list[str], list[str]]:
    frames = []
    used_paths = []
    datasets = set()
    warnings = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            warnings.append(f"skipping empty CSV: {path}")
            continue
        frame.columns = [normalize_token(column) for column in frame.columns]
        missing = REQUIRED_COLUMNS.union({metric}).difference(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for column in NUMERIC_COLUMNS.intersection(frame.columns):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["label"] = frame["label"].astype(str).str.strip()
        frame["mode"] = frame["mode"].map(normalize_mode)
        frame = frame.dropna(subset=["label", "mode", "queries", metric]).copy()
        if frame.empty:
            warnings.append(f"skipping CSV without usable rows: {path}")
            continue
        frame["outer_size"] = frame["queries"].astype(np.int64)
        frame["source_csv"] = str(path)
        frames.append(frame)
        used_paths.append(path)
        match = FILE_PATTERN.match(path.name)
        datasets.add(match.group("dataset") if match else path.stem)
    if not frames:
        raise ValueError("none of the input CSVs contained usable rows")

    data = pd.concat(frames, ignore_index=True)
    keys = ["label", "mode", "outer_size"]
    duplicate_mask = data.duplicated(keys, keep=False)
    if duplicate_mask.any():
        warnings.append(
            f"found {int(duplicate_mask.sum())} duplicate rows by {keys}; keeping the last"
        )
        data = data.drop_duplicates(keys, keep="last")
    return data, used_paths, sorted(datasets), warnings


def find_incomplete(
    data: pd.DataFrame, workloads: list[str], sizes: list[int], modes: list[str]
) -> list[str]:
    observed = set(
        data[["label", "outer_size", "mode"]].itertuples(index=False, name=None)
    )
    warnings = []
    for size in sizes:
        missing = [
            f"{display_workload(workload)}/{MODE_LABELS[mode]}"
            for workload in workloads
            for mode in modes
            if (workload, size, mode) not in observed
        ]
        if missing:
            preview = ", ".join(missing[:8])
            if len(missing) > 8:
                preview += f", ... (+{len(missing) - 8})"
            warnings.append(f"outer size {format_size(size)} is incomplete: {preview}")
    return warnings


def draw_figure(
    data: pd.DataFrame,
    output: Path,
    metric: str,
    modes: list[str],
    linear_y: bool,
) -> tuple[list[str], list[int]]:
    workloads = sorted(data["label"].unique().tolist(), key=workload_sort_key)
    sizes = sorted(int(value) for value in data["outer_size"].unique())
    ncols = min(3, len(workloads))
    nrows = (len(workloads) + ncols - 1) // ncols
    scale, ylabel, default_log = METRICS[metric]
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.1 * ncols, 3.8 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    size_positions = {size: index for index, size in enumerate(sizes)}
    for index, workload in enumerate(workloads):
        ax = axes[index // ncols][index % ncols]
        part = data[data["label"] == workload]
        for mode in modes:
            series = part[part["mode"] == mode].sort_values("outer_size")
            if series.empty:
                continue
            color, marker, linestyle = MODE_STYLES[mode]
            ax.plot(
                [size_positions[int(value)] for value in series["outer_size"]],
                series[metric] / scale,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.9,
                markersize=6.5,
                markeredgecolor="white",
                markeredgewidth=0.45,
            )
        ax.set_title(display_workload(workload), fontsize=SUBPLOT_TITLE_FONTSIZE, pad=7)
        if default_log and not linear_y:
            ax.set_yscale("log")
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([format_size(value) for value in sizes])
        ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
        ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.4, alpha=0.2)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    for index in range(len(workloads), nrows * ncols):
        axes[index // ncols][index % ncols].set_visible(False)

    handles = [
        Line2D(
            [0],
            [0],
            color=MODE_STYLES[mode][0],
            marker=MODE_STYLES[mode][1],
            linestyle=MODE_STYLES[mode][2],
            linewidth=1.9,
            markersize=6.5,
            label=MODE_LABELS[mode],
        )
        for mode in modes
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(6, len(handles)),
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=2.8,
        columnspacing=1.2,
    )
    fig.supxlabel("Outer relation size", fontsize=XLABEL_FONTSIZE, y=0.018)
    fig.supylabel(ylabel, fontsize=YLABEL_FONTSIZE, x=0.01)
    fig.tight_layout(rect=(0.04, 0.07, 1.0, 0.89), h_pad=1.2, w_pad=1.0)
    if output.suffix.lower() != ".pdf":
        raise ValueError(f"--output must end in .pdf: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return workloads, sizes


def write_report(
    path: Path,
    output: Path,
    input_paths: list[Path],
    metric: str,
    workloads: list[str],
    sizes: list[int],
    modes: list[str],
    warnings: list[str],
) -> None:
    lines = [
        "# Hybrid Join Visualization Report",
        "",
        "## Experiment target",
        "",
        f"Compare join strategies across workloads and outer relation sizes using `{metric}`.",
        "",
        "## Input files",
        "",
        *[f"- `{item}`" for item in input_paths],
        "",
        "## Schema normalization",
        "",
        "Column and strategy names were normalized, numeric fields were parsed, and the "
        "CSV `queries` field was used as outer relation size.",
        "",
        "## Coverage",
        "",
        f"- Workloads: {', '.join(display_workload(item) for item in workloads)}",
        f"- Outer sizes: {', '.join(format_size(item) for item in sizes)}",
        f"- Strategies: {', '.join(MODE_LABELS[item] for item in modes)}",
        "",
        "## Observations",
        "",
    ]
    if warnings:
        lines.append("Inputs are incomplete; missing points were omitted rather than imputed.")
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("All requested workload/size/strategy combinations were present.")
    lines.extend(
        [
            "",
            "## Output",
            "",
            f"- Combined PDF: `{output}`",
            f"- Plot source: `{Path(__file__).resolve()}`",
            "",
            "## Conclusion",
            "",
            "The figure supports comparison across outer sizes; conclusions involving "
            "incomplete sizes should be treated as provisional.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        modes = normalize_modes(args.modes)
        data, paths, datasets, warnings = load_results(discover_inputs(args), args.metric)
        if len(datasets) > 1:
            raise ValueError(
                f"multiple datasets matched ({', '.join(datasets)}); use --dataset-filter"
            )
        data = data[data["mode"].isin(modes)].copy()
        if data.empty:
            raise ValueError("no rows matched the requested join modes")
        workloads = sorted(data["label"].unique().tolist(), key=workload_sort_key)
        sizes = sorted(int(value) for value in data["outer_size"].unique())
        warnings.extend(find_incomplete(data, workloads, sizes, modes))
        if warnings and args.strict_completeness:
            raise ValueError("\n".join(warnings))
        workloads, sizes = draw_figure(
            data, args.output, args.metric, modes, args.linear_y
        )
        report = args.report or args.output.with_suffix(".md")
        write_report(report, args.output, paths, args.metric, workloads, sizes, modes, warnings)
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)
        print(args.output)
        print(report)
        return 0
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
