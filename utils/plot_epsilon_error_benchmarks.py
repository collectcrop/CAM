#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Iterable

TMP_MPLCONFIGDIR = Path("/tmp/matplotlib")
TMP_XDG_CACHE_HOME = Path("/tmp/xdg-cache")
TMP_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
TMP_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_XDG_CACHE_HOME))

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
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
XLABEL_FONTSIZE = 20
YLABEL_FONTSIZE = 20
TICK_FONTSIZE = 15
TICK_FONT = {"labelsize": TICK_FONTSIZE}


ESTIMATE_REQUIRED_COLUMNS = {"m", "epsilon", "cost", "ratio"}
BENCH_REQUIRED_COLUMNS = {
    "epsilon",
    "policy",
    "queries",
    "hit_ratio",
    "logical_ios",
    "throughput_qps",
}
MERGE_KEYS = ["dataset_key", "M", "epsilon", "policy"]
PREFERRED_POLICIES = ["FIFO", "LRU", "LFU", "NONE"]
COLOR_MAP = {
    "FIFO": "tab:blue",
    "LRU": "tab:orange",
    "LFU": "tab:green",
    "NONE": "tab:gray",
}
DEFAULT_OUTPUT_DIR = Path("data/outputs/figures/epsilon_error_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge learned-index estimate logs with benchmark CSVs and draw "
            "error plots against epsilon."
        )
    )
    parser.add_argument(
        "--estimate-paths",
        nargs="+",
        type=Path,
        required=True,
        help="Explicit estimate log paths, for example: build/log/books_10M_uint64_unique_FIFO.log",
    )
    parser.add_argument(
        "--bench-paths",
        nargs="+",
        type=Path,
        required=True,
        help="Explicit benchmark CSV paths, for example: build/log/books_10M_M60_bench.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used for merged CSVs, reports, and plot images.",
    )
    parser.add_argument(
        "--dataset-filter",
        default=None,
        help="Optional substring used to filter filenames and normalized dataset keys.",
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
        help="Optional cache budget allowlist, for example: 10 20 40.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional prefix for output artifact filenames.",
    )
    return parser.parse_args()


def normalized_header(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_")


def normalize_dataset_key(name: str) -> str:
    key = name.strip()
    key = re.sub(r"_(?:u?int\d+)(?:_(?:unique|sorted))?$", "", key, flags=re.IGNORECASE)
    key = re.sub(r"_(?:unique|sorted|keys|data)$", "", key, flags=re.IGNORECASE)
    key = re.sub(r"__+", "_", key)
    return key


def should_keep_path(path: Path, dataset_filter: str | None) -> bool:
    if dataset_filter is None:
        return True
    needle = dataset_filter.lower()
    candidates = {
        path.name.lower(),
        path.stem.lower(),
        normalize_dataset_key(path.stem).lower(),
    }
    return any(needle in candidate for candidate in candidates)


def resolve_explicit_paths(paths: Iterable[Path], dataset_filter: str | None, source_name: str) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{source_name} path does not exist: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"{source_name} path is not a file: {path}")
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if should_keep_path(path, dataset_filter):
            resolved.append(path)
    if not resolved:
        raise ValueError(f"No {source_name} files remained after applying filters.")
    return resolved


def infer_policy_from_estimate_log(path: Path) -> str | None:
    match = re.match(r"(?P<dataset>.+)_(?P<policy>FIFO|LRU|LFU|NONE)\.log$", path.name, flags=re.IGNORECASE)
    return match.group("policy").upper() if match else None


def infer_dataset_from_estimate_log(path: Path) -> str:
    match = re.match(r"(?P<dataset>.+)_(?P<policy>FIFO|LRU|LFU|NONE)\.log$", path.name, flags=re.IGNORECASE)
    base = match.group("dataset") if match else path.stem
    return normalize_dataset_key(base)


def infer_dataset_and_m_from_bench(path: Path) -> tuple[str, int | None]:
    match = re.match(r"(?P<dataset>.+)_M(?P<m>\d+)_bench\.csv$", path.name, flags=re.IGNORECASE)
    if match is None:
        return normalize_dataset_key(path.stem), None
    return normalize_dataset_key(match.group("dataset")), int(match.group("m"))


def load_estimate_logs(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        policy = infer_policy_from_estimate_log(path)
        if policy is None:
            continue

        df = pd.read_csv(path)
        df.columns = [normalized_header(column) for column in df.columns]
        if not ESTIMATE_REQUIRED_COLUMNS.issubset(df.columns):
            raise ValueError(f"Unexpected estimate log format: {path}")

        df = df.rename(
            columns={
                "m": "M",
                "cost": "estimated_avg_logical_ios",
                "ratio": "estimated_hit_ratio",
            }
        )
        df["M"] = pd.to_numeric(df["M"], errors="coerce")
        df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce")
        df["estimated_avg_logical_ios"] = pd.to_numeric(df["estimated_avg_logical_ios"], errors="coerce")
        df["estimated_hit_ratio"] = pd.to_numeric(df["estimated_hit_ratio"], errors="coerce")
        df = df.dropna(subset=["M", "epsilon", "estimated_avg_logical_ios", "estimated_hit_ratio"]).copy()
        df["M"] = df["M"].astype(int)
        df["epsilon"] = df["epsilon"].astype(int)
        df["dataset_key"] = infer_dataset_from_estimate_log(path)
        df["policy"] = policy
        df["estimate_source"] = str(path)
        frames.append(
            df[
                [
                    "dataset_key",
                    "policy",
                    "M",
                    "epsilon",
                    "estimated_avg_logical_ios",
                    "estimated_hit_ratio",
                    "estimate_source",
                ]
            ]
        )

    if not frames:
        raise FileNotFoundError("No estimate logs matched the expected schema and naming pattern.")
    return pd.concat(frames, ignore_index=True)


def load_bench_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        df.columns = [normalized_header(column) for column in df.columns]
        if not BENCH_REQUIRED_COLUMNS.issubset(df.columns):
            raise ValueError(f"Unexpected benchmark CSV format: {path}")

        dataset_key, inferred_m = infer_dataset_and_m_from_bench(path)
        if "m" not in df.columns:
            if inferred_m is None:
                raise ValueError(f"Unable to infer M from benchmark filename: {path}")
            df["m"] = inferred_m

        df["m"] = pd.to_numeric(df["m"], errors="coerce")
        df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce")
        df["queries"] = pd.to_numeric(df["queries"], errors="coerce")
        df["hit_ratio"] = pd.to_numeric(df["hit_ratio"], errors="coerce")
        df["logical_ios"] = pd.to_numeric(df["logical_ios"], errors="coerce")
        df["throughput_qps"] = pd.to_numeric(df["throughput_qps"], errors="coerce")
        if "avg_logical_ios" in df.columns:
            df["avg_logical_ios"] = pd.to_numeric(df["avg_logical_ios"], errors="coerce")

        df = df.dropna(
            subset=["m", "epsilon", "queries", "hit_ratio", "logical_ios", "throughput_qps", "policy"]
        ).copy()
        df["M"] = df["m"].astype(int)
        df["epsilon"] = df["epsilon"].astype(int)
        df["policy"] = df["policy"].astype(str).str.upper()
        df["dataset_key"] = dataset_key
        df["bench_source"] = str(path)
        if "avg_logical_ios" not in df.columns:
            df["avg_logical_ios"] = df["logical_ios"] / df["queries"]

        frames.append(
            df[
                [
                    "dataset_key",
                    "policy",
                    "M",
                    "epsilon",
                    "queries",
                    "logical_ios",
                    "avg_logical_ios",
                    "hit_ratio",
                    "throughput_qps",
                    "bench_source",
                ]
            ].rename(
                columns={
                    "logical_ios": "actual_total_logical_ios",
                    "avg_logical_ios": "actual_avg_logical_ios",
                    "hit_ratio": "actual_hit_ratio",
                    "throughput_qps": "actual_throughput_qps",
                }
            )
        )

    if not frames:
        raise FileNotFoundError("No benchmark CSVs matched the expected schema.")
    return pd.concat(frames, ignore_index=True)


def filter_frame(
    df: pd.DataFrame,
    dataset_filter: str | None,
    policies: set[str] | None,
    m_values: set[int] | None,
    source_column: str,
) -> pd.DataFrame:
    filtered = df.copy()
    if dataset_filter is not None:
        needle = dataset_filter.lower()
        filtered = filtered[
            filtered["dataset_key"].astype(str).str.lower().str.contains(needle)
            | filtered[source_column].astype(str).str.lower().str.contains(needle)
        ]
    if policies is not None:
        filtered = filtered[filtered["policy"].isin(policies)]
    if m_values is not None:
        filtered = filtered[filtered["M"].isin(m_values)]
    return filtered


def validate_unique_keys(df: pd.DataFrame, source_name: str) -> None:
    duplicates = df[df.duplicated(subset=MERGE_KEYS, keep=False)]
    if not duplicates.empty:
        keys = duplicates[MERGE_KEYS].drop_duplicates().to_dict("records")
        raise ValueError(f"Duplicate {source_name} rows found for merge keys: {keys[:5]}")


def print_merge_diagnostics(estimates: pd.DataFrame, benches: pd.DataFrame, merged: pd.DataFrame) -> None:
    est_m = sorted(estimates["M"].dropna().astype(int).unique().tolist())
    bench_m = sorted(benches["M"].dropna().astype(int).unique().tolist())
    merged_m = sorted(merged["M"].dropna().astype(int).unique().tolist())
    print(
        f"[merge] estimate M={est_m}, bench M={bench_m}, merged M={merged_m}",
        file=sys.stderr,
    )

    estimate_keys = estimates[MERGE_KEYS].drop_duplicates()
    bench_keys = benches[MERGE_KEYS].drop_duplicates()
    key_diff = pd.merge(estimate_keys, bench_keys, on=MERGE_KEYS, how="outer", indicator=True)
    only_estimate = key_diff[key_diff["_merge"] == "left_only"]
    only_bench = key_diff[key_diff["_merge"] == "right_only"]
    if not only_estimate.empty:
        by_m = only_estimate.groupby("M").size().sort_index().to_dict()
        print(f"[merge] keys only in estimate (missing in bench) by M: {by_m}", file=sys.stderr)
    if not only_bench.empty:
        by_m = only_bench.groupby("M").size().sort_index().to_dict()
        print(f"[merge] keys only in bench (missing in estimate) by M: {by_m}", file=sys.stderr)


def merge_frames(estimates: pd.DataFrame, benches: pd.DataFrame) -> pd.DataFrame:
    validate_unique_keys(estimates, "estimate")
    validate_unique_keys(benches, "benchmark")

    merged = pd.merge(
        estimates,
        benches,
        how="inner",
        on=MERGE_KEYS,
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError(
            "Estimate logs and benchmark CSVs did not overlap on dataset_key, M, epsilon, and policy."
        )

    merged["estimated_total_logical_ios"] = merged["estimated_avg_logical_ios"] * merged["queries"]
    print_merge_diagnostics(estimates, benches, merged)
    return merged.sort_values(["dataset_key", "M", "policy", "epsilon"]).reset_index(drop=True)


def compute_error_metrics(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["estimated_minus_actual_total_logical_ios"] = (
        enriched["estimated_total_logical_ios"] - enriched["actual_total_logical_ios"]
    )
    enriched["estimated_minus_actual_avg_logical_ios"] = (
        enriched["estimated_avg_logical_ios"] - enriched["actual_avg_logical_ios"]
    )
    enriched["estimated_minus_actual_logical_io_pct_error"] = (
        enriched["estimated_minus_actual_total_logical_ios"] / enriched["actual_total_logical_ios"]
    )
    enriched["absolute_logical_io_pct_error"] = enriched["estimated_minus_actual_logical_io_pct_error"].abs()
    enriched["estimated_minus_actual_hit_ratio"] = enriched["estimated_hit_ratio"] - enriched["actual_hit_ratio"]
    enriched["absolute_hit_ratio_error"] = enriched["estimated_minus_actual_hit_ratio"].abs()
    return enriched


def build_prefix(dataset_key: str, output_prefix: str | None, multiple_datasets: bool) -> str:
    if output_prefix and multiple_datasets:
        return f"{output_prefix}_{dataset_key}"
    if output_prefix:
        return output_prefix
    return dataset_key


def ordered_policies(df: pd.DataFrame) -> list[str]:
    seen = set(df["policy"].dropna().astype(str))
    ordered = [policy for policy in PREFERRED_POLICIES if policy in seen]
    extras = sorted(seen - set(PREFERRED_POLICIES))
    return ordered + extras


def subplot_layout(count: int) -> tuple[int, int]:
    if count <= 1:
        return 1, 1
    ncols = 2
    nrows = math.ceil(count / ncols)
    return nrows, ncols


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
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def apply_tick_font(axis) -> None:
    axis.tick_params(axis="both", **TICK_FONT)
    tick_size = TICK_FONT["labelsize"]
    axis.xaxis.get_offset_text().set_fontsize(tick_size)
    axis.yaxis.get_offset_text().set_fontsize(tick_size)


def apply_outer_labels(axis, idx: int, nrows: int, ncols: int, xlabel: str, ylabel: str) -> None:
    row = idx // ncols
    col = idx % ncols
    axis.set_xlabel(xlabel if row == nrows - 1 else "", fontsize=XLABEL_FONTSIZE)
    axis.set_ylabel(ylabel if col == 0 else "", fontsize=YLABEL_FONTSIZE)


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


def plot_metric_vs_epsilon(
    dataset_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
    legend_output_path: Path,
    *,
    signed: bool,
) -> None:
    m_values = sorted(dataset_df["M"].unique().tolist())
    fig, axes, nrows, ncols = make_axes(m_values)
    policies = ordered_policies(dataset_df)
    legend_map: dict[str, object] = {}

    for idx, (axis, m_value) in enumerate(zip(axes, m_values)):
        part_m = dataset_df[dataset_df["M"] == m_value].copy()
        for policy in policies:
            part = part_m[part_m["policy"] == policy].sort_values("epsilon")
            if part.empty:
                continue
            color = COLOR_MAP.get(policy, None)
            line, = axis.plot(
                part["epsilon"],
                part[metric],
                color=color,
                marker="o",
                linestyle="-",
                linewidth=2,
                markersize=5,
                label=policy,
            )
            legend_map.setdefault(policy, line)

        if signed:
            axis.axhline(0.0, color="black", linestyle=":", linewidth=1.2, alpha=0.8)

        axis.set_title(f"M = {m_value}MB", fontsize=TITLE_FONTSIZE - 7)
        apply_outer_labels(axis, idx, nrows, ncols, "Epsilon", ylabel)
        axis.grid(alpha=0.3)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        apply_tick_font(axis)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_path)
    save_legend_figure(list(legend_map.values()), list(legend_map.keys()), legend_output_path)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = []
    for row in rows:
        formatted = []
        for value in row:
            if value is None:
                formatted.append("")
            elif isinstance(value, float):
                formatted.append(f"{value:.6f}".rstrip("0").rstrip("."))
            else:
                formatted.append(str(value))
        body.append("| " + " | ".join(formatted) + " |")
    return "\n".join([header, divider, *body])


def coverage_rows(dataset_df: pd.DataFrame) -> list[list[object]]:
    summary = (
        dataset_df.groupby(["M", "policy"], as_index=False)
        .agg(
            epsilon_points=("epsilon", "count"),
            mean_abs_logical_io_pct_error=("absolute_logical_io_pct_error", "mean"),
            max_abs_logical_io_pct_error=("absolute_logical_io_pct_error", "max"),
            mean_abs_hit_ratio_error=("absolute_hit_ratio_error", "mean"),
            max_abs_hit_ratio_error=("absolute_hit_ratio_error", "max"),
        )
        .sort_values(["M", "policy"])
    )
    return summary.values.tolist()


def write_report(
    dataset_df: pd.DataFrame,
    output_path: Path,
    estimate_paths: list[Path],
    bench_paths: list[Path],
    figure_paths: list[Path],
) -> None:
    dataset_key = str(dataset_df["dataset_key"].iloc[0])
    lines = [
        "# Epsilon Error Benchmark Report",
        "",
        "## Inputs",
        "",
        f"- Dataset: {dataset_key}",
        f"- Estimate logs: {len(estimate_paths)}",
        f"- Bench CSVs: {len(bench_paths)}",
        f"- Merged rows: {len(dataset_df)}",
        f"- Source code: utils/plot_epsilon_error_benchmarks.py",
        "",
        "## Error Definitions",
        "",
        "- Signed logical I/O error is `estimated_total_logical_ios - actual_total_logical_ios`.",
        "- Absolute logical I/O error is `|estimated - actual| / actual`.",
        "- Signed hit-ratio error is `estimated_hit_ratio - actual_hit_ratio`.",
        "- Absolute hit-ratio error is `|estimated_hit_ratio - actual_hit_ratio|`.",
        "",
        "### Estimate log files",
        "",
    ]
    lines.extend(f"- {path}" for path in estimate_paths)
    lines.extend(["", "### Benchmark CSV files", ""])
    lines.extend(f"- {path}" for path in bench_paths)
    lines.extend(
        [
            "",
            "## Coverage Summary",
            "",
            markdown_table(
                [
                    "M",
                    "policy",
                    "epsilon_points",
                    "mean_abs_logical_io_pct_error",
                    "max_abs_logical_io_pct_error",
                    "mean_abs_hit_ratio_error",
                    "max_abs_hit_ratio_error",
                ],
                coverage_rows(dataset_df),
            ),
            "",
            "## Figures",
            "",
        ]
    )
    lines.extend(f"- {path}" for path in figure_paths)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset_outputs(dataset_df: pd.DataFrame, output_dir: Path, prefix: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_csv_path = output_dir / f"{prefix}_merged_error_metrics.csv"
    logical_signed_path = output_dir / f"{prefix}_logical_io_signed_error_vs_epsilon.png"
    logical_signed_legend_path = output_dir / f"{prefix}_logical_io_signed_error_vs_epsilon_legend.png"
    logical_abs_path = output_dir / f"{prefix}_logical_io_abs_error_vs_epsilon.png"
    logical_abs_legend_path = output_dir / f"{prefix}_logical_io_abs_error_vs_epsilon_legend.png"
    hit_signed_path = output_dir / f"{prefix}_hit_ratio_signed_error_vs_epsilon.png"
    hit_signed_legend_path = output_dir / f"{prefix}_hit_ratio_signed_error_vs_epsilon_legend.png"
    hit_abs_path = output_dir / f"{prefix}_hit_ratio_abs_error_vs_epsilon.png"
    hit_abs_legend_path = output_dir / f"{prefix}_hit_ratio_abs_error_vs_epsilon_legend.png"
    report_path = output_dir / f"{prefix}_error_report.md"

    dataset_df.to_csv(merged_csv_path, index=False)
    plot_metric_vs_epsilon(
        dataset_df,
        metric="estimated_minus_actual_logical_io_pct_error",
        ylabel="Est. - Real Logical IO Error",
        output_path=logical_signed_path,
        legend_output_path=logical_signed_legend_path,
        signed=True,
    )
    plot_metric_vs_epsilon(
        dataset_df,
        metric="absolute_logical_io_pct_error",
        ylabel="Absolute Logical IO Error",
        output_path=logical_abs_path,
        legend_output_path=logical_abs_legend_path,
        signed=False,
    )
    plot_metric_vs_epsilon(
        dataset_df,
        metric="estimated_minus_actual_hit_ratio",
        ylabel="Est. - Real Hit Ratio",
        output_path=hit_signed_path,
        legend_output_path=hit_signed_legend_path,
        signed=True,
    )
    plot_metric_vs_epsilon(
        dataset_df,
        metric="absolute_hit_ratio_error",
        ylabel="Absolute Hit Ratio Error",
        output_path=hit_abs_path,
        legend_output_path=hit_abs_legend_path,
        signed=False,
    )
    write_report(
        dataset_df=dataset_df,
        output_path=report_path,
        estimate_paths=[Path(path) for path in sorted(dataset_df["estimate_source"].dropna().unique().tolist())],
        bench_paths=[Path(path) for path in sorted(dataset_df["bench_source"].dropna().unique().tolist())],
        figure_paths=[
            logical_signed_path,
            logical_signed_legend_path,
            logical_abs_path,
            logical_abs_legend_path,
            hit_signed_path,
            hit_signed_legend_path,
            hit_abs_path,
            hit_abs_legend_path,
            merged_csv_path,
        ],
    )
    return [
        merged_csv_path,
        logical_signed_path,
        logical_signed_legend_path,
        logical_abs_path,
        logical_abs_legend_path,
        hit_signed_path,
        hit_signed_legend_path,
        hit_abs_path,
        hit_abs_legend_path,
        report_path,
    ]


def main() -> None:
    args = parse_args()
    estimate_paths = resolve_explicit_paths(args.estimate_paths, args.dataset_filter, "estimate")
    bench_paths = resolve_explicit_paths(args.bench_paths, args.dataset_filter, "benchmark")

    estimates = load_estimate_logs(estimate_paths)
    benches = load_bench_csvs(bench_paths)
    policies = {policy.upper() for policy in args.policies} if args.policies else None
    m_values = set(args.m_values) if args.m_values else None
    if m_values:
        print(f"[filter] requested M values: {sorted(m_values)}", file=sys.stderr)
    estimates = filter_frame(estimates, args.dataset_filter, policies, m_values, "estimate_source")
    benches = filter_frame(benches, args.dataset_filter, policies, m_values, "bench_source")
    if estimates.empty:
        raise ValueError("No estimate rows remained after applying the requested filters.")
    if benches.empty:
        raise ValueError("No benchmark rows remained after applying the requested filters.")

    merged = merge_frames(estimates, benches)
    merged = compute_error_metrics(merged)
    dataset_keys = sorted(merged["dataset_key"].dropna().unique().tolist())
    multiple_datasets = len(dataset_keys) > 1

    for dataset_key in dataset_keys:
        dataset_df = merged[merged["dataset_key"] == dataset_key].copy()
        prefix = build_prefix(str(dataset_key), args.output_prefix, multiple_datasets)
        artifact_paths = write_dataset_outputs(dataset_df, args.output_dir, prefix)
        for path in artifact_paths:
            print(path)


if __name__ == "__main__":
    main()
