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

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, ScalarFormatter
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover - runtime environment guard
    raise SystemExit(
        "This script requires matplotlib and pandas. "
        "Set PYTHON_BIN in config.sh to a Python environment with those packages."
    ) from exc

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
# global fonts
TITLE_FONTSIZE = 30     
XLABEL_FONTSIZE = 30    
YLABEL_FONTSIZE = 30    
TICK_FONTSIZE   = 25    
TICK_FONT = {"labelsize": TICK_FONTSIZE}
MARKER_SIZE_PRIMARY = 8
MARKER_SIZE_SECONDARY = 7
MARKER_SIZE_ERROR = 9
MIN_PLOTTED_EPSILON = 8


ESTIMATE_REQUIRED_COLUMNS = {"m", "epsilon", "cost", "ratio"}
BENCH_REQUIRED_COLUMNS = {
    "epsilon",
    "policy",
    "hit_ratio",
    "logical_ios",
    "throughput_qps",
}
BENCH_QUERY_COUNT_COLUMNS = ("queries", "ranges")
MERGE_KEYS = ["dataset_key", "M", "epsilon", "policy"]
PREFERRED_POLICIES = ["FIFO", "LRU", "LFU", "NONE"]
COLOR_MAP = {
    "FIFO": "tab:blue",
    "LRU": "tab:orange",
    "LFU": "tab:green",
    "NONE": "tab:gray",
}
DEFAULT_OUTPUT_DIR = Path("data/outputs/figures/epsilon_analysis")
DEFAULT_FITCAM_ROOT = Path("build/log/fitcam_q30")
FITCAM_REQUIRED_COLUMNS = {
    "policy",
    "epsilon",
    "actual_avg_ios",
    "m",
    "estimated_cost",
    "corrected_cost",
}
REVISION_LOG_REQUIRED_COLUMNS = {"m", "epsilon", "cost", "ratio"}
REAL_FITCAM_REQUIRED_COLUMNS = {"epsilon", "avg_ios"}
REAL_SUMMARY_REQUIRED_COLUMNS = {"epsilon", "policy", "queries"}
FITCAM_CURVE_COLOR_MAP = {
    "real": "black",
    "estimate": "blue",
    "calibrated": "red",
    "error_before": "blue",
    "error_after": "red",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge learned-index estimate logs with benchmark CSVs and draw "
            "matplotlib figures aligned with build/visualize/pgm_logical_ios_visualization.ipynb."
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
    parser.add_argument(
        "--fitcam-root",
        type=Path,
        default=DEFAULT_FITCAM_ROOT,
        help="Root directory used to discover q30 fitCAM outputs. Default: build/log/fitcam_q30",
    )
    parser.add_argument(
        "--revision-log-dir",
        type=Path,
        default=None,
        help=(
            "Directory used to discover revised estimate logs named "
            "${DATA_FILE}_${POLICY}_revision.log. "
            "Default: parent directory of --fitcam-root, e.g., build/log."
        ),
    )
    parser.add_argument(
        "--skip-fitcam",
        action="store_true",
        help="Skip auto-discovery and plotting of q30 fitCAM corrected-vs-real figures.",
    )
    parser.add_argument(
        "--only-logical-ios",
        action="store_true",
        help="Only write the main logical_ios_vs_epsilon PDF; skip legends, reports, and other figures.",
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
    match = re.match(
        r"(?P<dataset>.+)_M(?P<m>\d+)(?:_range)?_bench(?:_[^.]+)?\.csv$",
        path.name,
        flags=re.IGNORECASE,
    )
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
        query_count_column = next((column for column in BENCH_QUERY_COUNT_COLUMNS if column in df.columns), None)
        if query_count_column is None:
            raise ValueError(
                f"Unexpected benchmark CSV format: {path}; expected one of "
                f"{', '.join(BENCH_QUERY_COUNT_COLUMNS)}"
            )
        workload_type = "range" if query_count_column == "ranges" else "point"

        dataset_key, inferred_m = infer_dataset_and_m_from_bench(path)
        if "m" not in df.columns:
            if inferred_m is None:
                raise ValueError(f"Unable to infer M from benchmark filename: {path}")
            df["m"] = inferred_m

        df["m"] = pd.to_numeric(df["m"], errors="coerce")
        df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce")
        df["queries"] = pd.to_numeric(df[query_count_column], errors="coerce")
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
        df["workload_type"] = workload_type
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
                    "workload_type",
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
        print_merge_diagnostics(estimates, benches, merged)
        raise ValueError(
            "Estimate logs and benchmark CSVs did not overlap on dataset_key, M, epsilon, and policy."
        )

    query_counts = (
        benches.dropna(subset=["queries"])
        .groupby(["dataset_key", "M", "policy"], as_index=False)
        .agg(reference_queries=("queries", "first"))
    )
    merged = merged.merge(query_counts, how="left", on=["dataset_key", "M", "policy"])
    merged["queries"] = merged["queries"].fillna(merged["reference_queries"])
    merged = merged.drop(columns=["reference_queries"])
    merged = merged.dropna(subset=["queries"]).copy()
    if merged.empty:
        print_merge_diagnostics(estimates, benches, merged)
        raise ValueError(
            "Benchmark CSVs did not provide query/range counts for any estimate rows after filtering."
        )

    workload_types = (
        benches.dropna(subset=["workload_type"])
        .groupby(["dataset_key", "M", "policy"], as_index=False)
        .agg(reference_workload_type=("workload_type", "first"))
    )
    merged = merged.merge(workload_types, how="left", on=["dataset_key", "M", "policy"])
    merged["workload_type"] = merged["workload_type"].fillna(merged["reference_workload_type"])
    merged["workload_type"] = merged["workload_type"].fillna("estimate_only")
    merged = merged.drop(columns=["reference_workload_type"])

    merged["estimated_total_logical_ios"] = merged["estimated_avg_logical_ios"] * merged["queries"]
    merged["logical_io_error"] = merged["actual_total_logical_ios"] - merged["estimated_total_logical_ios"]
    merged["logical_io_abs_pct_error"] = (
        merged["logical_io_error"].abs() / merged["actual_total_logical_ios"]
    )
    merged["hit_ratio_error"] = merged["actual_hit_ratio"] - merged["estimated_hit_ratio"]
    merged["hit_ratio_abs_error"] = merged["hit_ratio_error"].abs()
    print_merge_diagnostics(estimates, benches, merged)
    return merged.sort_values(["dataset_key", "M", "policy", "epsilon"]).reset_index(drop=True)


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


def apply_scientific_y(axis) -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    axis.yaxis.set_major_formatter(formatter)


def apply_percent_y(axis, ratio_values: pd.Series) -> None:
    max_value = pd.to_numeric(ratio_values, errors="coerce").dropna().max()
    xmax = 100.0 if pd.notna(max_value) and max_value > 1.5 else 1.0
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=xmax, decimals=0))


def apply_outer_labels(axis, idx: int, nrows: int, ncols: int, xlabel: str, ylabel: str) -> None:
    row = idx // ncols
    col = idx % ncols
    axis.set_xlabel(xlabel if row == nrows - 1 else "", fontsize=XLABEL_FONTSIZE)
    axis.set_ylabel(ylabel if col == 0 else "", fontsize=YLABEL_FONTSIZE)


def apply_outer_right_ylabel(axis_right, idx: int, ncols: int, ylabel: str) -> None:
    col = idx % ncols
    axis_right.set_ylabel(ylabel if col == ncols - 1 else "", fontsize=YLABEL_FONTSIZE)


def filter_min_epsilon(df: pd.DataFrame) -> pd.DataFrame:
    if "epsilon" not in df.columns:
        return df
    return df[df["epsilon"] >= MIN_PLOTTED_EPSILON].copy()


def series_rows(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    return (
        df.dropna(subset=["epsilon", value_column])
        .sort_values("epsilon")
        .copy()
    )


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


def infer_policy_from_fitcam_curve_csv(path: Path) -> str | None:
    match = re.match(
        r"(?P<dataset>.+)_(?P<policy>FIFO|LRU|LFU|NONE)_q30_fitcam_corrected_vs_real\.csv$",
        path.name,
        flags=re.IGNORECASE,
    )
    return match.group("policy").upper() if match else None


def load_real_fitcam_rows(real_dir: Path, dataset_key: str, policy: str) -> pd.DataFrame:
    if not real_dir.exists() or not real_dir.is_dir():
        return pd.DataFrame(columns=["policy", "M", "epsilon", "actual_avg_ios", "real_fitcam_source"])

    pattern = re.compile(
        rf"^{re.escape(dataset_key)}_M(?P<m>\d+)_{policy.lower()}_fitcam_real\.csv$",
        flags=re.IGNORECASE,
    )
    frames: list[pd.DataFrame] = []
    for path in sorted(real_dir.glob("*.csv")):
        match = pattern.match(path.name)
        if match is None:
            continue
        df = pd.read_csv(path)
        df.columns = [normalized_header(column) for column in df.columns]
        if not REAL_FITCAM_REQUIRED_COLUMNS.issubset(df.columns):
            raise ValueError(f"Unexpected real_fitcam CSV format: {path}")
        df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce")
        df["avg_ios"] = pd.to_numeric(df["avg_ios"], errors="coerce")
        df = df.dropna(subset=["epsilon", "avg_ios"]).copy()
        df["epsilon"] = df["epsilon"].astype(int)
        df["M"] = int(match.group("m"))
        df["policy"] = policy.upper()
        df["real_fitcam_source"] = str(path.resolve())
        frames.append(
            df[
                [
                    "policy",
                    "M",
                    "epsilon",
                    "avg_ios",
                    "real_fitcam_source",
                ]
            ].rename(columns={"avg_ios": "actual_avg_ios"})
        )

    if not frames:
        return pd.DataFrame(columns=["policy", "M", "epsilon", "actual_avg_ios", "real_fitcam_source"])
    return pd.concat(frames, ignore_index=True)


def load_fitcam_query_counts(summary_dir: Path, dataset_key: str, policy: str) -> pd.DataFrame:
    if not summary_dir.exists() or not summary_dir.is_dir():
        return pd.DataFrame(columns=["policy", "M", "epsilon", "queries", "fitcam_summary_source"])

    pattern = re.compile(
        rf"^{re.escape(dataset_key)}_M(?P<m>\d+)_q30_summary\.csv$",
        flags=re.IGNORECASE,
    )
    frames: list[pd.DataFrame] = []
    for path in sorted(summary_dir.glob("*.csv")):
        match = pattern.match(path.name)
        if match is None:
            continue
        df = pd.read_csv(path)
        df.columns = [normalized_header(column) for column in df.columns]
        if not REAL_SUMMARY_REQUIRED_COLUMNS.issubset(df.columns):
            raise ValueError(f"Unexpected fitcam real_summary CSV format: {path}")
        df["policy"] = df["policy"].astype(str).str.upper()
        df = df[df["policy"] == policy.upper()].copy()
        df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce")
        df["queries"] = pd.to_numeric(df["queries"], errors="coerce")
        df = df.dropna(subset=["epsilon", "queries"]).copy()
        df["epsilon"] = df["epsilon"].astype(int)
        df["queries"] = df["queries"].astype(int)
        df["M"] = int(match.group("m"))
        df["fitcam_summary_source"] = str(path.resolve())
        frames.append(df[["policy", "M", "epsilon", "queries", "fitcam_summary_source"]])

    if not frames:
        return pd.DataFrame(columns=["policy", "M", "epsilon", "queries", "fitcam_summary_source"])
    return pd.concat(frames, ignore_index=True)


def choose_real_fitcam_dir(fitcam_dir: Path, corrected_csv: Path, policy: str) -> Path | None:
    candidates = [
        corrected_csv.parent / "real_fitcam",
        fitcam_dir / policy / "real_fitcam",
        fitcam_dir / "real_fitcam",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def load_fitcam_curve_frame(
    corrected_csv: Path,
    fitcam_dir: Path,
    dataset_key: str,
    policy: str,
    full_reference_df: pd.DataFrame,
    allowed_m_values: set[int] | None,
) -> pd.DataFrame:
    df = pd.read_csv(corrected_csv)
    df.columns = [normalized_header(column) for column in df.columns]
    if not FITCAM_REQUIRED_COLUMNS.issubset(df.columns):
        raise ValueError(f"Unexpected fitCAM corrected-vs-real CSV format: {corrected_csv}")

    df["policy"] = df["policy"].astype(str).str.upper()
    df = df[df["policy"] == policy.upper()].copy()
    df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce")
    df["m"] = pd.to_numeric(df["m"], errors="coerce")
    df["actual_avg_ios"] = pd.to_numeric(df["actual_avg_ios"], errors="coerce")
    df["estimated_cost"] = pd.to_numeric(df["estimated_cost"], errors="coerce")
    df["corrected_cost"] = pd.to_numeric(df["corrected_cost"], errors="coerce")
    df = df.dropna(subset=["epsilon", "m", "actual_avg_ios", "estimated_cost", "corrected_cost"]).copy()
    df["epsilon"] = df["epsilon"].astype(int)
    df["M"] = df["m"].astype(int)
    if allowed_m_values is not None:
        df = df[df["M"].isin(allowed_m_values)].copy()
    if df.empty:
        return df
    df["fitcam_source"] = str(corrected_csv.resolve())
    df["dataset_key"] = dataset_key

    real_fitcam_dir = choose_real_fitcam_dir(fitcam_dir, corrected_csv, policy)
    if real_fitcam_dir is not None:
        real_df = load_real_fitcam_rows(real_fitcam_dir, dataset_key, policy)
        if not real_df.empty:
            df = df.merge(
                real_df,
                how="left",
                on=["policy", "M", "epsilon"],
                suffixes=("", "_from_real_fitcam"),
            )
            df["actual_avg_ios"] = df["actual_avg_ios_from_real_fitcam"].fillna(df["actual_avg_ios"])
            df = df.drop(columns=["actual_avg_ios_from_real_fitcam"])
        else:
            df["real_fitcam_source"] = ""
    else:
        df["real_fitcam_source"] = ""

    reference_df = full_reference_df.copy()
    reference_df["policy"] = reference_df["policy"].astype(str).str.upper()
    reference_df["M"] = pd.to_numeric(reference_df["M"], errors="coerce").astype(int)
    reference_df["epsilon"] = pd.to_numeric(reference_df["epsilon"], errors="coerce").astype(int)
    reference_df["queries"] = pd.to_numeric(reference_df["queries"], errors="coerce")
    reference_df["actual_total_logical_ios"] = pd.to_numeric(
        reference_df["actual_total_logical_ios"], errors="coerce"
    )
    reference_df = reference_df.dropna(subset=["queries", "actual_total_logical_ios"]).copy()
    reference_df = reference_df.rename(columns={"queries": "full_queries"})

    df = df.merge(
        reference_df[["policy", "M", "epsilon", "full_queries", "actual_total_logical_ios"]],
        how="left",
        on=["policy", "M", "epsilon"],
    )
    if df["full_queries"].isna().any() or df["actual_total_logical_ios"].isna().any():
        missing = (
            df[df["full_queries"].isna() | df["actual_total_logical_ios"].isna()][["policy", "M", "epsilon"]]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(f"Missing full-workload benchmark reference for fitCAM rows: {missing[:5]}")

    df["actual_total_ios"] = df["actual_total_logical_ios"]
    df["estimated_total_ios"] = df["estimated_cost"] * df["full_queries"]
    df["corrected_total_ios"] = df["corrected_cost"] * df["full_queries"]
    df["error_before"] = df["estimated_total_ios"] - df["actual_total_ios"]
    df["error_after"] = df["corrected_total_ios"] - df["actual_total_ios"]
    return df.sort_values(["M", "epsilon"]).reset_index(drop=True)


def discover_fitcam_curve_sources(
    fitcam_dir: Path,
    dataset_key: str,
    policies: set[str] | None,
) -> dict[str, Path]:
    if not fitcam_dir.exists() or not fitcam_dir.is_dir():
        return {}

    selected: dict[str, tuple[int, Path]] = {}
    for path in sorted(fitcam_dir.glob("**/*_q30_fitcam_corrected_vs_real.csv")):
        policy = infer_policy_from_fitcam_curve_csv(path)
        if policy is None:
            continue
        if policies is not None and policy not in policies:
            continue
        score = 1 if path.parent.name.upper() == policy else 0
        previous = selected.get(policy)
        if previous is None or score > previous[0]:
            selected[policy] = (score, path.resolve())
    return {policy: item[1] for policy, item in selected.items()}




def infer_policy_from_revision_log(path: Path) -> str | None:
    match = re.match(
        r"(?P<dataset>.+)_(?P<policy>FIFO|LRU|LFU|NONE)_revision\.log$",
        path.name,
        flags=re.IGNORECASE,
    )
    return match.group("policy").upper() if match else None


def infer_dataset_from_revision_log(path: Path) -> str | None:
    match = re.match(
        r"(?P<dataset>.+)_(?P<policy>FIFO|LRU|LFU|NONE)_revision\.log$",
        path.name,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return normalize_dataset_key(match.group("dataset"))


def discover_revision_log_sources(
    revision_dir: Path,
    dataset_key: str,
    policies: set[str] | None,
) -> dict[str, Path]:
    """Discover build/log/${DATA_FILE}_${POLICY}_revision.log files."""
    if not revision_dir.exists() or not revision_dir.is_dir():
        return {}

    selected: dict[str, tuple[int, Path]] = {}
    for path in sorted(revision_dir.rglob("*_revision.log")):
        policy = infer_policy_from_revision_log(path)
        path_dataset_key = infer_dataset_from_revision_log(path)
        if policy is None or path_dataset_key is None:
            continue
        if policies is not None and policy not in policies:
            continue
        if path_dataset_key != dataset_key:
            continue
        score = 1 if path.parent.resolve() == revision_dir.resolve() else 0
        previous = selected.get(policy)
        if previous is None or score > previous[0]:
            selected[policy] = (score, path.resolve())
    return {policy: item[1] for policy, item in selected.items()}


def load_cost_log(path: Path, *, cost_name: str, ratio_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [normalized_header(column) for column in df.columns]
    if not REVISION_LOG_REQUIRED_COLUMNS.issubset(df.columns):
        raise ValueError(f"Unexpected estimate/revision log format: {path}")
    df = df.rename(columns={"m": "M", "cost": cost_name, "ratio": ratio_name})
    df["M"] = pd.to_numeric(df["M"], errors="coerce")
    df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce")
    df[cost_name] = pd.to_numeric(df[cost_name], errors="coerce")
    df[ratio_name] = pd.to_numeric(df[ratio_name], errors="coerce")
    df = df.dropna(subset=["M", "epsilon", cost_name, ratio_name]).copy()
    df["M"] = df["M"].astype(int)
    df["epsilon"] = df["epsilon"].astype(int)
    return df[["M", "epsilon", cost_name, ratio_name]]


def original_log_for_revision(revision_log: Path) -> Path:
    return revision_log.with_name(revision_log.name.replace("_revision.log", ".log"))


def load_revision_curve_frame(
    revision_log: Path,
    dataset_key: str,
    policy: str,
    full_reference_df: pd.DataFrame,
    allowed_m_values: set[int] | None,
) -> pd.DataFrame:
    """Build calibrated-vs-real metrics from a revised full estimate log.

    The revised log stores corrected per-query cost in `cost`.  The original
    estimate log is read from the same directory by replacing `_revision.log`
    with `.log`. If it is absent, `cost - ratio` is used as a fallback because
    fitCAM writes the predicted residual into `ratio`.
    """
    rev_df = load_cost_log(revision_log, cost_name="corrected_cost", ratio_name="predicted_residual")
    original_log = original_log_for_revision(revision_log)
    if original_log.exists():
        est_df = load_cost_log(original_log, cost_name="estimated_cost", ratio_name="estimated_hit_ratio")
        df = rev_df.merge(est_df[["M", "epsilon", "estimated_cost", "estimated_hit_ratio"]], on=["M", "epsilon"], how="inner")
        if df.empty:
            raise ValueError(f"Revision log and original estimate log do not overlap: {revision_log}, {original_log}")
        df["estimate_source"] = str(original_log.resolve())
    else:
        df = rev_df.copy()
        df["estimated_cost"] = df["corrected_cost"] - df["predicted_residual"]
        df["estimated_hit_ratio"] = np.nan
        df["estimate_source"] = ""
        print(
            f"[fitCAM] original estimate log not found for {revision_log}; "
            "using corrected_cost - predicted_residual as estimated_cost.",
            file=sys.stderr,
        )

    df["policy"] = policy.upper()
    df["dataset_key"] = dataset_key
    df["fitcam_source"] = str(revision_log.resolve())
    if allowed_m_values is not None:
        df = df[df["M"].isin(allowed_m_values)].copy()
    if df.empty:
        return df

    reference_df = full_reference_df.copy()
    reference_df["policy"] = reference_df["policy"].astype(str).str.upper()
    reference_df["M"] = pd.to_numeric(reference_df["M"], errors="coerce").astype(int)
    reference_df["epsilon"] = pd.to_numeric(reference_df["epsilon"], errors="coerce").astype(int)
    reference_df["queries"] = pd.to_numeric(reference_df["queries"], errors="coerce")
    reference_df["actual_total_logical_ios"] = pd.to_numeric(reference_df["actual_total_logical_ios"], errors="coerce")
    reference_policy_df = reference_df[reference_df["policy"] == policy.upper()].copy()

    query_df = (
        reference_policy_df.dropna(subset=["queries"])
        .groupby(["policy", "M"], as_index=False)
        .agg(full_queries=("queries", "first"))
    )
    actual_df = reference_policy_df.dropna(subset=["actual_total_logical_ios"]).copy()

    df = df.merge(
        actual_df[["policy", "M", "epsilon", "actual_total_logical_ios"]],
        how="outer",
        on=["policy", "M", "epsilon"],
    )
    df = df.merge(query_df, how="left", on=["policy", "M"])
    df = df.dropna(subset=["full_queries"]).copy()
    if df.empty:
        raise ValueError(f"Revision log has no workload query counts in benchmark rows: {revision_log}")
    df["dataset_key"] = df["dataset_key"].fillna(dataset_key)
    df["policy"] = df["policy"].fillna(policy.upper())
    df["fitcam_source"] = df["fitcam_source"].fillna(str(revision_log.resolve()))

    df["actual_total_ios"] = df["actual_total_logical_ios"]
    df["estimated_total_ios"] = df["estimated_cost"] * df["full_queries"]
    df["corrected_total_ios"] = df["corrected_cost"] * df["full_queries"]
    df["error_before"] = df["estimated_total_ios"] - df["actual_total_ios"]
    df["error_after"] = df["corrected_total_ios"] - df["actual_total_ios"]
    return df.sort_values(["M", "epsilon"]).reset_index(drop=True)

def plot_fitcam_policy_curves(
    policy_df: pd.DataFrame,
    output_path: Path,
    legend_output_path: Path,
) -> None:
    policy_df = filter_min_epsilon(policy_df)
    m_values = sorted(policy_df["M"].unique().tolist())
    fig, axes, nrows, ncols = make_axes(m_values)
    legend_map: dict[str, object] = {}

    for idx, (axis, m_value) in enumerate(zip(axes, m_values)):
        axis_right = axis.twinx()
        part = filter_min_epsilon(policy_df[policy_df["M"] == m_value]).sort_values("epsilon")
        if part.empty:
            continue

        real_part = series_rows(part, "actual_total_ios")
        estimate_part = series_rows(part, "estimated_total_ios")
        calibrated_part = series_rows(part, "corrected_total_ios")
        err_before_part = series_rows(part, "error_before")
        err_after_part = series_rows(part, "error_after")

        if not real_part.empty:
            real_line, = axis.plot(
                real_part["epsilon"],
                real_part["actual_total_ios"],
                color=FITCAM_CURVE_COLOR_MAP["real"],
                marker="o",
                linestyle="-",
                linewidth=2,
                markersize=MARKER_SIZE_PRIMARY,
                label="real",
            )
            legend_map.setdefault("real", real_line)
        if not estimate_part.empty:
            estimate_line, = axis.plot(
                estimate_part["epsilon"],
                estimate_part["estimated_total_ios"],
                color=FITCAM_CURVE_COLOR_MAP["estimate"],
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
                calibrated_part["epsilon"],
                calibrated_part["corrected_total_ios"],
                color=FITCAM_CURVE_COLOR_MAP["calibrated"],
                marker=None,
                linestyle="-",
                linewidth=2,
                solid_capstyle="round",
                solid_joinstyle="round",
                label="calibrated estimate",
            )
            legend_map.setdefault("calibrated estimate", calibrated_line)
        if not err_before_part.empty:
            err_before_line, = axis_right.plot(
                err_before_part["epsilon"],
                err_before_part["error_before"],
                color=FITCAM_CURVE_COLOR_MAP["error_before"],
                marker="x",
                linestyle="None",
                markersize=MARKER_SIZE_ERROR,
                label="error before calibration",
            )
            legend_map.setdefault("error before calibration", err_before_line)
        if not err_after_part.empty:
            err_after_line, = axis_right.plot(
                err_after_part["epsilon"],
                err_after_part["error_after"],
                color=FITCAM_CURVE_COLOR_MAP["error_after"],
                marker="x",
                linestyle="None",
                markersize=MARKER_SIZE_ERROR,
                label="error after calibration",
            )
            legend_map.setdefault("error after calibration", err_after_line)
        axis_right.axhline(0.0, color="tab:gray", linestyle=":", linewidth=1.2, alpha=0.8)

        axis.set_title(f"M = {m_value}MB", fontsize=TITLE_FONTSIZE)
        apply_outer_labels(axis, idx, nrows, ncols, "Epsilon", "Total I/Os")
        apply_outer_right_ylabel(axis_right, idx, ncols, "Error (Total I/Os)")
        axis.grid(alpha=0.3)
        apply_scientific_y(axis)
        apply_scientific_y(axis_right)
        apply_tick_font(axis)
        apply_tick_font(axis_right)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_path)
    save_legend_figure(list(legend_map.values()), list(legend_map.keys()), legend_output_path, ncol=3)


def write_fitcam_outputs(
    dataset_key: str,
    output_dir: Path,
    prefix: str,
    fitcam_root: Path,
    revision_log_dir: Path | None,
    policies: set[str] | None,
    m_values: set[int] | None,
    full_reference_df: pd.DataFrame,
) -> list[Path]:
    artifact_paths: list[Path] = []

    # Prefer revised full estimate logs produced by fitcam_local_runner.py:
    #   build/log/${DATA_FILE}_${POLICY}_revision.log
    revision_dir = (revision_log_dir or fitcam_root.parent).resolve()
    revision_sources = discover_revision_log_sources(revision_dir, dataset_key, policies)

    if revision_sources:
        for policy, revision_log in sorted(revision_sources.items()):
            policy_df = load_revision_curve_frame(
                revision_log=revision_log,
                dataset_key=dataset_key,
                policy=policy,
                full_reference_df=full_reference_df,
                allowed_m_values=m_values,
            )
            policy_df = filter_min_epsilon(policy_df)
            if policy_df.empty:
                continue

            merged_csv_path = output_dir / f"{prefix}_{policy.lower()}_fitcam_revision_curve_metrics.csv"
            figure_path = output_dir / f"{prefix}_{policy.lower()}_fitcam_revision_error_vs_epsilon.pdf"
            legend_path = output_dir / f"{prefix}_{policy.lower()}_fitcam_revision_error_vs_epsilon_legend.pdf"
            policy_df.to_csv(merged_csv_path, index=False)
            plot_fitcam_policy_curves(policy_df, figure_path, legend_path)
            artifact_paths.extend([merged_csv_path, figure_path, legend_path])
        return artifact_paths

    # Backward-compatible fallback: q30 corrected-vs-real CSVs.
    fitcam_dir = fitcam_root / dataset_key / "fit_output"
    if not fitcam_dir.exists():
        return []

    sources = discover_fitcam_curve_sources(fitcam_dir, dataset_key, policies)
    for policy, corrected_csv in sorted(sources.items()):
        policy_df = load_fitcam_curve_frame(
            corrected_csv=corrected_csv,
            fitcam_dir=fitcam_dir,
            dataset_key=dataset_key,
            policy=policy,
            full_reference_df=full_reference_df,
            allowed_m_values=m_values,
        )
        policy_df = filter_min_epsilon(policy_df)
        if policy_df.empty:
            continue

        merged_csv_path = output_dir / f"{prefix}_{policy.lower()}_fitcam_curve_metrics.csv"
        figure_path = output_dir / f"{prefix}_{policy.lower()}_fitcam_curve_error_vs_epsilon.pdf"
        legend_path = output_dir / f"{prefix}_{policy.lower()}_fitcam_curve_error_vs_epsilon_legend.pdf"
        policy_df.to_csv(merged_csv_path, index=False)
        plot_fitcam_policy_curves(policy_df, figure_path, legend_path)
        artifact_paths.extend([merged_csv_path, figure_path, legend_path])

    return artifact_paths

def plot_logical_io_vs_epsilon(
    dataset_df: pd.DataFrame,
    output_path: Path,
    legend_output_path: Path | None,
) -> None:
    dataset_df = filter_min_epsilon(dataset_df)
    m_values = sorted(dataset_df["M"].unique().tolist())
    fig, axes, nrows, ncols = make_axes(m_values)
    policies = ordered_policies(dataset_df)
    legend_map: dict[str, object] = {}

    for idx, (axis, m_value) in enumerate(zip(axes, m_values)):
        part_m = dataset_df[dataset_df["M"] == m_value].copy()
        for policy in policies:
            part = filter_min_epsilon(part_m[part_m["policy"] == policy]).sort_values("epsilon")
            if part.empty:
                continue
            color = COLOR_MAP.get(policy, None)
            actual_part = series_rows(part, "actual_total_logical_ios")
            estimated_part = series_rows(part, "estimated_total_logical_ios")
            if not actual_part.empty:
                actual_line, = axis.plot(
                    actual_part["epsilon"],
                    actual_part["actual_total_logical_ios"],
                    color=color,
                    marker="o",
                    linestyle="-",
                    linewidth=2,
                    markersize=MARKER_SIZE_PRIMARY,
                    label=f"{policy} actual",
                )
                legend_map.setdefault(f"{policy} actual", actual_line)
            if not estimated_part.empty:
                estimated_line, = axis.plot(
                    estimated_part["epsilon"],
                    estimated_part["estimated_total_logical_ios"],
                    color=color,
                    marker=None,
                    linestyle="--",
                    linewidth=2,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    label=f"{policy} estimated",
                )
                legend_map.setdefault(f"{policy} estimated", estimated_line)
        axis.set_title(f"M = {m_value}MB", fontsize=TITLE_FONTSIZE)
        apply_outer_labels(axis, idx, nrows, ncols, "Epsilon", "Total I/Os")
        axis.grid(alpha=0.3)
        apply_scientific_y(axis)
        apply_tick_font(axis)

    # dataset_key = str(dataset_df["dataset_key"].iloc[0])
    # fig.suptitle(f"{dataset_key}: Logical IOs vs Estimated IOs", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_path)
    if legend_output_path is not None:
        save_legend_figure(list(legend_map.values()), list(legend_map.keys()), legend_output_path)


def plot_hit_ratio_vs_epsilon(dataset_df: pd.DataFrame, output_path: Path, legend_output_path: Path) -> None:
    dataset_df = filter_min_epsilon(dataset_df)
    m_values = sorted(dataset_df["M"].unique().tolist())
    fig, axes, nrows, ncols = make_axes(m_values)
    policies = ordered_policies(dataset_df)
    legend_map: dict[str, object] = {}

    for idx, (axis, m_value) in enumerate(zip(axes, m_values)):
        part_m = dataset_df[dataset_df["M"] == m_value].copy()
        for policy in policies:
            part = filter_min_epsilon(part_m[part_m["policy"] == policy]).sort_values("epsilon")
            if part.empty:
                continue
            color = COLOR_MAP.get(policy, None)
            actual_part = series_rows(part, "actual_hit_ratio")
            estimated_part = series_rows(part, "estimated_hit_ratio")
            if not actual_part.empty:
                actual_line, = axis.plot(
                    actual_part["epsilon"],
                    actual_part["actual_hit_ratio"],
                    color=color,
                    marker="o",
                    linestyle="-",
                    linewidth=2,
                    markersize=MARKER_SIZE_PRIMARY,
                    label=f"{policy} actual",
                )
                legend_map.setdefault(f"{policy} actual", actual_line)
            if not estimated_part.empty:
                estimated_line, = axis.plot(
                    estimated_part["epsilon"],
                    estimated_part["estimated_hit_ratio"],
                    color=color,
                    marker=None,
                    linestyle="--",
                    linewidth=2,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    label=f"{policy} estimated",
                )
                legend_map.setdefault(f"{policy} estimated", estimated_line)
        axis.set_title(f"M = {m_value}MB", fontsize=TITLE_FONTSIZE)
        apply_outer_labels(axis, idx, nrows, ncols, "Epsilon", "Cache Hit Ratio")
        axis.grid(alpha=0.3)
        ratio_values = pd.concat([part_m["actual_hit_ratio"], part_m["estimated_hit_ratio"]], ignore_index=True)
        apply_percent_y(axis, ratio_values)
        apply_tick_font(axis)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_path)
    save_legend_figure(list(legend_map.values()), list(legend_map.keys()), legend_output_path)


def plot_estimated_io_and_throughput(
    dataset_df: pd.DataFrame,
    output_path: Path,
    legend_output_path: Path,
) -> None:
    dataset_df = filter_min_epsilon(dataset_df)
    m_values = sorted(dataset_df["M"].unique().tolist())
    fig, axes, nrows, ncols = make_axes(m_values)
    policies = ordered_policies(dataset_df)
    legend_map: dict[str, object] = {}

    for idx, (axis, m_value) in enumerate(zip(axes, m_values)):
        axis_right = axis.twinx()
        part_m = dataset_df[dataset_df["M"] == m_value].copy()

        for policy in policies:
            part = filter_min_epsilon(part_m[part_m["policy"] == policy]).sort_values("epsilon")
            if part.empty:
                continue
            color = COLOR_MAP.get(policy, None)
            estimated_part = series_rows(part, "estimated_total_logical_ios")
            throughput_part = series_rows(part, "actual_throughput_qps")
            if not estimated_part.empty:
                left_line, = axis.plot(
                    estimated_part["epsilon"],
                    estimated_part["estimated_total_logical_ios"],
                    color=color,
                    marker=None,
                    linestyle="--",
                    linewidth=2,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    label=f"{policy} estimated IO",
                )
                legend_map.setdefault(f"{policy} estimated IO", left_line)
            if not throughput_part.empty:
                right_line, = axis_right.plot(
                    throughput_part["epsilon"],
                    throughput_part["actual_throughput_qps"],
                    color=color,
                    marker="o",
                    linestyle="-",
                    linewidth=2,
                    markersize=MARKER_SIZE_PRIMARY,
                    label=f"{policy} throughput",
                )
                legend_map.setdefault(f"{policy} throughput", right_line)

        axis.set_title(f"M = {m_value}MB", fontsize=TITLE_FONTSIZE)
        apply_outer_labels(axis, idx, nrows, ncols, "Epsilon", "Total I/Os")
        apply_outer_right_ylabel(axis_right, idx, ncols, "Throughput (qps)")
        axis.grid(alpha=0.3)
        apply_scientific_y(axis)
        apply_scientific_y(axis_right)
        apply_tick_font(axis)
        apply_tick_font(axis_right)

    # dataset_key = str(dataset_df["dataset_key"].iloc[0])
    # fig.suptitle(f"{dataset_key}: Estimated Logical IO and Actual Throughput", fontsize=14)
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
            mean_abs_logical_io_pct_error=("logical_io_abs_pct_error", "mean"),
            mean_abs_hit_ratio_error=("hit_ratio_abs_error", "mean"),
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
    workload_types = sorted(dataset_df.get("workload_type", pd.Series(["unknown"])).dropna().astype(str).unique())
    lines = [
        "# Epsilon Benchmark Report",
        "",
        "## Inputs",
        "",
        f"- Dataset: {dataset_key}",
        f"- Workload type(s): {', '.join(workload_types)}",
        f"- Estimate logs: {len(estimate_paths)}",
        f"- Bench CSVs: {len(bench_paths)}",
        f"- Merged rows: {len(dataset_df)}",
        f"- Plotting style reference: build/visualize/pgm_logical_ios_visualization.ipynb",
        f"- Source code: utils/plot_epsilon_benchmarks.py",
        "",
        "## Schema Normalization",
        "",
        "- Estimated `cost` is interpreted as estimated average logical IOs per query.",
        "- Benchmark `queries` is used directly for point workloads; range benchmark `ranges` is normalized to `queries` internally.",
        "- Estimated total logical IOs are computed as `cost * queries` after this normalization.",
        "- Actual IO uses `logical_ios`, not `physical_ios`.",
        "- Throughput plots use `epsilon` on the x-axis, estimated logical IO on the left y-axis, and actual throughput on the right y-axis.",
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
                ["M", "policy", "epsilon_points", "mean_abs_logical_io_pct_error", "mean_abs_hit_ratio_error"],
                coverage_rows(dataset_df),
            ),
            "",
            "## Figures",
            "",
        ]
    )
    lines.extend(f"- {path}" for path in figure_paths)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset_outputs(
    dataset_df: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    fitcam_root: Path,
    revision_log_dir: Path | None,
    policies: set[str] | None,
    m_values: set[int] | None,
    skip_fitcam: bool,
    only_logical_ios: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    logical_io_path = output_dir / f"{prefix}_logical_ios_vs_epsilon.pdf"
    if only_logical_ios:
        plot_logical_io_vs_epsilon(dataset_df, logical_io_path, None)
        return [logical_io_path]

    merged_csv_path = output_dir / f"{prefix}_merged_metrics.csv"
    logical_io_legend_path = output_dir / f"{prefix}_logical_ios_vs_epsilon_legend.pdf"
    hit_ratio_path = output_dir / f"{prefix}_hit_ratio_vs_epsilon.pdf"
    hit_ratio_legend_path = output_dir / f"{prefix}_hit_ratio_vs_epsilon_legend.pdf"
    throughput_path = output_dir / f"{prefix}_estimated_io_throughput_vs_epsilon.pdf"
    throughput_legend_path = output_dir / f"{prefix}_estimated_io_throughput_vs_epsilon_legend.pdf"
    report_path = output_dir / f"{prefix}_report.md"

    dataset_df.to_csv(merged_csv_path, index=False)
    plot_logical_io_vs_epsilon(dataset_df, logical_io_path, logical_io_legend_path)
    plot_hit_ratio_vs_epsilon(dataset_df, hit_ratio_path, hit_ratio_legend_path)
    plot_estimated_io_and_throughput(dataset_df, throughput_path, throughput_legend_path)
    fitcam_artifact_paths: list[Path] = []
    if not skip_fitcam:
        fitcam_m_values = m_values or set(dataset_df["M"].dropna().astype(int).unique().tolist())
        fitcam_artifact_paths = write_fitcam_outputs(
            dataset_key=str(dataset_df["dataset_key"].iloc[0]),
            output_dir=output_dir,
            prefix=prefix,
            fitcam_root=fitcam_root,
            revision_log_dir=revision_log_dir,
            policies=policies,
            m_values=fitcam_m_values,
            full_reference_df=dataset_df[["policy", "M", "epsilon", "queries", "actual_total_logical_ios"]].copy(),
        )
    write_report(
        dataset_df=dataset_df,
        output_path=report_path,
        estimate_paths=[Path(path) for path in sorted(dataset_df["estimate_source"].dropna().unique().tolist())],
        bench_paths=[Path(path) for path in sorted(dataset_df["bench_source"].dropna().unique().tolist())],
        figure_paths=[
            logical_io_path,
            logical_io_legend_path,
            hit_ratio_path,
            hit_ratio_legend_path,
            throughput_path,
            throughput_legend_path,
            merged_csv_path,
            *fitcam_artifact_paths,
        ],
    )
    return [
        merged_csv_path,
        logical_io_path,
        logical_io_legend_path,
        hit_ratio_path,
        hit_ratio_legend_path,
        throughput_path,
        throughput_legend_path,
        *fitcam_artifact_paths,
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
    merged = filter_min_epsilon(merged)
    if merged.empty:
        raise ValueError(f"No rows remained after filtering epsilon >= {MIN_PLOTTED_EPSILON}.")
    dataset_keys = sorted(merged["dataset_key"].dropna().unique().tolist())
    multiple_datasets = len(dataset_keys) > 1

    for dataset_key in dataset_keys:
        dataset_df = merged[merged["dataset_key"] == dataset_key].copy()
        prefix = build_prefix(str(dataset_key), args.output_prefix, multiple_datasets)
        artifact_paths = write_dataset_outputs(
            dataset_df=dataset_df,
            output_dir=args.output_dir,
            prefix=prefix,
            fitcam_root=args.fitcam_root,
            revision_log_dir=args.revision_log_dir,
            policies=policies,
            m_values=m_values,
            skip_fitcam=args.skip_fitcam,
            only_logical_ios=args.only_logical_ios,
        )
        for path in artifact_paths:
            print(path)


if __name__ == "__main__":
    main()
