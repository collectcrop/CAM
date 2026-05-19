#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

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

POLICIES = ("FIFO", "LRU", "LFU")
METHODS = ("sim", "log")
BAR_ORDER = (
    ("FIFO", "sim"),
    ("FIFO", "log"),
    ("LRU", "sim"),
    ("LRU", "log"),
    ("LFU", "sim"),
    ("LFU", "log"),
)
POLICY_COLORS = {
    "FIFO": "tab:blue",
    "LRU": "tab:orange",
    "LFU": "tab:green",
}
METHOD_LABELS = {
    "sim": "simulate",
    "log": "estimate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare epsilon-tuning runtime across query sizes with grouped bars "
            "(6 bars per query size: FIFO/LRU/LFU x sim/log)."
        )
    )
    parser.add_argument(
        "--cmp-dir",
        type=Path,
        default=Path("build/log/cmp"),
        help="Directory containing books_10M_M10_<q>Mquery_summary_* files.",
    )
    parser.add_argument(
        "--dataset-prefix",
        default="books_10M_M10",
        help="Common filename prefix before _<q>Mquery_summary_*.",
    )
    parser.add_argument(
        "--query-sizes",
        nargs="*",
        type=int,
        default=None,
        help="Query sizes in millions. Default: auto-discover from sim files.",
    )
    parser.add_argument(
        "--eps-start",
        type=int,
        default=None,
        help="Optional epsilon range start (inclusive).",
    )
    parser.add_argument(
        "--eps-end",
        type=int,
        default=None,
        help="Optional epsilon range end (inclusive).",
    )
    parser.add_argument(
        "--eps-step",
        type=int,
        default=None,
        help="Optional epsilon range step (positive integer).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output png path. Default: <cmp-dir>/<dataset-prefix>_querysize_epsilon_tuning_time_compare.png",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional output csv path for aggregated totals.",
    )
    parser.add_argument(
        "--legend-output",
        type=Path,
        default=None,
        help="Optional output path for a standalone legend figure.",
    )
    return parser.parse_args()


def load_sim(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"epsilon", "policy", "simulate_wall_ns", "index_build_ns"}
    if not required.issubset(df.columns):
        raise ValueError(f"Unexpected sim format: {path}")
    out = df.loc[:, ["epsilon", "policy", "simulate_wall_ns", "index_build_ns"]].copy()
    out["epsilon"] = pd.to_numeric(out["epsilon"], errors="coerce")
    out["policy"] = out["policy"].astype(str).str.upper()
    simulate_wall_ns = pd.to_numeric(out["simulate_wall_ns"], errors="coerce")
    index_build_ns = pd.to_numeric(out["index_build_ns"], errors="coerce")
    out["time_s"] = (simulate_wall_ns + index_build_ns) / 1e9
    out["method"] = "sim"
    out = out.dropna(subset=["epsilon", "policy", "time_s"])
    out["epsilon"] = out["epsilon"].astype(int)
    return out.loc[:, ["epsilon", "policy", "method", "time_s"]]


def load_log(path: Path, policy: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"epsilon", "time"}
    if not required.issubset(df.columns):
        raise ValueError(f"Unexpected log format: {path}")
    out = df.loc[:, ["epsilon", "time"]].copy()
    out["epsilon"] = pd.to_numeric(out["epsilon"], errors="coerce")
    out["time_s"] = pd.to_numeric(out["time"], errors="coerce")
    out["policy"] = policy
    out["method"] = "log"
    out = out.dropna(subset=["epsilon", "time_s"])
    out["epsilon"] = out["epsilon"].astype(int)
    return out.loc[:, ["epsilon", "policy", "method", "time_s"]]


def candidate_sim_paths(cmp_dir: Path, dataset_prefix: str, query_size_m: int) -> list[Path]:
    paths = [cmp_dir / f"{dataset_prefix}_{query_size_m}Mquery_summary_sim.csv"]
    if query_size_m == 1:
        paths.append(cmp_dir / f"{dataset_prefix}_summary_sim.csv")
    return paths


def candidate_log_paths(
    cmp_dir: Path, dataset_prefix: str, query_size_m: int, policy: str
) -> list[Path]:
    paths = [cmp_dir / f"{dataset_prefix}_{query_size_m}Mquery_summary_{policy}.log"]
    if query_size_m == 1:
        paths.append(cmp_dir / f"{dataset_prefix}_summary_{policy}.log")
    return paths


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def autodetect_query_sizes(cmp_dir: Path, dataset_prefix: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(dataset_prefix)}_(\d+)Mquery_summary_sim\.csv$")
    sizes: set[int] = set()
    for path in cmp_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match:
            sizes.add(int(match.group(1)))
    if not sizes:
        raise FileNotFoundError(
            f"No files matching {dataset_prefix}_<q>Mquery_summary_sim.csv found in {cmp_dir}"
        )
    return sorted(sizes)


def build_raw_frame(cmp_dir: Path, dataset_prefix: str, query_sizes: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for query_size_m in query_sizes:
        sim_path = first_existing(candidate_sim_paths(cmp_dir, dataset_prefix, query_size_m))
        if sim_path is None:
            raise FileNotFoundError(
                f"Missing sim file for {query_size_m}M query. "
                f"Tried: {[str(p) for p in candidate_sim_paths(cmp_dir, dataset_prefix, query_size_m)]}"
            )
        sim_df = load_sim(sim_path)
        sim_df["query_size_m"] = query_size_m
        frames.append(sim_df)

        for policy in POLICIES:
            log_path = first_existing(candidate_log_paths(cmp_dir, dataset_prefix, query_size_m, policy))
            if log_path is None:
                raise FileNotFoundError(
                    f"Missing log file for {query_size_m}M query, policy={policy}. "
                    f"Tried: {[str(p) for p in candidate_log_paths(cmp_dir, dataset_prefix, query_size_m, policy)]}"
                )
            log_df = load_log(log_path, policy=policy)
            log_df["query_size_m"] = query_size_m
            frames.append(log_df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df[all_df["policy"].isin(POLICIES)].copy()
    duplicated = all_df.duplicated(subset=["query_size_m", "policy", "method", "epsilon"], keep=False)
    if duplicated.any():
        d = all_df.loc[duplicated, ["query_size_m", "policy", "method", "epsilon"]].drop_duplicates()
        raise ValueError(
            "Duplicate rows for the same query-size/policy/method/epsilon: "
            f"{d.head(8).to_dict('records')}"
        )
    return all_df


def epsilon_space_from_intersection(df: pd.DataFrame) -> list[int]:
    eps_sets: list[set[int]] = []
    for query_size_m in sorted(df["query_size_m"].unique().tolist()):
        for policy in POLICIES:
            for method in METHODS:
                part = df[
                    (df["query_size_m"] == query_size_m)
                    & (df["policy"] == policy)
                    & (df["method"] == method)
                ]
                if part.empty:
                    raise ValueError(
                        f"Missing series for query_size={query_size_m}M, policy={policy}, method={method}."
                    )
                eps_sets.append(set(part["epsilon"].astype(int).tolist()))
    common = sorted(set.intersection(*eps_sets))
    if not common:
        raise ValueError("No shared epsilon points across all query-size/policy/method series.")
    return common


def epsilon_space_from_range(start: int, end: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError("--eps-step must be a positive integer.")
    if end < start:
        raise ValueError("--eps-end must be >= --eps-start.")
    return list(range(start, end + 1, step))


def resolve_epsilon_space(df: pd.DataFrame, args: argparse.Namespace) -> list[int]:
    explicit = [args.eps_start, args.eps_end, args.eps_step]
    if all(v is None for v in explicit):
        return epsilon_space_from_intersection(df)
    if any(v is None for v in explicit):
        raise ValueError("When setting epsilon range, provide --eps-start, --eps-end and --eps-step together.")
    return epsilon_space_from_range(int(args.eps_start), int(args.eps_end), int(args.eps_step))


def validate_epsilon_coverage(df: pd.DataFrame, eps: list[int]) -> None:
    eps_set = set(eps)
    missing_records: list[dict[str, object]] = []
    for query_size_m in sorted(df["query_size_m"].unique().tolist()):
        for policy in POLICIES:
            for method in METHODS:
                part = df[
                    (df["query_size_m"] == query_size_m)
                    & (df["policy"] == policy)
                    & (df["method"] == method)
                ]
                found = set(part["epsilon"].astype(int).tolist())
                missing = sorted(eps_set - found)
                if missing:
                    missing_records.append(
                        {
                            "query_size_m": query_size_m,
                            "policy": policy,
                            "method": method,
                            "missing_count": len(missing),
                            "missing_preview": missing[:8],
                        }
                    )
    if missing_records:
        raise ValueError(
            "Some series do not fully cover the requested epsilon space. "
            f"Examples: {missing_records[:6]}"
        )


def aggregate_total_time(df: pd.DataFrame, eps: list[int]) -> pd.DataFrame:
    filtered = df[df["epsilon"].isin(eps)].copy()
    grouped = (
        filtered.groupby(["query_size_m", "policy", "method"], as_index=False)
        .agg(total_time_s=("time_s", "sum"), epsilon_points=("epsilon", "count"))
        .sort_values(["query_size_m", "policy", "method"])
    )
    return grouped


def make_bar_values(agg_df: pd.DataFrame, query_sizes: list[int]) -> dict[tuple[str, str], list[float]]:
    out: dict[tuple[str, str], list[float]] = {}
    for key in BAR_ORDER:
        policy, method = key
        values: list[float] = []
        for query_size_m in query_sizes:
            part = agg_df[
                (agg_df["query_size_m"] == query_size_m)
                & (agg_df["policy"] == policy)
                & (agg_df["method"] == method)
            ]
            values.append(float(part["total_time_s"].iloc[0]) if not part.empty else np.nan)
        out[key] = values
    return out


def draw_bar_chart(
    agg_df: pd.DataFrame, query_sizes: list[int], output_path: Path
) -> None:
    x = np.arange(len(query_sizes), dtype=float)
    n_bars = len(BAR_ORDER)
    bar_width = 0.12

    fig, ax = plt.subplots(figsize=(12, 7))
    bar_values = make_bar_values(agg_df, query_sizes)

    for idx, (policy, method) in enumerate(BAR_ORDER):
        offset = (idx - (n_bars - 1) / 2.0) * bar_width
        values = bar_values[(policy, method)]
        color = POLICY_COLORS[policy]
        label = f"{policy}-{METHOD_LABELS[method]}"
        if method == "sim":
            ax.bar(
                x + offset,
                values,
                width=bar_width,
                label=label,
                color=color,
                edgecolor="black",
                linewidth=0.8,
                alpha=0.9,
            )
        else:
            ax.bar(
                x + offset,
                values,
                width=bar_width,
                label=label,
                color=color,
                edgecolor="black",
                linewidth=0.8,
                alpha=0.35,
                hatch="//",
            )

    ax.set_xlabel("Query Size", fontsize=XLABEL_FONTSIZE)
    ax.set_ylabel("Total Time (s)", fontsize=YLABEL_FONTSIZE)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{q}M" for q in query_sizes], fontsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", **TICK_FONT)
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def default_output_path(cmp_dir: Path, dataset_prefix: str) -> Path:
    return cmp_dir / f"{dataset_prefix}_querysize_epsilon_tuning_time_compare.pdf"


def default_legend_output_path(cmp_dir: Path, dataset_prefix: str) -> Path:
    return cmp_dir / f"{dataset_prefix}_querysize_epsilon_tuning_time_compare_legend.pdf"


def default_summary_csv_path(cmp_dir: Path, dataset_prefix: str) -> Path:
    return cmp_dir / f"{dataset_prefix}_querysize_epsilon_tuning_time_compare.csv"


def build_legend_handles() -> tuple[list[Patch], list[str]]:
    handles: list[Patch] = []
    labels: list[str] = []
    for policy, method in BAR_ORDER:
        color = POLICY_COLORS[policy]
        label = f"{policy}-{METHOD_LABELS[method]}"
        patch = Patch(
            facecolor=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9 if method == "sim" else 0.35,
            hatch=None if method == "sim" else "//",
        )
        handles.append(patch)
        labels.append(label)
    return handles, labels


def save_legend_figure(output_path: Path) -> None:
    handles, labels = build_legend_handles()
    fig = plt.figure(figsize=(12, 1.5))
    legend = fig.legend(
        handles,
        labels,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
    )
    for text in legend.get_texts():
        text.set_fontsize(LEGEND_FONTSIZE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cmp_dir = args.cmp_dir.resolve()
    query_sizes = sorted(args.query_sizes) if args.query_sizes else autodetect_query_sizes(cmp_dir, args.dataset_prefix)

    raw_df = build_raw_frame(cmp_dir=cmp_dir, dataset_prefix=args.dataset_prefix, query_sizes=query_sizes)
    eps = resolve_epsilon_space(raw_df, args)
    validate_epsilon_coverage(raw_df, eps)

    agg_df = aggregate_total_time(raw_df, eps)
    output_path = (args.output or default_output_path(cmp_dir, args.dataset_prefix)).resolve()
    legend_output_path = (
        args.legend_output or default_legend_output_path(cmp_dir, args.dataset_prefix)
    ).resolve()
    draw_bar_chart(agg_df, query_sizes=query_sizes, output_path=output_path)
    save_legend_figure(legend_output_path)

    summary_csv = (args.summary_csv or default_summary_csv_path(cmp_dir, args.dataset_prefix)).resolve()
    agg_out = agg_df.copy()
    agg_out["epsilon_min"] = min(eps)
    agg_out["epsilon_max"] = max(eps)
    agg_out["epsilon_step_info"] = (
        str(eps[1] - eps[0]) if len(eps) > 1 and np.all(np.diff(eps) == np.diff(eps)[0]) else "irregular"
    )
    agg_out.to_csv(summary_csv, index=False)

    print(output_path)
    print(legend_output_path)
    print(summary_csv)
    print(f"query_sizes={query_sizes}")
    print(f"epsilon_space={eps}")


if __name__ == "__main__":
    main()
