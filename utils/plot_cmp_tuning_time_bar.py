#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
            "Compare epsilon-tuning runtime from sim and log files with grouped bars "
            "(6 bars per dataset size: FIFO/LRU/LFU x sim/log)."
        )
    )
    parser.add_argument(
        "--cmp-dir",
        type=Path,
        default=Path("build/log/cmp"),
        help="Directory containing books_<size>M_M<M>_summary_sim.csv and *_<POLICY>.log files.",
    )
    parser.add_argument(
        "--dataset-sizes",
        nargs="+",
        type=int,
        default=[10, 100, 200],
        help="Dataset sizes used to build prefixes books_<size>M_M<M>_summary.",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=10,
        help="Memory budget M in filename pattern books_<size>M_M<M>_summary.",
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
        help="Output png path. Default: <cmp-dir>/books_M<M>_epsilon_tuning_time_compare.png",
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


def build_raw_frame(cmp_dir: Path, dataset_sizes: list[int], m_value: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for size in dataset_sizes:
        prefix = f"books_{size}M_M{m_value}_summary"
        sim_path = cmp_dir / f"{prefix}_sim.csv"
        if not sim_path.exists():
            raise FileNotFoundError(f"Missing file: {sim_path}")
        sim_df = load_sim(sim_path)
        sim_df["dataset_size"] = size
        frames.append(sim_df)

        for policy in POLICIES:
            log_path = cmp_dir / f"{prefix}_{policy}.log"
            if not log_path.exists():
                raise FileNotFoundError(f"Missing file: {log_path}")
            log_df = load_log(log_path, policy=policy)
            log_df["dataset_size"] = size
            frames.append(log_df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df[all_df["policy"].isin(POLICIES)].copy()

    duplicated = all_df.duplicated(
        subset=["dataset_size", "policy", "method", "epsilon"], keep=False
    )
    if duplicated.any():
        d = all_df.loc[
            duplicated, ["dataset_size", "policy", "method", "epsilon"]
        ].drop_duplicates()
        raise ValueError(
            "Duplicate rows for the same dataset/policy/method/epsilon: "
            f"{d.head(8).to_dict('records')}"
        )
    return all_df


def epsilon_space_from_intersection(df: pd.DataFrame) -> list[int]:
    eps_sets: list[set[int]] = []
    for size in sorted(df["dataset_size"].unique().tolist()):
        for policy in POLICIES:
            for method in METHODS:
                part = df[
                    (df["dataset_size"] == size)
                    & (df["policy"] == policy)
                    & (df["method"] == method)
                ]
                if part.empty:
                    raise ValueError(
                        f"Missing series for dataset={size}, policy={policy}, method={method}."
                    )
                eps_sets.append(set(part["epsilon"].astype(int).tolist()))
    common = sorted(set.intersection(*eps_sets))
    if not common:
        raise ValueError("No shared epsilon points across all dataset/policy/method series.")
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
    return epsilon_space_from_range(
        start=int(args.eps_start),
        end=int(args.eps_end),
        step=int(args.eps_step),
    )


def validate_epsilon_coverage(df: pd.DataFrame, eps: list[int]) -> None:
    eps_set = set(eps)
    missing_records: list[dict[str, object]] = []
    for size in sorted(df["dataset_size"].unique().tolist()):
        for policy in POLICIES:
            for method in METHODS:
                part = df[
                    (df["dataset_size"] == size)
                    & (df["policy"] == policy)
                    & (df["method"] == method)
                ]
                found = set(part["epsilon"].astype(int).tolist())
                missing = sorted(eps_set - found)
                if missing:
                    missing_records.append(
                        {
                            "dataset_size": size,
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
        filtered.groupby(["dataset_size", "policy", "method"], as_index=False)
        .agg(total_time_s=("time_s", "sum"), epsilon_points=("epsilon", "count"))
        .sort_values(["dataset_size", "policy", "method"])
    )
    return grouped


def make_bar_values(
    agg_df: pd.DataFrame, dataset_sizes: list[int]
) -> dict[tuple[str, str], list[float]]:
    out: dict[tuple[str, str], list[float]] = {}
    for key in BAR_ORDER:
        policy, method = key
        values: list[float] = []
        for size in dataset_sizes:
            part = agg_df[
                (agg_df["dataset_size"] == size)
                & (agg_df["policy"] == policy)
                & (agg_df["method"] == method)
            ]
            if part.empty:
                values.append(np.nan)
            else:
                values.append(float(part["total_time_s"].iloc[0]))
        out[key] = values
    return out


def draw_bar_chart(
    agg_df: pd.DataFrame, dataset_sizes: list[int], eps: list[int], output_path: Path
) -> None:
    x = np.arange(len(dataset_sizes), dtype=float)
    n_bars = len(BAR_ORDER)
    bar_width = 0.12

    fig, ax = plt.subplots(figsize=(12, 7))
    bar_values = make_bar_values(agg_df, dataset_sizes)

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

    ax.set_xlabel("Dataset Size", fontsize=XLABEL_FONTSIZE)
    ax.set_ylabel(
        f"Total Time (s)",
        fontsize=YLABEL_FONTSIZE,
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{size}MB" for size in dataset_sizes], fontsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", **TICK_FONT)
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def default_output_path(cmp_dir: Path, m_value: int) -> Path:
    return cmp_dir / f"books_M{m_value}_epsilon_tuning_time_compare.png"


def default_legend_output_path(cmp_dir: Path, m_value: int) -> Path:
    return cmp_dir / f"books_M{m_value}_epsilon_tuning_time_compare_legend.png"


def default_summary_csv_path(cmp_dir: Path, m_value: int) -> Path:
    return cmp_dir / f"books_M{m_value}_epsilon_tuning_time_compare.csv"


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
    dataset_sizes = list(args.dataset_sizes)

    raw_df = build_raw_frame(
        cmp_dir=cmp_dir,
        dataset_sizes=dataset_sizes,
        m_value=int(args.m),
    )
    eps = resolve_epsilon_space(raw_df, args)
    validate_epsilon_coverage(raw_df, eps)

    agg_df = aggregate_total_time(raw_df, eps)
    output_path = (args.output or default_output_path(cmp_dir, int(args.m))).resolve()
    legend_output_path = (
        args.legend_output or default_legend_output_path(cmp_dir, int(args.m))
    ).resolve()
    draw_bar_chart(agg_df, dataset_sizes=dataset_sizes, eps=eps, output_path=output_path)
    save_legend_figure(legend_output_path)

    summary_csv = (args.summary_csv or default_summary_csv_path(cmp_dir, int(args.m))).resolve()
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
    print(f"epsilon_space={eps}")


if __name__ == "__main__":
    main()
