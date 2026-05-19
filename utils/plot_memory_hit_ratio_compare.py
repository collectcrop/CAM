#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
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

XLABEL_FONTSIZE = 25
YLABEL_FONTSIZE = 25
TICK_FONTSIZE = 20
TICK_FONT = {"labelsize": TICK_FONTSIZE}

POLICIES = ("FIFO", "LRU", "LFU")
METHOD_ORDER = ("sim", "log")
METHOD_LABELS = {
    "sim": "simulate",
    "log": "estimate",
}
POLICY_COLORS = {
    "FIFO": "tab:blue",
    "LRU": "tab:orange",
    "LFU": "tab:green",
}
METHOD_STYLES = {
    "sim": {"linestyle": "-", "marker": "s"},
    "log": {"linestyle": "-.", "marker": "^"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot cache hit ratio vs memory (M) for fixed epsilon, "
            "using 200M memory-sweep sim csv and per-M summary log files."
        )
    )
    parser.add_argument(
        "--cmp-dir",
        type=Path,
        default=Path("build/log/cmp"),
        help="Directory containing sim and log files.",
    )
    parser.add_argument(
        "--sim-csv",
        type=Path,
        default=Path("build/log/cmp/books_10M_eps32_memory_sweep_sim_merged.csv"),
        help="Sim merged CSV path.",
    )
    parser.add_argument(
        "--dataset-tag",
        default="books_10M",
        help="Log filename prefix tag, e.g. books_200M or books_200MB.",
    )
    parser.add_argument(
        "--memory-list",
        nargs="+",
        type=int,
        default=[10, 20, 40, 60],
        help="Memory budgets in MB for x-axis and log-file lookup.",
    )
    parser.add_argument(
        "--epsilon",
        type=int,
        default=None,
        help="Fixed epsilon. If omitted, infer from sim csv (must be unique).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output png path. Default: <cmp-dir>/<dataset-tag>_eps<E>_memory_hit_ratio_compare.pdf",
    )
    parser.add_argument(
        "--legend-output",
        type=Path,
        default=None,
        help="Optional output path for a standalone legend figure.",
    )
    return parser.parse_args()


def infer_single_epsilon(sim_df: pd.DataFrame) -> int:
    eps = sorted(pd.to_numeric(sim_df["epsilon"], errors="coerce").dropna().astype(int).unique().tolist())
    if len(eps) != 1:
        raise ValueError(f"sim csv should contain exactly one epsilon when --epsilon is not set, got {eps}")
    return eps[0]


def load_sim(sim_path: Path, memory_list: list[int], epsilon: int | None) -> tuple[pd.DataFrame, int]:
    if not sim_path.exists():
        raise FileNotFoundError(f"Missing sim file: {sim_path}")
    sim = pd.read_csv(sim_path)
    required = {"epsilon", "policy", "memory_budget_bytes", "global_hit_ratio"}
    if not required.issubset(sim.columns):
        raise ValueError(f"Unexpected sim columns in {sim_path}; need {sorted(required)}")

    eps = infer_single_epsilon(sim) if epsilon is None else int(epsilon)
    part = sim.copy()
    part["epsilon"] = pd.to_numeric(part["epsilon"], errors="coerce")
    part = part[part["epsilon"] == eps].copy()
    if part.empty:
        raise ValueError(f"No sim rows for epsilon={eps} in {sim_path}")

    part["policy"] = part["policy"].astype(str).str.upper()
    part["memory_mb"] = (pd.to_numeric(part["memory_budget_bytes"], errors="coerce") / (1 << 20)).round().astype(int)
    part["hit_ratio"] = pd.to_numeric(part["global_hit_ratio"], errors="coerce")
    part["method"] = "sim"
    part = part[part["policy"].isin(POLICIES) & part["memory_mb"].isin(memory_list)]
    part = part.dropna(subset=["hit_ratio"])
    dedup = part.duplicated(subset=["policy", "memory_mb"], keep=False)
    if dedup.any():
        d = part.loc[dedup, ["policy", "memory_mb"]].drop_duplicates().to_dict("records")
        raise ValueError(f"Duplicate sim rows for (policy,memory_mb): {d}")
    return part.loc[:, ["policy", "memory_mb", "hit_ratio", "method"]], eps


def candidate_log_paths(cmp_dir: Path, dataset_tag: str, memory_mb: int, policy: str) -> list[Path]:
    tags = [dataset_tag]
    if dataset_tag.endswith("MB"):
        tags.append(dataset_tag[:-1])  # books_200MB -> books_200M
    if dataset_tag.endswith("M") and not dataset_tag.endswith("MB"):
        tags.append(dataset_tag + "B")  # books_200M -> books_200MB
    unique_tags = list(dict.fromkeys(tags))
    return [cmp_dir / f"{tag}_M{memory_mb}_summary_{policy}.log" for tag in unique_tags]


def load_log(cmp_dir: Path, dataset_tag: str, memory_list: list[int], epsilon: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for memory_mb in memory_list:
        for policy in POLICIES:
            selected: Path | None = None
            for p in candidate_log_paths(cmp_dir, dataset_tag, memory_mb, policy):
                if p.exists():
                    selected = p
                    break
            if selected is None:
                raise FileNotFoundError(
                    f"Missing log file for M={memory_mb}, policy={policy}. "
                    f"Tried: {[str(p) for p in candidate_log_paths(cmp_dir, dataset_tag, memory_mb, policy)]}"
                )

            df = pd.read_csv(selected)
            required = {"epsilon", "ratio"}
            if not required.issubset(df.columns):
                raise ValueError(f"Unexpected log columns in {selected}; need {sorted(required)}")
            df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce")
            point = df[df["epsilon"] == epsilon].copy()
            if point.empty:
                raise ValueError(f"No epsilon={epsilon} row in {selected}")
            if len(point) > 1:
                raise ValueError(f"Duplicate epsilon={epsilon} rows in {selected}")
            rows.append(
                {
                    "policy": policy,
                    "memory_mb": memory_mb,
                    "hit_ratio": float(point["ratio"].iloc[0]),
                    "method": "log",
                }
            )
    return pd.DataFrame(rows)


def validate_full_grid(df: pd.DataFrame, memory_list: list[int]) -> None:
    missing = []
    for policy in POLICIES:
        for method in METHOD_ORDER:
            for memory_mb in memory_list:
                part = df[
                    (df["policy"] == policy)
                    & (df["method"] == method)
                    & (df["memory_mb"] == memory_mb)
                ]
                if part.empty:
                    missing.append((policy, method, memory_mb))
    if missing:
        raise ValueError(f"Missing points: {missing[:12]}")


def plot_hit_ratio(merged: pd.DataFrame, memory_list: list[int], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    for policy in POLICIES:
        for method in METHOD_ORDER:
            part = merged[(merged["policy"] == policy) & (merged["method"] == method)].copy()
            if part.empty:
                continue
            part = part.sort_values("memory_mb")
            style = METHOD_STYLES[method]
            ax.plot(
                part["memory_mb"],
                part["hit_ratio"],
                label=f"{policy} - {METHOD_LABELS[method]}",
                color=POLICY_COLORS[policy],
                linewidth=2,
                markersize=7,
                **style,
            )

    ax.set_xlabel("Memory Budget (MB)", fontsize=XLABEL_FONTSIZE)
    ax.set_ylabel("Cache Hit Ratio", fontsize=YLABEL_FONTSIZE)
    ax.set_xticks(memory_list)
    ax.set_xticklabels([f"{m}" for m in memory_list])
    ax.tick_params(axis="both", **TICK_FONT)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def default_legend_output_path(cmp_dir: Path, dataset_tag: str, epsilon: int) -> Path:
    return cmp_dir / f"{dataset_tag}_eps{epsilon}_memory_hit_ratio_compare_legend.pdf"


def build_legend_handles() -> tuple[list[Line2D], list[str]]:
    handles: list[Line2D] = []
    labels: list[str] = []
    for policy in POLICIES:
        for method in METHOD_ORDER:
            style = METHOD_STYLES[method]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=POLICY_COLORS[policy],
                    linewidth=2,
                    markersize=5,
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                )
            )
            labels.append(f"{policy} - {METHOD_LABELS[method]}")
    return handles, labels


def save_legend_figure(output: Path) -> None:
    handles, labels = build_legend_handles()
    fig = plt.figure(figsize=(12, 1.5))
    legend = fig.legend(
        handles,
        labels,
        ncol=3,
        fontsize=TICK_FONTSIZE,
        loc="center",
        frameon=False,
    )
    for text in legend.get_texts():
        text.set_fontsize(TICK_FONTSIZE)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cmp_dir = args.cmp_dir.resolve()
    memory_list = list(args.memory_list)

    sim_df, eps = load_sim(args.sim_csv.resolve(), memory_list=memory_list, epsilon=args.epsilon)
    log_df = load_log(cmp_dir=cmp_dir, dataset_tag=args.dataset_tag, memory_list=memory_list, epsilon=eps)
    merged = pd.concat([sim_df, log_df], ignore_index=True)
    validate_full_grid(merged, memory_list=memory_list)
    merged = merged.sort_values(["policy", "method", "memory_mb"]).reset_index(drop=True)

    output = args.output
    if output is None:
        output = cmp_dir / f"{args.dataset_tag}_eps{eps}_memory_hit_ratio_compare.pdf"
    output = output.resolve()
    legend_output = (
        args.legend_output
        or default_legend_output_path(cmp_dir, args.dataset_tag, eps)
    ).resolve()
    plot_hit_ratio(merged=merged, memory_list=memory_list, output=output)
    save_legend_figure(legend_output)
    print(output)
    print(legend_output)


if __name__ == "__main__":
    main()
