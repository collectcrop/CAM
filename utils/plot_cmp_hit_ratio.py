#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Times", "Computer Modern Roman"],
                "axes.unicode_minus": False,
                "text.latex.preamble": r"\usepackage{amsmath}",
            })
# global fonts
TITLE_FONTSIZE = 25     
XLABEL_FONTSIZE = 20    
YLABEL_FONTSIZE = 20    
TICK_FONTSIZE   = 15    
TICK_FONT = {"labelsize": TICK_FONTSIZE}

POLICIES = ("FIFO", "LRU", "LFU")
METHOD_ORDER = ("simple_sim", "log_estimate")
METHOD_LABELS = {
    "simple_sim": "simulate",
    "log_estimate": "estimate",
}
POLICY_COLORS = {
    "FIFO": "tab:blue",
    "LRU": "tab:orange",
    "LFU": "tab:green",
}
METHOD_STYLES = {
    "simple_sim": {"linestyle": "-", "marker": "s"},
    "log_estimate": {"linestyle": "-.", "marker": "^"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cache hit ratio vs epsilon from cmp summary files."
    )
    parser.add_argument(
        "--cmp-dir",
        type=Path,
        default=Path("build/log/cmp"),
        help="Directory containing *_summary_sim.csv and *_summary_<POLICY>.log files.",
    )
    parser.add_argument(
        "--prefix",
        default="books_10M_M10_summary",
        help="Common filename prefix before suffixes like _sim.csv / _FIFO.log.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Default: <cmp-dir>/<prefix>_hit_ratio_compare.png",
    )
    parser.add_argument(
        "--epsilon-step",
        type=int,
        default=4,
        help="Sample epsilon points by fixed interval, keep eps where epsilon %% step == 0.",
    )
    return parser.parse_args()


def load_summary_csv(path: Path, method: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"epsilon", "policy", "global_hit_ratio"}
    if not required.issubset(df.columns):
        raise ValueError(f"Unexpected columns in {path}; need {sorted(required)}")

    out = df.loc[:, ["epsilon", "policy", "global_hit_ratio"]].copy()
    out["epsilon"] = pd.to_numeric(out["epsilon"], errors="coerce")
    out["policy"] = out["policy"].astype(str).str.upper()
    out["hit_ratio"] = pd.to_numeric(out["global_hit_ratio"], errors="coerce")
    out["method"] = method
    out = out.dropna(subset=["epsilon", "policy", "hit_ratio"])
    out["epsilon"] = out["epsilon"].astype(int)
    return out.loc[:, ["epsilon", "policy", "hit_ratio", "method"]]


def load_policy_log(path: Path, policy: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"epsilon", "ratio"}
    if not required.issubset(df.columns):
        raise ValueError(f"Unexpected columns in {path}; need {sorted(required)}")

    out = df.loc[:, ["epsilon", "ratio"]].copy()
    out["epsilon"] = pd.to_numeric(out["epsilon"], errors="coerce")
    out["hit_ratio"] = pd.to_numeric(out["ratio"], errors="coerce")
    out["policy"] = policy
    out["method"] = "log_estimate"
    out = out.dropna(subset=["epsilon", "hit_ratio"])
    out["epsilon"] = out["epsilon"].astype(int)
    return out.loc[:, ["epsilon", "policy", "hit_ratio", "method"]]


def build_merged_frame(cmp_dir: Path, prefix: str, epsilon_step: int = 4) -> pd.DataFrame:
    sim_path = cmp_dir / f"{prefix}_sim.csv"
    if not sim_path.exists():
        raise FileNotFoundError(f"Missing sim file: {sim_path}")

    sim_df = load_summary_csv(sim_path, method="simple_sim")
    frames = [sim_df]
    log_frames: list[pd.DataFrame] = []
    for policy in POLICIES:
        log_path = cmp_dir / f"{prefix}_{policy}.log"
        if not log_path.exists():
            raise FileNotFoundError(f"Missing log file: {log_path}")
        log_frames.append(load_policy_log(log_path, policy))
    frames.extend(log_frames)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged[merged["policy"].isin(POLICIES)].copy()
    # Keep epsilons that exist in both simple(sim) and log estimate for each policy.
    log_eps = pd.concat(log_frames, ignore_index=True).loc[:, ["policy", "epsilon"]].drop_duplicates()
    sim_eps = sim_df.loc[:, ["policy", "epsilon"]].drop_duplicates()
    common_eps = sim_eps.merge(log_eps, on=["policy", "epsilon"], how="inner")
    merged = merged.merge(common_eps, on=["policy", "epsilon"], how="inner")

    if epsilon_step <= 0:
        raise ValueError("epsilon_step must be a positive integer.")
    merged = merged[merged["epsilon"] % epsilon_step == 0]
    merged = merged.sort_values(["policy", "method", "epsilon"]).reset_index(drop=True)
    return merged


def plot_hit_ratio(merged: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    for policy in POLICIES:
        for method in METHOD_ORDER:
            part = merged[(merged["policy"] == policy) & (merged["method"] == method)].copy()
            if part.empty:
                continue
            style = METHOD_STYLES[method]
            ax.plot(
                part["epsilon"],
                part["hit_ratio"],
                label=f"{policy} - {METHOD_LABELS[method]}",
                color=POLICY_COLORS[policy],
                linewidth=2,
                markersize=4,
                **style,
            )

    ax.set_xlabel("epsilon", fontsize=XLABEL_FONTSIZE)
    ax.set_ylabel("cache hit ratio", fontsize=YLABEL_FONTSIZE)
    ax.tick_params(axis="both", **TICK_FONT)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(alpha=0.3)
    ax.legend(
        ncol=3,
        fontsize=TICK_FONTSIZE,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cmp_dir = args.cmp_dir.resolve()
    output = args.output or (cmp_dir / f"{args.prefix}_hit_ratio_compare.png")
    merged = build_merged_frame(cmp_dir=cmp_dir, prefix=args.prefix, epsilon_step=args.epsilon_step)
    plot_hit_ratio(merged=merged, output=output.resolve())
    print(output.resolve())


if __name__ == "__main__":
    main()
