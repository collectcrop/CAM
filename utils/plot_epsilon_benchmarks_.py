#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


POLICIES = ("FIFO", "LRU", "LFU", "NONE")
MERGE_KEYS = ["dataset", "policy", "M", "epsilon"]


def norm_col(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", s.strip().lower()).strip("_")


def norm_dataset(s: str) -> str:
    s = re.sub(r"_(?:u?int\d+)(?:_(?:unique|sorted))?$", "", s, flags=re.I)
    s = re.sub(r"_(?:unique|sorted|keys|data)$", "", s, flags=re.I)
    return re.sub(r"__+", "_", s)


def infer_policy(path: Path, suffix: str = r"\.log") -> str:
    m = re.match(rf"(?P<dataset>.+)_(?P<policy>{'|'.join(POLICIES)}){suffix}$", path.name, re.I)
    if not m:
        raise ValueError(f"Cannot infer policy from filename: {path}")
    return m.group("policy").upper()


def infer_dataset_from_estimate(path: Path) -> str:
    m = re.match(rf"(?P<dataset>.+)_(?P<policy>{'|'.join(POLICIES)})\.log$", path.name, re.I)
    if not m:
        raise ValueError(f"Cannot infer dataset from estimate filename: {path}")
    return norm_dataset(m.group("dataset"))


def infer_dataset_m_from_bench(path: Path) -> tuple[str, int | None]:
    # Expected: fb_10M_M10_range_bench.csv or books_10M_M60_bench.csv
    m = re.match(r"(?P<dataset>.+)_M(?P<M>\d+)(?:_range)?_bench(?:_[^.]+)?\.csv$", path.name, re.I)
    if not m:
        return norm_dataset(path.stem), None
    return norm_dataset(m.group("dataset")), int(m.group("M"))


def read_csv_normalized(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [norm_col(c) for c in df.columns]
    return df


def load_estimates(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        df = read_csv_normalized(path)
        required = {"m", "epsilon", "cost", "ratio"}
        if not required <= set(df.columns):
            raise ValueError(f"Bad estimate schema in {path}; need {required}, got {set(df.columns)}")

        out = df.rename(columns={
            "m": "M",
            "cost": "estimated_avg_ios",
            "ratio": "estimated_hit_ratio",
        })[["M", "epsilon", "estimated_avg_ios", "estimated_hit_ratio"]].copy()

        out["dataset"] = infer_dataset_from_estimate(path)
        out["policy"] = infer_policy(path)
        out["estimate_source"] = str(path)
        rows.append(out)

    ans = pd.concat(rows, ignore_index=True)
    return coerce_numeric(ans, ["M", "epsilon", "estimated_avg_ios", "estimated_hit_ratio"])


def load_benches(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        df = read_csv_normalized(path)
        required = {"epsilon", "policy", "hit_ratio", "logical_ios"}
        if not required <= set(df.columns):
            raise ValueError(f"Bad benchmark schema in {path}; need {required}, got {set(df.columns)}")

        count_col = "ranges" if "ranges" in df.columns else "queries" if "queries" in df.columns else None
        if count_col is None:
            raise ValueError(f"{path} must contain `queries` or `ranges`.")

        dataset, inferred_M = infer_dataset_m_from_bench(path)
        if "m" not in df.columns:
            if inferred_M is None:
                raise ValueError(f"Cannot infer M from {path}; add column `m` or rename file.")
            df["m"] = inferred_M

        out = pd.DataFrame({
            "dataset": dataset,
            "policy": df["policy"].astype(str).str.upper(),
            "M": df["m"],
            "epsilon": df["epsilon"],
            "queries": df[count_col],
            "actual_hit_ratio": df["hit_ratio"],
            "actual_total_ios": df["logical_ios"],
            "bench_source": str(path),
        })

        # Prefer explicit avg_logical_ios if benchmark has it.
        if "avg_logical_ios" in df.columns:
            out["actual_avg_ios"] = df["avg_logical_ios"]
        else:
            out["actual_avg_ios"] = pd.to_numeric(out["actual_total_ios"], errors="coerce") / pd.to_numeric(out["queries"], errors="coerce")

        rows.append(out)

    ans = pd.concat(rows, ignore_index=True)
    return coerce_numeric(ans, ["M", "epsilon", "queries", "actual_total_ios", "actual_avg_ios", "actual_hit_ratio"])


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=cols).copy()
    df["M"] = df["M"].astype(int)
    df["epsilon"] = df["epsilon"].astype(int)
    return df


def apply_filters(df: pd.DataFrame, dataset_filter: str | None, policies: set[str] | None, m_values: set[int] | None) -> pd.DataFrame:
    out = df.copy()
    if dataset_filter:
        out = out[out["dataset"].astype(str).str.contains(dataset_filter, case=False, regex=False)]
    if policies:
        out = out[out["policy"].isin(policies)]
    if m_values:
        out = out[out["M"].isin(m_values)]
    return out


def merge_metrics(est: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    for name, df in [("estimate", est), ("bench", bench)]:
        dup = df[df.duplicated(MERGE_KEYS, keep=False)]
        if not dup.empty:
            raise ValueError(f"Duplicate {name} rows for keys:\n{dup[MERGE_KEYS].drop_duplicates().head(10)}")

    merged = est.merge(bench, on=MERGE_KEYS, how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No overlap between estimate and benchmark on dataset/policy/M/epsilon.")

    merged["estimated_total_ios"] = merged["estimated_avg_ios"] * merged["queries"]
    merged["avg_error"] = merged["estimated_avg_ios"] - merged["actual_avg_ios"]
    merged["avg_abs_pct_error"] = (merged["avg_error"].abs() / merged["actual_avg_ios"]).replace([float("inf")], pd.NA)

    merged["total_error"] = merged["estimated_total_ios"] - merged["actual_total_ios"]
    merged["total_abs_pct_error"] = (merged["total_error"].abs() / merged["actual_total_ios"]).replace([float("inf")], pd.NA)

    return merged.sort_values(MERGE_KEYS).reset_index(drop=True)


def diagnose_units(df: pd.DataFrame) -> pd.DataFrame:
    # If total-ratio is about queries times avg-ratio, the plotting unit is probably mixed.
    rows = []
    for keys, g in df.groupby(["dataset", "policy", "M"]):
        avg_ratio = median_ratio(g["estimated_avg_ios"], g["actual_avg_ios"])
        total_ratio = median_ratio(g["estimated_total_ios"], g["actual_total_ios"])
        q = g["queries"].median()
        rows.append({
            "dataset": keys[0],
            "policy": keys[1],
            "M": keys[2],
            "median_queries": q,
            "median_est_over_actual_avg": avg_ratio,
            "median_est_over_actual_total": total_ratio,
            "suspect": classify_suspect(avg_ratio, total_ratio, q),
        })
    return pd.DataFrame(rows)


def median_ratio(a: pd.Series, b: pd.Series) -> float:
    r = pd.to_numeric(a, errors="coerce") / pd.to_numeric(b, errors="coerce")
    r = r.replace([float("inf"), -float("inf")], pd.NA).dropna()
    return float(r.median()) if not r.empty else float("nan")


def classify_suspect(avg_ratio: float, total_ratio: float, queries: float) -> str:
    if pd.isna(avg_ratio) or pd.isna(total_ratio):
        return "insufficient data"
    if 0.5 <= avg_ratio <= 2.0 and not (0.5 <= total_ratio <= 2.0):
        return "plot avg-vs-total mismatch likely"
    if 0.5 <= total_ratio <= 2.0 and not (0.5 <= avg_ratio <= 2.0):
        return "estimate cost may already be total"
    if 8 <= avg_ratio <= 12 or 8 <= total_ratio <= 12:
        return "near 10x scale mismatch"
    return "check model/data"


def plot_avg_ios(df: pd.DataFrame, output: Path) -> None:
    for (dataset, M), gM in df.groupby(["dataset", "M"]):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for policy, g in gM.groupby("policy"):
            g = g.sort_values("epsilon")
            ax.plot(g["epsilon"], g["actual_avg_ios"], marker="o", label=f"{policy} actual avg")
            ax.plot(g["epsilon"], g["estimated_avg_ios"], marker="s", linestyle="--", label=f"{policy} estimated avg")
        ax.set_title(f"{dataset}, M={M}")
        ax.set_xlabel("epsilon")
        ax.set_ylabel("avg logical I/Os per query")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{dataset}_M{M}_avg_ios_vs_epsilon.pdf", bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--estimate-paths", nargs="+", type=Path, required=True)
    p.add_argument("--bench-paths", nargs="+", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("data/outputs/figures/epsilon_analysis_clean"))
    p.add_argument("--dataset-filter")
    p.add_argument("--policies", nargs="*")
    p.add_argument("--m-values", nargs="*", type=int)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    policies = {p.upper() for p in args.policies} if args.policies else None
    m_values = set(args.m_values) if args.m_values else None

    est = apply_filters(load_estimates(args.estimate_paths), args.dataset_filter, policies, m_values)
    bench = apply_filters(load_benches(args.bench_paths), args.dataset_filter, policies, m_values)

    merged = merge_metrics(est, bench)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    merged.to_csv(args.output_dir / "merged_metrics.csv", index=False)
    diagnosis = diagnose_units(merged)
    diagnosis.to_csv(args.output_dir / "unit_diagnosis.csv", index=False)

    plot_avg_ios(merged, args.output_dir)

    print(f"wrote: {args.output_dir / 'merged_metrics.csv'}")
    print(f"wrote: {args.output_dir / 'unit_diagnosis.csv'}")
    print(diagnosis.to_string(index=False))


if __name__ == "__main__":
    main()
