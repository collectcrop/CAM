#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from io import StringIO
from pathlib import Path


DEFAULT_DATASETS = ["books", "fb", "osm", "wiki"]
DEFAULT_METHODS = ["CAM", "replay"]
REQUIRED_COLUMNS = {
    "method",
    "dataset",
    "dataset_label",
    "sample_rate_percent",
    "M",
    "total_estimate_time_s",
    "mean_accuracy",
}
ACTUAL_REQUIRED_COLUMNS = {
    "epsilon",
    "policy",
    "strategy",
    "budget_mode",
    "queries",
    "total_cache_misses",
    "index_build_ns",
    "simulate_wall_ns",
}


def split_tokens(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for token in value.replace(",", " ").split():
            token = token.strip()
            if token:
                out.append(token)
    return out


def parse_rates(values: list[str]) -> list[float]:
    rates: list[float] = []
    seen: set[float] = set()
    for token in split_tokens(values):
        cleaned = token[:-1] if token.endswith("%") else token
        rate = float(cleaned)
        if rate not in seen:
            rates.append(rate)
            seen.add(rate)
    return rates


def infer_summary_name(root: Path, kind: str, summary_name: str | None) -> str:
    if summary_name:
        return summary_name
    if kind == "point":
        return "point_cmp_summary.csv"
    if kind == "range":
        return "range_cmp_summary.csv"
    root_text = str(root)
    if "point_cmp" in root_text:
        return "point_cmp_summary.csv"
    return "range_cmp_summary.csv"


def infer_detail_name(root: Path, kind: str, detail_name: str | None) -> str:
    if detail_name:
        return detail_name
    if kind == "point":
        return "point_cmp_detail.csv"
    if kind == "range":
        return "range_cmp_detail.csv"
    root_text = str(root)
    if "point_cmp" in root_text:
        return "point_cmp_detail.csv"
    return "range_cmp_detail.csv"


def infer_actual_suffix(root: Path, kind: str, actual_suffix: str | None) -> str:
    if actual_suffix is not None:
        return actual_suffix
    if kind == "range":
        return "_range_actual.csv"
    if kind == "point":
        return "_actual.csv"
    root_text = str(root)
    if "range_cmp" in root_text:
        return "_range_actual.csv"
    return "_actual.csv"


def dataset_label(dataset: str) -> str:
    if dataset.startswith("fb_"):
        return "fb"
    if dataset.startswith("wiki_ts_"):
        return "wiki"
    if dataset.startswith("osm_cellids_"):
        return "osm"
    if dataset.startswith("books_"):
        return "books"
    return dataset.replace("_uint64_unique", "")


def rate_label(rate: float) -> str:
    if abs(rate - round(rate)) < 1e-9:
        return f"{int(round(rate))}%"
    return f"{rate:g}%"


def format_number(value: float | None, precision: int) -> str:
    if value is None:
        return "NA"
    return f"{value:.{precision}f}"


def q_error(prediction: float, truth: float) -> float | None:
    if prediction < 0 or truth < 0:
        return None
    if prediction == 0 and truth == 0:
        return 1.0
    if prediction == 0 or truth == 0:
        return float("inf")
    return max(prediction / truth, truth / prediction)


def row_matches_dataset(row: dict[str, str], datasets: set[str]) -> bool:
    dataset_label_ = row["dataset_label"]
    dataset_name = row["dataset"]
    return dataset_label_ in datasets or dataset_name in datasets


def load_summary(
    path: Path,
    *,
    datasets: set[str],
    rates: set[float],
    m_mib: int,
    metric: str,
) -> tuple[dict[tuple[str, float, str], tuple[float, float]], dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS.difference(columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        rows: dict[tuple[str, float, str], tuple[float, float]] = {}
        dataset_names: dict[str, str] = {}
        for row in reader:
            try:
                row_m = int(float(row["M"]))
                row_rate = float(row["sample_rate_percent"])
            except ValueError:
                continue
            if row_m != m_mib or row_rate not in rates:
                continue
            if not row_matches_dataset(row, datasets):
                continue

            dataset_label_ = row["dataset_label"]
            dataset_name = row["dataset"]
            dataset_names[dataset_label_] = dataset_name
            dataset_names[dataset_name] = dataset_name
            method = row["method"]
            if metric == "accuracy":
                metric_value = float(row["mean_accuracy"])
            elif "mean_q_error" in row and row["mean_q_error"] != "":
                metric_value = float(row["mean_q_error"])
            else:
                metric_value = float("nan")
            value = (
                float(row["total_estimate_time_s"]),
                metric_value,
            )
            rows[(dataset_label_, row_rate, method)] = value
            rows[(dataset_name, row_rate, method)] = value
    return rows, dataset_names


def load_detail_metric(
    path: Path,
    *,
    datasets: set[str],
    rates: set[float],
    m_mib: int,
) -> dict[tuple[str, float, str], float]:
    required = {
        "method",
        "dataset",
        "dataset_label",
        "sample_rate_percent",
        "M",
        "estimated_total_ios",
        "actual_total_ios",
    }
    groups: dict[tuple[str, float, str], list[float]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        missing = required.difference(columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            try:
                row_m = int(float(row["M"]))
                row_rate = float(row["sample_rate_percent"])
            except ValueError:
                continue
            if row_m != m_mib or row_rate not in rates:
                continue
            if not row_matches_dataset(row, datasets):
                continue
            qe = q_error(float(row["estimated_total_ios"]), float(row["actual_total_ios"]))
            if qe is None:
                continue
            dataset_label_ = row["dataset_label"]
            dataset_name = row["dataset"]
            method = row["method"]
            groups.setdefault((dataset_label_, row_rate, method), []).append(qe)
            groups.setdefault((dataset_name, row_rate, method), []).append(qe)
    return {key: sum(values) / len(values) for key, values in groups.items() if values}


def actual_path(
    root: Path,
    *,
    workload: str,
    dataset_name: str,
    m_mib: int,
    policy: str,
    suffix: str,
    wocache: bool,
) -> Path:
    actual_root = root / workload
    if wocache:
        actual_root = actual_root / "wocache"
    return actual_root / "actual" / dataset_name / f"{dataset_name}_{workload}_M{m_mib}_{policy.upper()}{suffix}"


def discover_dataset_names(root: Path, workload: str, m_mib: int, policy: str, suffix: str) -> dict[str, str]:
    actual_dir = root / workload / "actual"
    if not actual_dir.exists():
        return {}
    pattern = f"*_M{m_mib}_{policy.upper()}{suffix}"
    out: dict[str, str] = {}
    for path in actual_dir.glob(f"*/{pattern}"):
        dataset_name = path.parent.name
        label = dataset_label(dataset_name)
        out[label] = dataset_name
        out[dataset_name] = dataset_name
    return out


def load_actual_rows(path: Path, *, policy: str, strategy: str, budget_mode: str) -> dict[int, dict[str, float]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        missing = ACTUAL_REQUIRED_COLUMNS.difference(columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        rows: dict[int, dict[str, float]] = {}
        for row in reader:
            if row["policy"].upper() != policy.upper():
                continue
            if row["strategy"] != strategy:
                continue
            if row["budget_mode"].lower() != budget_mode.lower():
                continue
            eps = int(float(row["epsilon"]))
            rows[eps] = {
                "ios": float(row["total_cache_misses"]),
                "time_s": (float(row["index_build_ns"]) + float(row["simulate_wall_ns"])) / 1e9,
            }
    return rows


def lpm_value(
    root: Path,
    *,
    workload: str,
    dataset_name: str,
    m_mib: int,
    policy: str,
    lpm_policy: str,
    strategy: str,
    budget_mode: str,
    suffix: str,
    metric: str,
) -> tuple[float, float] | None:
    cached_path = actual_path(
        root,
        workload=workload,
        dataset_name=dataset_name,
        m_mib=m_mib,
        policy=policy,
        suffix=suffix,
        wocache=False,
    )
    lpm_path = actual_path(
        root,
        workload=workload,
        dataset_name=dataset_name,
        m_mib=m_mib,
        policy=lpm_policy,
        suffix=suffix,
        wocache=True,
    )
    if not cached_path.exists() or not lpm_path.exists():
        return None

    cached = load_actual_rows(cached_path, policy=policy, strategy=strategy, budget_mode=budget_mode)
    lpm = load_actual_rows(lpm_path, policy=lpm_policy, strategy=strategy, budget_mode=budget_mode)
    common_eps = sorted(set(cached).intersection(lpm))
    if not common_eps:
        return None

    metrics: list[float] = []
    total_time_s = 0.0
    for eps in common_eps:
        actual_ios = cached[eps]["ios"]
        lpm_ios = lpm[eps]["ios"]
        total_time_s += lpm[eps]["time_s"]
        if metric == "accuracy":
            if actual_ios <= 0:
                continue
            rel_err = abs(lpm_ios - actual_ios) / actual_ios
            metrics.append(min(1.0, max(0.0, 1.0 - rel_err)))
        else:
            qe = q_error(lpm_ios, actual_ios)
            if qe is not None:
                metrics.append(qe)
    if not metrics:
        return None
    return total_time_s, sum(metrics) / len(metrics)


def load_lpm_rows(
    root: Path,
    *,
    workload: str,
    dataset_order: list[str],
    known_dataset_names: dict[str, str],
    rates: list[float],
    m_mib: int,
    policy: str,
    lpm_policy: str,
    strategy: str,
    budget_mode: str,
    suffix: str,
    metric: str,
) -> tuple[dict[tuple[str, float, str], tuple[float, float]], list[str]]:
    rows: dict[tuple[str, float, str], tuple[float, float]] = {}
    warnings: list[str] = []
    discovered = discover_dataset_names(root, workload, m_mib, policy, suffix)
    dataset_names = {**discovered, **known_dataset_names}

    for dataset in dataset_order:
        dataset_name = dataset_names.get(dataset, dataset)
        value = lpm_value(
            root,
            workload=workload,
            dataset_name=dataset_name,
            m_mib=m_mib,
            policy=policy,
            lpm_policy=lpm_policy,
            strategy=strategy,
            budget_mode=budget_mode,
            suffix=suffix,
            metric=metric,
        )
        if value is None:
            warnings.append(f"missing or unmatched lpm actual rows: workload={workload} dataset={dataset}")
            continue
        label = dataset_label(dataset_name)
        for rate in rates:
            rows[(dataset, rate, "lpm")] = value
            rows[(label, rate, "lpm")] = value
            rows[(dataset_name, rate, "lpm")] = value
    return rows, warnings


def collect_rows(args: argparse.Namespace) -> tuple[list[str], list[dict[str, str]], list[str]]:
    workloads = split_tokens(args.workloads)
    dataset_order = split_tokens(args.datasets)
    datasets = set(dataset_order)
    methods = split_tokens(args.methods)
    if args.include_lpm and "lpm" not in methods:
        methods.append("lpm")
    rates = parse_rates(args.rates)
    rate_set = set(rates)
    summary_name = infer_summary_name(args.root, args.kind, args.summary_name)
    detail_name = infer_detail_name(args.root, args.kind, args.detail_name)
    actual_suffix = infer_actual_suffix(args.root, args.kind, args.actual_suffix)
    metric_col = "mean_accuracy" if args.metric == "accuracy" else "mean_q_error"

    headers = ["workload", "dataset", "rate"]
    for method in methods:
        headers.append(f"{method}_total_estimate_time_s")
        headers.append(f"{method}_{metric_col}")

    table_rows: list[dict[str, str]] = []
    warnings: list[str] = []

    for workload in workloads:
        summary_path = args.root / workload / "summary" / summary_name
        if not summary_path.exists():
            msg = f"missing summary: {summary_path}"
            if args.missing == "error":
                raise FileNotFoundError(msg)
            warnings.append(msg)
            if args.missing == "skip":
                continue
            summary_rows: dict[tuple[str, float, str], tuple[float, float]] = {}
            dataset_names: dict[str, str] = {}
        else:
            summary_rows, dataset_names = load_summary(
                summary_path,
                datasets=datasets,
                rates=rate_set,
                m_mib=args.memory,
                metric=args.metric,
            )
            if args.metric == "q_error":
                detail_path = args.root / workload / "summary" / detail_name
                if not detail_path.exists():
                    warnings.append(f"missing detail for q-error: {detail_path}")
                else:
                    detail_metrics = load_detail_metric(
                        detail_path,
                        datasets=datasets,
                        rates=rate_set,
                        m_mib=args.memory,
                    )
                    for key, metric_value in detail_metrics.items():
                        if key in summary_rows:
                            total_time_s, _ = summary_rows[key]
                            summary_rows[key] = (total_time_s, metric_value)

        if "lpm" in methods:
            lpm_rows, lpm_warnings = load_lpm_rows(
                args.root,
                workload=workload,
                dataset_order=dataset_order,
                known_dataset_names=dataset_names,
                rates=rates,
                m_mib=args.memory,
                policy=args.policy,
                lpm_policy=args.lpm_policy,
                strategy=args.strategy,
                budget_mode=args.budget_mode,
                suffix=actual_suffix,
                metric=args.metric,
            )
            summary_rows.update(lpm_rows)
            warnings.extend(lpm_warnings)

        for dataset in dataset_order:
            for rate in rates:
                out: dict[str, str] = {
                    "workload": workload,
                    "dataset": dataset,
                    "rate": rate_label(rate),
                }
                for method in methods:
                    value = summary_rows.get((dataset, rate, method))
                    if value is None:
                        out[f"{method}_total_estimate_time_s"] = "NA"
                        out[f"{method}_{metric_col}"] = "NA"
                    else:
                        total_time_s, metric_value = value
                        out[f"{method}_total_estimate_time_s"] = format_number(total_time_s, args.precision)
                        out[f"{method}_{metric_col}"] = format_number(metric_value, args.precision)
                table_rows.append(out)

    return headers, table_rows, warnings


def render_markdown(headers: list[str], rows: list[dict[str, str]]) -> str:
    def cell(value: str) -> str:
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def render_csv(headers: list[str], rows: list[dict[str, str]]) -> str:
    out = StringIO()
    writer = csv.DictWriter(out, fieldnames=headers, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().rstrip("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract total estimate time and accuracy/q-error from point/range cmp CSVs."
    )
    parser.add_argument("--root", type=Path, default=Path("build/log/range_cmp"))
    parser.add_argument("--kind", choices=["auto", "point", "range"], default="auto")
    parser.add_argument("--summary-name", default=None)
    parser.add_argument("--detail-name", default=None)
    parser.add_argument("--workloads", nargs="+", default=["w1", "w2", "w4", "w6"])
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--rates", nargs="+", default=["10", "30", "50", "100"])
    parser.add_argument("--memory", "--M", dest="memory", type=int, default=128)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--include-lpm", action="store_true", help="Append lpm from wocache actual CSVs.")
    parser.add_argument("--policy", default="LRU", help="Cached actual policy used as the accuracy reference.")
    parser.add_argument("--lpm-policy", default="NONE", help="No-cache actual policy used for lpm.")
    parser.add_argument("--strategy", default="all_in_once")
    parser.add_argument("--budget-mode", default="estimated")
    parser.add_argument("--actual-suffix", default=None)
    parser.add_argument("--metric", choices=["accuracy", "q_error"], default="q_error")
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--precision", type=int, default=6)
    parser.add_argument(
        "--missing",
        choices=["na", "skip", "error"],
        default="na",
        help="How to handle missing workload summary files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    headers, rows, warnings = collect_rows(args)
    for warning in warnings:
        print(f"[warn] {warning}", file=sys.stderr)

    if args.format == "csv":
        text = render_csv(headers, rows)
    else:
        text = render_markdown(headers, rows)

    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
        print(f"[write] {args.output}")


if __name__ == "__main__":
    main()
