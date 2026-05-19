#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare CAM-selected epsilon against PGM tuner epsilons under "
            "fixed cache splits of the same memory budget."
        )
    )
    parser.add_argument("--data", required=True, help="Dataset path or filename under --datasets-directory.")
    parser.add_argument("--queries", required=True, help="Query path or filename under --datasets-directory.")
    parser.add_argument("--keys", type=int, default=0, help="Number of keys. Default: infer from data file size.")
    parser.add_argument("--M", type=int, required=True, help="Total memory budget in MiB.")
    parser.add_argument(
        "--candidate-eps",
        default="4-128",
        help="Candidate epsilon set for CAM selection. Supports comma lists and ranges like 4-128.",
    )
    parser.add_argument(
        "--cache-ratios",
        default="0.25,0.50,0.75",
        help="Comma-separated cache fractions for PGM tuner baselines.",
    )
    parser.add_argument("--datasets-directory", default="/mnt/data/Dataset/public/SOSD")
    parser.add_argument("--cam-bin", default="./build/pgm_cam_covariance")
    parser.add_argument("--index-size-bin", default="./build/pgm_index_sizes")
    parser.add_argument("--tuner-bin", default="./build/tuner")
    parser.add_argument("--output-dir", default="build/log/pgm_tuner_cache_compare")
    parser.add_argument("--dataset-tag", default=None)
    parser.add_argument("--policies", default="FIFO,LRU,LFU")
    parser.add_argument("--strategies", default="all_in_once")
    parser.add_argument("--cam-policy", default="LRU", help="Cache policy used by CAM's estimator.")
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--seg-size", type=int, default=16)
    parser.add_argument("--ipp", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=4096)
    parser.add_argument("--tuner-tol", type=float, default=None)
    parser.add_argument("--tuner-ratio", type=float, default=None)
    parser.add_argument(
        "--cold-start-correction",
        action="store_true",
        help="Enable cold-start compulsory-miss correction in point-query CAM estimates.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    out = []
    normalized = value.replace(" ", ",")
    for token in normalized.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = [part.strip() for part in token.split("-")]
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"invalid range token: {token}")
            start = int(parts[0])
            end = int(parts[1])
            step = 1 if end >= start else -1
            out.extend(range(start, end + step, step))
        else:
            out.append(int(token))
    if not out:
        raise ValueError("empty integer list")
    return list(dict.fromkeys(out))


def parse_float_list(value: str) -> list[float]:
    out = []
    for token in value.replace(" ", ",").split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    if not out:
        raise ValueError("empty float list")
    return out


def resolve_input(path_text: str, datasets_directory: Path) -> Path:
    path = Path(path_text)
    if path.exists():
        return path.resolve()
    if not path.is_absolute():
        candidate = datasets_directory / path
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(path_text)


def resolve_executable(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute() or path.exists():
        return path
    return repo_root / path


def infer_key_count(path: Path) -> int:
    size = path.stat().st_size
    if size % 8 != 0:
        raise ValueError(f"dataset size is not a multiple of uint64: {path}")
    return size // 8


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def load_keys(path: Path, keys: int) -> np.ndarray:
    count = keys if keys > 0 else -1
    return np.fromfile(path, dtype=np.uint64, count=count)


def read_measured_index_size_csv(path: Path) -> dict[int, int]:
    sizes: dict[int, int] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"epsilon", "measured_index_bytes"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"unexpected index-size CSV header in {path}")
        for row in reader:
            sizes[int(row["epsilon"])] = int(row["measured_index_bytes"])
    return sizes


def load_measured_index_sizes(
    index_size_bin: Path,
    data_path: Path,
    keys: int,
    candidate_eps: list[int],
    output_csv: Path,
    dry_run: bool,
) -> dict[int, int]:
    if output_csv.exists():
        try:
            sizes = read_measured_index_size_csv(output_csv)
            missing = [epsilon for epsilon in candidate_eps if epsilon not in sizes]
            if not missing:
                print(f"[*] reuse existing CAM index-size CSV: {output_csv}")
                return {epsilon: sizes[epsilon] for epsilon in candidate_eps}
            print(
                "[*] existing CAM index-size CSV is incomplete; "
                f"missing epsilons: {missing[:8]}"
            )
        except Exception as exc:
            print(f"[*] existing CAM index-size CSV cannot be reused: {exc}")

    cmd = [
        str(index_size_bin),
        "--data",
        str(data_path),
        "--keys",
        str(keys),
        "--epsilons",
        ",".join(str(epsilon) for epsilon in candidate_eps),
        "--output",
        str(output_csv),
    ]
    run_command(cmd, dry_run=dry_run)
    if dry_run:
        return {epsilon: 0 for epsilon in candidate_eps}

    sizes = read_measured_index_size_csv(output_csv)
    missing = [epsilon for epsilon in candidate_eps if epsilon not in sizes]
    if missing:
        raise RuntimeError(f"missing measured index sizes for epsilons: {missing[:8]}")
    return sizes


def choose_cam_epsilon(
    repo_root: Path,
    index_size_bin: Path,
    data_path: Path,
    query_path: Path,
    keys: int,
    memory_mib: int,
    candidate_eps: list[int],
    policy: str,
    strategy: str,
    seg_size: int,
    ipp: int,
    page_size: int,
    output_csv: Path,
    index_size_csv: Path,
    dry_run: bool,
    cold_start_correction: bool,
) -> tuple[int, list[dict[str, object]]]:
    sys.path.insert(0, str(repo_root / "utils"))
    import optimalEpsilon  # noqa: PLC0415

    optimalEpsilon.BUDGET_MODE = "MEASURED"
    data = load_keys(data_path, keys)
    n = int(keys or data.shape[0])
    memory_bytes = int(memory_mib) * 1024 * 1024
    measured_index_sizes = load_measured_index_sizes(
        index_size_bin=index_size_bin,
        data_path=data_path,
        keys=n,
        candidate_eps=candidate_eps,
        output_csv=index_size_csv,
        dry_run=dry_run,
    )

    rows: list[dict[str, object]] = []
    for epsilon in candidate_eps:
        index_estimate = n * seg_size / (2.0 * epsilon)
        measured_index_bytes = measured_index_sizes[epsilon]
        if measured_index_bytes > memory_bytes:
            rows.append(
                {
                    "epsilon": epsilon,
                    "estimated_cost": math.inf,
                    "hit_ratio": 0.0,
                    "feasible": 0,
                    "estimated_index_bytes": index_estimate,
                    "measured_index_bytes": measured_index_bytes,
                    "cache_bytes": 0,
                    "cold_start_correction": int(cold_start_correction),
                    "steady_hit_ratio": 0.0,
                    "cold_miss_ratio": 0.0,
                    "expected_distinct_pages": 0.0,
                }
            )
            continue

        cache_bytes = memory_bytes - measured_index_bytes
        cost, hit_ratio, detail = optimalEpsilon.cost_function(
            epsilon,
            n,
            seg_size,
            memory_bytes,
            ipp,
            page_size,
            query_file=str(query_path),
            data_file=str(data_path),
            s=strategy,
            cache_policy=policy,
            data_arr=data,
            measured_index_bytes=measured_index_bytes,
            cold_start_correction=cold_start_correction,
            return_detail=True,
        )
        rows.append(
            {
                "epsilon": epsilon,
                "estimated_cost": float(cost),
                "hit_ratio": float(hit_ratio),
                "feasible": 1,
                "estimated_index_bytes": index_estimate,
                "measured_index_bytes": measured_index_bytes,
                "cache_bytes": cache_bytes,
                "cold_start_correction": int(cold_start_correction),
                "steady_hit_ratio": float(detail.get("steady_hit_ratio", hit_ratio)),
                "cold_miss_ratio": float(detail.get("cold_miss_ratio", 0.0)),
                "expected_distinct_pages": float(detail.get("expected_distinct_pages", 0.0)),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epsilon",
                "estimated_cost",
                "hit_ratio",
                "feasible",
                "estimated_index_bytes",
                "measured_index_bytes",
                "cache_bytes",
                "cold_start_correction",
                "steady_hit_ratio",
                "cold_miss_ratio",
                "expected_distinct_pages",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    feasible_rows = [row for row in rows if row["feasible"]]
    if not feasible_rows:
        raise RuntimeError("no feasible CAM epsilon in candidate set")

    best = min(feasible_rows, key=lambda row: (float(row["estimated_cost"]), int(row["epsilon"])))
    return int(best["epsilon"]), rows


def run_command(cmd: list[str], *, log_path: Path | None = None, dry_run: bool = False) -> str:
    printable = " ".join(cmd)
    print(f"[*] {printable}")
    if dry_run:
        return ""

    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    output = result.stdout + result.stderr
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with code {result.returncode}: {printable}\n{output}")
    return output


def parse_tuner_epsilon(output: str) -> int:
    matches = re.findall(r"Set epsilon to\s+(\d+)", output)
    if not matches:
        raise RuntimeError("failed to parse tuner epsilon from tuner output")
    return int(matches[-1])


def run_tuner(
    tuner_bin: Path,
    data_path: Path,
    space_bytes: int,
    log_path: Path,
    tol: float | None,
    ratio: float | None,
    dry_run: bool,
) -> int:
    cmd = [str(tuner_bin), "--u64", "--space", str(space_bytes)]
    if tol is not None:
        cmd.extend(["--tol", str(tol)])
    if ratio is not None:
        cmd.extend(["--ratio", str(ratio)])
    cmd.append(str(data_path))
    output = run_command(cmd, log_path=log_path, dry_run=dry_run)
    if dry_run:
        return 0
    return parse_tuner_epsilon(output)


def run_covariance(
    cam_bin: Path,
    data_path: Path,
    query_path: Path,
    keys: int,
    memory_mib: int,
    epsilon: int,
    policies: str,
    strategies: str,
    summary_out: Path,
    budget_mode: str,
    query_limit: int,
    cache_bytes: int | None,
    dry_run: bool,
) -> None:
    cmd = [
        str(cam_bin),
        "--data",
        str(data_path),
        "--queries",
        str(query_path),
        "--M",
        str(memory_mib),
        "--epsilons",
        str(epsilon),
        "--policies",
        policies,
        "--strategies",
        strategies,
        "--budget-mode",
        budget_mode,
        "--summary-out",
        str(summary_out),
    ]
    if keys > 0:
        cmd.extend(["--keys", str(keys)])
    if query_limit > 0:
        cmd.extend(["--query-limit", str(query_limit)])
    if cache_bytes is not None:
        cmd.extend(["--cache-bytes", str(cache_bytes)])
    run_command(cmd, dry_run=dry_run)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_combined_summary(output_path: Path, run_rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for run in run_rows:
        for key in run:
            if key not in fieldnames:
                fieldnames.append(key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_rows)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    datasets_directory = Path(args.datasets_directory).expanduser()
    data_path = resolve_input(args.data, datasets_directory)
    query_path = resolve_input(args.queries, datasets_directory)
    keys = int(args.keys or infer_key_count(data_path))
    candidate_eps = parse_int_list(args.candidate_eps)
    cache_ratios = parse_float_list(args.cache_ratios)
    policies_csv = args.policies.replace(" ", ",")
    dataset_tag = args.dataset_tag or data_path.stem
    output_dir = Path(args.output_dir).resolve() / f"{dataset_tag}_M{args.M}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cam_bin = resolve_executable(args.cam_bin, repo_root)
    index_size_bin = resolve_executable(args.index_size_bin, repo_root)
    tuner_bin = resolve_executable(args.tuner_bin, repo_root)
    cam_strategy = args.strategies.split(",")[0].strip()
    if cam_strategy.lower() == "all":
        cam_strategy = "all_in_once"

    cam_candidates_csv = output_dir / "cam_candidate_costs.csv"
    index_size_csv = output_dir / "cam_candidate_index_sizes.csv"
    cam_t0 = time.perf_counter()
    cam_epsilon, cam_candidate_rows = choose_cam_epsilon(
        repo_root=repo_root,
        index_size_bin=index_size_bin,
        data_path=data_path,
        query_path=query_path,
        keys=keys,
        memory_mib=args.M,
        candidate_eps=candidate_eps,
        policy=args.cam_policy,
        strategy=cam_strategy,
        seg_size=args.seg_size,
        ipp=args.ipp,
        page_size=args.page_size,
        output_csv=cam_candidates_csv,
        index_size_csv=index_size_csv,
        dry_run=args.dry_run,
        cold_start_correction=args.cold_start_correction,
    )
    cam_tuning_time_s = None if args.dry_run else time.perf_counter() - cam_t0
    cam_feasible_candidates = sum(1 for row in cam_candidate_rows if row.get("feasible"))
    print(f"[+] CAM epsilon={cam_epsilon} from {cam_candidates_csv}")

    run_metadata: list[dict[str, object]] = []
    cam_summary = output_dir / "cam_measured_summary.csv"
    run_covariance(
        cam_bin=cam_bin,
        data_path=data_path,
        query_path=query_path,
        keys=keys,
        memory_mib=args.M,
        epsilon=cam_epsilon,
        policies=policies_csv,
        strategies=args.strategies,
        summary_out=cam_summary,
        budget_mode="measured",
        query_limit=args.query_limit,
        cache_bytes=None,
        dry_run=args.dry_run,
    )
    run_metadata.append(
        {
            "selector": "CAM",
            "cache_ratio": "",
            "tuner_space_bytes": "",
            "epsilon": cam_epsilon,
            "tuning_time_s": fmt_seconds(cam_tuning_time_s),
            "tuning_time_source": "wall_clock_cam_estimator",
            "tuning_time_cached": 0,
            "candidate_count": len(cam_candidate_rows),
            "feasible_candidates": cam_feasible_candidates,
            "candidate_costs_path": str(cam_candidates_csv),
            "index_size_path": str(index_size_csv),
            "summary_path": str(cam_summary),
        }
    )

    memory_bytes = int(args.M) * 1024 * 1024
    tuning_rows: list[dict[str, object]] = [
        {
            "selector": "CAM",
            "cache_ratio": "",
            "epsilon": cam_epsilon,
            "tuning_time_s": fmt_seconds(cam_tuning_time_s),
            "tuning_time_source": "wall_clock_cam_estimator",
            "cached": 0,
            "log_path": str(cam_candidates_csv),
            "candidate_count": len(cam_candidate_rows),
            "feasible_candidates": cam_feasible_candidates,
            "tuner_space_bytes": "",
            "cache_bytes": "",
        }
    ]
    for ratio in cache_ratios:
        cache_bytes = int(round(memory_bytes * ratio))
        space_bytes = memory_bytes - cache_bytes
        ratio_tag = f"{int(round(ratio * 100)):02d}"
        tuner_log = output_dir / f"tuner_cache{ratio_tag}.log"
        tuner_t0 = time.perf_counter()
        epsilon = run_tuner(
            tuner_bin=tuner_bin,
            data_path=data_path,
            space_bytes=space_bytes,
            log_path=tuner_log,
            tol=args.tuner_tol,
            ratio=args.tuner_ratio,
            dry_run=args.dry_run,
        )
        tuner_tuning_time_s = None if args.dry_run else time.perf_counter() - tuner_t0
        tuning_rows.append(
            {
                "selector": "PGM_tuner",
                "cache_ratio": ratio,
                "epsilon": epsilon if not args.dry_run else "",
                "tuning_time_s": fmt_seconds(tuner_tuning_time_s),
                "tuning_time_source": "wall_clock_tuner",
                "cached": 0,
                "log_path": str(tuner_log),
                "candidate_count": "",
                "feasible_candidates": "",
                "tuner_space_bytes": space_bytes,
                "cache_bytes": cache_bytes,
            }
        )
        if args.dry_run:
            run_metadata.append(
                {
                    "selector": "PGM_tuner",
                    "cache_ratio": ratio,
                    "cache_bytes": cache_bytes,
                    "tuner_space_bytes": space_bytes,
                    "epsilon": "<from_tuner>",
                    "tuner_log": str(tuner_log),
                    "tuning_time_s": fmt_seconds(tuner_tuning_time_s),
                    "tuning_time_source": "wall_clock_tuner",
                    "tuning_time_cached": 0,
                    "summary_path": "",
                }
            )
            continue
        summary = output_dir / f"pgm_tuner_cache{ratio_tag}_summary.csv"
        run_covariance(
            cam_bin=cam_bin,
            data_path=data_path,
            query_path=query_path,
            keys=keys,
            memory_mib=args.M,
            epsilon=epsilon,
            policies=policies_csv,
            strategies=args.strategies,
            summary_out=summary,
            budget_mode="fixed-cache",
            query_limit=args.query_limit,
            cache_bytes=cache_bytes,
            dry_run=args.dry_run,
        )
        run_metadata.append(
            {
                "selector": "PGM_tuner",
                "cache_ratio": ratio,
                "cache_bytes": cache_bytes,
                "tuner_space_bytes": space_bytes,
                "epsilon": epsilon,
                "tuner_log": str(tuner_log),
                "tuning_time_s": fmt_seconds(tuner_tuning_time_s),
                "tuning_time_source": "wall_clock_tuner",
                "tuning_time_cached": 0,
                "summary_path": str(summary),
            }
        )

    plan_path = output_dir / "experiment_plan.csv"
    write_combined_summary(plan_path, run_metadata)
    tuning_path = output_dir / "tuning_time_summary.csv"
    write_combined_summary(tuning_path, tuning_rows)

    if not args.dry_run:
        combined_rows: list[dict[str, object]] = []
        for meta in run_metadata:
            for row in read_csv_rows(Path(str(meta["summary_path"]))):
                combined = dict(meta)
                combined.update(row)
                combined_rows.append(combined)
        combined_path = output_dir / "comparison_summary.csv"
        write_combined_summary(combined_path, combined_rows)
        print(f"[+] combined summary: {combined_path}")

    print(f"[+] tuning time summary: {tuning_path}")
    print(f"[+] experiment plan: {plan_path}")


if __name__ == "__main__":
    main()
