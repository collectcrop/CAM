#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
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
    parser.add_argument("--policies", default="LRU")
    parser.add_argument("--strategies", default="all_in_once")
    parser.add_argument("--cam-policy", default="LRU", help="Cache policy used by CAM's estimator.")
    parser.add_argument(
        "--cam-size-mode",
        choices=["estimated", "measured", "powerlaw"],
        default="estimated",
        help=(
            "Index-size mode used while CAM selects epsilon. "
            "'estimated' uses n*seg_size/(2*epsilon) and does not build candidate indexes; "
            "'measured' builds/reuses measured sizes for all candidate epsilons; "
            "'powerlaw' builds anchor epsilons, fits S(eps)=a*eps^-b+c, and estimates candidates."
        ),
    )
    parser.add_argument(
        "--cam-size-model-eps",
        default="4,8,16,32,64,128,256,512",
        help=(
            "Anchor epsilons built in --cam-size-mode powerlaw before fitting "
            "S(eps)=a*eps^-b+c."
        ),
    )
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
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "Number of independent experiment repeats. When repeats > 1, each repeat "
            "uses its own output directory and rebuilds index-size artifacts by default."
        ),
    )
    parser.add_argument(
        "--force-index-rebuild",
        action="store_true",
        help="Always rebuild CAM index-size CSVs instead of reusing existing files.",
    )
    parser.add_argument(
        "--reuse-index-size-csv",
        action="store_true",
        help=(
            "Allow reusing existing CAM index-size CSVs. By default multi-repeat runs "
            "rebuild them every repeat for a fair cold tuning cost."
        ),
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
    force_rebuild: bool = False,
) -> dict[int, int]:
    if output_csv.exists() and not force_rebuild:
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


def estimate_index_bytes(n: int, seg_size: int, epsilon: int) -> float:
    return float(n) * float(seg_size) / (2.0 * float(epsilon))


@dataclass(frozen=True)
class PowerLawIndexSizeModel:
    a: float
    b: float
    c: float
    fit_method: str
    anchor_sizes: dict[int, int]

    def predict(self, epsilon: int) -> float:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive: {epsilon}")
        value = self.a * math.pow(float(epsilon), -self.b) + self.c
        if not math.isfinite(value):
            return math.inf
        return max(0.0, value)


def fit_log_powerlaw_with_fixed_c(
    eps_arr: np.ndarray,
    size_arr: np.ndarray,
    c: float,
) -> tuple[float, float, float, float] | None:
    adjusted = size_arr - c
    if np.any(adjusted <= 0):
        return None

    slope, intercept = np.polyfit(np.log(eps_arr), np.log(adjusted), 1)
    a = max(0.0, float(math.exp(intercept)))
    b = max(0.0, -float(slope))
    pred = a * np.power(eps_arr, -b) + c
    rel = (pred - size_arr) / np.maximum(size_arr, 1.0)
    loss = float(np.mean(rel * rel))
    return a, b, c, loss


def fit_powerlaw_index_size_model(anchor_sizes: dict[int, int]) -> PowerLawIndexSizeModel:
    points = sorted(
        (int(epsilon), int(size))
        for epsilon, size in anchor_sizes.items()
        if int(epsilon) > 0 and int(size) > 0
    )
    if not points:
        return PowerLawIndexSizeModel(0.0, 0.0, 0.0, "degenerate_zero", dict(anchor_sizes))
    if len(points) == 1:
        return PowerLawIndexSizeModel(0.0, 0.0, float(points[0][1]), "single_point_constant", dict(anchor_sizes))

    eps_arr = np.array([epsilon for epsilon, _ in points], dtype=np.float64)
    size_arr = np.array([size for _, size in points], dtype=np.float64)
    min_size = float(np.min(size_arr))

    best = fit_log_powerlaw_with_fixed_c(eps_arr, size_arr, 0.0)

    if len(points) >= 3:
        try:
            from scipy.optimize import curve_fit  # noqa: PLC0415

            def curve(eps: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
                return a * np.power(eps, -b) + c

            if best is None:
                p0 = [max(1.0, float(size_arr[0])), 1.0, 0.0]
            else:
                p0 = [max(1.0, best[0]), max(0.01, best[1]), max(0.0, min(best[2], min_size * 0.5))]
            c_hi = max(0.0, min_size * (1.0 - 1e-9))
            params, _ = curve_fit(
                curve,
                eps_arr,
                size_arr,
                p0=p0,
                bounds=([0.0, 0.0, 0.0], [np.inf, 10.0, c_hi]),
                maxfev=20000,
            )
            a, b, c = (float(params[0]), float(params[1]), float(params[2]))
            if all(math.isfinite(value) for value in (a, b, c)):
                return PowerLawIndexSizeModel(a, b, c, "curve_fit", dict(anchor_sizes))
        except Exception as exc:
            print(f"[*] power-law curve_fit failed; falling back to log-grid fit: {exc}")

    c_grid_max = max(0.0, min_size * 0.95)
    for c in np.linspace(0.0, c_grid_max, 64):
        candidate = fit_log_powerlaw_with_fixed_c(eps_arr, size_arr, float(c))
        if candidate is None:
            continue
        if best is None or candidate[3] < best[3]:
            best = candidate

    if best is None:
        return PowerLawIndexSizeModel(0.0, 0.0, min_size, "constant_min", dict(anchor_sizes))
    return PowerLawIndexSizeModel(best[0], best[1], best[2], "log_grid", dict(anchor_sizes))


def write_powerlaw_model_csv(output_csv: Path, model: PowerLawIndexSizeModel) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epsilon",
                "measured_index_bytes",
                "predicted_index_bytes",
                "absolute_error_bytes",
                "relative_error",
                "fit_method",
                "a",
                "b",
                "c",
            ],
        )
        writer.writeheader()
        for epsilon, measured in sorted(model.anchor_sizes.items()):
            predicted = model.predict(int(epsilon))
            if measured:
                absolute_error = predicted - float(measured)
                relative_error: float | str = absolute_error / float(measured)
            else:
                absolute_error = ""
                relative_error = ""
            writer.writerow(
                {
                    "epsilon": epsilon,
                    "measured_index_bytes": measured,
                    "predicted_index_bytes": predicted,
                    "absolute_error_bytes": absolute_error,
                    "relative_error": relative_error,
                    "fit_method": model.fit_method,
                    "a": model.a,
                    "b": model.b,
                    "c": model.c,
                }
            )


def load_powerlaw_index_size_model(
    index_size_bin: Path,
    data_path: Path,
    keys: int,
    model_eps: list[int],
    anchor_csv: Path,
    model_csv: Path,
    dry_run: bool,
    force_rebuild: bool = False,
) -> PowerLawIndexSizeModel:
    anchor_sizes = load_measured_index_sizes(
        index_size_bin=index_size_bin,
        data_path=data_path,
        keys=keys,
        candidate_eps=model_eps,
        output_csv=anchor_csv,
        dry_run=dry_run,
        force_rebuild=force_rebuild,
    )
    model = fit_powerlaw_index_size_model(anchor_sizes)
    write_powerlaw_model_csv(model_csv, model)
    print(
        "[+] CAM power-law size model: "
        f"S(eps)={model.a:.6g}*eps^-{model.b:.6g}+{model.c:.6g} "
        f"({model.fit_method})"
    )
    return model


def load_selected_measured_index_size(
    index_size_bin: Path,
    data_path: Path,
    keys: int,
    epsilon: int,
    output_csv: Path,
    dry_run: bool,
    force_rebuild: bool = False,
) -> int:
    sizes = load_measured_index_sizes(
        index_size_bin=index_size_bin,
        data_path=data_path,
        keys=keys,
        candidate_eps=[epsilon],
        output_csv=output_csv,
        dry_run=dry_run,
        force_rebuild=force_rebuild,
    )
    return int(sizes[epsilon])


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
    size_mode: str,
    size_model_eps: list[int],
    powerlaw_anchor_csv: Path,
    powerlaw_model_csv: Path,
    dry_run: bool,
    cold_start_correction: bool,
    force_index_rebuild: bool = False,
) -> tuple[int, list[dict[str, object]], dict[str, object]]:
    sys.path.insert(0, str(repo_root / "utils"))
    import optimalEpsilon  # noqa: PLC0415

    size_mode = size_mode.lower()
    if size_mode not in {"estimated", "measured", "powerlaw"}:
        raise ValueError(f"unknown CAM size mode: {size_mode}")

    optimalEpsilon.BUDGET_MODE = "MEASURED" if size_mode in {"measured", "powerlaw"} else "ESTIMATED"
    data = load_keys(data_path, keys)
    n = int(keys or data.shape[0])
    memory_bytes = int(memory_mib) * 1024 * 1024
    measured_index_sizes: dict[int, int] = {}
    powerlaw_model: PowerLawIndexSizeModel | None = None
    size_model_info: dict[str, object] = {}

    if size_mode == "measured":
        measured_index_sizes = load_measured_index_sizes(
            index_size_bin=index_size_bin,
            data_path=data_path,
            keys=n,
            candidate_eps=candidate_eps,
            output_csv=index_size_csv,
            dry_run=dry_run,
            force_rebuild=force_index_rebuild,
        )
    elif size_mode == "powerlaw":
        powerlaw_model = load_powerlaw_index_size_model(
            index_size_bin=index_size_bin,
            data_path=data_path,
            keys=n,
            model_eps=size_model_eps,
            anchor_csv=powerlaw_anchor_csv,
            model_csv=powerlaw_model_csv,
            dry_run=dry_run,
            force_rebuild=force_index_rebuild,
        )
        size_model_info = {
            "size_model": "powerlaw",
            "size_model_anchor_eps": ",".join(str(epsilon) for epsilon in size_model_eps),
            "size_model_anchor_path": str(powerlaw_anchor_csv),
            "size_model_path": str(powerlaw_model_csv),
            "size_model_fit_method": powerlaw_model.fit_method,
            "size_model_a": powerlaw_model.a,
            "size_model_b": powerlaw_model.b,
            "size_model_c": powerlaw_model.c,
        }

    query_cache, query_count = optimalEpsilon.prepare_query_position_cache(str(query_path), data)

    rows: list[dict[str, object]] = []
    for epsilon in candidate_eps:
        estimated_index_bytes = estimate_index_bytes(n, seg_size, epsilon)
        powerlaw_index_bytes: float | str = ""
        measured_index_bytes: int | str = ""
        selection_index_bytes = estimated_index_bytes
        measured_index_arg: float | None = None

        if size_mode == "measured":
            measured_index_arg = float(measured_index_sizes[epsilon])
            measured_index_bytes = measured_index_sizes[epsilon]
            selection_index_bytes = measured_index_arg
        elif size_mode == "powerlaw":
            if powerlaw_model is None:
                raise RuntimeError("power-law model was not initialized")
            powerlaw_index_bytes = powerlaw_model.predict(epsilon)
            selection_index_bytes = float(powerlaw_index_bytes)
            measured_index_arg = selection_index_bytes
            if epsilon in powerlaw_model.anchor_sizes:
                measured_index_bytes = powerlaw_model.anchor_sizes[epsilon]

        if selection_index_bytes > memory_bytes:
            rows.append(
                {
                    "epsilon": epsilon,
                    "selection_size_mode": size_mode,
                    "estimated_cost": math.inf,
                    "hit_ratio": 0.0,
                    "feasible": 0,
                    "selection_index_bytes": selection_index_bytes,
                    "estimated_index_bytes": estimated_index_bytes,
                    "powerlaw_index_bytes": powerlaw_index_bytes,
                    "measured_index_bytes": measured_index_bytes,
                    "cache_bytes": 0,
                    "cold_start_correction": int(cold_start_correction),
                    "steady_hit_ratio": 0.0,
                    "cold_miss_ratio": 0.0,
                    "expected_distinct_pages": 0.0,
                }
            )
            continue

        cache_bytes = memory_bytes - selection_index_bytes
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
            H=query_cache,
            Q=query_count,
            measured_index_bytes=measured_index_arg,
            cold_start_correction=cold_start_correction,
            return_detail=True,
        )
        rows.append(
            {
                "epsilon": epsilon,
                "selection_size_mode": size_mode,
                "estimated_cost": float(cost),
                "hit_ratio": float(hit_ratio),
                "feasible": 1,
                "selection_index_bytes": selection_index_bytes,
                "estimated_index_bytes": estimated_index_bytes,
                "powerlaw_index_bytes": powerlaw_index_bytes,
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
                "selection_size_mode",
                "estimated_cost",
                "hit_ratio",
                "feasible",
                "selection_index_bytes",
                "estimated_index_bytes",
                "powerlaw_index_bytes",
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
    return int(best["epsilon"]), rows, size_model_info


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


def with_repeat_fields(
    rows: list[dict[str, object]],
    repeat_id: int,
    repeat_count: int,
) -> list[dict[str, object]]:
    if repeat_count <= 1:
        return rows
    tagged: list[dict[str, object]] = []
    for row in rows:
        tagged_row = {"repeat": repeat_id, "total_repeats": repeat_count}
        tagged_row.update(row)
        tagged.append(tagged_row)
    return tagged


def is_blank(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none"}


def parse_numeric(value: object) -> float | None:
    if is_blank(value):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def format_aggregate_number(value: float) -> float | int:
    rounded = round(value)
    if abs(value - rounded) < 1e-9:
        return int(rounded)
    return value


def mode_value(values: list[object]) -> object:
    counts: dict[str, int] = {}
    originals: dict[str, object] = {}
    for value in values:
        if is_blank(value):
            continue
        key = str(value).strip()
        counts[key] = counts.get(key, 0) + 1
        originals.setdefault(key, value)
    if not counts:
        return ""

    def sort_key(item: tuple[str, int]) -> tuple[int, float, str]:
        key, count = item
        numeric = parse_numeric(key)
        numeric_key = numeric if numeric is not None else math.inf
        return (-count, numeric_key, key)

    winner = sorted(counts.items(), key=sort_key)[0][0]
    return originals[winner]


def distinct_values(values: list[object]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if is_blank(value):
            continue
        text = str(value).strip()
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


def aggregate_repeat_rows(
    rows: list[dict[str, object]],
    group_fields: list[str],
) -> list[dict[str, object]]:
    if not rows:
        return []

    groups: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")).strip() for field in group_fields)
        groups.setdefault(key, []).append(row)

    all_fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in all_fields:
                all_fields.append(field)

    skip_fields = set(group_fields) | {"repeat", "total_repeats"}
    categorical_numeric_fields = {"checksum"}
    aggregated: list[dict[str, object]] = []
    for _, group_rows in groups.items():
        out: dict[str, object] = {}
        first = group_rows[0]
        for field in group_fields:
            out[field] = first.get(field, "")
        out["repeat_count"] = len(group_rows)

        for field in all_fields:
            if field in skip_fields:
                continue
            values = [row.get(field, "") for row in group_rows]
            nonblank = [value for value in values if not is_blank(value)]
            if field == "epsilon":
                numeric = [number for number in (parse_numeric(value) for value in values) if number is not None]
                out[field] = mode_value(values)
                out["epsilon_values"] = ";".join(distinct_values(values))
                if numeric:
                    out["epsilon_mean"] = format_aggregate_number(float(np.mean(numeric)))
                    out["epsilon_std"] = float(np.std(numeric, ddof=0)) if len(numeric) > 1 else 0.0
                continue

            if field in categorical_numeric_fields:
                uniques = distinct_values(values)
                out[field] = uniques[0] if len(uniques) == 1 else "varies"
                continue

            numeric = [number for number in (parse_numeric(value) for value in values) if number is not None]
            if numeric and len(numeric) == len(nonblank):
                out[field] = format_aggregate_number(float(np.mean(numeric)))
                if len(numeric) > 1:
                    out[f"{field}_std"] = float(np.std(numeric, ddof=0))
                continue

            uniques = distinct_values(values)
            if not uniques:
                out[field] = ""
            elif len(uniques) == 1:
                out[field] = uniques[0]
            else:
                out[field] = "varies"

        aggregated.append(out)

    def row_sort_key(row: dict[str, object]) -> tuple[str, float, str, str]:
        selector = str(row.get("selector", ""))
        ratio = parse_numeric(row.get("cache_ratio", ""))
        ratio_key = ratio if ratio is not None else -1.0
        policy = str(row.get("policy", ""))
        strategy = str(row.get("strategy", ""))
        return selector, ratio_key, policy, strategy

    return sorted(aggregated, key=row_sort_key)


def run_one_repeat(
    args: argparse.Namespace,
    repo_root: Path,
    data_path: Path,
    query_path: Path,
    keys: int,
    candidate_eps: list[int],
    size_model_eps: list[int],
    cache_ratios: list[float],
    policies_csv: str,
    output_dir: Path,
    repeat_id: int,
    repeat_count: int,
    force_index_rebuild: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cam_bin = resolve_executable(args.cam_bin, repo_root)
    index_size_bin = resolve_executable(args.index_size_bin, repo_root)
    tuner_bin = resolve_executable(args.tuner_bin, repo_root)
    cam_strategy = args.strategies.split(",")[0].strip()
    if cam_strategy.lower() == "all":
        cam_strategy = "all_in_once"

    print(f"[+] repeat {repeat_id}/{repeat_count}: output={output_dir}")
    memory_bytes = int(args.M) * 1024 * 1024
    cam_candidates_csv = output_dir / "cam_candidate_costs.csv"
    index_size_csv = output_dir / "cam_candidate_index_sizes.csv"
    powerlaw_anchor_csv = output_dir / "cam_powerlaw_anchor_index_sizes.csv"
    powerlaw_model_csv = output_dir / "cam_powerlaw_size_model.csv"
    selected_index_size_csv = output_dir / "cam_selected_index_size.csv"
    cam_t0 = time.perf_counter()
    cam_epsilon, cam_candidate_rows, cam_size_model_info = choose_cam_epsilon(
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
        size_mode=args.cam_size_mode,
        size_model_eps=size_model_eps,
        powerlaw_anchor_csv=powerlaw_anchor_csv,
        powerlaw_model_csv=powerlaw_model_csv,
        dry_run=args.dry_run,
        cold_start_correction=args.cold_start_correction,
        force_index_rebuild=force_index_rebuild,
    )
    cam_tuning_time_s = None if args.dry_run else time.perf_counter() - cam_t0
    cam_feasible_candidates = sum(1 for row in cam_candidate_rows if row.get("feasible"))
    print(f"[+] CAM epsilon={cam_epsilon} from {cam_candidates_csv}")

    selected_index_size_time_s: float | None = None
    measured_sizes = {
        int(row["epsilon"]): int(row["measured_index_bytes"])
        for row in cam_candidate_rows
        if row.get("measured_index_bytes") != ""
    }
    if args.cam_size_mode == "measured" and not args.dry_run:
        cam_selected_measured_index_bytes = measured_sizes[cam_epsilon]
        selected_index_size_path = index_size_csv
        selected_index_size_time_s = 0.0
    elif args.cam_size_mode == "powerlaw" and cam_epsilon in measured_sizes and not args.dry_run:
        cam_selected_measured_index_bytes = measured_sizes[cam_epsilon]
        selected_index_size_path = powerlaw_anchor_csv
        selected_index_size_time_s = 0.0
    else:
        selected_size_t0 = time.perf_counter()
        cam_selected_measured_index_bytes = load_selected_measured_index_size(
            index_size_bin=index_size_bin,
            data_path=data_path,
            keys=keys,
            epsilon=cam_epsilon,
            output_csv=selected_index_size_csv,
            dry_run=args.dry_run,
            force_rebuild=force_index_rebuild,
        )
        selected_index_size_time_s = None if args.dry_run else time.perf_counter() - selected_size_t0
        selected_index_size_path = selected_index_size_csv

    if not args.dry_run and cam_selected_measured_index_bytes > memory_bytes:
        raise RuntimeError(
            "CAM selected epsilon is infeasible after measuring the actual index size: "
            f"epsilon={cam_epsilon}, measured_index_bytes={cam_selected_measured_index_bytes}, "
            f"memory_budget_bytes={memory_bytes}"
        )

    cam_refined_cache_bytes = max(0, memory_bytes - cam_selected_measured_index_bytes)
    cam_uses_refined_cache = args.cam_size_mode in {"estimated", "powerlaw"}
    cam_eval_budget_mode = "fixed-cache" if cam_uses_refined_cache else "measured"
    cam_eval_cache_bytes = cam_refined_cache_bytes if cam_uses_refined_cache else None
    cam_tuning_time_source = {
        "estimated": "wall_clock_cam_estimator",
        "measured": "wall_clock_cam_measured_index_sweep",
        "powerlaw": "wall_clock_cam_powerlaw_size_model",
    }[args.cam_size_mode]
    cam_index_size_path = {
        "estimated": "",
        "measured": str(index_size_csv),
        "powerlaw": str(powerlaw_anchor_csv),
    }[args.cam_size_mode]

    run_metadata: list[dict[str, object]] = []
    cam_summary_name = {
        "estimated": "cam_refined_summary.csv",
        "measured": "cam_measured_summary.csv",
        "powerlaw": "cam_powerlaw_refined_summary.csv",
    }[args.cam_size_mode]
    cam_summary = output_dir / cam_summary_name
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
        budget_mode=cam_eval_budget_mode,
        query_limit=args.query_limit,
        cache_bytes=cam_eval_cache_bytes,
        dry_run=args.dry_run,
    )
    cam_metadata = {
        "selector": "CAM",
        "cache_ratio": "",
        "tuner_space_bytes": "",
        "epsilon": cam_epsilon,
        "cam_size_mode": args.cam_size_mode,
        "tuning_time_s": fmt_seconds(cam_tuning_time_s),
        "tuning_time_source": cam_tuning_time_source,
        "tuning_time_cached": 0,
        "selected_index_size_time_s": fmt_seconds(selected_index_size_time_s),
        "candidate_count": len(cam_candidate_rows),
        "feasible_candidates": cam_feasible_candidates,
        "candidate_costs_path": str(cam_candidates_csv),
        "index_size_path": cam_index_size_path,
        "selected_index_size_path": str(selected_index_size_path),
        "selected_measured_index_bytes": cam_selected_measured_index_bytes,
        "refined_cache_bytes": cam_refined_cache_bytes,
        "evaluation_budget_mode": cam_eval_budget_mode,
        "summary_path": str(cam_summary),
    }
    cam_metadata.update(cam_size_model_info)
    run_metadata.append(cam_metadata)

    cam_tuning_row = {
        "selector": "CAM",
        "cache_ratio": "",
        "epsilon": cam_epsilon,
        "cam_size_mode": args.cam_size_mode,
        "tuning_time_s": fmt_seconds(cam_tuning_time_s),
        "tuning_time_source": cam_tuning_time_source,
        "cached": 0,
        "selected_index_size_time_s": fmt_seconds(selected_index_size_time_s),
        "log_path": str(cam_candidates_csv),
        "candidate_count": len(cam_candidate_rows),
        "feasible_candidates": cam_feasible_candidates,
        "tuner_space_bytes": "",
        "cache_bytes": cam_refined_cache_bytes,
        "index_size_path": cam_index_size_path,
        "selected_index_size_path": str(selected_index_size_path),
        "selected_measured_index_bytes": cam_selected_measured_index_bytes,
        "evaluation_budget_mode": cam_eval_budget_mode,
    }
    cam_tuning_row.update(cam_size_model_info)
    tuning_rows: list[dict[str, object]] = [cam_tuning_row]
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

    tagged_metadata = with_repeat_fields(run_metadata, repeat_id, repeat_count)
    tagged_tuning_rows = with_repeat_fields(tuning_rows, repeat_id, repeat_count)
    plan_path = output_dir / "experiment_plan.csv"
    write_combined_summary(plan_path, tagged_metadata)
    tuning_path = output_dir / "tuning_time_summary.csv"
    write_combined_summary(tuning_path, tagged_tuning_rows)

    combined_rows: list[dict[str, object]] = []
    if not args.dry_run:
        for meta in run_metadata:
            for row in read_csv_rows(Path(str(meta["summary_path"]))):
                combined = dict(meta)
                combined.update(row)
                combined_rows.append(combined)
        combined_rows = with_repeat_fields(combined_rows, repeat_id, repeat_count)
        combined_path = output_dir / "comparison_summary.csv"
        write_combined_summary(combined_path, combined_rows)
        print(f"[+] combined summary: {combined_path}")

    print(f"[+] tuning time summary: {tuning_path}")
    print(f"[+] experiment plan: {plan_path}")
    return tagged_metadata, tagged_tuning_rows, combined_rows


def write_multi_repeat_outputs(
    output_dir: Path,
    all_metadata: list[dict[str, object]],
    all_tuning_rows: list[dict[str, object]],
    all_comparison_rows: list[dict[str, object]],
    dry_run: bool,
) -> None:
    plan_path = output_dir / "experiment_plan.csv"
    write_combined_summary(plan_path, all_metadata)
    plan_repeats_path = output_dir / "experiment_plan_repeats.csv"
    write_combined_summary(plan_repeats_path, all_metadata)
    tuning_repeats_path = output_dir / "tuning_time_repeats.csv"
    write_combined_summary(tuning_repeats_path, all_tuning_rows)

    tuning_mean_rows = aggregate_repeat_rows(all_tuning_rows, ["selector", "cache_ratio"])
    tuning_mean_path = output_dir / "tuning_time_summary.csv"
    write_combined_summary(tuning_mean_path, tuning_mean_rows)
    tuning_mean_alias = output_dir / "tuning_time_summary_mean.csv"
    write_combined_summary(tuning_mean_alias, tuning_mean_rows)

    if not dry_run:
        comparison_repeats_path = output_dir / "comparison_summary_repeats.csv"
        write_combined_summary(comparison_repeats_path, all_comparison_rows)
        comparison_mean_rows = aggregate_repeat_rows(
            all_comparison_rows,
            ["selector", "cache_ratio", "policy", "strategy"],
        )
        comparison_mean_path = output_dir / "comparison_summary.csv"
        write_combined_summary(comparison_mean_path, comparison_mean_rows)
        comparison_mean_alias = output_dir / "comparison_summary_mean.csv"
        write_combined_summary(comparison_mean_alias, comparison_mean_rows)
        print(f"[+] combined repeat details: {comparison_repeats_path}")
        print(f"[+] averaged comparison summary: {comparison_mean_path}")

    print(f"[+] experiment plan: {plan_path}")
    print(f"[+] repeat plan details: {plan_repeats_path}")
    print(f"[+] tuning repeat details: {tuning_repeats_path}")
    print(f"[+] averaged tuning summary: {tuning_mean_path}")


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.force_index_rebuild and args.reuse_index_size_csv:
        print("[*] --force-index-rebuild overrides --reuse-index-size-csv")

    repo_root = Path(__file__).resolve().parents[1]
    datasets_directory = Path(args.datasets_directory).expanduser()
    data_path = resolve_input(args.data, datasets_directory)
    query_path = resolve_input(args.queries, datasets_directory)
    keys = int(args.keys or infer_key_count(data_path))
    candidate_eps = parse_int_list(args.candidate_eps)
    size_model_eps = parse_int_list(args.cam_size_model_eps)
    cache_ratios = parse_float_list(args.cache_ratios)
    policies_csv = args.policies.replace(" ", ",")
    dataset_tag = args.dataset_tag or data_path.stem
    output_dir = Path(args.output_dir).resolve() / f"{dataset_tag}_M{args.M}"
    output_dir.mkdir(parents=True, exist_ok=True)

    force_index_rebuild = args.force_index_rebuild or (args.repeats > 1 and not args.reuse_index_size_csv)
    if args.repeats > 1 and force_index_rebuild:
        print("[+] multi-repeat mode: rebuilding CAM index-size artifacts in every repeat")

    all_metadata: list[dict[str, object]] = []
    all_tuning_rows: list[dict[str, object]] = []
    all_comparison_rows: list[dict[str, object]] = []
    for repeat_id in range(1, args.repeats + 1):
        repeat_output_dir = output_dir if args.repeats == 1 else output_dir / f"repeat_{repeat_id}"
        metadata, tuning_rows, comparison_rows = run_one_repeat(
            args=args,
            repo_root=repo_root,
            data_path=data_path,
            query_path=query_path,
            keys=keys,
            candidate_eps=candidate_eps,
            size_model_eps=size_model_eps,
            cache_ratios=cache_ratios,
            policies_csv=policies_csv,
            output_dir=repeat_output_dir,
            repeat_id=repeat_id,
            repeat_count=args.repeats,
            force_index_rebuild=force_index_rebuild,
        )
        all_metadata.extend(metadata)
        all_tuning_rows.extend(tuning_rows)
        all_comparison_rows.extend(comparison_rows)

    if args.repeats > 1:
        write_multi_repeat_outputs(
            output_dir=output_dir,
            all_metadata=all_metadata,
            all_tuning_rows=all_tuning_rows,
            all_comparison_rows=all_comparison_rows,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
