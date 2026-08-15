#!/usr/bin/env python3
"""
Measure the affine I/O model parameter alpha on a local Linux storage target.

Model:
    T(x) = s + t*x
    C(x) = T(x)/s = 1 + alpha*x
    alpha = t/s

By default x is measured in bytes, so alpha has units byte^-1.
The script also reports alpha * page_size and the half-bandwidth size 1/alpha.

Requirements:
    - Linux
    - fio installed and available in PATH
    - target must already exist
    - target is READ ONLY in this benchmark; fio is invoked with --readonly

Example:
    python3 measure_affine_alpha.py /path/to/large/existing/file
    sudo python3 measure_affine_alpha.py /dev/nvme0n1 --page-size 4096

For a regular file, use a large, already-populated file. Do not benchmark a sparse file.
"""

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


DEFAULT_SIZES = [
    4 * 1024,
    8 * 1024,
    16 * 1024,
    32 * 1024,
    64 * 1024,
    128 * 1024,
    256 * 1024,
    512 * 1024,
    1 * 1024 * 1024,
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
    16 * 1024 * 1024,
]


def human_bytes(n: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if abs(x) < 1024.0 or u == units[-1]:
            return f"{x:.3f} {u}"
        x /= 1024.0
    return f"{x:.3f} TiB"


def parse_size(s: str) -> int:
    s = s.strip().lower()
    suffixes = {
        "k": 1024, "kb": 1024, "kib": 1024,
        "m": 1024**2, "mb": 1024**2, "mib": 1024**2,
        "g": 1024**3, "gb": 1024**3, "gib": 1024**3,
        "b": 1,
    }
    for suffix in sorted(suffixes, key=len, reverse=True):
        if s.endswith(suffix):
            return int(float(s[:-len(suffix)]) * suffixes[suffix])
    return int(s)


def extract_mean_latency_ns(job: dict) -> float:
    r = job["read"]

    # Newer fio JSON normally contains total latency ("lat_ns").
    if "lat_ns" in r and isinstance(r["lat_ns"], dict) and "mean" in r["lat_ns"]:
        return float(r["lat_ns"]["mean"])

    # Fall back to completion latency if total latency is unavailable.
    if "clat_ns" in r and isinstance(r["clat_ns"], dict) and "mean" in r["clat_ns"]:
        return float(r["clat_ns"]["mean"])

    # Compatibility with older fio output units.
    if "lat_us" in r and isinstance(r["lat_us"], dict) and "mean" in r["lat_us"]:
        return float(r["lat_us"]["mean"]) * 1e3
    if "clat_us" in r and isinstance(r["clat_us"], dict) and "mean" in r["clat_us"]:
        return float(r["clat_us"]["mean"]) * 1e3

    # Last-resort estimate from runtime / completed IOs.
    total_ios = r.get("total_ios")
    runtime_ms = r.get("runtime")
    if total_ios and runtime_ms is not None:
        return float(runtime_ms) * 1e6 / float(total_ios)

    raise RuntimeError("Could not find a usable latency field in fio JSON output.")


def run_fio(target: str, bs: int, number_ios: int, seed: int) -> float:
    cmd = [
        "fio",
        "--name=affine-alpha",
        f"--filename={target}",
        "--rw=randread",
        f"--bs={bs}",
        f"--ba={4096}",
        "--direct=1",
        "--ioengine=psync",
        "--iodepth=1",
        "--numjobs=1",
        f"--number_ios={number_ios}",
        "--random_distribution=random",
        "--randrepeat=0",
        f"--randseed={seed}",
        "--readonly",
        "--group_reporting=1",
        "--output-format=json",
    ]

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "fio failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr:\n{p.stderr}\n"
            f"stdout:\n{p.stdout[:2000]}"
        )

    try:
        data = json.loads(p.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Could not parse fio JSON output: {e}\nOutput:\n{p.stdout[:2000]}")

    if not data.get("jobs"):
        raise RuntimeError("fio JSON contains no jobs.")

    return extract_mean_latency_ns(data["jobs"][0])


def linear_regression(xs, ys):
    """OLS fit y = intercept + slope*x; returns intercept, slope, R^2."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("Need at least two points.")

    xbar = statistics.fmean(xs)
    ybar = statistics.fmean(ys)
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("All x values are identical.")

    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sxx
    intercept = ybar - slope * xbar

    yhat = [intercept + slope * x for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return intercept, slope, r2


def main():
    ap = argparse.ArgumentParser(
        description="Measure affine-model alpha from random direct-read latency vs I/O size."
    )
    ap.add_argument("target", help="Existing regular file or raw block device to READ.")
    ap.add_argument("--page-size", type=parse_size, default=4096,
                    help="Page size used in alpha*k test (default: 4096 bytes).")
    ap.add_argument("--number-ios", type=int, default=64,
                    help="Random reads per I/O size per repetition (default: 64, matching the paper).")
    ap.add_argument("--repeats", type=int, default=3,
                    help="Repetitions per I/O size; median is fitted (default: 3).")
    ap.add_argument("--max-io", type=parse_size, default=16 * 1024 * 1024,
                    help="Largest I/O size to include (default: 16MiB, matching the paper).")
    ap.add_argument("--min-io", type=parse_size, default=4 * 1024,
                    help="Smallest I/O size to include (default: 4KiB).")
    ap.add_argument("--csv", default="affine_alpha_results.csv",
                    help="CSV output path (default: affine_alpha_results.csv).")
    args = ap.parse_args()

    if shutil.which("fio") is None:
        sys.exit("ERROR: fio was not found in PATH. Install fio first.")

    if not os.path.exists(args.target):
        sys.exit(f"ERROR: target does not exist: {args.target}")

    sizes = [x for x in DEFAULT_SIZES if args.min_io <= x <= args.max_io]
    if len(sizes) < 2:
        sys.exit("ERROR: need at least two I/O sizes in the selected range.")

    print("=== Affine I/O alpha benchmark ===")
    print(f"Target      : {args.target}")
    print(f"Page size   : {args.page_size} bytes ({human_bytes(args.page_size)})")
    print(f"Reads/size  : {args.number_ios}")
    print(f"Repeats     : {args.repeats}")
    print("Mode        : random read, O_DIRECT, synchronous QD=1")
    print()

    rows = []
    for i, bs in enumerate(sizes):
        samples = []
        print(f"[{i+1:2d}/{len(sizes)}] bs={human_bytes(bs):>11} :", end="", flush=True)
        for rep in range(args.repeats):
            lat_ns = run_fio(args.target, bs, args.number_ios, seed=12345 + i * 100 + rep)
            samples.append(lat_ns)
            print(f" {lat_ns/1e3:.2f}us", end="", flush=True)
        med = statistics.median(samples)
        rows.append((bs, med, samples))
        print(f"  -> median {med/1e3:.2f}us")

    xs = [float(r[0]) for r in rows]   # bytes
    ys = [float(r[1]) for r in rows]   # ns
    s_ns, t_ns_per_byte, r2 = linear_regression(xs, ys)

    if s_ns <= 0:
        print("\nWARNING: fitted setup cost s <= 0. The affine model is not trustworthy over this range.")
        alpha_per_byte = float("nan")
        alpha_k = float("nan")
        half_bw = float("nan")
    else:
        alpha_per_byte = t_ns_per_byte / s_ns
        alpha_k = alpha_per_byte * args.page_size
        half_bw = 1.0 / alpha_per_byte if alpha_per_byte > 0 else float("inf")

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["io_size_bytes", "median_latency_ns"] +
                   [f"repeat_{i+1}_latency_ns" for i in range(args.repeats)])
        for bs, med, samples in rows:
            w.writerow([bs, med, *samples])

    print("\n=== Fit: T(x) = s + t*x ===")
    print(f"s (setup/intercept)  = {s_ns:.3f} ns = {s_ns/1e3:.3f} us")
    print(f"t (slope)            = {t_ns_per_byte:.9f} ns/byte")
    print(f"R^2                  = {r2:.6f}")
    print(f"alpha = t/s          = {alpha_per_byte:.12e} byte^-1")
    print(f"alpha (per 4KiB)     = {alpha_per_byte * 4096:.9f}  # comparable to Table 2 if 4KiB is one unit")

    print("\n=== Quantity relevant to C_O - C_A ===")
    print(f"k (page size)        = {args.page_size} bytes")
    print(f"alpha * k            = {alpha_k:.9f}")
    if math.isfinite(half_bw):
        print(f"1/alpha              = {half_bw:.1f} bytes = {human_bytes(half_bw)}")

    if math.isnan(alpha_k):
        verdict = "UNRELIABLE: invalid affine fit"
    elif alpha_k < 1:
        verdict = "alpha*k < 1  =>  C_O - C_A > 0  (for epsilon/Cipp > 0)"
    elif alpha_k > 1:
        verdict = "alpha*k > 1  =>  C_O - C_A < 0  (for epsilon/Cipp > 0)"
    else:
        verdict = "alpha*k ~= 1 =>  C_O ~= C_A"

    print(f"\nVERDICT: {verdict}")
    print(f"CSV saved to: {args.csv}")

    if r2 < 0.95:
        print(
            "\nNOTE: R^2 is below 0.95. Your device/workload is not well described by one "
            "global affine line over this I/O-size range. Re-run with --max-io set close "
            "to the largest all-at-once request size used by your index."
        )


if __name__ == "__main__":
    main()