#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import struct
import sys
from pathlib import Path


UINT64_SIZE = 8
COPY_BUFFER_SIZE = 64 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an RMI training file by adding the uint64 count header "
            "expected by src/rmi/src/load.rs."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Source uint64 dataset.")
    parser.add_argument("--output", type=Path, required=True, help="Headered output file for RMI training.")
    parser.add_argument(
        "--input-header",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Whether --input already has the leading uint64 count header.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite --output even when it already matches the source size and mtime.",
    )
    return parser.parse_args()


def read_first_u64(path: Path) -> int:
    with path.open("rb") as f:
        data = f.read(UINT64_SIZE)
    if len(data) != UINT64_SIZE:
        raise ValueError(f"dataset is smaller than one uint64: {path}")
    return struct.unpack("<Q", data)[0]


def source_payload_info(path: Path, input_header: str) -> tuple[int, int]:
    size = path.stat().st_size
    if size <= 0 or size % UINT64_SIZE != 0:
        raise ValueError(f"dataset size is not a positive multiple of uint64: {path}")

    total_u64 = size // UINT64_SIZE
    first = read_first_u64(path)
    has_header = first + 1 == total_u64
    if input_header == "yes" and not has_header:
        raise ValueError(f"--input-header=yes but {path} does not contain a valid count header")
    if input_header == "no" and has_header:
        raise ValueError(f"--input-header=no but {path} appears to already contain a count header")

    if input_header == "yes" or (input_header == "auto" and has_header):
        return first, UINT64_SIZE
    return total_u64, 0


def output_is_current(output: Path, input_path: Path, count: int) -> bool:
    if not output.exists():
        return False
    expected_size = (count + 1) * UINT64_SIZE
    try:
        if output.stat().st_size != expected_size:
            return False
        if output.stat().st_mtime < input_path.stat().st_mtime:
            return False
        return read_first_u64(output) == count
    except OSError:
        return False


def write_headered(input_path: Path, output: Path, count: int, payload_offset: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp")
    try:
        with input_path.open("rb") as src, tmp.open("wb") as dst:
            dst.write(struct.pack("<Q", count))
            src.seek(payload_offset)
            shutil.copyfileobj(src, dst, length=COPY_BUFFER_SIZE)
        tmp.replace(output)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path == output:
        raise ValueError("--input and --output must be different paths")

    count, payload_offset = source_payload_info(input_path, args.input_header)
    if not args.force and output_is_current(output, input_path, count):
        print(f"[prepare-rmi-data][skip] {output} already current ({count} keys)")
        return

    write_headered(input_path, output, count, payload_offset)
    print(f"[prepare-rmi-data][write] {input_path} -> {output} ({count} keys)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[prepare-rmi-data][error] {exc}", file=sys.stderr)
        raise
