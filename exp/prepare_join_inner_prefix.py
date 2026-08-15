#!/usr/bin/env python3
"""Create a key-only inner relation from an exact prefix of a source dataset."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

MIB = 1024 * 1024
COPY_CHUNK_BYTES = MIB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the first N MiB of a raw fixed-width-key dataset into a new "
            "dataset file. Existing matching prefixes are reused."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size-mib", type=int, default=10)
    parser.add_argument("--record-bytes", type=int, default=8)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace an existing output that is not the requested prefix.",
    )
    return parser.parse_args()


def matches_source_prefix(source: Path, output: Path, size_bytes: int) -> bool:
    if not output.is_file() or output.stat().st_size != size_bytes:
        return False
    remaining = size_bytes
    with source.open("rb") as source_file, output.open("rb") as output_file:
        while remaining:
            want = min(COPY_CHUNK_BYTES, remaining)
            if source_file.read(want) != output_file.read(want):
                return False
            remaining -= want
    return True


def write_prefix(source: Path, output: Path, size_bytes: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with source.open("rb") as source_file, tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            remaining = size_bytes
            while remaining:
                chunk = source_file.read(min(COPY_CHUNK_BYTES, remaining))
                if not chunk:
                    raise EOFError(
                        f"source ended with {remaining} requested bytes remaining: {source}"
                    )
                temp_file.write(chunk)
                remaining -= len(chunk)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, output)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if source == output:
        raise ValueError("--source and --output must be different files")
    if args.size_mib <= 0:
        raise ValueError("--size-mib must be positive")
    if args.record_bytes <= 0:
        raise ValueError("--record-bytes must be positive")
    if not source.is_file():
        raise FileNotFoundError(f"source dataset does not exist: {source}")

    size_bytes = args.size_mib * MIB
    if size_bytes % args.record_bytes:
        raise ValueError(
            f"requested {size_bytes} bytes is not divisible by record width "
            f"{args.record_bytes}"
        )
    if source.stat().st_size < size_bytes:
        raise ValueError(
            f"source has {source.stat().st_size} bytes, fewer than requested {size_bytes}"
        )

    records = size_bytes // args.record_bytes
    if matches_source_prefix(source, output, size_bytes):
        print(
            f"[=] prefix dataset already valid: {output} "
            f"bytes={size_bytes} records={records}"
        )
        return 0
    if output.exists() and not args.force:
        raise FileExistsError(
            f"output exists but is not the requested source prefix: {output}; "
            "pass --force to replace it atomically"
        )

    write_prefix(source, output, size_bytes)
    print(f"[+] wrote prefix dataset: {output} bytes={size_bytes} records={records}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
