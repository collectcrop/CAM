#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a prefix-limited RMI collector CSV and fix #num_queries metadata."
    )
    parser.add_argument("--input", type=Path, required=True, help="Source RMI collector CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Limited output CSV.")
    parser.add_argument("--limit", type=int, required=True, help="Maximum number of query rows to keep.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")

    comments: list[str] = []
    header: str | None = None
    rows: list[str] = []

    with args.input.open("r", encoding="utf-8", newline="") as src:
        for line in src:
            if line.startswith("#"):
                comments.append(line)
                continue
            header = line
            break

        if header is None:
            raise ValueError(f"{args.input} does not contain a CSV header")

        for line in src:
            if not line.strip():
                continue
            rows.append(line)
            if len(rows) >= args.limit:
                break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    wrote_num_queries = False
    with args.output.open("w", encoding="utf-8", newline="") as dst:
        for line in comments:
            if line.startswith("#num_queries,"):
                dst.write(f"#num_queries,{len(rows)}\n")
                wrote_num_queries = True
            else:
                dst.write(line)
        if not wrote_num_queries:
            dst.write(f"#num_queries,{len(rows)}\n")
        dst.write(header)
        dst.writelines(rows)

    print(f"[limit_rmi_records] wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
