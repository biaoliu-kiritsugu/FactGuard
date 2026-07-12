#!/usr/bin/env python3
"""Combine prediction shards, validate uniqueness, and sort by row index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=4200)
    args = parser.parse_args()

    rows: dict[int, dict] = {}
    for path in sorted(args.input_dir.glob("part-*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                row_idx = int(row["row_idx"])
                if row_idx in rows:
                    raise ValueError(f"Duplicate row_idx={row_idx}")
                rows[row_idx] = row
    if len(rows) != args.expected or set(rows) != set(range(args.expected)):
        missing = sorted(set(range(args.expected)) - set(rows))
        raise ValueError(f"Expected {args.expected} rows, got {len(rows)}; missing={missing[:20]}")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as writer:
        for row_idx in sorted(rows):
            writer.write(json.dumps(rows[row_idx], ensure_ascii=False) + "\n")
    print(f"Merged {len(rows)} rows into {args.output_file}")


if __name__ == "__main__":
    main()
