#!/usr/bin/env python3
"""Convert FactGuard JSONL into verl's message-based Parquet format.

The conversion is streaming so the 2.4 GB training JSONL is never held in
memory in full.  Each row has a two-turn conversation; verl's SFT loss is
computed only on the assistant turn by ``FactGuardSFTDataset``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


MESSAGE_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("role", pa.string(), nullable=False),
            pa.field("content", pa.string(), nullable=False),
        ]
    )
)
SCHEMA = pa.schema(
    [
        pa.field("messages", MESSAGE_TYPE, nullable=False),
        pa.field("uid", pa.string()),
        pa.field("source", pa.string()),
        pa.field("is_positive", pa.bool_()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="FactGuard merged JSONL")
    parser.add_argument("--dst", type=Path, required=True, help="Output Parquet file")
    parser.add_argument("--batch-size", type=int, default=256, help="Rows per Parquet write batch")
    parser.add_argument("--limit", type=int, default=None, help="Convert at most N valid rows (for smoke tests)")
    return parser.parse_args()


def to_row(raw: dict) -> dict | None:
    user_text = raw.get("input")
    assistant_text = raw.get("output")
    if not isinstance(user_text, str) or not user_text.strip():
        return None
    if not isinstance(assistant_text, str) or not assistant_text.strip():
        return None
    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "uid": str(raw["uid"]) if raw.get("uid") is not None else None,
        "source": str(raw["source"]) if raw.get("source") is not None else None,
        "is_positive": raw.get("is_positive") if isinstance(raw.get("is_positive"), bool) else None,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not args.src.is_file():
        raise FileNotFoundError(args.src)

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.dst.with_name(f".{args.dst.name}.tmp-{os.getpid()}")
    writer: pq.ParquetWriter | None = None
    rows: list[dict] = []
    kept = dropped = invalid_json = 0

    def flush() -> None:
        nonlocal writer
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(tmp_path, SCHEMA, compression="zstd")
        writer.write_table(table)
        rows.clear()

    try:
        with args.src.open("r", encoding="utf-8") as src:
            for line_number, line in enumerate(src, 1):
                if args.limit is not None and kept >= args.limit:
                    break
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    invalid_json += 1
                    dropped += 1
                    continue
                row = to_row(raw)
                if row is None:
                    dropped += 1
                    continue
                rows.append(row)
                kept += 1
                if len(rows) >= args.batch_size:
                    flush()
                if kept % 1000 == 0:
                    print(f"converted {kept} rows (source line {line_number})", flush=True)
        flush()
        if writer is None:
            raise RuntimeError("No valid input/output rows were found")
        writer.close()
        writer = None
        tmp_path.replace(args.dst)
    finally:
        if writer is not None:
            writer.close()
        if tmp_path.exists():
            tmp_path.unlink()

    print(
        f"done: kept={kept} dropped={dropped} invalid_json={invalid_json} "
        f"output={args.dst} size={args.dst.stat().st_size} bytes"
    )


if __name__ == "__main__":
    main()
