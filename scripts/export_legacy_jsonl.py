#!/usr/bin/env python3
"""Export normalized FactGuard Parquet shards to the legacy input/output JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_prompt(language: str, document: str, question: str) -> str:
    language_instruction = (
        "Answer in English." if language == "en" else "Answer in Chinese."
    )
    return (
        f"Document:\n{document}\n\n"
        "Answer the question using only the document.\n"
        f"Question: {question}\n{language_instruction}"
    )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as writer:
        for pattern in args.input:
            paths = sorted(pattern.parent.glob(pattern.name))
            if not paths:
                raise FileNotFoundError(pattern)
            for path in paths:
                columns = [
                    "id",
                    "document_id",
                    "document",
                    "question",
                    "response",
                    "language",
                    "source_file",
                    "answerability",
                ]
                for batch in pq.ParquetFile(path).iter_batches(columns=columns):
                    for row in batch.to_pylist():
                        record = {
                            "source": row["source_file"],
                            "uid": row["document_id"],
                            "input": build_prompt(row["language"], row["document"], row["question"]),
                            "output": row["response"],
                            "is_positive": row["answerability"] == "answerable",
                            "public_id": row["id"],
                        }
                        writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1
    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
