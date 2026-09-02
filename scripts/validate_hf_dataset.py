#!/usr/bin/env python3
"""Validate an exported FactGuard Hugging Face dataset repository."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from prepare_hf_dataset import SCHEMA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("hf_dataset"))
    parser.add_argument("--expected-train", type=int, default=19100)
    parser.add_argument("--expected-validation", type=int, default=1920)
    parser.add_argument("--expected-test", type=int, default=4200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = {
        "train": args.expected_train,
        "validation": args.expected_validation,
        "test": args.expected_test,
    }
    observed: dict[str, dict] = {}
    seen_ids: set[str] = set()
    split_document_ids: dict[str, set[str]] = {}
    split_document_hashes: dict[str, set[str]] = {}
    split_questions: dict[str, set[str]] = {}

    for split, expected_rows in expected.items():
        single_file = args.dataset_dir / "data" / f"{split}.parquet"
        files = (
            [single_file]
            if single_file.is_file()
            else sorted((args.dataset_dir / "data").glob(f"{split}-*.parquet"))
        )
        if not files:
            raise FileNotFoundError(f"No {split} Parquet file(s) found")
        counts = Counter()
        row_count = 0
        document_ids: set[str] = set()
        document_hashes: set[str] = set()
        questions: set[str] = set()
        for path in files:
            table = pq.read_table(path)
            if table.schema != SCHEMA:
                raise ValueError(f"Schema mismatch in {path}\n{table.schema}\n!=\n{SCHEMA}")
            for row in table.to_pylist():
                row_count += 1
                if row["id"] in seen_ids:
                    raise ValueError(f"Duplicate public id: {row['id']}")
                seen_ids.add(row["id"])
                document_ids.add(row["document_id"])
                document_hashes.add(
                    hashlib.sha256(row["document"].encode("utf-8")).hexdigest()
                )
                questions.add(row["question"])
                for field in ("document", "question", "response"):
                    if not row[field]:
                        raise ValueError(f"{path}: {row['id']} has empty {field}")
                if row["answerability"] == "answerable":
                    if row["category"] != "answerable" or row["perturbation"] != "none":
                        raise ValueError(f"{row['id']}: invalid answerable labels")
                else:
                    if row["category"] not in {"lack_of_evidence", "misleading_evidence"}:
                        raise ValueError(f"{row['id']}: invalid unanswerable category")
                counts[(row["language"], row["domain"], row["perturbation"])] += 1
        if row_count != expected_rows:
            raise ValueError(f"{split}: expected {expected_rows} rows, observed {row_count}")
        observed[split] = {"num_examples": row_count, "groups": dict(counts)}
        split_document_ids[split] = document_ids
        split_document_hashes[split] = document_hashes
        split_questions[split] = questions
        print(f"{split}: {row_count} rows across {len(files)} Parquet file(s)")

    info_path = args.dataset_dir / "release_stats.json"
    with info_path.open(encoding="utf-8") as reader:
        info = json.load(reader)
    for split, expected_rows in expected.items():
        if info["splits"][split]["num_examples"] != expected_rows:
            raise ValueError(f"{info_path}: stale count for {split}")

    overlap: dict[str, dict[str, int]] = {}
    split_names = list(expected)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            pair = f"{left}__{right}"
            overlap[pair] = {
                "shared_document_ids": len(
                    split_document_ids[left] & split_document_ids[right]
                ),
                "shared_exact_documents": len(
                    split_document_hashes[left] & split_document_hashes[right]
                ),
                "shared_exact_questions": len(
                    split_questions[left] & split_questions[right]
                ),
            }
    print(f"Cross-split overlap audit: {overlap}")
    if any(metrics["shared_exact_documents"] for metrics in overlap.values()):
        print(
            "WARNING: exact document text appears across splits. Preserve these "
            "splits only for paper-result reproducibility; use a document-disjoint "
            "split for new generalization claims."
        )
    print(f"Validation passed: {len(seen_ids)} unique examples")


if __name__ == "__main__":
    main()
