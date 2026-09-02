#!/usr/bin/env python3
"""Convert FactGuard's legacy merged JSONL files into Hugging Face Parquet.

The exporter intentionally keeps a compact, English-keyed public schema. Large
internal generation traces and duplicate copies of the full document are not
included. Conversion is streaming and writes one compressed Parquet file per
split by default, with optional sharding for larger future releases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("document", pa.string(), nullable=False),
        pa.field("question", pa.string(), nullable=False),
        pa.field("response", pa.string(), nullable=False),
        pa.field("answerability", pa.string(), nullable=False),
        pa.field("category", pa.string(), nullable=False),
        pa.field("perturbation", pa.string(), nullable=False),
        pa.field("language", pa.string(), nullable=False),
        pa.field("domain", pa.string(), nullable=False),
        pa.field("source_file", pa.string(), nullable=False),
        pa.field("legacy_length_bucket", pa.string(), nullable=False),
        pa.field("document_char_length", pa.int64(), nullable=False),
        pa.field("question_char_length", pa.int64(), nullable=False),
        pa.field("response_char_length", pa.int64(), nullable=False),
        pa.field("evidence", pa.string(), nullable=False),
        pa.field("original_question", pa.string(), nullable=False),
        pa.field("original_answer", pa.string(), nullable=False),
        pa.field("unanswerable_reason", pa.string(), nullable=False),
        pa.field("original_entity", pa.string(), nullable=False),
        pa.field("substituted_entity", pa.string(), nullable=False),
        pa.field("unsupported_condition", pa.string(), nullable=False),
        pa.field("question_type", pa.string(), nullable=False),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=Path("data/merged_train.jsonl"))
    parser.add_argument(
        "--validation-file",
        "--dev-file",
        dest="validation_file",
        type=Path,
        default=Path("data/merged_dev.jsonl"),
        help="Legacy development split, published as the Hugging Face validation split.",
    )
    parser.add_argument("--test-file", type=Path, default=Path("data/merged_test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("hf_dataset"))
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=None,
        help=(
            "Optional maximum examples per Parquet shard. By default, each "
            "split is written to one file: train.parquet, "
            "validation.parquet, and test.parquet."
        ),
    )
    parser.add_argument(
        "--compression",
        choices=("zstd", "snappy", "gzip", "none"),
        default="zstd",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=9,
        help="Used by codecs such as zstd; ignored when unsupported.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete existing <output-dir>/data/*.parquet files.",
    )
    return parser.parse_args()


def text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def infer_subtype(source: str) -> str:
    if "dandian" in source:
        return "dandian"
    if "misattr" in source:
        return "misattr"
    if "impossible" in source:
        return "impossible"
    raise ValueError(f"Cannot infer FactGuard subtype from source={source!r}")


def infer_language(source: str, origin: dict[str, Any]) -> str:
    meta = origin.get("meta")
    if isinstance(meta, dict) and meta.get("lang") in {"en", "zh"}:
        return meta["lang"]
    if "_zh_" in source:
        return "zh"
    if "_en_" in source:
        return "en"
    raise ValueError(f"Cannot infer language from source={source!r}")


def infer_domain(source: str, origin: dict[str, Any]) -> str:
    meta = origin.get("meta")
    if isinstance(meta, dict) and isinstance(meta.get("domain"), str):
        return meta["domain"]
    match = re.search(r"_(?:en|zh)_([^._]+)", source)
    return match.group(1) if match else "unknown"


def normalize_row(raw: dict[str, Any], split: str, row_index: int) -> dict[str, Any]:
    origin = raw.get("origin")
    if not isinstance(origin, dict):
        raise TypeError(f"{split} row {row_index}: origin must be an object")

    source = text(raw.get("source"))
    subtype = infer_subtype(source)
    is_positive = raw.get("is_positive")
    if not isinstance(is_positive, bool):
        raise TypeError(f"{split} row {row_index}: is_positive must be boolean")

    if subtype == "dandian":
        document = text(origin.get("doc") if is_positive else origin.get("refuse_doc"))
        question = text(origin.get("问题"))
        original_question = question
        original_answer = text(origin.get("改写后答案") or origin.get("答案"))
        evidence = text(origin.get("答案依据"))
        reason = "" if is_positive else text(origin.get("拒答回复语"))
        question_type = text(origin.get("question_type") or origin.get("问答对类型"))
    elif subtype == "misattr":
        document = text(origin.get("doc"))
        original_question = text(origin.get("原始问题"))
        question = original_question if is_positive else text(origin.get("问题"))
        original_answer = text(origin.get("正确答案"))
        evidence = text(origin.get("依据"))
        reason = "" if is_positive else text(origin.get("纠正"))
        question_type = ""
    else:
        document = text(origin.get("doc"))
        original_question = text(origin.get("问题"))
        question = text(origin.get("增加条件的问题"))
        original_answer = ""
        evidence = text(origin.get("片段"))
        reason = text(origin.get("解释"))
        question_type = ""

    response = text(raw.get("output"))
    required = {
        "document": document,
        "question": question,
        "response": response,
        "document_id": text(raw.get("uid")),
        "source_file": source,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"{split} row {row_index}: empty required fields {missing}")

    if is_positive:
        answerability = "answerable"
        category = "answerable"
        perturbation = "none"
    elif subtype == "dandian":
        answerability = "unanswerable"
        category = "lack_of_evidence"
        perturbation = "evidence_removal"
    elif subtype == "misattr":
        answerability = "unanswerable"
        category = "misleading_evidence"
        perturbation = "entity_substitution"
    else:
        answerability = "unanswerable"
        category = "misleading_evidence"
        perturbation = "impossible_condition"

    return {
        "id": f"{split}-{row_index:06d}",
        "document_id": required["document_id"],
        "document": document,
        "question": question,
        "response": response,
        "answerability": answerability,
        "category": category,
        "perturbation": perturbation,
        "language": infer_language(source, origin),
        "domain": infer_domain(source, origin),
        "source_file": source,
        "legacy_length_bucket": text(origin.get("len_range")),
        "document_char_length": len(document),
        "question_char_length": len(question),
        "response_char_length": len(response),
        "evidence": evidence,
        "original_question": original_question,
        "original_answer": original_answer,
        "unanswerable_reason": reason,
        "original_entity": text(origin.get("替换前实体")),
        "substituted_entity": text(origin.get("替换后实体")),
        "unsupported_condition": text(origin.get("增加的条件")),
        "question_type": question_type,
    }


def update_stats(stats: dict[str, Any], row: dict[str, Any]) -> None:
    stats["num_examples"] += 1
    stats["document_chars"] += row["document_char_length"]
    for field in ("answerability", "category", "perturbation", "language", "domain"):
        stats[field][row[field]] += 1


def write_shard(
    rows: list[dict[str, Any]],
    path: Path,
    compression: str | None,
    compression_level: int,
) -> int:
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    kwargs: dict[str, Any] = {
        "compression": compression,
        "use_dictionary": [
            "answerability",
            "category",
            "perturbation",
            "language",
            "domain",
            "source_file",
            "legacy_length_bucket",
            "question_type",
        ],
        "write_statistics": True,
    }
    if compression in {"zstd", "gzip"}:
        kwargs["compression_level"] = compression_level
    pq.write_table(table, path, **kwargs)
    return path.stat().st_size


def parquet_writer(
    path: Path,
    compression: str | None,
    compression_level: int,
) -> pq.ParquetWriter:
    kwargs: dict[str, Any] = {
        "compression": compression,
        "use_dictionary": [
            "answerability",
            "category",
            "perturbation",
            "language",
            "domain",
            "source_file",
            "legacy_length_bucket",
            "question_type",
        ],
        "write_statistics": True,
    }
    if compression in {"zstd", "gzip"}:
        kwargs["compression_level"] = compression_level
    return pq.ParquetWriter(path, SCHEMA, **kwargs)


def convert_split(
    source_path: Path,
    split: str,
    data_dir: Path,
    rows_per_shard: int,
    compression: str | None,
    compression_level: int,
) -> dict[str, Any]:
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    stats: dict[str, Any] = {
        "num_examples": 0,
        "num_bytes": 0,
        "document_chars": 0,
        "answerability": Counter(),
        "category": Counter(),
        "perturbation": Counter(),
        "language": Counter(),
        "domain": Counter(),
        "shards": [],
        "_document_ids": set(),
        "_document_hashes": set(),
        "_questions": set(),
    }
    rows: list[dict[str, Any]] = []
    shard_index = 0
    single_path = data_dir / f"{split}.parquet"
    single_writer = (
        parquet_writer(single_path, compression, compression_level)
        if rows_per_shard is None
        else None
    )
    write_batch_size = rows_per_shard or 1000

    def flush() -> None:
        nonlocal shard_index
        if not rows:
            return
        if single_writer is not None:
            single_writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
            rows.clear()
            return
        shard_name = (
            f"{split}-{shard_index:05d}.parquet"
        )
        shard_path = data_dir / shard_name
        size = write_shard(rows, shard_path, compression, compression_level)
        stats["num_bytes"] += size
        stats["shards"].append({"path": f"data/{shard_name}", "num_examples": len(rows), "num_bytes": size})
        print(f"{split}: wrote {shard_name} ({len(rows)} examples, {size / 2**20:.1f} MiB)")
        rows.clear()
        shard_index += 1

    try:
        with source_path.open("r", encoding="utf-8") as reader:
            for row_index, line in enumerate(reader):
                if not line.strip():
                    continue
                raw = json.loads(line)
                row = normalize_row(raw, split, row_index)
                rows.append(row)
                update_stats(stats, row)
                stats["_document_ids"].add(row["document_id"])
                stats["_document_hashes"].add(
                    hashlib.sha256(row["document"].encode("utf-8")).hexdigest()
                )
                stats["_questions"].add(row["question"])
                if len(rows) >= write_batch_size:
                    flush()
        flush()
    finally:
        if single_writer is not None:
            single_writer.close()

    if rows_per_shard is None:
        size = single_path.stat().st_size
        stats["num_bytes"] = size
        stats["shards"] = [
            {
                "path": f"data/{single_path.name}",
                "num_examples": stats["num_examples"],
                "num_bytes": size,
            }
        ]
        print(
            f"{split}: wrote {single_path.name} "
            f"({stats['num_examples']} examples, {size / 2**20:.1f} MiB)"
        )

    for key in ("answerability", "category", "perturbation", "language", "domain"):
        stats[key] = dict(sorted(stats[key].items()))
    return stats


def schema_summary() -> dict[str, str]:
    return {field.name: str(field.type) for field in SCHEMA}


def write_release_stats(
    output_dir: Path,
    split_stats: dict[str, dict[str, Any]],
    split_overlap: dict[str, dict[str, int]],
) -> None:
    info = {
        "dataset_name": "FactGuard-Bench",
        "format": "parquet",
        "features": schema_summary(),
        "splits": split_stats,
        "totals": {
            "num_examples": sum(item["num_examples"] for item in split_stats.values()),
            "num_bytes": sum(item["num_bytes"] for item in split_stats.values()),
        },
        "split_overlap": split_overlap,
        "notes": [
            "Character lengths are provided for transparent filtering and are not tokenizer token counts.",
            "The public schema omits internal model traces and duplicate full-document fields.",
            "The legacy development split is published as the Hugging Face validation split.",
        ],
    }
    with (output_dir / "release_stats.json").open("w", encoding="utf-8") as writer:
        json.dump(info, writer, ensure_ascii=False, indent=2)
        writer.write("\n")


def main() -> None:
    args = parse_args()
    if args.rows_per_shard is not None and args.rows_per_shard <= 0:
        raise ValueError("--rows-per-shard must be positive")

    output_dir = args.output_dir.resolve()
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if not args.keep_existing:
        for old_file in data_dir.glob("*.parquet"):
            old_file.unlink()

    compression = None if args.compression == "none" else args.compression
    split_sources = {
        "train": args.train_file,
        "validation": args.validation_file,
        "test": args.test_file,
    }
    split_stats = {
        split: convert_split(
            path,
            split,
            data_dir,
            args.rows_per_shard,
            compression,
            args.compression_level,
        )
        for split, path in split_sources.items()
    }
    split_overlap: dict[str, dict[str, int]] = {}
    split_names = list(split_stats)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            split_overlap[f"{left}__{right}"] = {
                "shared_document_ids": len(
                    split_stats[left]["_document_ids"]
                    & split_stats[right]["_document_ids"]
                ),
                "shared_exact_documents": len(
                    split_stats[left]["_document_hashes"]
                    & split_stats[right]["_document_hashes"]
                ),
                "shared_exact_questions": len(
                    split_stats[left]["_questions"] & split_stats[right]["_questions"]
                ),
            }
    for stats in split_stats.values():
        stats["num_unique_document_ids"] = len(stats.pop("_document_ids"))
        stats["num_unique_exact_documents"] = len(stats.pop("_document_hashes"))
        stats["num_unique_questions"] = len(stats.pop("_questions"))
    write_release_stats(output_dir, split_stats, split_overlap)
    if any(any(metrics.values()) for metrics in split_overlap.values()):
        print(f"WARNING: cross-split overlap detected: {split_overlap}")
    print(
        "Done: "
        f"{sum(s['num_examples'] for s in split_stats.values())} examples, "
        f"{sum(s['num_bytes'] for s in split_stats.values()) / 2**30:.2f} GiB"
    )


if __name__ == "__main__":
    main()
