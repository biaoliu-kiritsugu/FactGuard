#!/usr/bin/env python3
"""Analyze the distribution of string-field lengths in JSONL files."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def get_string_lengths(obj: Any) -> list[int]:
    """Recursively collect lengths of all string values in a JSON object."""
    lengths: list[int] = []
    if isinstance(obj, str):
        lengths.append(len(obj))
    elif isinstance(obj, dict):
        for value in obj.values():
            lengths.extend(get_string_lengths(value))
    elif isinstance(obj, list):
        for value in obj:
            lengths.extend(get_string_lengths(value))
    return lengths


def get_length_category(length: int) -> str:
    if length <= 16 * 1024:
        return "0K-16K"
    if length <= 32 * 1024:
        return "16K-32K"
    if length <= 64 * 1024:
        return "32K-64K"
    if length <= 128 * 1024:
        return "64K-128K"
    return "128K+"


def analyze_jsonl_file(file_path: str | Path) -> dict:
    """Analyze one JSONL file."""
    path = Path(file_path)
    if not path.is_file():
        print(f"File not found: {path}")
        return {}

    print(f"Analyzing file: {path}")
    distribution: dict[str, int] = defaultdict(int)
    lengths_over_128k: list[int] = []
    total_fields = 0

    with path.open("r", encoding="utf-8") as reader:
        for line_number, line in enumerate(reader, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                for length in get_string_lengths(row):
                    category = get_length_category(length)
                    distribution[category] += 1
                    total_fields += 1
                    if category == "128K+":
                        lengths_over_128k.append(length)
            except json.JSONDecodeError:
                print(f"Invalid JSON on line {line_number}")
            except Exception as exc:
                print(f"Error processing line {line_number}: {exc}")

    return {
        "distribution": dict(distribution),
        "total_fields": total_fields,
        "average_length_over_128k": (
            sum(lengths_over_128k) / len(lengths_over_128k)
            if lengths_over_128k
            else None
        ),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python analyze_multiple_jsonl.py <file1.jsonl> [file2.jsonl ...]")
        raise SystemExit(1)

    combined: dict[str, int] = defaultdict(int)
    total_fields = 0
    weighted_over_128k = 0.0
    over_128k_count = 0

    for file_path in sys.argv[1:]:
        result = analyze_jsonl_file(file_path)
        if not result:
            continue
        total_fields += result["total_fields"]
        for category, count in result["distribution"].items():
            combined[category] += count
        average = result["average_length_over_128k"]
        count = result["distribution"].get("128K+", 0)
        if average is not None:
            weighted_over_128k += average * count
            over_128k_count += count

    print("\nLength distribution:")
    for category in ["0K-16K", "16K-32K", "32K-64K", "64K-128K", "128K+"]:
        count = combined.get(category, 0)
        percentage = count / total_fields * 100 if total_fields else 0.0
        print(f"{category}: {count} ({percentage:.1f}%)")
    if over_128k_count:
        print(
            "\nAverage length above 128K: "
            f"{weighted_over_128k / over_128k_count:.2f} characters"
        )
    else:
        print("\nNo fields longer than 128K characters")


if __name__ == "__main__":
    main()
