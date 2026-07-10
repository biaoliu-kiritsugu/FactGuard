#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert FactGuard merged_train.jsonl (fields: input / output / is_positive / ...)
into LLaMA-Factory "alpaca" format: {"instruction", "input", "output"}.

Usage:
    python train/prepare_data.py \
        --src /path/to/merged_train.jsonl \
        --dst train/data/factguard_train.json

Notes:
- FactGuard's `input` already contains "Document:\n...\n\nQuestion: ..." so we map it
  directly to the alpaca `input` field and leave `instruction` empty (matching the way
  the model is fed at eval time). `output` is the target answer / reasoned rejection.
- Output is a single JSON array (LLaMA-Factory default for the alpaca loader).
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to merged_train.jsonl")
    ap.add_argument("--dst", required=True, help="output .json (alpaca list)")
    args = ap.parse_args()

    out = []
    kept = dropped = 0
    with open(args.src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            inp = d.get("input", "")
            oup = d.get("output", "")
            if not inp or not oup:
                dropped += 1
                continue
            out.append({"instruction": "", "input": inp, "output": oup})
            kept += 1

    with open(args.dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"kept={kept} dropped={dropped} -> {args.dst}")


if __name__ == "__main__":
    main()
