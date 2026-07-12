#!/usr/bin/env python3
"""Merge exported PEFT adapters into standalone Hugging Face checkpoints."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_dir = args.model_dir / "lora_adapter"
    if not adapter_dir.is_dir():
        raise FileNotFoundError(adapter_dir)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload(safe_merge=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.output_dir)
    for name in ("chat_template.jinja", "generation_config.json", "processor_config.json"):
        src = args.model_dir / name
        dst = args.output_dir / name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
