#!/usr/bin/env python3
"""Restore Pixtral processor files omitted by verl's text-only checkpoint export."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_model", type=Path)
    parser.add_argument("target_models", type=Path, nargs="+")
    args = parser.parse_args()

    processor_config = json.loads((args.source_model / "processor_config.json").read_text(encoding="utf-8"))
    image_processor = processor_config["image_processor"]
    for target in args.target_models:
        shutil.copy2(args.source_model / "processor_config.json", target / "processor_config.json")
        (target / "preprocessor_config.json").write_text(
            json.dumps(image_processor, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Prepared processor files in {target}")


if __name__ == "__main__":
    main()
