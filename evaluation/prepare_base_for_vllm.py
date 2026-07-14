#!/usr/bin/env python3
"""Create a lightweight vLLM-compatible view of an original Ministral model."""

import argparse
import os
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base_model", type=Path)
    p.add_argument("compatible_config_model", type=Path)
    p.add_argument("output_dir", type=Path)
    args = p.parse_args()

    base = args.base_model.resolve()
    config_source = (args.compatible_config_model / "config.json").resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # The download contains both Hugging Face shards and a native Mistral
    # consolidated checkpoint. Exposing both makes vLLM select the native
    # parser; keep only the HF representation in this compatibility view.
    excluded = {"config.json", "consolidated.safetensors", "params.json"}
    for stale in ("consolidated.safetensors", "params.json"):
        target = out / stale
        if target.is_symlink():
            target.unlink()

    for source in base.iterdir():
        target = out / source.name
        if source.name in excluded or target.exists() or target.is_symlink():
            continue
        os.symlink(source, target, target_is_directory=source.is_dir())
    shutil.copy2(config_source, out / "config.json")
    print(f"Prepared {out} (weights symlinked from {base})")


if __name__ == "__main__":
    main()
