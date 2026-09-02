#!/usr/bin/env python3
"""Upload the prepared `hf_dataset/` directory to the Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default="kilizi/FactGuard",
        help="Hugging Face dataset repository (default: kilizi/FactGuard)",
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("hf_dataset"))
    parser.add_argument("--private", action="store_true", help="Create the dataset repository as private")
    parser.add_argument(
        "--commit-message",
        default="Publish FactGuard-Bench dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(args.dataset_dir)
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install with `python -m pip install huggingface-hub`."
        ) from exc
    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=args.dataset_dir,
        commit_message=args.commit_message,
        # Remove the obsolete custom statistics file. Hugging Face reserves
        # this name for standard DatasetInfo metadata, so retaining the old
        # payload makes `load_dataset()` fail before reading the Parquet files.
        delete_patterns="dataset_infos.json",
    )
    print(f"Uploaded {args.dataset_dir} to dataset repository {args.repo_id}")


if __name__ == "__main__":
    main()
