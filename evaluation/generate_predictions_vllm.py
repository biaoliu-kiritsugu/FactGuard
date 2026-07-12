#!/usr/bin/env python3
"""Generate FactGuard test predictions from a local Hugging Face checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, default=Path("data/merged_test.jsonl"))
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def load_done(path: Path) -> set[int]:
    done: set[int] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                done.add(int(json.loads(line)["row_idx"]))
    return done


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must satisfy 0 <= shard-index < num-shards")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.output_file)

    rows: list[tuple[int, dict]] = []
    with args.test_file.open(encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            if args.limit is not None and row_idx >= args.limit:
                break
            if row_idx % args.num_shards == args.shard_index and row_idx not in done:
                rows.append((row_idx, json.loads(line)))
    if not rows:
        print(f"Already complete: {args.output_file}")
        return

    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    with args.output_file.open("a", encoding="utf-8") as writer:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            conversations = [[{"role": "user", "content": row["input"]}] for _, row in batch]
            outputs = llm.chat(
                conversations,
                sampling_params=sampling,
                use_tqdm=True,
                tokenization_kwargs={
                    "truncate_prompt_tokens": args.max_model_len - args.max_new_tokens,
                    "truncation_side": "left",
                },
            )
            for (row_idx, row), output in zip(batch, outputs, strict=True):
                record = {
                    "row_idx": row_idx,
                    "uid": row.get("uid"),
                    "source": row.get("source"),
                    "completion": output.outputs[0].text,
                }
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            writer.flush()
            print(f"Completed {min(start + len(batch), len(rows))}/{len(rows)} pending rows", flush=True)

    print(f"Wrote {len(rows)} predictions to {args.output_file}")


if __name__ == "__main__":
    main()
