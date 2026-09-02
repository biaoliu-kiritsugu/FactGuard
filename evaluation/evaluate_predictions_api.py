#!/usr/bin/env python3
"""Evaluate FactGuard predictions with an OpenAI-compatible judge API.

This uses the English R1 -> R2 -> conditional R3 prompts in ``judge_utils.py``
against an already-running OpenAI-compatible server. Outputs are append-only
and resumable by row_idx.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import jsonlines
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge_utils import (
    R2_PROMPT,
    build_examples,
    build_r1_messages,
    build_r3_messages,
    extract_json,
    finalize_r3,
    get_type,
    load_done_rows,
    needs_r3,
)


def validate_r2(result, ex_type):
    decision_key = "clarified" if ex_type in ("misattr", "impossible") else "refusal_detected"
    if not isinstance(result.get(decision_key), bool):
        raise ValueError(f"invalid R2 result for {ex_type}: {result!r}")
    if not isinstance(result.get("excerpt"), str):
        raise ValueError(f"invalid R2 excerpt for {ex_type}: {result!r}")
    if ex_type == "impossible" and not isinstance(result.get("analysis"), str):
        raise ValueError(f"invalid R2 result for impossible: {result!r}")
    return result


def validate_r3(result):
    if not isinstance(result.get("same_conclusion"), bool):
        raise ValueError(f"invalid R3 result: {result!r}")
    if not isinstance(result.get("analysis"), str):
        raise ValueError(f"invalid R3 analysis: {result!r}")
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", type=Path, required=True)
    p.add_argument("--test-file", type=Path, default=Path("data/merged_test.jsonl"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="qwen2.5-72b-judge")
    p.add_argument("--concurrency", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--retries", type=int, default=5)
    return p.parse_args()


async def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_rows(args.output)
    examples = build_examples(args.preds, args.test_file, done)
    typed = [(ex, get_type(ex)) for ex in examples]
    print(f"{args.preds.name}: done={len(done)}, pending={len(typed)}", flush=True)
    if not typed:
        return

    client = AsyncOpenAI(api_key="EMPTY", base_url=args.base_url, max_retries=0)
    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    completed = 0

    async def call(messages, *, json_mode=False):
        last_error = None
        for attempt in range(args.retries):
            try:
                kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
                rsp = await client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=0,
                    max_tokens=args.max_tokens,
                    **kwargs,
                )
                return rsp.choices[0].message.content
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(min(2 ** attempt, 16))
        raise RuntimeError(f"judge API failed after {args.retries} attempts: {last_error}")

    async def call_valid_json(messages, validator):
        last_error = None
        for attempt in range(args.retries):
            try:
                raw = await call(messages, json_mode=True)
                parsed = validator(extract_json(raw))
                return raw, parsed
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(min(2 ** attempt, 16))
        raise RuntimeError(f"invalid judge JSON after {args.retries} attempts: {last_error}")

    writer = jsonlines.open(args.output, mode="a", flush=True)

    async def evaluate_one(ex, ex_type):
        nonlocal completed
        try:
            # Keep one concurrency slot for the complete judging pipeline. If the
            # semaphore only wraps individual API calls, thousands of R1 calls can
            # jump ahead of the corresponding R2/R3 calls and delay every result.
            async with sem:
                r1_messages = build_r1_messages(ex, ex_type)
                r1 = await call(r1_messages)
                r2_messages = r1_messages + [
                    {"role": "assistant", "content": r1},
                    {"role": "user", "content": R2_PROMPT[ex_type]},
                ]
                _, result = await call_valid_json(
                    r2_messages, lambda parsed: validate_r2(parsed, ex_type)
                )
                if needs_r3(ex_type, result):
                    r3, _ = await call_valid_json(
                        build_r3_messages(ex, ex_type), validate_r3
                    )
                    finalize_r3(ex_type, result, r3)
            out = {
                "row_idx": ex["row_idx"],
                "uid": ex["uid"],
                "source": ex["source"],
                "type": ex_type,
                "eval_result": result,
            }
            async with write_lock:
                writer.write(out)
                completed += 1
                if completed % 100 == 0:
                    print(f"{args.preds.name}: +{completed}/{len(typed)}", flush=True)
        except Exception as exc:
            print(f"[error] row_idx={ex['row_idx']}: {exc}", flush=True)

    try:
        await asyncio.gather(*(evaluate_one(ex, t) for ex, t in typed))
    finally:
        writer.close()
        await client.close()
    print(f"{args.preds.name}: finished, added={completed}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
