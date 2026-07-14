#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE_URL=${BASE_URL:-http://127.0.0.1:8000/v1}
JUDGE_MODEL=${JUDGE_MODEL:-qwen2.5-72b-judge}
CONCURRENCY=${CONCURRENCY:-64}

for experiment in \
    ministral3_3b_sft ministral3_3b_lora \
    ministral3_8b_sft ministral3_8b_lora; do
    python "${ROOT_DIR}/evaluation/evaluate_predictions_api.py" \
        --preds "${ROOT_DIR}/evaluation/predictions/${experiment}_preds.jsonl" \
        --test-file "${ROOT_DIR}/data/merged_test.jsonl" \
        --output "${ROOT_DIR}/evaluation/judge_results/${experiment}_qwen_judge.jsonl" \
        --base-url "${BASE_URL}" \
        --model "${JUDGE_MODEL}" \
        --concurrency "${CONCURRENCY}"
done
