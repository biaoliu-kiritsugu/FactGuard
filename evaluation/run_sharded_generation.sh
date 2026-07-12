#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: bash evaluation/run_sharded_generation.sh <model-dir> <experiment-name>" >&2
    exit 1
fi

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_DIR=$1
EXPERIMENT=$2
NUM_GPUS=${NUM_GPUS:-8}
PART_DIR="${ROOT_DIR}/evaluation/predictions/${EXPERIMENT}.parts"
OUTPUT_FILE="${ROOT_DIR}/evaluation/predictions/${EXPERIMENT}_preds.jsonl"
mkdir -p "${PART_DIR}"

pids=()
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    CUDA_VISIBLE_DEVICES=${gpu} python "${ROOT_DIR}/evaluation/generate_predictions_vllm.py" \
        --model "${MODEL_DIR}" \
        --test-file "${ROOT_DIR}/data/merged_test.jsonl" \
        --output-file "${PART_DIR}/part-${gpu}.jsonl" \
        --tensor-parallel-size 1 \
        --max-model-len 131072 \
        --max-new-tokens 1024 \
        --batch-size 32 \
        --num-shards "${NUM_GPUS}" \
        --shard-index "${gpu}" \
        >"${PART_DIR}/part-${gpu}.log" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done
if (( failed )); then
    echo "At least one generation shard failed; inspect ${PART_DIR}/part-*.log" >&2
    exit 1
fi

python "${ROOT_DIR}/evaluation/merge_prediction_shards.py" \
    --input-dir "${PART_DIR}" \
    --output-file "${OUTPUT_FILE}" \
    --expected 4200
