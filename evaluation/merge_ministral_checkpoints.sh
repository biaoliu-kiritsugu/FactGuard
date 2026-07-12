#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"${ROOT_DIR}/train/output_verl/128k"}
MERGED_ROOT=${MERGED_ROOT:-"${ROOT_DIR}/evaluation/merged_models"}

mkdir -p "${MERGED_ROOT}"

for model in ministral3_3b ministral3_8b; do
    for method in sft lora; do
        experiment="${model}_${method}"
        run_dir="${CHECKPOINT_ROOT}/${experiment}"
        step=$(<"${run_dir}/latest_checkpointed_iteration.txt")
        checkpoint_dir="${run_dir}/global_step_${step}"
        target_dir="${MERGED_ROOT}/${experiment}"

        echo "Merging ${experiment} from ${checkpoint_dir} to ${target_dir}"
        python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "${checkpoint_dir}" \
            --target_dir "${target_dir}"
    done
done
