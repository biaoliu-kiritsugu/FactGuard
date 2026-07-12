#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MERGED_ROOT="${ROOT_DIR}/evaluation/merged_models"

bash "${ROOT_DIR}/evaluation/run_sharded_generation.sh" \
    "${MERGED_ROOT}/ministral3_3b_sft" ministral3_3b_sft
bash "${ROOT_DIR}/evaluation/run_sharded_generation.sh" \
    "${MERGED_ROOT}/ministral3_3b_lora_merged" ministral3_3b_lora
bash "${ROOT_DIR}/evaluation/run_sharded_generation.sh" \
    "${MERGED_ROOT}/ministral3_8b_sft" ministral3_8b_sft
bash "${ROOT_DIR}/evaluation/run_sharded_generation.sh" \
    "${MERGED_ROOT}/ministral3_8b_lora_merged" ministral3_8b_lora
