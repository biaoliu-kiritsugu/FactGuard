#!/usr/bin/env bash
# Run cross-family SFT/LoRA fine-tuning for the FactGuard rebuttal experiment.
#
# Purpose: replicate the paper's Qwen2.5 SFT/LoRA recipe on TWO other model families
#          (InternLM2.5 series: 1.8B/7B/20B; GLM-4: 9B/32B) to show the reliability
#          gains are NOT specific to Qwen (rebuttal to the "circularity" critique).
#
# Hyperparameters are identical to the paper (see any config in train/configs/):
#   Full SFT : AdamW, lr=2e-5, 2 epochs, warmup_ratio=0.1, weight_decay=0.1
#   LoRA     : rank=8, alpha=16, target = query/value (fused) projections
#
# Prereqs on the remote server:
#   1) pip install "llamafactory[torch,metrics,deepspeed]"  (or clone LLaMA-Factory)
#   2) The FactGuard repo checked out; run this script FROM the factguard_code/ dir.
#   3) merged_train.jsonl available (path passed via TRAIN_SRC below).
#
# Usage:
#   cd factguard_code
#   bash train/run_train.sh prepare              # build alpaca json once
#   bash train/run_train.sh internlm2_5_7b sft   # single run
#   bash train/run_train.sh all                  # every config, sequentially
#
set -euo pipefail

cd "$(dirname "$0")/.."   # -> factguard_code/

TRAIN_SRC="${TRAIN_SRC:-data/merged_train.jsonl}"
DST="train/data/factguard_train.json"

ALL_CONFIGS=(
  internlm2_5_1_8b
  internlm2_5_7b
  internlm2_5_20b
  glm4_9b
  glm4_32b
)

prepare() {
  echo ">> preparing dataset from ${TRAIN_SRC}"
  python train/prepare_data.py --src "${TRAIN_SRC}" --dst "${DST}"
}

run_one() {
  local model="$1" method="$2"
  local cfg="train/configs/${model}_${method}.yaml"
  if [[ ! -f "${cfg}" ]]; then
    echo "!! config not found: ${cfg}" >&2; exit 1
  fi
  echo ">> [$(date '+%F %T')] training ${model} (${method})  cfg=${cfg}"
  # Use all visible GPUs; torchrun handles multi-GPU. Override GPUS to limit.
  llamafactory-cli train "${cfg}" 2>&1 | tee "train/output/${model}_${method}.log"
}

main() {
  mkdir -p train/output
  local arg1="${1:-}"; local arg2="${2:-}"

  if [[ "${arg1}" == "prepare" ]]; then prepare; exit 0; fi

  # ensure dataset exists
  [[ -f "${DST}" ]] || prepare

  if [[ "${arg1}" == "all" ]]; then
    for m in "${ALL_CONFIGS[@]}"; do
      run_one "${m}" sft
      run_one "${m}" lora
    done
  elif [[ -n "${arg1}" && -n "${arg2}" ]]; then
    run_one "${arg1}" "${arg2}"
  else
    echo "usage: bash train/run_train.sh [prepare | all | <model> <sft|lora>]"
    echo "models: ${ALL_CONFIGS[*]}"
    exit 1
  fi
}

main "$@"
