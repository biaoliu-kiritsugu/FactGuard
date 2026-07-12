#!/usr/bin/env bash
# Run FactGuard full SFT or LoRA with verl on one 8xH200 node.
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

MODEL_ROOT=${MODEL_ROOT:-"$(dirname "${ROOT_DIR}")/hf_models"}
TRAIN_JSONL=${TRAIN_JSONL:-"${ROOT_DIR}/data/merged_train.jsonl"}
TRAIN_PARQUET=${TRAIN_PARQUET:-"${ROOT_DIR}/train/data/factguard_train.parquet"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${ROOT_DIR}/train/output_verl"}
NUM_GPUS=${NUM_GPUS:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29621}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
LR=${LR:-2e-5}
EPOCHS=${EPOCHS:-2}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
RESUME_MODE=${RESUME_MODE:-auto}
SAVE_FREQ=${SAVE_FREQ:-after_each_epoch}
PROJECT_NAME=${PROJECT_NAME:-factguard-verl-sft}
SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-"${ROOT_DIR}/train/swanlog"}

usage() {
    cat <<'EOF'
Usage:
  bash train/run_verl_train.sh prepare
  bash train/run_verl_train.sh <model> <sft|lora> [Hydra overrides ...]
  bash train/run_verl_train.sh all
  bash train/run_verl_train.sh ministral-all

Models:
  internlm2_5_1_8b, internlm2_5_7b, internlm2_5_20b, glm4_9b, glm4_32b,
  ministral3_3b, ministral3_8b, ministral3_14b

Examples:
  bash train/run_verl_train.sh prepare
  bash train/run_verl_train.sh internlm2_5_7b lora
  RESUME_MODE=disable bash train/run_verl_train.sh glm4_32b sft
  SP_SIZE=8 MAX_LENGTH=131072 MAX_TOKEN_LEN_PER_GPU=16384 bash train/run_verl_train.sh glm4_9b lora

Useful environment overrides:
  MODEL_ROOT, TRAIN_PARQUET, OUTPUT_ROOT, NUM_GPUS, GLOBAL_BATCH_SIZE,
  SP_SIZE, MAX_LENGTH, MAX_TOKEN_LEN_PER_GPU, RESUME_MODE, SAVE_FREQ,
  SWANLAB_LOG_DIR, SWANLAB_MODE, CUDA_VISIBLE_DEVICES, DRY_RUN=1
EOF
}

prepare_data() {
    python train/prepare_verl_data.py --src "${TRAIN_JSONL}" --dst "${TRAIN_PARQUET}"
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

if [[ $1 == prepare ]]; then
    prepare_data
    exit 0
fi

if [[ $1 == all ]]; then
    # Current production queue: GLM only, from the smallest to the largest.
    # InternLM is excluded because its bundled remote code produces NaN logits
    # with the installed Transformers 5.x runtime.
    for model in glm4_9b glm4_32b; do
        for method in sft lora; do
            bash "$0" "${model}" "${method}"
        done
    done
    exit 0
fi

if [[ $1 == ministral-all ]]; then
    # Run Ministral 3 from small to large; each full SFT is followed by LoRA.
    for model in ministral3_3b ministral3_8b ministral3_14b; do
        for method in sft lora; do
            bash "$0" "${model}" "${method}"
        done
    done
    exit 0
fi

if [[ $# -lt 2 ]]; then
    usage
    exit 1
fi

MODEL_KEY=$1
METHOD=$2
shift 2

case "${MODEL_KEY}" in
    internlm2_5_1_8b)
        MODEL_DIR=internlm2_5-1_8b-chat
        DEFAULT_SP_SIZE=1
        DEFAULT_MAX_LENGTH=32768
        DEFAULT_MAX_TOKENS=32768
        LORA_TARGETS='[wqkv]'
        ;;
    internlm2_5_7b)
        MODEL_DIR=internlm2_5-7b-chat
        DEFAULT_SP_SIZE=1
        DEFAULT_MAX_LENGTH=32768
        DEFAULT_MAX_TOKENS=32768
        LORA_TARGETS='[wqkv]'
        ;;
    internlm2_5_20b)
        MODEL_DIR=internlm2_5-20b-chat
        DEFAULT_SP_SIZE=1
        DEFAULT_MAX_LENGTH=32768
        DEFAULT_MAX_TOKENS=32768
        LORA_TARGETS='[wqkv]'
        ;;
    glm4_9b)
        MODEL_DIR=glm-4-9b-chat
        DEFAULT_SP_SIZE=8
        DEFAULT_MAX_LENGTH=131072
        DEFAULT_MAX_TOKENS=16384
        LORA_TARGETS='[query_key_value]'
        ;;
    glm4_32b)
        MODEL_DIR=GLM-4-32B-0414
        DEFAULT_SP_SIZE=8
        DEFAULT_MAX_LENGTH=131072
        DEFAULT_MAX_TOKENS=16384
        # This model has separate Q/K/V projections; query_key_value does not exist.
        LORA_TARGETS='[q_proj,v_proj]'
        ;;
    ministral3_3b)
        MODEL_DIR=Ministral-3-3B-Instruct-2512-BF16
        DEFAULT_SP_SIZE=8
        DEFAULT_MAX_LENGTH=131072
        DEFAULT_MAX_TOKENS=16384
        LORA_TARGETS='[q_proj,v_proj]'
        ;;
    ministral3_8b)
        MODEL_DIR=Ministral-3-8B-Instruct-2512-BF16
        DEFAULT_SP_SIZE=8
        DEFAULT_MAX_LENGTH=131072
        DEFAULT_MAX_TOKENS=16384
        LORA_TARGETS='[q_proj,v_proj]'
        ;;
    ministral3_14b)
        MODEL_DIR=Ministral-3-14B-Instruct-2512-BF16
        DEFAULT_SP_SIZE=8
        DEFAULT_MAX_LENGTH=131072
        DEFAULT_MAX_TOKENS=16384
        LORA_TARGETS='[q_proj,v_proj]'
        ;;
    *)
        echo "Unknown model: ${MODEL_KEY}" >&2
        usage
        exit 1
        ;;
esac

if [[ ${METHOD} != sft && ${METHOD} != lora ]]; then
    echo "Method must be 'sft' or 'lora', got: ${METHOD}" >&2
    exit 1
fi

MODEL_PATH="${MODEL_ROOT}/${MODEL_DIR}"
SP_SIZE=${SP_SIZE:-${DEFAULT_SP_SIZE}}
MAX_LENGTH=${MAX_LENGTH:-${DEFAULT_MAX_LENGTH}}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-${DEFAULT_MAX_TOKENS}}
EXPERIMENT_NAME="${MODEL_KEY}_${METHOD}"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}"

if [[ ! -f ${MODEL_PATH}/config.json ]]; then
    echo "Model not found: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -f ${TRAIN_PARQUET} ]]; then
    echo "Training Parquet not found: ${TRAIN_PARQUET}" >&2
    echo "Run: bash train/run_verl_train.sh prepare" >&2
    exit 1
fi
if (( NUM_GPUS % SP_SIZE != 0 )); then
    echo "NUM_GPUS (${NUM_GPUS}) must be divisible by SP_SIZE (${SP_SIZE})" >&2
    exit 1
fi
if (( GLOBAL_BATCH_SIZE % (NUM_GPUS / SP_SIZE) != 0 )); then
    echo "GLOBAL_BATCH_SIZE must be divisible by the data-parallel size" >&2
    exit 1
fi

mkdir -p \
    "${OUTPUT_DIR}" \
    "${ROOT_DIR}/train/.cache/hf_modules" \
    "${ROOT_DIR}/train/.cache/flashinfer" \
    "${ROOT_DIR}/train/.cache/torch_extensions"
export HF_MODULES_CACHE=${HF_MODULES_CACHE:-"${ROOT_DIR}/train/.cache/hf_modules"}
# This installed FlashInfer version derives its cache from
# FLASHINFER_WORKSPACE_BASE (not FLASHINFER_WORKSPACE_DIR).
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-"${ROOT_DIR}/train/.cache/flashinfer"}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-"${ROOT_DIR}/train/.cache/torch_extensions"}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export FACTGUARD_LORA_DROPOUT=${LORA_DROPOUT}
export SWANLAB_LOG_DIR
export SWANLAB_MODE=${SWANLAB_MODE:-cloud}

LORA_ARGS=(model.lora_rank=0)
if [[ ${METHOD} == lora ]]; then
    LORA_ARGS=(
        "model.lora_rank=${LORA_RANK}"
        "model.lora_alpha=${LORA_ALPHA}"
        "model.target_modules=${LORA_TARGETS}"
    )
fi

if [[ ${MODEL_KEY} == ministral3_* ]]; then
    # Ulysses sequence parallelism communicates through verl's patched
    # FlashAttention path. Mistral3 otherwise defaults to SDPA and produces
    # full-sequence outputs on every SP rank.
    MODEL_OVERRIDES=(+model.override_config.attn_implementation=flash_attention_2)
else
    MODEL_OVERRIDES=(+model.override_config.attn_implementation=sdpa)
fi
if [[ ${MODEL_KEY} == glm4_9b ]]; then
    # ChatGLM calls this field multi_query_group_num; verl's Ulysses check uses
    # the standard Hugging Face num_key_value_heads name.
    MODEL_OVERRIDES+=(+model.override_config.num_key_value_heads=2)
    # The bundled ChatGLM remote code reads max_length while the checkpoint
    # config stores the same native context window as seq_length.
    MODEL_OVERRIDES+=(+model.override_config.max_length=131072)
fi

CMD=(
    torchrun --nnodes=1 --node_rank=0 "--nproc_per_node=${NUM_GPUS}"
    "--master_addr=${MASTER_ADDR}" "--master_port=${MASTER_PORT}"
    -m verl.trainer.sft_trainer
    "data.train_files=${TRAIN_PARQUET}"
    data.val_files=null
    "data.train_batch_size=${GLOBAL_BATCH_SIZE}"
    "data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU}"
    data.use_dynamic_bsz=false
    "data.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
    "data.max_length=${MAX_LENGTH}"
    data.pad_mode=no_padding
    data.truncation=left
    data.messages_key=messages
    "data.num_workers=${NUM_WORKERS}"
    "data.custom_cls.path=${ROOT_DIR}/train/verl_sft_dataset.py"
    data.custom_cls.name=FactGuardSFTDataset
    "model.path=${MODEL_PATH}"
    model.trust_remote_code=true
    model.external_lib=train.verl_lora_patch
    model.enable_gradient_checkpointing=true
    model.use_remove_padding=true
    engine=fsdp
    engine.strategy=fsdp
    "engine.fsdp_size=${NUM_GPUS}"
    "engine.ulysses_sequence_parallel_size=${SP_SIZE}"
    engine.model_dtype=bfloat16
    engine.dtype=bfloat16
    engine.use_torch_compile=false
    optim=fsdp
    "optim.lr=${LR}"
    "optim.lr_warmup_steps_ratio=${WARMUP_RATIO}"
    "optim.weight_decay=${WEIGHT_DECAY}"
    optim.lr_scheduler_type=cosine
    "trainer.total_epochs=${EPOCHS}"
    trainer.total_training_steps=null
    "trainer.default_local_dir=${OUTPUT_DIR}"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.save_freq=${SAVE_FREQ}"
    trainer.test_freq=-1
    trainer.logger='[console,file,swanlab]'
    "trainer.resume_mode=${RESUME_MODE}"
    "${LORA_ARGS[@]}"
    "${MODEL_OVERRIDES[@]}"
    "$@"
)

echo "Experiment: ${EXPERIMENT_NAME}"
echo "Model:      ${MODEL_PATH}"
echo "Data:       ${TRAIN_PARQUET}"
echo "Output:     ${OUTPUT_DIR}"
echo "GPUs/SP/DP: ${NUM_GPUS}/${SP_SIZE}/$((NUM_GPUS / SP_SIZE))"
echo "Batch:      ${GLOBAL_BATCH_SIZE} global"
echo "Max length: ${MAX_LENGTH}; token budget/GPU: ${MAX_TOKEN_LEN_PER_GPU}"
echo "Optimizer:  AdamW lr=${LR}, epochs=${EPOCHS}, warmup=${WARMUP_RATIO}, wd=${WEIGHT_DECAY}"
if [[ ${METHOD} == lora ]]; then
    echo "LoRA:       rank=${LORA_RANK}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}, targets=${LORA_TARGETS}"
fi

if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf 'Command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

"${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
