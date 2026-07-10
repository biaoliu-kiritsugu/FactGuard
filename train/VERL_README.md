# FactGuard SFT / LoRA with verl

The launcher targets the installed verl `0.9.0.dev` SPMD SFT trainer and a
single node with 8 H200 GPUs. It uses the models already downloaded under
`../hf_models`.

## 1. Convert the dataset

```bash
bash train/run_verl_train.sh prepare
```

This streams `data/merged_train.jsonl` into
`train/data/factguard_train.parquet`. Each row is represented as `messages`
with one user turn and one assistant turn.

## 2. Train one model

```bash
# Full-parameter SFT
bash train/run_verl_train.sh internlm2_5_7b sft

# LoRA
bash train/run_verl_train.sh internlm2_5_7b lora
```

Supported model keys are `internlm2_5_1_8b`, `internlm2_5_7b`,
`internlm2_5_20b`, `glm4_9b`, and `glm4_32b`. Run every model/method pair
sequentially with:

```bash
bash train/run_verl_train.sh all
```

The paper-aligned settings are fixed by default:

- AdamW learning rate: `2e-5`
- epochs: `2`
- warmup ratio: `0.1`
- weight decay: `0.1`
- cosine LR schedule
- LoRA: rank `8`, alpha `16`, dropout `0.05`
- global batch size: `128` (same as 1 sample/GPU x 16 accumulation x 8 GPUs)

InternLM uses its fused `wqkv` LoRA projection and GLM-4-9B uses
`query_key_value`. GLM-4-32B-0414 does not contain `query_key_value`; it uses
the stricter paper-aligned `q_proj` and `v_proj` modules.

Checkpoints and logs are saved below `train/output_verl/<model>_<method>/`.
The default `RESUME_MODE=auto` resumes the latest valid checkpoint in that
directory. Use `RESUME_MODE=disable` to start a new run in an empty output
directory.

The launcher also redirects Hugging Face remote-code, FlashInfer JIT, and
PyTorch extension caches into `train/.cache`, which is writable in this
environment.

## Tuning memory and sequence parallelism

The defaults choose Ulysses SP sizes based on model size and native context.
They are conservative starting points for 8 H200 GPUs. Override them without
editing the script, for example:

```bash
SP_SIZE=4 MAX_TOKEN_LEN_PER_GPU=8192 \
  bash train/run_verl_train.sh glm4_9b lora
```

`MAX_TOKEN_LEN_PER_GPU * SP_SIZE` must be at least as large as a desired single
sequence. `MAX_LENGTH` defaults to 32K for InternLM2.5 and GLM-4-32B, and 128K
for GLM-4-9B. Over-length FactGuard samples retain the final question and
assistant answer; loss is applied only to assistant tokens.

To inspect the complete torchrun/Hydra invocation without starting training:

```bash
DRY_RUN=1 bash train/run_verl_train.sh glm4_32b lora
```
