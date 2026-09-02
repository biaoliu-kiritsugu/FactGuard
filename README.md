# FactGuard

Official implementation and data-release utilities for:

> **Towards Reliable Long-Context Reasoning: Detecting Unanswerable Questions via FactGuard**

FactGuard studies whether long-context language models can distinguish questions
that are answerable from the supplied document from questions that should be
rejected. The benchmark contains:

- **Answerable** questions grounded in the document;
- **Lack of Evidence** questions created by removing the relevant evidence;
- **Misleading Evidence** questions created through entity substitution or
  unsupported-condition insertion.

The repository contains the data-generation prototype, supervised fine-tuning
launchers, vLLM inference, multi-stage LLM-as-a-Judge evaluation, and analysis
utilities used in the paper.

## Repository layout

```text
.
├── data/                         # Local legacy JSONL files (not tracked by Git)
├── hf_dataset/                   # Upload-ready Hugging Face dataset repository
│   ├── README.md                 # Dataset card
│   ├── release_stats.json        # Generated statistics and schema summary
│   └── data/*.parquet            # One Parquet file per split
├── scripts/
│   ├── prepare_hf_dataset.py     # Legacy JSONL -> normalized Parquet
│   └── validate_hf_dataset.py    # Release integrity checks
├── factguard_generation/          # FactGuard synthetic-data generation package
├── train/                        # LLaMA-Factory and verl SFT/LoRA recipes
├── evaluation/                   # vLLM generation and judge API pipeline
├── compute_metrics.py            # Main benchmark metrics
└── docs/                         # Reproduction and release notes
```

## Quick start

### 1. Install the lightweight release/evaluation dependencies

```bash
python -m pip install -r requirements.txt
```

Install the optional training or generation dependencies only when needed:

```bash
python -m pip install -r requirements-generation.txt
python -m pip install -r requirements-evaluation.txt
```

### 2. Load the released dataset

Load the published Hugging Face dataset with:

```python
from datasets import load_dataset

dataset = load_dataset("kilizi/FactGuard")
print(dataset["train"][0])
```

For a local checkout:

```python
from datasets import load_dataset

dataset = load_dataset(
    "parquet",
    data_files={
        "train": "hf_dataset/data/train.parquet",
        "validation": "hf_dataset/data/validation.parquet",
        "test": "hf_dataset/data/test.parquet",
    },
)
```

Each row contains normalized `document`, `question`, and `response` fields plus
the language, domain, answerability category, evidence, and adversarial
transformation metadata. See [`hf_dataset/README.md`](hf_dataset/README.md) for
the complete schema.

### Data-generation package

The data-construction package is named `factguard_generation`:

```bash
python -m pip install -e factguard_generation
cp factguard_generation/.env.example factguard_generation/.env
```

See [`factguard_generation/README.md`](factguard_generation/README.md) for
generation commands and environment configuration.

### 3. Generate model predictions

```bash
python evaluation/generate_predictions_vllm.py \
  --model /path/to/model \
  --test-file data/merged_test.jsonl \
  --output-file evaluation/predictions/model_preds.jsonl
```

For one process per GPU:

```bash
bash evaluation/run_sharded_generation.sh /path/to/model experiment_name
```

### 4. Evaluate with an OpenAI-compatible judge

Start an OpenAI-compatible server for the judge model, then run:

```bash
python evaluation/evaluate_predictions_api.py \
  --preds evaluation/predictions/model_preds.jsonl \
  --test-file data/merged_test.jsonl \
  --output evaluation/judge_results/model_qwen_judge.jsonl \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen2.5-72b-judge
```

The judge pipeline evaluates refusal/clarification behavior, answer consistency,
and, for correctly rejected unanswerable questions, whether the stated reason
matches the actual evidentiary defect.

Aggregate the main benchmark metrics with:

```bash
python compute_metrics.py \
  --eval-file evaluation/judge_results/model_qwen_judge.jsonl \
  --test-file data/merged_test.jsonl
```

### 5. Fine-tune

- LLaMA-Factory recipes: [`train/README.md`](train/README.md)
- verl long-context recipes: [`train/VERL_README.md`](train/VERL_README.md)

## Reproducibility notes

- Legacy data construction uses character-count buckets. Model context limits
  and training truncation are measured in tokenizer tokens.
- The data-generation code was developed against an OpenAI-compatible local
  Qwen service. Configure endpoints through environment variables; never commit
  credentials.
- Some scripts under `factguard_generation/` are retained as research artifacts. The supported
  public release path is documented in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).
- Before publication, complete the source-corpus review described in
  [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md).

## Citation

```bibtex
@inproceedings{zhang-etal-2026-factguard,
  title     = {Towards Reliable Long-Context Reasoning: Detecting Unanswerable Questions via FactGuard},
  author    = {Zhang, Qian-Wen and Liu, Biao and Li, Fang and Wang, Jie and Qiao, Lingfeng and Yu, Yifei and Yin, Di and Sun, Xing},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

## License

The source code is released under the **MIT License**; see [`LICENSE`](LICENSE).
FactGuard-Bench is released separately under **CC BY 4.0**; see
[`DATA_LICENSE`](DATA_LICENSE).
