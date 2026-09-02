# Reproducibility guide

## Supported release path

The most reproducible path in this repository starts from the released
FactGuard-Bench data:

1. Load the normalized Hugging Face Parquet files.
2. Convert the training split for LLaMA-Factory or verl.
3. Fine-tune a model with the provided configurations.
4. Generate predictions with vLLM.
5. Evaluate predictions with the OpenAI-compatible judge pipeline.
6. Aggregate metrics with `compute_metrics.py`.

The `factguard_generation/` directory contains the original research
data-generation code under a public-facing package name. Install it with:

```bash
python -m pip install -e factguard_generation
```

It is released for transparency, but parts reflect the environment in which
the benchmark was built.

## Dataset categories

| Public category | Legacy name | Construction |
|---|---|---|
| `answerable` | positive `dandian` or `misattr` row | Original document and grounded question |
| `lack_of_evidence` | negative `dandian` row | Evidence-bearing fragment removed |
| `entity_substitution` | negative `misattr` row | Key entity replaced with a similar unsupported entity |
| `impossible_condition` | `impossible` row | Unsupported constraint inserted into the question |

`entity_substitution` and `impossible_condition` are grouped as
`misleading_evidence` in the paper.

## Training

The released dataset fields are `document`, `question`, and `response`. The
legacy train launchers consume prompts whose instructions are written in
English:

```text
Document:
{document}

Answer the question using only the document.
Question: {question}
Answer in English.
```

For Chinese examples, the instruction text remains English and the final line
is `Answer in Chinese.`. Use `scripts/export_legacy_jsonl.py` if a launcher
expects the original `input`/`output` schema.

The verl dataset keeps the tail of over-length examples because the question is
placed after the document. Loss is applied only to assistant tokens.

## Evaluation protocol

`evaluation/evaluate_predictions_api.py` implements:

1. Refusal or clarification detection;
2. Answer consistency for answerable responses;
3. Reason grounding for correctly rejected unanswerable responses.

Outputs are append-only and resumable by `row_idx`.

## Known distinctions from the paper text

- Data stratification in the legacy scripts uses character counts, whereas
  model context windows use tokenizer token counts.
- `train/run_verl_train.sh` defaults to LoRA rank 128 for the later
  cross-family experiments. Check the exact experiment configuration before
  comparing with a paper table that reports rank 8.
- The public generation prototype requires an OpenAI-compatible model service
  and external raw corpora. Raw source corpora are not redistributed by the
  code repository.
