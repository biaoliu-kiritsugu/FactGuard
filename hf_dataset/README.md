---
annotations_creators:
- machine-generated
language:
- en
- zh
language_creators:
- found
license: cc-by-4.0
multilinguality:
- multilingual
pretty_name: FactGuard-Bench
size_categories:
- 10K<n<100K
source_datasets:
- Pile of Law
- Project Gutenberg
- TigerBot law corpus
- open-copyright Chinese books
task_categories:
- question-answering
- text-generation
task_ids:
- closed-domain-qa
- open-domain-qa
tags:
- long-context
- unanswerable-questions
- hallucination
- abstention
- evidence-grounding
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.parquet
  - split: validation
    path: data/validation.parquet
  - split: test
    path: data/test.parquet
---

# FactGuard-Bench

FactGuard-Bench is a bilingual long-context benchmark for evaluating and
improving whether language models answer only when the supplied document
contains sufficient evidence. It contains English and Chinese examples from
the book and legal domains, with contexts extending to approximately 128K in
the legacy character-based construction buckets.

The benchmark accompanies:

> **Towards Reliable Long-Context Reasoning: Detecting Unanswerable Questions via FactGuard**

## Dataset summary

The public release contains:

| Split | Examples | Answerable | Unanswerable |
|---|---:|---:|---:|
| Train | 19,100 | 6,168 | 12,932 |
| Validation | 1,920 | 317 | 1,603 |
| Test | 4,200 | 700 | 3,500 |
| Total | 25,220 | 7,185 | 18,035 |

## Task formulation

Every example provides a document and a question. A model should either:

1. answer using evidence from the document, or
2. reject or clarify the question when the document does not support it.

Unanswerable examples cover:

- **Lack of Evidence**: the answer-bearing evidence is removed;
- **Entity Substitution**: a supported entity is replaced by a similar but
  unsupported entity;
- **Impossible Condition**: the question is augmented with an unsupported
  constraint.

Entity substitution and impossible-condition insertion form the paper's
broader **Misleading Evidence** category.

## Loading

```python
from datasets import load_dataset

dataset = load_dataset("kilizi/FactGuard")
```

Local loading:

```python
from datasets import load_dataset

dataset = load_dataset(
    "parquet",
    data_files={
        "train": "data/train.parquet",
        "validation": "data/validation.parquet",
        "test": "data/test.parquet",
    },
)
```

## Data fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable public example identifier within this release |
| `document_id` | string | MD5-based identifier inherited from the source document |
| `document` | string | Document shown to the evaluated model |
| `question` | string | Answerable or adversarial question |
| `response` | string | Reference answer or reasoned rejection |
| `answerability` | string | `answerable` or `unanswerable` |
| `category` | string | `answerable`, `lack_of_evidence`, or `misleading_evidence` |
| `perturbation` | string | `none`, `evidence_removal`, `entity_substitution`, or `impossible_condition` |
| `language` | string | `en` or `zh` |
| `domain` | string | `book` or `law` |
| `source_file` | string | Legacy source/split filename for traceability |
| `legacy_length_bucket` | string | Character-based bucket used during construction |
| `document_char_length` | int64 | Number of Unicode code points in `document` |
| `question_char_length` | int64 | Number of Unicode code points in `question` |
| `response_char_length` | int64 | Number of Unicode code points in `response` |
| `evidence` | string | Original evidence or source passage, when available |
| `original_question` | string | Question before an adversarial transformation |
| `original_answer` | string | Answer to the original supported question, when available |
| `unanswerable_reason` | string | Specific missing/misaligned evidence explanation |
| `original_entity` | string | Entity before substitution |
| `substituted_entity` | string | Unsupported replacement entity |
| `unsupported_condition` | string | Inserted unsupported condition |
| `question_type` | string | Legacy QA-generation type, when available |

Empty strings indicate fields that do not apply to a particular perturbation.

## Example

```python
example = dataset["test"][0]
prompt = (
    f"Document:\n{example['document']}\n\n"
    f"Please Answer the Question based on the document: {example['question']}"
)
```

For Chinese examples:

```python
prompt = (
    f"文档:\n{example['document']}\n\n"
    f"请根据文档回答问题: {example['question']}"
)
```

## Construction

FactGuard-Bench was synthesized from long-form English and Chinese book and
legal documents. A Qwen2.5-72B-Instruct-based multi-stage workflow generated
grounded questions, adversarial transformations, reasoned rejection targets,
and automatic quality checks. See the paper and code repository for complete
prompts and processing details.

## Evaluation

The paper uses a multi-stage LLM-as-a-Judge protocol:

1. detect refusal or clarification;
2. compare answer content with the reference for answerable examples;
3. verify that a correct rejection identifies the actual evidentiary defect.

The test split contains 700 answerable and 3,500 unanswerable examples.

## Limitations

- Questions and responses are machine-generated and can contain residual
  generation or annotation errors.
- The benchmark covers two languages and two primary domains; results should
  not be treated as representative of every language or application.
- Length buckets used during construction are character based, not tokenizer
  invariant.
- The dataset is intended to measure document-grounded behavior. It does not
  establish whether a claim is globally true outside the supplied document.
- The released train/validation/test assignment reproduces the paper's legacy
  sampling process and is not document-disjoint. Exact overlap statistics are
  recorded in `release_stats.json` and printed by the validation script.
  Preserve these splits when reproducing paper results, but use a newly
  generated document-disjoint split for claims about generalization to unseen
  documents.
- Source documents can contain outdated, offensive, or otherwise sensitive
  material inherited from books and legal corpora.

## License and redistribution

FactGuard-Bench is released under the **Creative Commons Attribution 4.0
International License (CC BY 4.0)**. Users may share and adapt the dataset,
including for commercial purposes, provided that appropriate attribution is
given and modifications are indicated.

Suggested attribution:

> FactGuard-Bench, from “Towards Reliable Long-Context Reasoning: Detecting
> Unanswerable Questions via FactGuard,” The FactGuard Authors, 2026.

The dataset repository includes a `LICENSE` notice. Third-party names,
trademarks, and material explicitly identified as third-party content remain
subject to their respective rights.

## Citation

```bibtex
@article{zhang2026factguard,
  title={Towards Reliable Long-Context Reasoning: Detecting Unanswerable Questions via FactGuard},
  author={Zhang, Qian-Wen and Liu, Biao and Li, Fang and Wang, Jie and Qiao, Lingfeng and Yu, Yifei and Yin, Di and Sun, Xing},
  year={2026}
}
```
