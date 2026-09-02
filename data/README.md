# Legacy local data

This directory contains the original internal JSONL representation consumed by
the training and evaluation scripts:

```text
merged_train.jsonl
merged_dev.jsonl
merged_test.jsonl
```

These files are intentionally ignored by the code repository because they are
large. The public dataset artifact should be published separately from
`../hf_dataset/`.

## Legacy schema

Each row contains:

- `source`: generation source and split name;
- `uid`: source-document identifier;
- `input`: formatted document-plus-question prompt;
- `output`: target answer or reasoned rejection;
- `origin`: detailed internal generation metadata;
- `is_positive`: whether the displayed question is answerable.

Use `../scripts/prepare_hf_dataset.py` to create the normalized public schema.
Do not publish the legacy files without reviewing the large `origin` objects:
they contain internal generation traces and duplicate copies of documents.
