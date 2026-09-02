# FactGuard data generation

This package contains the research pipeline used to construct FactGuard-Bench.
The directory, distribution, and Python import package consistently use the
public-facing name `factguard_generation`.

The package covers:

- source-document loading;
- grounded answerable QA generation;
- evidence-removal examples;
- entity-substitution examples;
- impossible-condition examples;
- retrieval-based checks;
- filtering, deduplication, sampling, and dataset merging.

## Install

From the repository root:

```bash
python -m pip install -e factguard_generation
```

Copy and edit the environment template:

```bash
cp factguard_generation/.env.example factguard_generation/.env
```

The main variables are:

```bash
FACTGUARD_DATA_DIR=/absolute/path/to/raw_and_intermediate_data
VLLM_BASE_URL=http://127.0.0.1:8000/v1
VLLM_API_KEY=EMPTY
```

All model-facing instructions in the released generation and evaluation code
are written in English. For Chinese source examples, the English instruction
asks the model to produce Chinese questions or responses.

## Generation examples

Run one construction method on one source corpus:

```bash
python -m factguard_generation.generation.evidence_removal \
  --dataset gutenberg \
  --output-dir /path/to/output

python -m factguard_generation.generation.entity_substitution \
  --dataset pile-of-law \
  --output-dir /path/to/output

python -m factguard_generation.generation.impossible_condition \
  --dataset chinese-law \
  --output-dir /path/to/output
```

Run the same module over all four paper corpora:

```bash
bash factguard_generation/run_all_datasets.sh \
  factguard_generation.generation.evidence_removal \
  --output-dir /path/to/output
```

## Status

This is research code released for transparency. It requires the original raw
corpora and an OpenAI-compatible model endpoint. The normalized, supported
dataset release path is documented in the repository-level README.
