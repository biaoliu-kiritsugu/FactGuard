# Public release checklist

## Required before making the repositories public

- [x] Choose and add a code license (`LICENSE`): MIT.
- [ ] Complete a source-by-source dataset license review.
- [x] Set the dataset license to CC BY 4.0 in `hf_dataset/README.md`.
- [x] Add the dataset license notice in `DATA_LICENSE`.
- [x] Add `hf_dataset/LICENSE` to the standalone Hugging Face dataset package.
- [ ] Confirm that redistribution of every document is permitted, especially
      the Chinese book and legal sources.
- [ ] Run a privacy/PII scan and manually inspect flagged examples.
- [ ] Decide whether to release only the paper-reproduction split or also add a
      document-disjoint split. The current paper split has document overlap;
      this is disclosed in the dataset card and validator output.
- [x] Confirm that no API keys, tokens, internal hostnames, or employee-local
      paths remain in tracked files.
- [x] Replace Hugging Face namespace placeholders with `kilizi/FactGuard`.
- [x] Add the paper title, authors, EMNLP 2026 venue, and BibTeX.
- [ ] Add the final DOI and ACL Anthology URL when available.
- [x] Run `python scripts/validate_hf_dataset.py --dataset-dir hf_dataset`.
- [x] Run `python -m compileall` on supported Python modules.
- [x] Run shell syntax checks with `bash -n`.
- [x] Upload Parquet files through Git LFS or `huggingface_hub`.
- [x] Test `load_dataset("kilizi/FactGuard")` from a clean environment.

## Recommended repository split

Use two public repositories:

1. **GitHub code repository**: source code, prompts, configs, and documentation.
2. **Hugging Face dataset repository**: the contents of `hf_dataset/`.

Keeping the large data files out of the GitHub repository makes cloning and
versioning substantially easier.

## License note

The intended dataset license is CC BY 4.0. The released rows contain
source-document text, so the source-by-source redistribution review remains
necessary even after selecting the dataset license.
