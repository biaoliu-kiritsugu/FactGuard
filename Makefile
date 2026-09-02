PYTHON ?= python

.PHONY: hf-data validate compile shellcheck

hf-data:
	$(PYTHON) scripts/prepare_hf_dataset.py \
		--train-file data/merged_train.jsonl \
		--validation-file data/merged_dev.jsonl \
		--test-file data/merged_test.jsonl \
		--output-dir hf_dataset

validate:
	$(PYTHON) scripts/validate_hf_dataset.py --dataset-dir hf_dataset

compile:
	$(PYTHON) -m compileall -q \
		evaluation factguard_generation/src factguard_generation/cli scripts train \
		compute_metrics.py compute_impossible_analysis.py compute_misattr_nls.py

shellcheck:
	@for file in evaluation/*.sh train/*.sh factguard_generation/run_all_datasets.sh; do \
		bash -n "$$file" || exit 1; \
	done
