#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash factguard_generation/run_all_datasets.sh <module-or-script> [extra args...]" >&2
    echo "Example: bash factguard_generation/run_all_datasets.sh factguard_generation.generation.evidence_removal --output-dir outputs" >&2
    exit 1
fi

target=$1
shift

run_one() {
    local dataset=$1
    if [[ "${target}" == *.py || "${target}" == */* ]]; then
        python "${target}" --dataset "${dataset}" "$@"
    else
        python -m "${target}" --dataset "${dataset}" "$@"
    fi
}

run_one ancient-book "$@"
run_one chinese-law "$@"
run_one gutenberg "$@"
run_one pile-of-law "$@"
