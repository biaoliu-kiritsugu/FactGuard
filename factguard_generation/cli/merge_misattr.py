import jsonlines
from factguard_generation.env import FACTGUARD_DATA_DIR


def read_failed_examples(filename):
    examples = set()
    for example in jsonlines.open(
        FACTGUARD_DATA_DIR.joinpath(
            "generation", "misattr_clean", filename + ".failed.jsonl"
        ),
        "r",
    ):
        examples.add(example["uid"])
    print("Failed examples:", len(examples))
    return examples


def read_original_examples(filename):
    examples = dict()
    for example in jsonlines.open(
        FACTGUARD_DATA_DIR.joinpath("generation", "misattr_bak", filename)
    ):
        if "log" in example:
            continue
        examples[example["uid"]] = example
    print("Original examples:", len(examples))
    return examples


def read_new_examples(filename):
    examples = dict()
    for example in jsonlines.open(FACTGUARD_DATA_DIR.joinpath("generation", filename), "r"):
        if "log" in example:
            continue
        examples[example["uid"]] = example
    print("New examples:", len(examples))
    return examples


for filename in [
    "data_misattr_zh_book_v2.jsonl",
    "data_misattr_zh_law_v2.jsonl",
    "data_misattr_en_book_v2.jsonl",
    "data_misattr_en_law_v2.jsonl",
]:
    merged = {}
    failed_examples = read_failed_examples(filename)
    original_examples = read_original_examples(filename)
    new_examples = read_new_examples(filename)

    writer = jsonlines.open(FACTGUARD_DATA_DIR / "misattr_merged" / filename, "w")
    for uid, example in original_examples.items():
        if uid in failed_examples:
            continue
        merged[uid] = example

    out_count = 0
    for uid, example in new_examples.items():
        if uid not in failed_examples:
            out_count += 1
        merged[uid] = example

    writer.write_all(merged.values())
    print(f"{out_count=}, {filename}:{len(merged)=}")
