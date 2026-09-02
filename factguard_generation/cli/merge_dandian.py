import jsonlines
from factguard_generation.env import FACTGUARD_DATA_DIR


def read_failed_examples(filename):
    examples = set()
    for example in jsonlines.open(
        FACTGUARD_DATA_DIR.joinpath(
            "generation", "dandian_clean", filename + ".failed.jsonl"
        ),
        "r",
    ):
        examples.add(example["uid"])
    print("Failed examples:", len(examples))
    return examples


def read_original_examples(filename):
    examples = dict()
    for example in jsonlines.open(
        FACTGUARD_DATA_DIR.joinpath("generation", "dandian_bak", filename)
    ):
        if "log" in example:
            continue
        examples[example["uid"]] = example
    print("Original examples:", len(examples))
    return examples


for filename in [
    "dandian_zh_book.jsonl",
    "dandian_zh_law.jsonl",
    "dandian_en_book.jsonl",
    "dandian_en_law.jsonl",
]:
    merged = {}
    failed_examples = read_failed_examples(filename)
    original_examples = read_original_examples(filename)

    writer = jsonlines.open(FACTGUARD_DATA_DIR / "dandian_merged" / filename, "w")
    for uid, example in original_examples.items():
        if uid in failed_examples:
            continue
        merged[uid] = example

    out_count = 0

    writer.write_all(merged.values())
    print(f"{out_count=}, {filename}:{len(merged)=}")
