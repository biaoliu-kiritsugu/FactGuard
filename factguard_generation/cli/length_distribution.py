import jsonlines
from factguard_generation.env import FACTGUARD_DATA_DIR
from collections import defaultdict


length_range = defaultdict(int)


def length_to_range(length):
    if length <= 8192:
        return "0-8K"
    if length <= 16 * 1024:
        return "8K-16K"
    if length <= 32 * 1024:
        return "16K-32K"
    if length <= 64 * 1024:
        return "32K-64K"
    if length <= 96 * 1024:
        return "64K-96K"
    if length <= 128 * 1024:
        return "96K-128K"
    else:
        return "128K+"


for filename in FACTGUARD_DATA_DIR.joinpath("generation").glob("*.jsonl"):
    print(filename)
    for example in jsonlines.open(filename, "r"):
        if "log" in example:
            continue
        doc = example.get("doc", None) or example["meta"]["doc"]
        length_range[length_to_range(len(doc))] += 1
print(length_range)
print(sum(length_range.values()))

