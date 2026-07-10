import argparse
import jsonlines

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input-files", type=str, required=True, nargs="+", help="Input files"
)
args = parser.parse_args()

num_exmaples = {"zh": 0, "en": 0}
doc_ids = {"zh": set(), "en": set()}
for filename in args.input_files:
    for example in jsonlines.open(filename, "r"):
        if "log" in example:
            continue
        source = example["source"]
        if "_zh" in source:
            doc_ids["zh"].add(example["uid"])
            num_exmaples["zh"] += 1
        elif "_en" in source:
            doc_ids["en"].add(example["uid"])
            num_exmaples["en"] += 1
        else:
            raise ValueError(f"Unknown source: {source}")
print(f"Total examples: {num_exmaples}")
print(f"Total unique doc ids: en: {len(doc_ids['en'])} zh: {len(doc_ids['zh'])}")
