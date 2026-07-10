import jsonlines
from collections import defaultdict
import rich
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--data-files", type=str, nargs="+", required=True)
args = parser.parse_args()

length_range = defaultdict(int)


def length_to_range(length):
    if length <= 16 * 1024:
        return "0-16K"
    if length <= 32 * 1024:
        return "16K-32K"
    if length <= 64 * 1024:
        return "32K-64K"
    if length <= 128 * 1024:
        return "64K-128K"
    else:
        return "128K+"


# per file stat
per_file_stat = {}
for filename in map(Path, args.data_files):
    print(filename)
    file_length_range = defaultdict(int)
    for example in jsonlines.open(filename, "r"):
        if "log" in example:
            continue
        doc = example.get("doc", None) or example["meta"]["doc"]
        range_key = length_to_range(len(doc))
        length_range[range_key] += 1
        file_length_range[range_key] += 1
    per_file_stat[filename.name] = dict(file_length_range)

rich.print("Length Range Distribution:")


# Custom sort function to sort based on the numeric value at the start of the key
def sort_key(item):
    key = item[0]
    if key == "128K+":
        return float("inf")  # Ensure "128K+" is always at the end
    return int(key.split("-")[0].rstrip("K"))


sorted_length_range = dict(sorted(length_range.items(), key=sort_key))
sorted_per_file_stat = {
    k: dict(sorted(v.items(), key=sort_key)) for k, v in sorted(per_file_stat.items())
}

rich.print(sorted_length_range)
rich.print("Total Documents Count:")
rich.print(sum(length_range.values()))
rich.print("Per File Statistics:")
rich.print(sorted_per_file_stat)
