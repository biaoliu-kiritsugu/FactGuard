"""
对于每个文件，按照文档的长度进行采样
"""

import argparse
import jsonlines
import random
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input-files", type=str, required=True, nargs="+", help="Input files"
)
parser.add_argument("--output-dir", type=str, required=True, help="Output directory")

args = parser.parse_args()
args.output_dir = Path(args.output_dir)


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


def sample_in_file(filename):
    filename = Path(filename)
    file_length_range = defaultdict(int)
    example_group_by_length = defaultdict(list)
    output_filename_test = args.output_dir / (filename.name + "_test.jsonl")
    output_filename_train = args.output_dir / (filename.name + "_train.jsonl")
    output_filename_dev = args.output_dir / (filename.name + "_dev.jsonl")
    test_writer = jsonlines.open(output_filename_test, "w")
    train_writer = jsonlines.open(output_filename_train, "w")
    dev_writer = jsonlines.open(output_filename_dev, "w")

    for example in jsonlines.open(filename, "r"):
        if "log" in example:
            continue
        doc = example.get("doc", None) or example["meta"]["doc"]
        range_key = length_to_range(len(doc))
        example_group_by_length[range_key].append(example)
        file_length_range[range_key] += 1
    print(f"File: {filename}")
    for range_key, examples in example_group_by_length.items():
        random.shuffle(examples)
        test_count = 100
        dev_count = 50
        if "_zh" in filename.name:
            if range_key == "32K-64K":
                test_count = 50
                dev_count = 20
            if range_key == "64K-128K":
                test_count = 50
                dev_count = 10
        elif "dandian_en_law" in filename.name:
            if range_key == "32K-64K" or range_key == "64K-128K":
                test_count = 100
                dev_count = 20

        valina_size = len(examples)
        examples_dict = {}
        for ex in examples:
            examples_dict[ex["uid"]] = ex
        examples = list(examples_dict.values())
        print(f"dedup examples from {valina_size} to size {len(examples)}")
        for example in examples[:test_count]:
            test_writer.write(example)
        for example in examples[test_count : test_count + dev_count]:
            dev_writer.write(example)
        for example in examples[
            test_count + dev_count : test_count + dev_count + random.randint(800, 1200)
        ]:
            train_writer.write(example)
        print(
            f"file : {filename.name}, range: {range_key},  test: {test_count}, dev: {dev_count}, train: {len(examples) - test_count - dev_count}"
        )


for filename in args.input_files:
    sample_in_file(filename)
