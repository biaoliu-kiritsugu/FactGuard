import jsonlines
import pathlib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--filenames", type=str, required=True, nargs="+")
parser.add_argument("--src-dirs", type=str, nargs="+", required=True)
parser.add_argument("--output-dir", type=str, required=True)
args = parser.parse_args()


def read_file(filename):
    examples = dict()
    for example in jsonlines.open(filename):
        if "log" in example:
            continue
        examples[example["uid"]] = example
    print(f"read {len(examples)} examples from {filename}")
    return examples


for filename in args.filenames:
    merged = {}
    for directory in args.src_dirs:
        directory = pathlib.Path(directory)
        examples = read_file(directory / filename)
        merged.update(examples)

    writer = jsonlines.open(pathlib.Path(args.output_dir) / filename, "w")
    writer.write_all(merged.values())
    print(f"{filename} number of examples: {len(merged)}")
