import jsonlines
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("--data-files", type=str, nargs="+", required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

writer = jsonlines.open(args.output, "w")

for filename in map(Path, args.data_files):
    print(filename)

    with jsonlines.open(filename, "r") as reader:
        for i, example in enumerate(reader):
            out = {
                "question_id": i,
                "uid": example["uid"],
                "source": example["source"],
                "question": example["input"],
            }
            writer.write(out)
