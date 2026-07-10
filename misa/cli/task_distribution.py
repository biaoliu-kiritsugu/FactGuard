import argparse
import jsonlines

parser = argparse.ArgumentParser()
parser.add_argument("--files", type=str, nargs="+")
args = parser.parse_args()

task_data = {"dandian": 0, "misleading": 0, "answerable": 0}

for filename in args.files:
    for example in jsonlines.open(filename, "r"):
        if example["is_positive"]:
            task_data["answerable"] += 1
            continue

        if "dandian" in example["source"]:
            task_data["dandian"] += 1
        else:
            task_data["misleading"] += 1
print(task_data)