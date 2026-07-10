import jsonlines
from argparse import ArgumentParser
from dataclasses import dataclass

parser = ArgumentParser()
parser.add_argument(
    "--input-files", type=str, required=True, nargs="+", help="Input files"
)
args = parser.parse_args()


@dataclass
class MisattrStat:
    total = 0
    low_quality_question = 0
    infer_question = 0
    common_sense = 0

    def print(self):
        print("Total:", self.total)
        print("Low quality question:", self.low_quality_question)
        print("Infer question:", self.infer_question)
        print("Common sense question:", self.common_sense)

    def misattr_stat(self, filename):
        for example in jsonlines.open(filename):
            self.total += 1
            if "log" not in example:
                continue
            log = "\n".join(example["log"])
            if any(
                map(
                    lambda keyowrd: keyowrd in log,
                    (
                        "替换前的实体没有出现在原始问题中",
                        "实体相同",
                        "替换后的实体没有出现在问题中",
                    ),
                )
            ):
                self.low_quality_question += 1
            elif "答案可以通过推理得出" in log:
                self.infer_question += 1
            elif "答案判断为回答修改后的问题终止" in log:
                self.common_sense += 1
            else:
                print("Unknown log:", log)


misattr = MisattrStat()
for filename in args.input_files:
    if "misattr" in filename:
        print("misattr_stat", filename)
        misattr.misattr_stat(filename)
misattr.print()
