import argparse
import jsonlines
from dataclasses import dataclass

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", type=str, default="data/evaluate/result/qwen2.5_preds.txt"
)
args = parser.parse_args()


@dataclass
class Metric:
    dandian_count: int = 0
    dandian_correct_count: int = 0
    dandian_data_error = 0

    answerable_count: int = 0
    answerable_allow_count: int = 0
    answerable_correct_count: int = 0

    misattr_count: int = 0
    misattr_clarify_count: int = 0
    misattr_misuse_count: int = 0

    impossible_count: int = 0
    impossible_clarify_count: int = 0

    def print_result(self):
        # print dandian accuracy
        print(
            f"dandian accuracy: {self.dandian_correct_count/(self.dandian_count - self.dandian_data_error + 1e-6)}"
        )
        # print dandian support
        print(f"dandian numbers : {self.dandian_count=},{self.dandian_data_error=}")
        # print misattr rate
        print(
            f"misattr clarify rete: {self.misattr_clarify_count/(self.misattr_count + 1e-6)}"
        )
        print(
            f"misattr misuse rate: {self.misattr_misuse_count/(self.misattr_count + 1e-6)}"
        )
        # print support
        print(
            f"misattr numbers : {self.misattr_count=}, {self.misattr_clarify_count=}, {self.misattr_misuse_count=}"
        )

        # print answerable accuracy
        print(
            f"answerable allow ratio: {self.answerable_allow_count/(self.answerable_count + 1e-6)} answerable accuracy: {self.answerable_correct_count/(self.answerable_count + 1e-6)}"
        )
        # print answerable support
        print(
            f"answerable numbers : {self.answerable_count=},{self.answerable_correct_count=}, {self.answerable_allow_count=}"
        )

        # print impossible clarify rate
        print(
            f"impossible clarify rate: {self.impossible_clarify_count/(self.impossible_count + 1e-6)}"
        )
        # print impossible support
        print(
            f"impossible numbers : {self.impossible_count=},{self.impossible_clarify_count=}"
        )


metric = Metric()


for example in jsonlines.open(args.input):
    eval_result = example["eval_result"]
    if example["type"] == "answerable":
        metric.answerable_count += 1
        refusal = eval_result.get("refusal_detected")
        if refusal is None:
            refusal = eval_result.get("是否有指出") == "是"
        if not refusal:
            metric.answerable_allow_count += 1
            same_conclusion = eval_result.get("same_conclusion")
            if same_conclusion is None:
                same_conclusion = eval_result.get("是否有相同结论") == "是"
            if same_conclusion:
                metric.answerable_correct_count += 1

    elif example["type"] == "dandian":
        metric.dandian_count += 1
        refusal = eval_result.get("refusal_detected")
        if refusal is None:
            refusal = eval_result.get("是否有指出") == "是"
        if refusal:
            metric.dandian_correct_count += 1
        else:
            if "same_conclusion" in eval_result or "是否有相同结论" in eval_result:
                metric.dandian_data_error += 1
    elif example["type"] == "misattr":
        metric.misattr_count += 1
        clarified = eval_result.get("clarified")
        if clarified is None:
            clarified = eval_result.get("是否有澄清") == "是"
        if clarified:
            metric.misattr_clarify_count += 1
        elif eval_result.get("confused_entity", eval_result.get("是否混淆", False)):
            metric.misattr_misuse_count += 1
    elif example["type"] == "impossible":
        metric.impossible_count += 1
        clarified = eval_result.get("clarified")
        if clarified is None:
            clarified = eval_result.get("是否有澄清") == "是"
        if clarified:
            metric.impossible_clarify_count += 1


metric.print_result()
