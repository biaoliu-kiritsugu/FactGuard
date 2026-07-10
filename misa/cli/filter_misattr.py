import os
from llmtask.utility import (
    extract_json_text,
    QueuedTasks,
    Writer,
)

import rich

from misattribution.generation.dedup import DedupReader
from misattribution.env import MISA_DATA_DIR
from openai import OpenAI
from argparse import ArgumentParser
import pathlib
import jsonlines

parser = ArgumentParser()
parser.add_argument("--input-file", type=str, required=True)

args = parser.parse_args()


if os.environ.get("TI_TASK_ID", None):
    base_url = "http://127.0.0.1:8081/v1"
else:
    base_url = "YOUR_VLLM_BASE_URL"

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url=base_url,
)
filename = pathlib.Path(args.input_file).name

filename_failed = MISA_DATA_DIR.joinpath(
    "generation", "misattr_clean", filename + ".failed.jsonl"
)
filename_passed = MISA_DATA_DIR.joinpath(
    "generation", "misattr_clean", filename + ".passed.jsonl"
)
failed_writer = Writer(
    filename_failed,
    mode="a",
)
passed_writer = Writer(
    filename_passed,
    mode="a",
)


def examples():
    dedup = DedupReader(filename_failed, "uid")
    dedup.add_file(filename_passed)
    for i, example in enumerate(jsonlines.open(args.input_file, "r"), 1):
        if i % 1000 == 0:
            print("Processed", i, "examples")
        if example not in dedup:
            yield example


def pretty_print_message(messages):
    table = rich.table.Table(show_header=True)
    table.add_column("Role", style="bold")
    table.add_column("Content")

    for message in messages:
        role_style = "green" if message["role"] == "assistant" else "blue"
        table.add_row(
            f"[{role_style}]{message['role'].title()}[/{role_style}]",
            message["content"],
        )

    rich.print(table)


def llm_call(messages, extra_body=None, **kwargs):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    rsp = client.chat.completions.create(
        model="qwen2.5",
        stream=False,
        messages=messages,
        extra_body=extra_body,
        **kwargs,
    )
    return rsp.choices[0].message.content


def worker(example):
    if "log" in example:
        failed_writer.write({"uid": example["uid"]})
        return

    prompt = f"""
下面两个实体是否是同一个实体:
1. {example["替换前实体"]}
2. {example["替换后实体"]}
请以下面的json格式返回:
```json
{{
  "是否同一实体": "是" | "否"
}}
"""
    result = llm_call(prompt)
    result_json = extract_json_text(result, parse=True)
    # pretty_print_message(
    #     [{"role": "user", "content": prompt}, {"role": "assistant", "content": result}]
    # )
    if result_json["是否同一实体"] == "是":
        failed_writer.write({"uid": example["uid"]})
    else:
        passed_writer.write({"uid": example["uid"]})


if __name__ == "__main__":
    tasks = QueuedTasks(num_threads=128, worker=worker)
    tasks.submit_jobs(examples())
    tasks.wait()
    passed_writer.finish()
    failed_writer.finish()
