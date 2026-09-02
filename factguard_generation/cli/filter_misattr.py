import os
from factguard_generation.utility import (
    extract_json_text,
    QueuedTasks,
    Writer,
)

import rich

from factguard_generation.generation.dedup import DedupReader
from factguard_generation.env import FACTGUARD_DATA_DIR
from openai import OpenAI
from argparse import ArgumentParser
import pathlib
import jsonlines

parser = ArgumentParser()
parser.add_argument("--input-file", type=str, required=True)

args = parser.parse_args()


base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")

client = OpenAI(
    api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
    base_url=base_url,
)
filename = pathlib.Path(args.input_file).name

filename_failed = FACTGUARD_DATA_DIR.joinpath(
    "generation", "misattr_clean", filename + ".failed.jsonl"
)
filename_passed = FACTGUARD_DATA_DIR.joinpath(
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
Determine whether the following expressions refer to the same entity:
1. {example["替换前实体"]}
2. {example["替换后实体"]}
Return JSON:
```json
{{
  "same_entity": true | false
}}
"""
    result = llm_call(prompt)
    result_json = extract_json_text(result, parse=True)
    # pretty_print_message(
    #     [{"role": "user", "content": prompt}, {"role": "assistant", "content": result}]
    # )
    if result_json["same_entity"] is True:
        failed_writer.write({"uid": example["uid"]})
    else:
        passed_writer.write({"uid": example["uid"]})


if __name__ == "__main__":
    tasks = QueuedTasks(num_threads=128, worker=worker)
    tasks.submit_jobs(examples())
    tasks.wait()
    passed_writer.finish()
    failed_writer.finish()
