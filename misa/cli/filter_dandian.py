import os

import jsonlines
import rich
from llmtask.utility import QueuedTasks, Writer, extract_json_text
from openai import OpenAI

from misattribution.generation.dedup import DedupReader
from argparse import ArgumentParser
import pathlib
from misattribution.env import MISA_DATA_DIR

if os.environ.get("TI_TASK_ID", None):
    base_url = "http://127.0.0.1:8081/v1"
else:
    base_url = "YOUR_VLLM_BASE_URL"

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url=base_url,
)

parser = ArgumentParser()
parser.add_argument("--input-file", type=str, required=True)

args = parser.parse_args()

filename = pathlib.Path(args.input_file).name

filename_failed = MISA_DATA_DIR.joinpath(
    "generation", "dandian_clean", filename + ".failed.jsonl"
)
filename_passed = MISA_DATA_DIR.joinpath(
    "generation", "dandian_clean", filename + ".passed.jsonl"
)

failed_writer = Writer(
    filename_failed,
    mode="a",
)
passed_writer = Writer(
    filename_passed,
    mode="a",
)


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


def llm_call(messages, extra_body=None, *, model="qwen2.5", **kwargs):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    rsp = client.chat.completions.create(
        model=model,
        stream=False,
        messages=messages,
        extra_body=extra_body,
        **kwargs,
    )
    return rsp.choices[0].message.content


def examples():
    dedup = DedupReader(filename_failed, "uid")
    dedup.add_file(filename_passed)
    print("Dedup size", len(dedup))
    for i, example in enumerate(jsonlines.open(args.input_file, "r"), 1):
        if "log" in example:
            continue
        if i % 1000 == 0:
            print("Processed", i, "examples")
        if example not in dedup:
            yield example


def get_completion(example):
    def make_example(doc, q):
        question = f"""文档:
{doc}

请根据文档回答问题: {q}"""

        if "_en" in filename:
            question = f"""Document:
{doc}

Please Answer the Question based on the document: {q}"""
        return question

    question = make_example(example["refuse_doc"], example["问题"])
    answer = llm_call(question)
    return answer


count = 0


def process_dandian(example):
    global count
    answer = get_completion(example)

    prompt = f"""
下面是针对某篇文档的问题和两个答案:

问题: {example["问题"]}
答案1: {answer}
答案2: {example["改写后答案"]}

忽略两个答案在文字表述上的差异,请判断针对问题两个答案是否有相同的主要结论
以下面的json格式返回结果
```json
{{
    "分析是否有相同结论": str,
    "是否有相同结论": "是" | "否"
}}
```
"""
    state = llm_call(prompt)
    state_json = extract_json_text(state, parse=True)
    if state_json["是否有相同结论"] == "是":
        failed_writer.write({"uid": example["uid"], "answer": answer})
        count += 1
        if count % 100 == 0:
            print(count)
    else:
        passed_writer.write({"uid": example["uid"], "answer": answer})


runner = QueuedTasks(worker=process_dandian, num_threads=2)
runner.submit_jobs(examples())
runner.wait()
passed_writer.finish()
failed_writer.finish()