import os

import jsonlines
import rich
from factguard_generation.utility import QueuedTasks, Writer, extract_json_text
from openai import OpenAI

from factguard_generation.generation.dedup import DedupReader
from argparse import ArgumentParser
import pathlib
from factguard_generation.env import FACTGUARD_DATA_DIR

base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")

client = OpenAI(
    api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
    base_url=base_url,
)

parser = ArgumentParser()
parser.add_argument("--input-file", type=str, required=True)

args = parser.parse_args()

filename = pathlib.Path(args.input_file).name

filename_failed = FACTGUARD_DATA_DIR.joinpath(
    "generation", "dandian_clean", filename + ".failed.jsonl"
)
filename_passed = FACTGUARD_DATA_DIR.joinpath(
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
        language_instruction = (
            "Answer in English." if "_en" in filename else "Answer in Chinese."
        )
        question = f"""Document:
{doc}

Answer the question using only the document. If the document does not contain
sufficient evidence, explicitly say so.
Question: {q}
{language_instruction}"""
        return question

    question = make_example(example["refuse_doc"], example["问题"])
    answer = llm_call(question)
    return answer


count = 0


def process_dandian(example):
    global count
    answer = get_completion(example)

    prompt = f"""
Below are a question and two answers:

Question: {example["问题"]}
Answer 1: {answer}
Answer 2: {example["改写后答案"]}

Ignore stylistic differences and determine whether the answers have the same
main conclusion. Return JSON:
```json
{{
    "analysis": str,
    "same_conclusion": true | false
}}
```
"""
    state = llm_call(prompt)
    state_json = extract_json_text(state, parse=True)
    if state_json["same_conclusion"] is True:
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
