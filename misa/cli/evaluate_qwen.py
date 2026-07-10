from misattribution.env import MISA_DATA_DIR
from misattribution.utility import QueuedTasks, Writer
from misattribution.generation.dedup import DedupReader
from openai import OpenAI
import os
import jsonlines
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
args = parser.parse_args()

if os.environ.get("TI_TASK_ID", None):
    base_url = "http://127.0.0.1:8081/v1"
else:
    base_url = "YOUR_VLLM_BASE_URL"

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url=base_url,
)

output_filename = MISA_DATA_DIR / "evaluate" / f"{args.model}_preds.txt"
writer = Writer(output_filename, "a", flush=True)


def examples():
    dedup = DedupReader(output_filename, lambda ex: (ex["uid"], ex["source"]))
    print(f"dedup size : {len(dedup)}")
    for i, example in enumerate(
        jsonlines.open(MISA_DATA_DIR / "split" / "merged_test.jsonl", "r"), 1
    ):
        if i % 10 == 0:
            print(f"at example {i}")
        if example in dedup:
            print(f"skip {(example['source'], example['uid'])}")
            continue
        yield example


def llm_call(messages, extra_body=None, **kwargs):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    rsp = client.chat.completions.create(
        model=args.model,
        stream=False,
        messages=messages,
        extra_body=extra_body,
        temperature=0.0,
        max_completion_tokens=1024,
        **kwargs,
    )
    return rsp.choices[0].message.content


def work(example):
    example["completion"] = llm_call(example["input"])
    writer.write(example)


task = QueuedTasks(work, 4)
task.submit_jobs(examples())
task.wait()
writer.finish()
