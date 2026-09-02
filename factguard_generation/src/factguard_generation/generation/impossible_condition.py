import os
import traceback
from factguard_generation.utility import (
    extract_json_text,
    QueuedTasks,
    Writer,
)
import pathlib

import rich

from factguard_generation.generation.dataset import Example
from factguard_generation.generation.dedup import DedupReader
from openai import OpenAI
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    choices=["gutenberg", "pile-of-law", "chinese-law", "ancient-book"],
    default="ancient-book",
)
parser.add_argument("--dedup_src_files", type=str, default=None, nargs="+")
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--num_workers", type=int, default=8)

args = parser.parse_args()
if args.dataset == "gutenberg":
    from factguard_generation.generation.dataset import Gutenberg as Dataset
elif args.dataset == "pile-of-law":
    from factguard_generation.generation.dataset import PileOfLaw as Dataset
elif args.dataset == "chinese-law":
    from factguard_generation.generation.dataset import ChineseLaw as Dataset
elif args.dataset == "ancient-book":
    from factguard_generation.generation.dataset import AncientBook as Dataset
else:
    raise ValueError(f"Unknown dataset {args.dataset}")

DEBUG = False

base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
DEBUG = os.environ.get("FACTGUARD_DEBUG", "").lower() in {"1", "true", "yes"}
num_workers = args.num_workers

client = OpenAI(
    api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
    base_url=base_url,
)


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


class Trace(list):
    def __init__(self, *args, **kwargs):
        debug = kwargs.pop("debug", False)
        super().__init__(*args, **kwargs)
        self.debug = debug

    def append(self, object) -> None:
        if self.debug:
            try:
                rich.print(object)
            except Exception:
                print(object)

        return super().append(object)

    def show_log(self):
        rich.print("\n".join(self))


OUT_FILENAME = pathlib.Path(args.output_dir).joinpath(
    f"data_impossible_{Dataset.meta.lang}_{Dataset.meta.domain}_v2.jsonl"
)
writer = Writer(OUT_FILENAME, "a")


def examples():
    dedup = DedupReader(OUT_FILENAME, "uid", filter_func=lambda x: "log" in x)
    if args.dedup_src_files:
        for file in args.dedup_src_files:
            dedup.add_file(file)
    print("Deduplication:", len(dedup))
    for example in Dataset(max_length=128 * 1024, min_length=32 * 1024):
        if dedup.contains_key(example.unique_id):
            print("skip", example.unique_id)
            continue
        yield example


def generate_question(example: Example, trace):
    context = {}

    lang_constrain = (
        "Write all JSON string values in English."
        if Dataset.meta.lang == "en"
        else "Write all JSON string values in Chinese."
    )

    step_2 = f"""**Document begins**

{example.document}
    
**Document ends**

Select a passage and ask one question about it.
Requirements:
1. The answer must be stated in or logically derivable from the document.
2. Ask exactly one question.


{lang_constrain}
Return JSON:
```json
{{
    "evidence": string,
    "question": string,
    "answer": string
}}
```
""".strip()
    # msg = llm_call(step_2)
    # context = extract_json_text(msg, parse=True)
    # trace.append(f"[red]问题生成: [/red]:\n{msg}")
    # assert all(
    #     map(
    #         lambda key: key in context,
    #         ("问题", "原文片段"),
    #     )
    # )
    # context["原始问题"] = context["问题"]
    # context["原文片段"] = context["原文片段"]

    step_3 = f"""**Document begins**
{example.document}
**Document ends**

Create a question grounded in one passage from the document, then insert one
false or unsupported constraint so that the modified question cannot be
answered or inferred from the document.

Requirements:
1. Integrate the condition naturally into the question; do not append it as an
   inverted or trailing conditional clause.
2. Do not present the condition as a hypothetical assumption.
3. The added condition must not merely express a cause, purpose, or motivation.
4. The result must be a fluent, single interrogative sentence.

Example:
Original question: "After what battle did Union forces return to Jacksonville?"
Unsupported condition: "permanently"
Modified question: "After what battle did Union forces permanently leave Jacksonville?"
Explanation: The document says that the forces returned to Jacksonville and
retained control; it does not state that they permanently left after a battle.

Reason carefully and return JSON:
```json
{{
    "evidence": string,
    "condition_reasoning": string,
    "unsupported_condition": string,
    "draft_modified_question": string,
    "validation": string,
    "modified_question": string,
    "original_question": string
}}
```
{lang_constrain}
""".strip()

    msg_context = [
        {"role": "user", "content": step_3},
    ]
    msg = llm_call(msg_context)
    trace.append(f"[red]增加条件[/red]:\n{msg}")
    msg_json = extract_json_text(msg, parse=True)
    context = {
        "片段": msg_json["evidence"],
        "思考如何增加条件": msg_json["condition_reasoning"],
        "增加的条件": msg_json["unsupported_condition"],
        "初步拟定增加条件的问题": msg_json["draft_modified_question"],
        "检查初步拟定增加条件的问题": msg_json["validation"],
        "增加条件的问题": msg_json["modified_question"],
        "问题": msg_json["original_question"],
    }
    check_prompt = f"""
Determine whether the added condition "{context["增加的条件"]}" itself contains
the full answer or part of the answer to the following question.

{context["增加条件的问题"]}
Return JSON:
```json
{{
  "analysis": string,
  "contains_answer": true | false
}}
```
"""
    check_result = llm_call(check_prompt)
    trace.append(f"[red]检查增加条件[/red]:\n{check_result}")
    check_result_json = extract_json_text(check_result, parse=True)
    if check_result_json["contains_answer"] is True:
        trace.append("[red]增加的条件包含问题的答案[/red]")
        return None

    msg_context.append({"role": "assistant", "content": msg})
    step_3_2 = """
For the modified question:
1. Explain why the added condition is unsupported by the document.
2. Write a response that first clarifies why the modified question cannot be
   answered, and then answers the supported original question when possible.
Return JSON:
```json
{
  "response": string,
  "explanation": string
}
```
"""
    msg_context.append({"role": "user", "content": step_3_2})

    exp = llm_call(msg_context)
    trace.append(f"[red]解释[/red]:\n{exp}")
    exp_json = extract_json_text(exp, parse=True)
    assert ("response" in exp_json) and ("explanation" in exp_json)
    context["回答增加条件的问题"] = exp_json["response"]
    context["解释"] = exp_json["explanation"]
    return context


def get_current_answer(example: Example, struct, trace):
    lang_constrain = (
        "Answer in English." if Dataset.meta.lang == "en" else "Answer in Chinese."
    )
    q = struct["增加条件的问题"]
    prompt = f"""**Document begins**

{example.document}

**Document ends**
Question: {q}
Answer using only the document. If the answer cannot be found, politely decline
and explain what information is missing.
{lang_constrain}
""".strip()
    message = llm_call(prompt)
    trace.append(f"[red]当前答案[/red]:\n{message}")
    return message


def judge_answer_type(example, struct, trace):
    prompt = f"""
**Document begins**
{example.document}
**Document ends**

Original question: {struct["问题"]}
Modified question: {struct["增加条件的问题"]}

The modified question adds the condition "{struct["增加的条件"]}".

Use only the document and do not assume facts that are not stated.

Determine whether the modified question can be answered or inferred from the
document. Return JSON:
```json
{{
    "reasoning": "brief justification",
    "answerable": true | false
}}
```
"""
    message = llm_call(prompt)
    trace.append(f"[red]答案类型:{message}[/red]")
    message = extract_json_text(message, parse=True)
    try:
        if message["answerable"] is True:
            trace.append("[red]答案可以通过推理得出[/red]")
            struct["type"] = "推理"
            struct["answer"] = message
            return False
    except Exception as e:
        trace.append(f"[red]答案推理判断json解析失败 {e}[/red]")
        return False
    return True


def consumer(example):
    trace = Trace(debug=DEBUG)

    def work():
        try:
            struct = generate_question(example, trace)
            if struct is None:
                writer.write(
                    {"uid": example.unique_id, "meta": Dataset.metadata(), "log": trace}
                )
                return
            if not judge_answer_type(example, struct, trace):
                struct["type"] = "infer"
                writer.write(
                    {
                        "uid": example.unique_id,
                        "meta": Dataset.metadata(),
                        "doc": example.document,
                        **struct,
                        "log": trace,
                    }
                )
                return
            answer = get_current_answer(example, struct, trace)
            struct["type"] = "common"
            writer.write(
                {
                    "uid": example.unique_id,
                    "meta": Dataset.metadata(),
                    "len_range": example.len_range,
                    "doc": example.document,
                    **struct,
                    "现有模型答案": answer,
                }
            )

        except Exception:
            trace.append(traceback.format_exc())
            return

    work()
    trace.show_log()


if __name__ == "__main__":
    tasks = QueuedTasks(num_threads=num_workers, worker=consumer)
    tasks.submit_jobs(examples())
    tasks.wait()
    writer.finish()
