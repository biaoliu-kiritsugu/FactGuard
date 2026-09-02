import json
import traceback
from factguard_generation.utility import extract_json_text, QueuedTasks, Writer
import pathlib
import os

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
parser.add_argument("--num_workers", type=int, default=4)


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


DEBUG = os.environ.get("FACTGUARD_DEBUG", "").lower() in {"1", "true", "yes"}
base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
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
    f"data_misattr_{Dataset.meta.lang}_{Dataset.meta.domain}_v2.jsonl"
)
print(f"processing {OUT_FILENAME}")
writer = Writer(OUT_FILENAME, "a")


def examples():
    dedup = DedupReader(OUT_FILENAME, "uid", filter_func=lambda x: "log" in x)
    if args.dedup_src_files:
        for filename in args.dedup_src_files:
            dedup.add_file(filename)

    print("Dedup examples:", len(dedup))
    for example in Dataset(max_length=128 * 1024, min_length=32 * 1024):
        if not dedup.contains_key(example.unique_id):
            yield example


def filter_question(example, struct, trace):
    q = struct["问题"]
    if any(keyword in q for keyword in ("如果", "替换", "instead")):
        trace.append("[red]终止原因:[/red]\n包含关键词")
        return True

    if not q.endswith("？") and not q.endswith("?"):
        trace.append("[red]终止原因:\n[/red]不是疑问句")
        return True

    origin = struct["替换前实体"]
    # if origin not in example["document"]:
    #     trace.append("[red]终止原因:\n[/red]替换前的实体没有出现在原文中")
    #     return True

    if origin not in struct["原始问题"]:
        trace.append("[red]终止原因:\n[/red]替换前的实体没有出现在原始问题中")
        return True
    if origin in q:
        trace.append("[red]终止原因:\n[/red]替换前的实体出现在最终问题中了")
        return True

    new = struct["替换后实体"]
    if new not in q:
        trace.append("[red]终止原因:\n[/red]替换后的实体没有出现在问题中")
        return True
    # if new in example["document"]:
    #     trace.append("[red]终止原因:\n[/red]替换后的实体在原文档中有出现")
    #     return True
    # judge_question_type(example, struct, trace)
    struct["origin"] = origin
    struct["new"] = new
    return False


def generate_question(example: Example, trace):
    context = {}
    step_1 = f"""**Document begins**

{example.document}
    
**Document ends**

Identify one salient entity in the document. The entity may be a time, place,
event, weather phenomenon, object, organization, or similar concept, but it
must not be a person's name. Return JSON:
```json
{{
  "entity": str
}}
```
""".strip()
    msg = llm_call(step_1, temperature=0.5)
    trace.append(f"[red]实体[/red]:\n{msg}")
    context["替换前实体"] = extract_json_text(msg, parse=True)["entity"]
    # if context["替换前实体"] in all_examples[example.unique_id]:
    #     trace.append("[red]终止原因:\n[/red]已经存在")
    #     return

    lang_constrain = (
        "Write the question and evidence in English."
        if Dataset.meta.lang == "en"
        else "Write the question and evidence in Chinese."
    )

    step_2 = f"""**Document begins**

{example.document}
    
**Document ends**

Create one question grounded in the document. The question must explicitly
contain "{context["替换前实体"]}". Also quote the supporting passage from the
document.

{lang_constrain}
Return JSON:
```json
{{
    "question": str,
    "evidence": str
}}
```
""".strip()
    msg = llm_call(step_2, temperature=0.5)
    ex = extract_json_text(msg, parse=True)
    question = ex["question"]
    clue = ex["evidence"]
    trace.append(f"[red]问题[/red]:\n{question}\n[red]依据[/red]:\n{clue}")
    context["原始问题"] = question
    context["依据"] = clue
    if context["替换前实体"] not in context["原始问题"]:
        trace.append("[red]终止原因:\n[/red]替换前的实体没有出现在原始问题中")
        return

    step_3 = f"""**Document begins**

{example.document}

**Document ends**

Propose a replacement for the entity "{context['替换前实体']}".
Requirements:
1. The replacement entity must not occur in the document.
2. It must have the same entity type and be semantically similar.
3. It should be lexically similar where possible.
4. It must clearly denote a different entity.
Return JSON:
```json
{{
  "entity": str
}}
```
""".strip()
    msg = llm_call(step_3)
    trace.append(f"[red]替换后实体[/red]:\n{msg}")
    msg = extract_json_text(msg, parse=True)
    context["替换后实体"] = msg["entity"]

    same_entity = llm_call(f"""
Determine whether the following two expressions refer to the same entity.
Entity 1: {context['替换前实体']}
Entity 2: {context['替换后实体']}

Return JSON:
```json
{{
  "same_entity": true | false
}}
""")
    trace.append(f"[red]实体判断[/red]:\n{same_entity}")
    if extract_json_text(same_entity, parse=True)["same_entity"] is True:
        trace.append("[red]终止原因:\n[/red]实体相同")
        return

    step_4 = f"""
Question: {context['原始问题']}

Replace "{context['替换前实体']}" with "{context['替换后实体']}" in the
question. Preserve all other meaning and return JSON:
```json
{{
  "question": str
}}
```
""".strip()
    msg = llm_call(step_4)
    trace.append(f"[red]替换后问题[/red]:\n{msg}")
    msg = extract_json_text(msg, parse=True)
    context["问题"] = msg["question"]
    if context["替换后实体"] not in context["问题"]:
        trace.append("[red]终止原因:\n[/red]替换后的实体没有出现在问题中")
        return
    return context


def get_current_answer(example: Example, struct, trace):
    q = struct["问题"]
    language_instruction = (
        "Answer in English." if Dataset.meta.lang == "en" else "Answer in Chinese."
    )
    prompt = f"""**Document begins**

{example.document}

**Document ends**
Question: {q}

Answer using only the document. If the answer cannot be found, politely decline
and explain what information is missing.
{language_instruction}
""".strip()
    message = llm_call(prompt)
    trace.append(f"[red]当前答案[/red]:\n{message}")
    return message


def generate_refuse(example, struct, trace):
    if Dataset.meta.lang == "en":
        constrain = "Write the response in English."
    else:
        constrain = "Write the response in Chinese."
    prompt = f"""
**Document begins**
{example.document}
**Document ends**

Question: {struct['问题']}

The document contains information about **{struct['替换前实体']}**, not
**{struct['替换后实体']}**. Generate a response that explicitly clarifies this
mismatch instead of answering the unsupported question.

Examples:
1. The document discusses the CPI increase in the United States, not Japan.
2. The document does not provide market-index data for March 15, 2025; the
   supplied data concerns June 12, 2024.
3. The document does not say that Huawei integrated GPT-4o; it discusses Apple.
4. The document does not provide the market size of India's flavor industry; it
   discusses China instead.

Write a natural and polished clarification in the style of these examples.
{constrain}

Response:
""".strip()
    message = llm_call(prompt)
    trace.append(f"[red]纠正[/red]:\n{message}")
    struct["纠正"] = message
    return


def generate_answer(example, struct, trace):
    origin_question = struct["问题"].replace(struct["替换后实体"], struct["替换前实体"])

    language_instruction = (
        "Answer in English." if Dataset.meta.lang == "en" else "Answer in Chinese."
    )
    prompt = f"""**Document begins**

{example.document}

**Document ends**
Question: {origin_question}
Answer using only the document.
{language_instruction}
""".strip()
    message = llm_call(prompt)
    trace.append(f"[red]原始问题答案:[/red]\n {message}")
    struct["正确答案"] = message
    return


def combine_refuse_answer(example, struct, trace):
    prompt = f"""
Merge the following two passages into one coherent response. Preserve the
format and factual detail of the second passage.

Passage 1:
{struct["纠正"]}

Passage 2:
{struct["正确答案"]}

Merged response:
""".strip()
    message = llm_call(prompt)
    trace.append(f"[red]最终答案:[/red]\n{message}")
    struct["final"] = message
    return True


def judge_answer_type(example, struct, answer, trace):
    prompt = f"""
**Document begins**
{example.document}
**Document ends**

Original question: {struct["原始问题"]}
Modified question: {struct["问题"]}

The modified question contains "{struct['替换后实体']}", which is not the
"{struct['替换前实体']}" entity discussed by the document and original
question.

Use only information stated in the document. Do not assume unsupported facts.

Determine whether the modified question can be answered or logically inferred
from the document. Return JSON:
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

    # 回答问题1的问题说明模型搞错了
    prompt = f"""
**Document begins**
{example.document}
**Document ends**

Question 1 (modified): {struct["问题"]}
Question 2 (original): {struct["原始问题"]}

Response:
{answer}

Determine which question the response actually answers and explain why.
Return JSON:
```json
{{
    "targets": "modified_question" | "original_question",
    "reasoning": "brief justification"
}}
```
"""
    message = llm_call(prompt)
    rich.print(f"[red]问题1和问题2的回答判断[/red]: {message}")
    message = extract_json_text(message, parse=True)
    try:
        if message["targets"] == "original_question":
            return True
        else:
            trace.append("[red]答案判断为回答修改后的问题终止[/red]")
            return False

    except json.decoder.JSONDecodeError:
        trace.append("[red]答案判断json解析失败终止[/red]")
    return False


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
            answer = get_current_answer(example, struct, trace)
            if not judge_answer_type(example, struct, answer, trace):
                writer.write(
                    {
                        "uid": example.unique_id,
                        "meta": Dataset.metadata(),
                        "doc": example.document,
                        **struct,
                        "log": trace,
                    }
                )
            struct["type"] = "common"
            generate_refuse(example, struct, trace)
            generate_answer(example, struct, trace)
            if not combine_refuse_answer(example, struct, trace):
                writer.write(
                    {"uid": example.unique_id, "meta": Dataset.metadata(), "log": trace}
                )
                return
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
