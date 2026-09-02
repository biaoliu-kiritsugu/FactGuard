import json
import os
import random
from argparse import ArgumentParser

import rich
import pathlib
from factguard_generation.utility import (
    QueuedTasks,
    Writer,
    extract_json_text,
    THIKINIG_PROMPT_ZH,
    THIKINIG_PROMPT_EN,
    extract_thinking_final_answer,
)
from openai import OpenAI

from factguard_generation.env import FACTGUARD_DATA_DIR
from factguard_generation.generation.dedup import DedupReader
from factguard_generation.generation.dataset import Example

parser = ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    choices=["gutenberg", "pile-of-law", "chinese-law", "ancient-book"],
    default="gutenberg",
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


output_filename = (
    pathlib.Path(args.output_dir)
    / f"dandian_{Dataset.meta.lang}_{Dataset().meta.domain}.jsonl"
)

writer = Writer(output_filename, mode="a")


def load_fewshot_examples():
    with open(
        FACTGUARD_DATA_DIR / "misc" / "任务分类器query.json",
        "r",
        encoding="utf-8",
    ) as read_fp:
        fewshot_examples = json.load(read_fp)
        return fewshot_examples


fewshot_examples = load_fewshot_examples()

base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
num_workers = args.num_workers

client = OpenAI(
    api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
    base_url=base_url,
)


def llm_call(messages, extra_body=None):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    rsp = client.chat.completions.create(
        model="qwen2.5", stream=False, messages=messages, extra_body=extra_body
    )
    return rsp.choices[0].message.content


def segment_document(text, min_length=500, max_length=1000, sep="\n"):
    # 按照换行符分割文本
    lines = text.split(sep)
    segments = []
    current_segment = ""
    for line in lines:
        # 如果当前片段加上新行的长度超过最大长度，并且当前片段不为空则将当前片段加入片段列表
        if current_segment and len(current_segment) + len(line) + 1 > max_length:
            segments.append(current_segment)
            current_segment = line
        else:
            # 如果当前片段不为空，则添加换行符
            if current_segment:
                current_segment += sep
            current_segment += line

        # 如果当前片段的长度已经超过最小长度，则将其加入片段列表
        if len(current_segment) >= min_length:
            segments.append(current_segment)
            current_segment = ""

    # 添加最后一个片段
    if current_segment:
        segments.append(current_segment)
    filtered_segments = []

    for item in segments:
        if len(item) > 5000:
            continue
        if len(item) > min_length:
            filtered_segments.append(item)
    return filtered_segments


def examples():
    dedup = DedupReader(output_filename, "uid", filter_func=lambda x: "log" in x)
    if args.dedup_src_files:
        for filename in args.dedup_src_files:
            dedup.add_file(filename)
    print(f"dedup size : {len(dedup)}")
    for i, example in enumerate(Dataset(max_length=128 * 1024)):
        if dedup.contains_key(example.unique_id):
            print("skip", example.unique_id)
            continue
        if i == 500:
            break
        yield example


def generate_question(segment_text: str):
    # choose 5 random example per question type
    example_text = []
    for question_type, category_examples in fewshot_examples.items():
        random.shuffle(category_examples)
        examples = category_examples[:5]
        examples = "\n".join(
            map(
                lambda s: f"{s[0]}. {s[1]}",
                enumerate(examples[:5], 1),
            )
        )
        typed_example = f"""The following are example QA pairs of type {question_type}:
{examples}"""
        example_text.append(typed_example)
    example_text = "\n".join(example_text)

    lang_constraint = ""
    if Dataset().meta.lang == "en":
        lang_constraint = "Write the question, answer, and evidence in English."
    else:
        lang_constraint = "Write the question, answer, and evidence in Chinese."

    prompt = f"""The following text is a segment from a long document. Generate one QA pair similar to the examples below.
Requirements:
1. For entity-extraction questions, avoid questions whose answer contains multiple entities.
2. Avoid summarization, open-ended, or broad reasoning questions. Ask only about factual information in the text.
3. Ensure that the answer is valid and can be found in the text.
4. Make the question complete, clear, and unambiguous.
Text:
{segment_text}

{example_text}

Return the question, answer, an exact supporting passage from the text, and the QA type. Return exactly one QA pair using the following JSON schema:
```json
{{
    "question": str,
    "answer": str,
    "evidence": str,
    "question_type": "entity_extraction" | "numeric_extraction" | "content_extraction"
}}
```
{lang_constraint}
""".strip()
    messages = [
        {
            "role": "system",
            "content": THIKINIG_PROMPT_EN
            if Dataset().meta.lang == "en"
            else THIKINIG_PROMPT_ZH,
        },
        {"role": "user", "content": prompt},
    ]
    result = llm_call(messages)
    final_answer = extract_thinking_final_answer(result)
    json_text = extract_json_text(final_answer)
    rich.print(f"问题生成: \n{result}")
    ex = json.loads(json_text)
    for key in ("question", "answer", "evidence"):
        assert key in ex, f"missing key {key}, in the result {ex}"
    return {
        "问题": ex["question"],
        "答案": ex["answer"],
        "答案依据": ex["evidence"],
        "question_type": ex.get("question_type", question_type),
    }


def refine_answer(context):
    lang_constraint = ""
    if Dataset().meta.lang == "en":
        lang_constraint = "Write the rewritten answer in English."
    else:
        lang_constraint = "Write the rewritten answer in Chinese."
    prompt = f"""Evaluate and rewrite the answer using the document segment below.
1. Determine whether the question is grounded in the segment and whether the answer correctly and directly answers it. Set `assessment` to `invalid` if either condition fails; otherwise set it to `valid`.
2. Rewrite the answer so that it starts with a direct answer and then explains the reasoning using the quoted evidence: {context["答案依据"]}. Integrate both parts naturally. The segment comes from the middle of a longer document, so do not refer to its position.

Document segment:
{context["segment"]}

Question:
{context["问题"]}

Answer:
{context["答案"]}

Reason step by step and return the following JSON:
```json
{{
    "assessment_reasoning": str,
    "assessment": "valid" | "invalid",
    "answer_reasoning": str,
    "rewritten_answer": str
}}
```
{lang_constraint}
""".strip()
    messages = [
        {
            "role": "system",
            "content": THIKINIG_PROMPT_EN
            if Dataset().meta.lang == "en"
            else THIKINIG_PROMPT_ZH,
        },
        {"role": "user", "content": prompt},
    ]
    result = llm_call(messages)
    rich.print(f"[red]问题和答案评估[/red]: \n {result}")
    final_answer = extract_thinking_final_answer(result)
    json_text = extract_json_text(final_answer)
    ex = json.loads(json_text)
    for key in ("assessment", "answer_reasoning", "rewritten_answer"):
        assert key in ex
    if ex["assessment"] == "invalid":
        context["log"] = "问题和答案评估不正确"
        return False

    context["改写后答案"] = ex["rewritten_answer"]

    return True


def answer_with_refuse_doc(context):
    language_instruction = (
        "Answer in English." if Dataset().meta.lang == "en" else "Answer in Chinese."
    )
    question = f"""Document:
{context["refuse_doc"]}

Answer the question using only the document. If the document does not contain
sufficient evidence, explicitly say so and explain why.

Question: {context["问题"]}
{language_instruction}"""

    refuse_doc_answer = llm_call(question)
    rich.print(f"[red]剩余文档答案[/red]: \n{refuse_doc_answer}")
    context["refuse_doc_answer"] = refuse_doc_answer


def judge_answer_correctness(context):
    prompt = f"""
Below are a document, a question, and an answer.
Document:
{context["doc"]}

Question: {context["问题"]}
Answer: {context["改写后答案"]}

Determine whether the answer is correct according to the document. Return JSON
using the following schema:
```json
{{
    "correct": true | false
}}
```
""".strip()
    result = llm_call(prompt)
    rich.print(f"[red]判断答案是否正确:[/red]:\n{result}")
    ex = extract_json_text(result)
    ex = json.loads(ex)
    if ex["correct"] is True:
        return True
    context["log"] = "答案不正确"
    return False


def judge_answer_do_refusable(context):
    prompt = f"""
Below are a question and two answers to that question:

Question: {context["问题"]}
Answer 1: {context["改写后答案"]}
Answer 2: {context["refuse_doc_answer"]}

Ignore stylistic differences and determine whether the two answers have the
same main conclusion. Return JSON:
```json
{{
    "analysis": str,
    "same_conclusion": true | false
}}
```
""".strip()
    result = llm_call(prompt)
    rich.print(f"[red]分析是否有相同结论:[/red]\n{result}")
    ex = extract_json_text(result)
    ex = json.loads(ex)

    if ex["same_conclusion"] is False:
        return True
    else:
        context["log"] = "两个答案有相同结论"


def add_refuse_answer(context):
    lang_constraint = ""
    if Dataset().meta.lang == "en":
        lang_constraint = "Write the rewritten question and rejection in English."
    else:
        lang_constraint = "Write the rewritten question and rejection in Chinese."
    prompt = f"""
Original question: {context["问题"]}
Rewrite the question and construct its target response:
1. Preserve the meaning of the original question while explicitly grounding it
   in the supplied document, using wording such as "according to the document"
   or "based on the text."
2. Assume the supplied document does not contain the answer. Write a relevant,
   reasoned rejection and briefly describe what the document actually discusses
   to demonstrate why the requested answer cannot be found.
3. Return exactly one result using the following JSON schema:
```json
{{
    "rewritten_question": str,
    "reasoned_rejection": str
}}
```
{lang_constraint}
"""
    result = llm_call(prompt)
    ex = extract_json_text(result)
    rich.print(f"[red]refuse answer:[/red]\n {result}")
    ex = json.loads(ex)
    context["新问题"] = ex["rewritten_question"]
    context["拒答回复语"] = ex["reasoned_rejection"]


def pipeline(example: Example):
    if args.dataset == "ancient-book":
        sep = "。"
    else:
        sep = "\n"
    segments = segment_document(example.document, sep=sep)
    if not segments:
        writer.write(
            {
                "uid": example.unique_id,
                "meta": Dataset.metadata(),
                "log": "no segment found",
            }
        )
        return
    segments = list(enumerate(segments))
    candidate_segments = random.sample(segments, min(1, len(segments)))
    for idx, segment_text in candidate_segments:
        context = generate_question(segment_text)
        context["uid"] = example.unique_id
        context["meta"] = Dataset.metadata()
        context["len_range"] = example.len_range
        context["segment_idx"] = idx
        assert segment_text in example.document
        context["refuse_doc"] = example.document.replace(segment_text, "")
        context["doc"] = example.document
        context["segment"] = segment_text
        if not refine_answer(context):
            continue
        if not judge_answer_correctness(context):
            continue
        answer_with_refuse_doc(context)
        if not judge_answer_do_refusable(context):
            continue
        add_refuse_answer(context)
        writer.write(context)
        rich.print(f"writed {writer.writed_count}")
        break


with writer:
    task = QueuedTasks(num_threads=num_workers, worker=pipeline)
    task.submit_jobs(examples())
    task.wait()
