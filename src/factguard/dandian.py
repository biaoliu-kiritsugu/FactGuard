import json
import os
import random
from argparse import ArgumentParser

import rich
import pathlib
from factguard.utility import (
    QueuedTasks,
    Writer,
    extract_json_text,
    THIKINIG_PROMPT_ZH,
    THIKINIG_PROMPT_EN,
    extract_thinking_final_answer,
)
from openai import OpenAI

from factguard import env
from factguard.dedup import DedupReader
from factguard.dataset import Example

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
args = parser.parse_args()
if args.dataset == "gutenberg":
    from factguard.dataset import Gutenberg as Dataset
elif args.dataset == "pile-of-law":
    from factguard.dataset import PileOfLaw as Dataset
elif args.dataset == "chinese-law":
    from factguard.dataset import ChineseLaw as Dataset
elif args.dataset == "ancient-book":
    from factguard.dataset import AncientBook as Dataset
else:
    raise ValueError(f"Unknown dataset {args.dataset}")

DEBUG = False


client = OpenAI(
    api_key=env.VLLM_API_KEY,
    base_url=env.VLLM_API_URL,
)


output_filename = (
    pathlib.Path(args.output_dir)
    / f"dandian_{Dataset.meta.lang}_{Dataset().meta.domain}.jsonl"
)

writer = Writer(output_filename, mode="a")


def load_fewshot_examples():
    with open(
        env.MISA_DATA_DIR / "misc" / "任务分类器query.json",
        "r",
        encoding="utf-8",
    ) as read_fp:
        fewshot_examples = json.load(read_fp)
        return fewshot_examples


fewshot_examples = load_fewshot_examples()


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
        typed_example = f"""下面是{question_type}类型的问答对示例数据:
{examples}"""
        example_text.append(typed_example)
    example_text = "\n".join(example_text)

    lang_constraint = ""
    if Dataset().meta.lang == "en":
        lang_constraint = "问题、答案、答案依据请使用英文。"

    prompt = f"""下面的文本是某个长文档的一部分，请根据文本内容生成一个下面示例中类似的问答对。
要求:
1. 如果生成的是实体信息抽取的问答对，请避免答案为多个实体的问题
2. 避免总结、推理性的，开放性的问题，只针对文本中的事实信息进行提问
3. 确保生成的答案是合理的，且可以从文本中找到答案
4. 问题描述清晰完整，不要出现歧义
文本:
{segment_text}

{example_text}


请给出问题、答案和原文中的片段作为提出问题的依据，并给出问答对类型。只需要给出一个问答对。最后以下面的json格式输出:
```json
{{
    "问题": str,
    "答案": str,
    "答案依据": str
    "问答对类型": "实体信息抽取" | "数值信息抽取" | "内容抽取"
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
    for key in ("问题", "答案", "答案依据"):
        assert key in ex, f"missing key {key}, in the result {ex}"
    ex["question_type"] = question_type
    return ex


def refine_answer(context):
    lang_constraint = ""
    if Dataset().meta.lang == "en":
        lang_constraint = "改写后的答案请使用英文。"
    prompt = f"""根据下面的文档片段、问题、答案。
1. 问题和答案评估: 请判断问题是否针对文档片段，答案是否合理，可以明确回答问题。如果问题不是针对文档片段或者答案不合理，请判断为"不正确"，如果都无误判断为"正确"。
2. 请对答案进行改写: 先给出明确答案，再根据文档的信息给出推理思考的过程，思考过程中带上原文中引用的片段{context["答案依据"]}，最后将两者顺序融合。注意：文档只是长文的一个中间片段，新答案不要出现段落位置等表述。

文档片段:
{context["segment"]}

问题:
{context["问题"]}

答案:
{context["答案"]}

请一步一步思考，以下面的json格式输出:
```json
{{
    "问题和答案评估推理": str,
    "问题和答案评估": str,
    "答案推理过程": str,
    "改写后答案": str
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
    for key in ("问题和答案评估", "答案推理过程", "改写后答案"):
        assert key in ex
    if ex["问题和答案评估"] == "不正确":
        context["log"] = "问题和答案评估不正确"
        return False

    context["改写后答案"] = ex["改写后答案"]

    return True


def answer_with_refuse_doc(context):
    if Dataset().meta.lang == "zh":
        question = f"""文档:
{context["refuse_doc"]}

请根据文档回答问题: {context["问题"]}"""
    else:
        question = f"""Document:
{context["refuse_doc"]}

Please Answer the Question based on the document: {context["问题"]}"""

    refuse_doc_answer = llm_call(question)
    rich.print(f"[red]剩余文档答案[/red]: \n{refuse_doc_answer}")
    context["refuse_doc_answer"] = refuse_doc_answer


def judge_answer_correctness(context):
    prompt = f"""
下面是一个文档和针对文档的问题、答案。
文档:
{context["doc"]}

问题: {context["问题"]}
答案: {context["改写后答案"]}

请判断答案是否正确。以json格式返回:
```json
{{
    "答案正确性": "正确" | "不正确"
}}
```
""".strip()
    result = llm_call(prompt)
    rich.print(f"[red]判断答案是否正确:[/red]:\n{result}")
    ex = extract_json_text(result)
    ex = json.loads(ex)
    if ex["答案正确性"] == "正确":
        return True
    context["log"] = "答案不正确"
    return False


def judge_answer_do_refusable(context):
    prompt = f"""
下面是针对某篇文档的问题和两个答案:

问题: {context["问题"]}
答案1: {context["改写后答案"]}
答案2: {context["refuse_doc_answer"]}

忽略两个答案在文字表述上的差异,请判断针对问题两个答案是否有相同的主要结论
以下面的json格式返回结果
```json
{{
    "分析是否有相同结论": str,
    "是否有相同结论": "是" | "否"
}}
```
""".strip()
    result = llm_call(prompt)
    rich.print(f"[red]分析是否有相同结论:[/red]\n{result}")
    ex = extract_json_text(result)
    ex = json.loads(ex)

    if ex["是否有相同结论"] == "否":
        return True
    else:
        context["log"] = "两个答案有相同结论"


def add_refuse_answer(context):
    lang_constraint = ""
    if Dataset().meta.lang == "en":
        lang_constraint = "新问题、拒答回复语请使用英文回答。"
    prompt = f"""
已知原始问题: {context["问题"]}
请按下面的要求改写问题和答案:
1. 改写原始问题：问题语义不变，问题表达上融合【基于文档回答，文档中，本文，请严格按照文章分析，根据上文，参考文档】等等相似含义内容。
2. 基于新问题，假设提供了文档、但文档无答案的情况下，给出合理、呼应问题的拒答回复语，然后介绍文档的主要内容，以证明文中无法找到答案
3. 只输出一种改写结果即可。要求json格式输出，输出示例：
```json
{{
    "新问题": str, 
    "拒答回复语": str
}}
```
{lang_constraint}
"""
    result = llm_call(prompt)
    ex = extract_json_text(result)
    rich.print(f"[red]refuse answer:[/red]\n {result}")
    ex = json.loads(ex)
    context["新问题"] = ex["新问题"]
    context["拒答回复语"] = ex["拒答回复语"]


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
