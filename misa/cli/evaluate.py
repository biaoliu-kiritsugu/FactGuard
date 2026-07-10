import json
import os

import jsonlines
import rich
from llmtask.utility import QueuedTasks, Writer, extract_json_text
from openai import OpenAI

from misattribution.generation.dedup import DedupReader
from misattribution.utility import data_dir
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--input", type=str, default="Llama-3.3-70B-Instruct_preds.txt")
args = parser.parse_args()

if os.environ.get("TI_TASK_ID", None):
    base_url = "http://127.0.0.1:8081/v1"
else:
    base_url = "YOUR_VLLM_BASE_URL"

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url=base_url,
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


model_output_dir = data_dir / "evaluate"
FILENAME = args.input

writer = Writer(data_dir / "evaluate" / "result" / FILENAME, mode="a")
dedup = DedupReader(
    data_dir / "evaluate" / "result" / FILENAME, lambda x: (x["uid"], x["source"])
)


def examples():
    with jsonlines.open(model_output_dir.joinpath(FILENAME), "r") as reader:
        for example in reader:
            if example in dedup:
                print(f"skip {(example['uid'], example['source'])}")
                continue
            if example["is_positive"]:
                example["type"] = "answerable"
            elif "dandian" in example["source"]:
                example["type"] = "dandian"
            elif "impossible" in example["source"]:
                example["type"] = "impossible"
            elif "misattr" in example["source"]:
                example["type"] = "misattr"
            else:
                assert False

            answer_key = None
            for key in ("answer_chatgpt4o", "answer_claude3.5", "completion"):
                if key in example:
                    answer_key = key
                    break
            assert answer_key is not None

            if answer_key != "completion":
                example["completion"] = example[answer_key]["choices"][0]["message"][
                    "content"
                ]
            yield example


def process_misattr(example):
    prompt = f"""
下面是针对某篇文档的问题和答案

问题: {example['origin']["问题"]}
答案: {example["completion"]}


已知文中没有出现"{example["origin"]["替换后实体"]}",出现的是"{example["origin"]["替换前实体"]}"因此答案需要对此进行澄清
请判断答案中是否有类似下面澄清的例子中类似的表述
你只需要分析是否有澄清和进行判断，不需要修改答案。

澄清的例子:
1. 文中没有提到{example["origin"]["替换后实体"]}
2. 关于{example["origin"]["替换后实体"]}的xxx无法回答
3. {example["origin"]["替换后实体"]}({example["origin"]["替换前实体"]})

请注意：答案中仅仅提到了{example["origin"]["替换前实体"]}不算澄清
"""
    messages = [{"role": "user", "content": prompt}]
    completion = llm_call(messages)
    messages.append({"role": "assistant", "content": completion})

    prompt = """
把上面的结果转换为json格式
```json
{
  "是否有澄清": "是" | "否",
  "对应片段": "如果指出来了或者有类似澄清,给出片段内容，否则给出空值"
}
```
"""
    messages.append({"role": "user", "content": prompt})
    result = llm_call(messages)
    messages.pop()
    messages.append({"role": "assistant", "content": result})
    json_text = extract_json_text(result)
    ex = json.loads(json_text)
    if (
        ex["是否有澄清"] == "否"
        and example["origin"]["替换后实体"] in example["completion"]
    ):
        ex["是否混淆"] = True
    else:
        ex["是否混淆"] = False

    messages.append(
        {
            "role": "background",
            "content": f"替换前实体: {example['origin']['替换前实体']}\n替换后实体: {example['origin']['替换后实体']}\n是否混淆: {ex['是否混淆']}",
        }
    )
    pretty_print_message(messages)
    example["eval_result"] = ex
    out = {
        "uid": example["uid"],
        "source": example["source"],
        "type": example["type"],
        "eval_result": ex,
    }
    writer.write(out)


def process_dandian(example):
    prompt = f"""
下面是针对某篇文档的问题和答案

问题: {example['origin']["问题"]}
答案: {example["completion"]}

请问答案中是否有指出问题无法回答或者相关信息不存在,请给出答案中对应片段
"""
    message = [{"role": "user", "content": prompt}]
    analysis = llm_call(message)
    message.append({"role": "assistant", "content": analysis})
    prompt = """
把上面的结果转换为json格式
```json
{
  "是否有指出": "是" | "否",
  "对应片段": "如果指出来了或者有类似澄清,给出片段内容，否则给出空值"
}
```
"""
    message.append({"role": "user", "content": prompt})
    result = llm_call(message)
    message.append({"role": "assistant", "content": result})
    json_text = extract_json_text(result)
    ex = json.loads(json_text)
    if ex["是否有指出"] == "否":
        prompt = f"""
下面是针对某篇文档的问题和两个答案:

问题: {example['origin']["问题"]}
答案1: {example["completion"]}
答案2: {example["origin"]["改写后答案"]}

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
            ex.update(state_json)
            ex["数据有误"] = True
        message.append({"role": "user", "content": prompt})
        message.append({"role": "assistant", "content": state})
    pretty_print_message(message)
    example["eval_result"] = ex
    out = {
        "uid": example["uid"],
        "source": example["source"],
        "type": example["type"],
        "eval_result": ex,
    }
    writer.write(out)


def process_impossible(example):
    prompt = f"""
下面是针对某篇文档的问题和答案

问题: {example['origin']["增加条件的问题"]}
答案: {example["completion"]}


已知条件:
{example['origin']["解释"]}

因此问题中的"{example['origin']["增加的条件"]}"是不存在的。答案需要对此进行讨论
包括和指出问题中和"{example['origin']["增加的条件"]}"相关的部分无法回答或者相关信息不存在。
请判断答案中是否有类似的表述.
你只需要分析和判断答案中是否有这样的表述，不需要修改答案。
"""
    messages = [{"role": "user", "content": prompt}]
    completion = llm_call(messages)
    messages.append({"role": "assistant", "content": completion})

    prompt = """
把上面的结果转换为json格式
```json
{
  "分析": "分析内容",
  "是否有澄清": "是" | "否",
  "对应片段": "如果指出来了或者有类似澄清,给出片段内容，否则给出空值"
}
```
"""
    messages.append({"role": "user", "content": prompt})
    result = llm_call(messages)
    messages.pop()
    messages.append({"role": "assistant", "content": result})
    json_text = extract_json_text(result)
    ex = json.loads(json_text)

    # messages.insert(
    #     0,
    #     {
    #         "role": "background",
    #         "content": f"增加的条件: {example['origin']['增加的条件']}",
    #     },
    # )
    pretty_print_message(messages)
    example["eval_result"] = ex
    out = {
        "uid": example["uid"],
        "source": example["source"],
        "type": example["type"],
        "eval_result": ex,
    }
    writer.write(out)


def process_answerable(example):
    prompt = f"""
下面是针对某篇文档的问题和答案

问题: {example['origin']["问题"]}
答案: {example["completion"]}

请问答案中是否有指出问题无法回答或者相关信息不存在,请给出答案中对应片段
"""
    message = [{"role": "user", "content": prompt}]
    analysis = llm_call(message)
    message.append({"role": "assistant", "content": analysis})
    prompt = """
把上面的结果转换为json格式
```json
{
  "是否有指出": "是" | "否",
  "对应片段": "如果指出来了或者有类似说法,给出片段内容，否则给出空值"
}
```
"""
    message.append({"role": "user", "content": prompt})
    result = llm_call(message)
    message.append({"role": "assistant", "content": result})
    json_text = extract_json_text(result)
    ex = json.loads(json_text)
    example["eval_result"] = ex
    out = {
        "uid": example["uid"],
        "source": example["source"],
        "type": example["type"],
        "eval_result": ex,
    }
    if ex["是否有指出"] == "否":
        prompt = f"""
下面是针对某篇文档的问题和两个答案:

问题: {example['origin']["问题"]}
答案1: {example["completion"]}
答案2: {example["output"]}

忽略两个答案在文字表述上的差异,请判断针对问题两个答案是否有相同的主要结论
以下面的json格式返回结果
```json
{{
    "分析是否有相同结论": str,
    "是否有相同结论": "是" | "否"
}}
```
""".strip()
        state = llm_call(prompt)
        state_json = extract_json_text(state, parse=True)
        ex.update(state_json)
    
    writer.write(out)


prompt_map = {
    "answerable": process_answerable,
    "misattr": process_misattr,
    "dandian": process_dandian,
    "impossible": process_impossible,
}


def worker(example):
    prompt_map[example["type"]](example)


with writer:
    runner = QueuedTasks(worker=worker, num_threads=64)
    runner.submit_jobs(examples())
    runner.wait()
