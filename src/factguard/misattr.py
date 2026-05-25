import json
import traceback
from factguard.utility import extract_json_text, QueuedTasks, Writer
import pathlib

import rich


from factguard.dataset import Example
from factguard.dedup import DedupReader
from openai import OpenAI
from argparse import ArgumentParser
from factguard import  env




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
    step_1 = f"""**文本开始**

{example.document}
    
**文本结束**

请输出的文本中主要的一个实体，实体类型可以是时间、地点、事件、天气、物品等，实体不能是人名。以json 格式返回:
```json
{{
  "实体": str
}}
```
""".strip()
    msg = llm_call(step_1, temperature=0.5)
    trace.append(f"[red]实体[/red]:\n{msg}")
    context["替换前实体"] = extract_json_text(msg, parse=True)["实体"]
    # if context["替换前实体"] in all_examples[example.unique_id]:
    #     trace.append("[red]终止原因:\n[/red]已经存在")
    #     return

    lang_constrain = "问题请使用英文" if Dataset.meta.lang == "en" else ""

    step_2 = f"""**文本开始**

{example.document}
    
**文本结束**

请根据原文内容，提出一个问题，问题文本中要包含"{context["替换前实体"]}"。并引用原文作为提问的依据。{lang_constrain}

以下面的json格式返回
```json
{{
    "问题": str,
    "依据": str
}}
```
""".strip()
    msg = llm_call(step_2, temperature=0.5)
    ex = extract_json_text(msg, parse=True)
    question = ex["问题"]
    clue = ex["依据"]
    trace.append(f"[red]问题[/red]:\n{question}\n[red]依据[/red]:\n{clue}")
    context["原始问题"] = question
    context["依据"] = clue
    if context["替换前实体"] not in context["原始问题"]:
        trace.append("[red]终止原因:\n[/red]替换前的实体没有出现在原始问题中")
        return

    step_3 = f"""**文本开始**

{example.document}

**文本结束**

请将文本中的"{context['替换前实体']}"替换为类似的实体，确保，
要求:
1. 替换后的实体没有在文本中出现
2. 替换后的实体和替换前的实体类型相同，语义相近
3. 替换后的实体和替换前的实体在字面上应该尽量接近
4. 替换后的实体应该明显不是替换前的实体
结果以json 格式返回:
```json
{{
  "实体": str
}}
```
""".strip()
    msg = llm_call(step_3)
    trace.append(f"[red]替换后实体[/red]:\n{msg}")
    msg = extract_json_text(msg, parse=True)
    context["替换后实体"] = msg["实体"]

    same_entity = llm_call(f"""
请判断下面两个实体是否是同一个实体，以json格式返回:
实体1: {context['替换前实体']}
实体2: {context['替换后实体']}

```json
{{
  "是否同一实体": "是" | "否"
}}
""")
    trace.append(f"[red]实体判断[/red]:\n{same_entity}")
    if extract_json_text(same_entity, parse=True)["是否同一实体"] == "是":
        trace.append("[red]终止原因:\n[/red]实体相同")
        return

    step_4 = f"""
问题:{context['原始问题']}

将问题中的"{context['替换前实体']}"替换为"{context['替换后实体']}"，以 json 格式返回:
```json
{{
  "问题": str
}}
```
""".strip()
    msg = llm_call(step_4)
    trace.append(f"[red]替换后问题[/red]:\n{msg}")
    msg = extract_json_text(msg, parse=True)
    context["问题"] = msg["问题"]
    if context["替换后实体"] not in context["问题"]:
        trace.append("[red]终止原因:\n[/red]替换后的实体没有出现在问题中")
        return
    return context


def get_current_answer(example: Example, struct, trace):
    q = struct["问题"]
    prompt = f"""**文本开始**"

{example.document}

**文本结束**
问题: {q}
要求：如果文档/链接中找不到问题的答案，请礼貌拒绝、并尽可能给出一些解释或建议。
""".strip()
    message = llm_call(prompt)
    trace.append(f"[red]当前答案[/red]:\n{message}")
    return message


def generate_refuse(example, struct, trace):
    if Dataset.meta.lang == "en":
        constrain = "回复请使用英文"
    else:
        constrain = ""
    prompt = f"""
**文档开始**
{example.document}
**文档结束**

问题: {struct['问题']}

已知文档中没有出现**{struct['替换后实体']}**相关的内容,出现的是**{struct['替换前实体']}**。请仿照样例生成回复，不要回答问题。

下面是几个回复的样例:
例1: 根据原文，文中介绍了**美国**的CPI同比上涨的情况，不是**日本**。
例2: 原文中并没有提供**2025年3月15日**A股市场的主要指数表现的信息。提供的数据和分析都是针对**2024年6月12日**的市场情况。
例3: 在所提供的文本中，并没有提到**华为**接入GPT-4o，而是提到了**苹果**接入GPT-4o。
例4: 文档中没有提供关于**印度**香精香料行业的市场规模的具体信息。文档主要讨论了**中国**香精香料行业的情况

请仿照上面的例子回复，注意润色表达方式。 {constrain}

回复:
""".strip()
    message = llm_call(prompt)
    trace.append(f"[red]纠正[/red]:\n{message}")
    struct["纠正"] = message
    return


def generate_answer(example, struct, trace):
    origin_question = struct["问题"].replace(struct["替换后实体"], struct["替换前实体"])

    prompt = f"""**文本开始**"

{example.document}

**文本结束**
问题: {origin_question}
""".strip()
    message = llm_call(prompt)
    trace.append(f"[red]原始问题答案:[/red]\n {message}")
    struct["正确答案"] = message
    return


def combine_refuse_answer(example, struct, trace):
    prompt = f"""
把下面两段回答合并成一个回答,保持第二段的原始格式。

第一段:
{struct["纠正"]}

第二段:
{struct["正确答案"]}

合并结果:
""".strip()
    message = llm_call(prompt)
    trace.append(f"[red]最终答案:[/red]\n{message}")
    struct["final"] = message
    return True


def judge_answer_type(example, struct, answer, trace):
    prompt = f"""
**文档开始**
{example.document}
**文档结束**

原始问题:{struct["原始问题"]}
新问题: {struct["问题"]}

注意: 新问题中的"{struct['替换后实体']}" 不是文档和原始问题中提到的"{struct['替换前实体']}"

推理的依据仅限于文档中提到的内容，不要假设文档中不存在的内容

请根据文档判断, 是否可以仅通过文档中介绍的内容推理得到新问题的答案。以json格式回复:
```json
{{
    "判断原因思考": "思考问题能否通过原文内容推理出答案",
    "能否通过推理得到答案": "能" | "不能"
}}
```
"""
    message = llm_call(prompt)
    trace.append(f"[red]答案类型:{message}[/red]")
    message = extract_json_text(message, parse=True)
    try:
        ans = message["能否通过推理得到答案"]
        if ans == "能":
            trace.append("[red]答案可以通过推理得出[/red]")
            struct["type"] = "推理"
            struct["answer"] = message
            return False
    except Exception as e:
        trace.append(f"[red]答案推理判断json解析失败 {e}[/red]")
        return False

    # 回答问题1的问题说明模型搞错了
    prompt = f"""
**文档开始**
{example.document}
**文档结束**


问题1: {struct["问题"]}
问题2: {struct["原始问题"]}

回答:
{answer}

请判断: 
回答是针对问题1的回答还是针对问题2的回答，请说明原因
以下面的json格式给出: 
```json
{{
    "回答针对":"问题1" | "问题2",
    "判断原因":"判断的原因"
}}
```
"""
    message = llm_call(prompt)
    rich.print(f"[red]问题1和问题2的回答判断[/red]: {message}")
    message = extract_json_text(message, parse=True)
    try:
        if message["回答针对"] in ("问题2", "问题2的回答"):
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
    tasks = QueuedTasks(num_threads=args.num_workers, worker=consumer)
    tasks.submit_jobs(examples())
    tasks.wait()
    writer.finish()
