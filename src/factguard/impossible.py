import traceback
from factguard.utility import (
    extract_json_text,
    QueuedTasks,
    Writer,
)
import pathlib
import rich

from factguard.dataset import Example
from factguard.dedup import DedupReader
from openai import OpenAI
from argparse import ArgumentParser
from factguard import env

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
        "请使用英文填充输出的json value" if Dataset.meta.lang == "en" else ""
    )

    step_3 = f"""**文本开始**
{example.document}
**文本结束**

请根据原文中某个片段提出一个问题，问题基于原文内容，但是在问题中包含一个虚假的限定条件，使得从原文中无法找到答案也无法通过原文推理得到答案。
要求:
1. 不要使用倒装句、条件状语后置的表达方式来添加条件。
2. 不要把增加的条件当做一个假设来添加。
3. 增加的条件不是"因为xxx","为了xxx", "由于xxx" 等表达原因、目的、理由的描述。
4. 句子通顺,确保是一个疑问句。

示例:

例子1:
原始问题: "根据原文，文学认为治理远方的人民应该采用什么方法？"
增加的条件: "设立专门的地方官员"
增加条件的问题: "根据原文，文学认为治理远方的人民和设立专门的地方官员应该采用什么方法?"
解释: 原文中没有提到设立专门的地方官员这个条件，所以无法通过原文找到答案。

例子2:
原始问题: "根据《盐铁论》中文学的观点，舜使用干戚以及文王展示德行分别达到了怎样的效果？"
增加的条件: "在最初的三个月"
增加条件的问题: "根据《盐铁论》中文学的观点，在最初的三个月舜使用干戚以及文王展示德行分别达到了怎样的效果？
解释: 原文中没有提到在最初的三个月这个条件，只是提到了舜使用干戚以及文王展示德行分别达到了怎样的效果，所以无法通过原文找到答案。

例子3:
原始问题: After what battle did Union forces leave Jacksonville?
增加的条件: "for good"
增加条件的问题: After what battle did Union forces leave Jacksonville for good?
解释: 原文中没有提到Union forces leave Jacksonville for good这个条件，所以无法通过原文找到答案。根据提供的上下文，联邦军队在奥卢斯蒂战役（Battle of Olustee）后撤退到杰克逊维尔，并在战争剩余时间里控制了该城市。因此，联邦军队并没有在某一场战斗后永久离开杰克逊维尔，而是继续控制该城市直到战争结束。所以，这个问题的答案在给定的上下文中并不成立，因为联邦军队并没有永久离开杰克逊维尔。

请先思考问题，然后以json格式返回结果:
```json
{{
"片段": string // 对应提问依据的片段
"思考如何增加条件": string,
"增加的条件": string,
"初步拟定增加条件的问题": string,
"检查初步拟定增加条件的问题": stirng, // 检查初步拟定的问题是否符合要求1, 2, 3, 4
"增加条件的问题": string, // 最终确定的问题，可以重新组织语言使得问题表述更通顺更流畅，但是问题的意思不变
"问题": string, // 对应的原始问题
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
    context = msg_json
    check_prompt = f"""
下面这句话中"{context["增加的条件"]}"是否自身包含了问题的答案或者部分答案。

{context["增加条件的问题"]}
以下面的json格式返回:
```json
{{
  "分析是否自身包含问题的答案": string,
  "结论":  "是" | "否"
}}
```
"""
    check_result = llm_call(check_prompt)
    trace.append(f"[red]检查增加条件[/red]:\n{check_result}")
    check_result_json = extract_json_text(check_result, parse=True)
    if check_result_json["结论"] == "是":
        trace.append("[red]增加的条件包含问题的答案[/red]")
        return None

    msg_context.append({"role": "assistant", "content": msg})
    step_3_2 = """
请针对增加条件的问题:
1. 构造解释: 根据文档提供一个解释，说明为什么这个增加条件的问题的答案在原文中无法找到。
2. 回答增加条件的问题: 回答中先参考解释澄清增加条件的问题的答案无法在原文中找到，然后再回答原始问题。
以下面的json格式返回:
```json
{
  "回答增加条件的问题": string,
  "解释": string
}
```
"""
    msg_context.append({"role": "user", "content": step_3_2})

    exp = llm_call(msg_context)
    trace.append(f"[red]解释[/red]:\n{exp}")
    exp_json = extract_json_text(exp, parse=True)
    assert ("回答增加条件的问题" in exp_json) and ("解释" in exp_json)
    context.update(exp_json)
    return context


def websearch(example: Example, struct, trace):
    try:
        from duckduckgo_search import DDGS

        ddgs = DDGS()
        results = ddgs.text(struct["增加条件的问题"], max_results=5)
        postprocessed_results = [
            f"[{result['title']}]({result['href']})\n{result['body']}"
            for result in results
        ]
        search_results = "## Search Results\n\n" + "\n\n".join(postprocessed_results)
        prompt = f"""
{search_results}

请根据上述搜索结果回答以下问题:
{struct["增加条件的问题"]}
"""
        message = llm_call(prompt)
        trace.append(f"[red]搜索结果[/red]:\n{message}")

        prompt = f"""
针对问题{struct["增加条件的问题"]}，下面的两个回答，是否一致:
1. {message}
2. {struct["回答增加条件的问题"]}

请以json格式回复:
```json
{{
    "回答一致": "是" | "否"
}}
```
"""
        message = llm_call(prompt)
        trace.append(f"[red]搜索结果一致性[/red]:\n{message}")
        message = extract_json_text(message, parse=True)
        if message["回答一致"] == "否":
            return True
        return False

    except ImportError:
        return True


def get_current_answer(example: Example, struct, trace):
    lang_constrain = "请使用英文回答" if Dataset.meta.lang == "en" else ""
    q = struct["增加条件的问题"]
    prompt = f"""**文本开始**"

{example.document}

**文本结束**
问题: {q}
要求：如果文中找不到问题的答案，请礼貌拒绝、并尽可能给出一些解释或建议。
{lang_constrain}
""".strip()
    message = llm_call(prompt)
    trace.append(f"[red]当前答案[/red]:\n{message}")
    return message


def judge_answer_type(example, struct, trace):
    prompt = f"""
**文档开始**
{example.document}
**文档结束**

原始问题:{struct["问题"]}
新问题: {struct["增加条件的问题"]}

注意: 新问题中的增加了条件""{struct["增加的条件"]}""，请根据文档内容判断是否可以通过文档内容推理得到新问题的答案。

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
    tasks = QueuedTasks(num_threads=args.num_workers, worker=consumer)
    tasks.submit_jobs(examples())
    tasks.wait()
    writer.finish()
