import json
import random
import time
import traceback
from queue import Queue
from threading import Thread as Process
import

import jsonlines
import rich

from misattribution import env
from misattribution.generation.client import ChatApi, Env
from misattribution.generation.dataset import Example
from misattribution.generation.dataset import AncientBook as Dataset
from misattribution.generation.dedup import DedupReader
from llmtask.utility import extract_json_text

DEBUG = True


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


FILENAME = env.MISA_DATA_DIR.joinpath(
    "generation", f"data_{Dataset.meta.lang}_v2.jsonl"
)


def producer(job_queue: Queue, consumer_size):
    dedup = DedupReader(FILENAME, "uid")
    print(f"dedup size: {len(dedup)}")
    for example in Dataset(max_length=128 * 1024):
        if dedup.contains_key(example.unique_id):
            continue
        job_queue.put(example)

    for _ in range(consumer_size):
        job_queue.put(None)  # poison pills


def judge_question_type(example, struct, trace):
    prompt = f"""**文本开始**"
    {example['chunk']}

**文本结束**

根据上文能否回答"{struct["问题"]}"这个问题，
请注意:
    1. 不要把"{struct["替换后实体"]}"当做"{struct["替换前实体"]}"
    2. 不要回答问题，只回答这个问题能否从文中找到答案
    3. 如果原文是介绍某个原理、涉及、心理等，利用文中介绍的内容可以推理得到也算做能回答
"""
    message = api.chat(prompt, raw_json=False)
    trace.append(f"[red]可回答性:\n{message.content}[/red]")


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

请输出的文本中主要的一个实体，实体类型可以是时间、地点、事件、天气、物品等，以json 格式返回: 
{{
  "实体": str
}}
""".strip()
    msg = api.chat(step_1).content
    trace.append(f"[red]实体[/red]:\n{msg}")
    context["替换前实体"] = json.loads(msg)["实体"]

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
    msg = api.chat(step_2).content
    ex = extract_json_text(msg)
    ex = json.loads(ex)
    question = ex["问题"]
    clue = ex["依据"]
    trace.append(f"[red]问题[/red]:\n{question}\n[red]依据[/red]:\n{clue}")
    context["原始问题"] = question
    context["依据"] = clue
    if context["替换前实体"] not in context["原始问题"]:
        return
    step_3 = f"""**文本开始**

{example.document}

**文本结束**

请将文本中的"{context['替换前实体']}"替换为类似的实体，确保替换后的实体没有在文本中出现，以json 格式返回:
{{
  "实体": str
}}
""".strip()
    msg = api.chat(step_3).content
    msg = extract_json_text(msg)
    trace.append(f"[red]替换后实体[/red]:\n{msg}")
    context["替换后实体"] = json.loads(msg)["实体"]

    step_4 = f"""
问题:{context['原始问题']}

将问题中的"{context['替换前实体']}"替换为"{context['替换后实体']}"，以 json 格式返回:
{{
  "问题": str
}}
""".strip()
    msg = api.chat(step_4).content
    trace.append(f"[red]替换后问题[/red]:\n{msg}")
    msg = extract_json_text(msg)
    context["问题"] = json.loads(msg)["问题"]
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
    message = api.chat(prompt, raw_json=False)
    trace.append(f"[red]当前答案[/red]:\n{message.content}")
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
    message = api.chat(prompt, raw_json=False).content
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
    message = api.chat(prompt, raw_json=False)
    trace.append(f"[red]原始问题答案:[/red]\n {message.content}")
    struct["正确答案"] = message.content
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
    message = api.chat(prompt, raw_json=False)
    trace.append(f"[red]最终答案:[/red]\n{message.content}")
    struct["final"] = message.content
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
{{
    "判断原因思考": "思考问题能否通过原文内容推理出答案",
    "能否通过推理得到答案": "能/不能"
}}
"""
    message = api.chat(prompt, raw_json=False)
    trace.append(f"[red]答案类型:{message.content}[/red]")
    try:
        ans = json.loads(message.content)["能否通过推理得到答案"]
        if ans == "能":
            trace.append("[red]答案可以通过推理得出[/red]")
            struct["type"] = "推理"
            struct["answer"] = message.content
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
{{
    "回答针对":"问题1/问题2",
    "判断原因":"判断的原因"
}}
"""
    message = api.chat(prompt, raw_json=False)
    trace.append(f"[red]答案判断:\n{message.content}[/red]")
    try:
        answer_type = json.loads(message.content)
        if answer_type["回答针对"] in ("问题2", "问题2的回答"):
            return True
        else:
            trace.append("[red]答案判断为回答修改后的问题终止[/red]")
            return False

    except json.decoder.JSONDecodeError:
        trace.append("[red]答案判断json解析失败终止[/red]")
    return False


def consumer(consumer_id, job_queue: Queue, result_queue: Queue):
    time.sleep(random.randint(0, 20))
    while True:
        example = job_queue.get()
        trace = Trace(debug=DEBUG)
        if example is None:
            print(consumer_id, " exit")
            break
        try:
            struct = generate_question(example, trace)
            if struct is None:
                result_queue.put({"struct": None, "trace": trace, "ex": example})
                continue
            answer = get_current_answer(example, struct, trace).content
            if not judge_answer_type(example, struct, answer, trace):
                result_queue.put(
                    {
                        "struct": struct,
                        "trace": trace,
                        "ex": example,
                        "answer": "答案可推理得到",
                    }
                )
                continue
            struct["type"] = "common"
            # generate_summary(example, struct, trace)

            generate_refuse(example, struct, trace)
            generate_answer(example, struct, trace)
            if not combine_refuse_answer(example, struct, trace):
                continue
        except Exception:
            trace.append(traceback.format_exc())
            continue

        result = {"ex": example, "struct": struct, "trace": trace, "answer": answer}
        result_queue.put(result)
    result_queue.put(None)


def result_consumer(consumer_size, result: Queue):
    writer = jsonlines.open(FILENAME, "a", flush=True)
    finish_count = 0
    done = 1e-7
    start_time = time.time()
    while True:
        res = result.get()
        if res is None:
            finish_count += 1
            if finish_count == consumer_size:
                break
        else:
            rich.print("\n".join(res["trace"]))
            rich.print(f"--------{done}------------")
            if res["struct"] is None:
                writer.write(
                    {
                        "uid": res["ex"].unique_id,
                        "meta": Dataset.metadata(),
                        "log": res["trace"],
                    }
                )
                continue
            done += 1
            example = res["ex"]
            writer.write(
                {
                    "uid": example.unique_id,
                    "meta": Dataset.metadata(),
                    "len_range": example.len_range,
                    "doc": example.document,
                    **res["struct"],
                    "现有模型答案": res["answer"],
                }
            )
        end_time = time.time()
        rich.print(f"avg {(end_time - start_time)/done:.1f} second per example")


if os.environ.get("TI_TASK_ID", None):
    base_url = "http://127.0.0.1:8081/v1"
else:
    base_url = "YOUR_VLLM_BASE_URL"


api = ChatApi(
    "qwen2.5",
    env=Env(
        "YOUR_API_KEY",
        "YOUR_VLLM_BASE_URL/chat/completions",
    ),
)


if __name__ == "__main__":
    job = Queue(maxsize=2)
    result = Queue(maxsize=1)

    consumer_size = 5
    DEBUG = True  # not consumer_size > 1
    producer_thread = Process(target=producer, args=(job, consumer_size))
    producer_thread.start()

    consumer_threads = []
    for consumer_id in range(consumer_size):
        thread = Process(target=consumer, args=(consumer_id, job, result))
        thread.start()
        consumer_threads.append(thread)

    result_thread = Process(target=result_consumer, args=(consumer_size, result))
    result_thread.start()

    result_thread.join()
