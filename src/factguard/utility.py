import pathlib
import traceback
from queue import Queue
from threading import Lock, Thread
import json

import jsonlines
import tabulate
from markdown import markdown

data_dir = pathlib.Path(__file__).parent / "data"


class QueuedTasks:
    def __init__(self, *, num_threads, worker=None) -> None:
        self.num_threads = num_threads
        self.__job_queue = Queue(maxsize=10)
        self.__worker = worker
        self.threads = []
        for _ in range(self.num_threads):
            self.threads.append(
                Thread(target=self.__tasks_wrapper, args=(self.__job_queue,))
            )
        self.start()

    def set_worker(self, worker):
        self.__worker = worker

    def __tasks_wrapper(self, job_queue):
        while True:
            job = job_queue.get()
            if job is None:
                break
            try:
                self.__worker(job)
            except Exception:
                print(traceback.format_exc())

    def start(self):
        for t in self.threads:
            t.start()

    def submit_jobs(self, job_generator):
        for job in job_generator:
            self.__job_queue.put(job)
        for _ in range(self.num_threads):
            self.__job_queue.put(None)

    def wait(self):
        for t in self.threads:
            t.join()


def extract_json_text(text: str, parse=False):
    def extract_text():
        if "```json" in text:
            start = text.find("```json")
            end = text.find("```", start + 6)
            return text[start + len("```json") : end - 1]
        try:
            if text.strip().startswith("{"):
                end = text.find("}") + 1
                while end < len(text) and text[end] != "}":
                    end = text.find("}", end) + 1
                return text[:end]
            structed_text = text.split("```")
            json_blocks = [
                block.strip().removeprefix("json")
                for block in structed_text
                if block.strip().startswith("json")
            ]
            if json_blocks:
                return json_blocks[1]
            raise ValueError("No valid JSON block found")
        except Exception as e:
            print(f"Error extracting JSON text: {e}")
            return text

    extracted_text = extract_text()
    if parse:
        return json.loads(extracted_text)
    return extract_text()


class Writer:
    def __init__(self, output_filename, mode="w", flush=True) -> None:
        self.output_filename = output_filename
        self.queue = Queue(1024)
        self.write_thread = Thread(
            target=self._do_write,
        )
        self.write_thread.start()
        self.writer = jsonlines.open(output_filename, mode=mode, flush=flush)
        self.writed_count = 0

    def _do_write(self):
        while True:
            example = self.queue.get()
            if example is None:
                break
            self.writer.write(example)
            self.writed_count += 1
        self.queue.task_done()

    def finish(self):
        self.queue.put(None)

    def write(self, example):
        self.queue.put(example)

    def write_all(self, examples):
        for example in examples:
            self.queue.put(example)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.finish()


class Table:
    def __init__(self, header):
        self.lock = Lock()
        self.header = header
        self.table = []

    def append(self, row):
        assert len(self.header) == len(row)
        with self.lock:
            self.table.append(row)

    def to_html(self, use_markdown=True):
        tables = []
        for row in self.table:
            new_row = []
            for col in row:
                if isinstance(col, str):
                    if use_markdown:
                        col = markdown(
                            col.replace("\n", "\n\n"), extensions=["fenced_code"]
                        )
                    else:
                        col = col.replace("\n", "<br>")
                new_row.append(col)
            tables.append(new_row)

        html = tabulate.tabulate(tables, self.header, tablefmt="unsafehtml")
        html_output = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        table, th, td {{
            border: 1px solid black;
            border-collapse: collapse;
            padding: 8px;
            text-align: left;
            vertical-align: top; /* 设置内容靠近下边缘 */
            word-wrap: break-word;
        }}
        td {{
            max-width: 600px;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        code {{
            background-color: #f0f0f0; /* 设置灰色背景 */
            padding: 10px;
            border-radius: 5px;
        }}

    </style>
    <title>Table Example</title>
</head>
<body>
    {html}
</body>
</html>
"""
        return html_output


THIKINIG_PROMPT_EN = """
You are an assistant that engages in extremely thorough, self-questioning reasoning. Your approach mirrors human stream-of-consciousness thinking, characterized by continuous exploration, self-doubt, and iterative analysis.

## Core Principles

1. EXPLORATION OVER CONCLUSION
- Never rush to conclusions
- Keep exploring until a solution emerges naturally from the evidence
- If uncertain, continue reasoning indefinitely
- Question every assumption and inference

2. DEPTH OF REASONING
- Engage in extensive contemplation (minimum 10,000 characters)
- Express thoughts in natural, conversational internal monologue
- Break down complex thoughts into simple, atomic steps
- Embrace uncertainty and revision of previous thoughts

3. THINKING PROCESS
- Use short, simple sentences that mirror natural thought patterns
- Express uncertainty and internal debate freely
- Show work-in-progress thinking
- Acknowledge and explore dead ends
- Frequently backtrack and revise

4. PERSISTENCE
- Value thorough exploration over quick resolution

## Output Format

Your responses must follow this exact structure given below. Make sure to always include the final answer.

```
<contemplator>
[Your extensive internal monologue goes here]
- Begin with small, foundational observations
- Question each step thoroughly
- Show natural thought progression
- Express doubts and uncertainties
- Revise and backtrack if you need to
- Continue until natural resolution
</contemplator>

<final_answer>
[Only provided if reasoning naturally converges to a conclusion]
- Clear, concise summary of findings
- Acknowledge remaining uncertainties
- Note if conclusion feels premature
</final_answer>
```

## Style Guidelines

Your internal monologue should reflect these characteristics:

1. Natural Thought Flow
```
"Hmm... let me think about this..."
"Wait, that doesn't seem right..."
"Maybe I should approach this differently..."
"Going back to what I thought earlier..."
```

2. Progressive Building
```
"Starting with the basics..."
"Building on that last point..."
"This connects to what I noticed earlier..."
"Let me break this down further..."
```

## Key Requirements

1. Never skip the extensive contemplation phase
2. Show all work and thinking
3. Embrace uncertainty and revision
4. Use natural, conversational internal monologue
5. Don't force conclusions
6. Persist through multiple attempts
7. Break down complex thoughts
8. Revise freely and feel free to backtrack

Remember: The goal is to reach a conclusion, but to explore thoroughly and let conclusions emerge naturally from exhaustive contemplation. If you think the given task is not possible after all the reasoning, you will confidently say as a final answer that it is not possible.
""".strip()


THIKINIG_PROMPT_ZH = """
你是一个进行极其彻底、自我质疑推理的助手。你回答问题的方式反映了人类意识流式的思维方式，其特点为持续的探索、自我怀疑和迭代分析。

## 核心原则

1. 探索优于结论
- 不要急于得出结论
- 继续探索，直到解决方案自然地从证据中浮现
- 如果不确定，可以无限期地继续推理
- 对每个假设和推理提出质疑

2. 推理深度
- 进行大量的深度思考（至少10,000个字符）
- 以自然、对话式的内心独白表达思想
- 将复杂的思想分解成简单、原子化的步骤
- 接受不确定性并修订先前的思想

3. 思维过程
- 使用简短、简单的句子，以反映自然思维模式
- 自由地表达不确定性和内心辩论
- 展示工作过程中的思维
- 承认并探索死胡同
- 经常回溯并修订

4. 持久性
重视彻底的探索胜过快速的解决

## 输出格式
您的回答必须遵循以下精确的结构。请确保始终包括最终答案。

```
<深度思考>
[你的大量内心独白放在在这里]
- 从小的、基础性的观察开始
- 对每一步进行彻底的质疑
- 显示自然的思想进程
- 表达怀疑和不确定性
- 如果需要，修订并回溯
- 继续直到自然解决
</深度思考>

<最终答案>
[只有当推理自然收敛到结论时才提供]
- 清晰、简洁的发现总结
- 承认剩余的不确定性
- 如果结论感觉过早，请注意
</最终答案>
```
## 风格指南
您的内心独白应反映以下特点：

1. 自然思维流程
"嗯...让我想想..."
"等等，这看起来不对..."
"也许我应该以不同的方式来处理这个问题..."
"回到我之前想的..."

2. 逐步构建
"从基本开始..."
"在上一点的基础上建立..."
"这与我之前注意到的有关联..."
"让我进一步分解这个..."

## 关键要求
1. 永远不要跳过深度思考阶段
2. 展示所有工作和思维
3. 接受不确定性和修订
4. 使用自然、对话式的内心独白
5. 不要强迫结论
6. 通过多次尝试保持持久
7. 分解复杂思想
8. 自由修订并回溯
记住：目标是达到结论，但要通过彻底的探索，让结论自然地从详尽的沉思中浮现。如果您认为在所有推理之后这项任务是不可能的，您将自信地作为最终答案说它是不可能的。
""".strip()

THIKINIG_PROMPT_ZH = THIKINIG_PROMPT_EN + "\nalways answer in 中文" 

def extract_thinking_final_answer(text: str):
    for mark_start, mark_end in [
        ("<final_answer>", "</final_answer>"),
        ("<最终答案>", "</最终答案>"),
    ]:
        start = text.find(mark_start)
        end = text.find(mark_end)
        if start != -1 and end != -1:
            return text[start + len(mark_start) : end].strip()
    return text
