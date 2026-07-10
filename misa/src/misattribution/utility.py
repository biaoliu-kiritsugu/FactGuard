import concurrent
import rich
import traceback
from threading import Thread
import jsonlines
from queue import Queue
import pathlib


data_dir = pathlib.Path(__file__).parent.parent.parent / "data"


def submit_tasks(executor, worker, example_generator, threads):
    tasks = []
    for example in example_generator:
        tasks.append(executor.submit(worker, example))
        if len(tasks) == threads:
            for future in concurrent.futures.as_completed(tasks):
                try:
                    future.result()
                except Exception:
                    rich.print(traceback.format_exc())
            tasks.clear()
    if len(tasks):
        for future in concurrent.futures.as_completed(tasks):
            try:
                future.result()
            except Exception:
                rich.print(traceback.format_exc())


class QueuedTasks:
    def __init__(self, worker, num_threads) -> None:
        self.num_threads = num_threads
        self.job_queue = Queue(maxsize=10)
        self.threads = []
        for _ in range(self.num_threads):
            self.threads.append(
                Thread(target=self.tasks_wrapper, args=(worker, self.job_queue))
            )
        self.start()

    def tasks_wrapper(self, worker, job_queue):
        while True:
            job = job_queue.get()
            if job is None:
                break
            try:
                worker(job)
            except Exception:
                print(traceback.format_exc())

    def start(self):
        for t in self.threads:
            t.start()

    def submit_jobs(self, job_generator):
        for job in job_generator:
            self.job_queue.put(job)
        for _ in range(self.num_threads):
            self.job_queue.put(None)

    def wait(self):
        for t in self.threads:
            t.join()


def extract_json_text(text: str):
    if "```json" in text:
        start = text.find("```json")
        end = text.find("```", start + 6)
        return text[start + len("```json") : end - 1]
    structed_text = text.split("```", 2)
    assert len(structed_text) == 3, repr(text)
    json_text = structed_text[1].strip().removeprefix("json")
    return json_text


class Writer:
    def __init__(self, output_filename, mode="w", flush=True) -> None:
        self.queue = Queue(1024)
        self.write_thread = Thread(
            target=self._do_write,
        )
        self.write_thread.start()
        self.writer = jsonlines.open(output_filename, mode=mode, flush=flush)

    def _do_write(self):
        while True:
            example = self.queue.get()
            if example is None:
                break
            self.writer.write(example)
        self.queue.task_done()

    def finish(self):
        self.queue.put(None)

    def write(self, example):
        self.queue.put(example)

    def __enter__(self):
        return self
    
    def __exit__(self, *args, **kwargs):
        self.finish()



def try_fix_json(json_str: str):
    if '"是否正确"' not in json_str:
        return
    index = json_str.find('"是否正确"')
    prev = json_str.rfind('",', 0, index)
    start = json_str.find('"分析":')
    if prev == -1 or start == -1:
        return

    analysis = (
        json_str[start + len('"分析":') : prev]
        .strip()
        .strip('"')
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )
    try:
        modified_str = (
            json_str[: start + len('"分析":')] + '"' + analysis + json_str[prev:]
        )
        j = json.loads(modified_str)
        j["fixed"] = True
        return j
    except json.decoder.JSONDecodeError:
        return


if __name__ == "__main__":
    import json

    s = """```json
{
  "分析": "该SQL查询语句的目的是计算所有书籍的5星评价人数的总和。查询中使用了`SUM`聚合函数来计算每本书的评价人数乘以其5星占比的总和，这是正确的计算方法。但是，SQL语句中的反引号（`）应该替换为双引号（"）或直接省略，因为PostgreSQL不使用反引号来引用列名。修正后的SQL语句应为：`SELECT SUM(评价人数 * \"5星占比\") FROM 书籍`。",
  "是否正确": "否"
}
```"""
    t = extract_json_text(s)
    print(t)
    try_fix_json(t)
    print(json.loads(t))
