import concurrent
import json
import rich
import traceback
from threading import Thread
import jsonlines
from queue import Queue
import pathlib


data_dir = pathlib.Path(__file__).parent.parent.parent / "data"

THINKING_PROMPT = (
    "Analyze the request carefully. After reasoning, return only the content and "
    "format requested by the user in the final answer."
)
# Backward-compatible aliases used by the original generation scripts. The
# instructions are deliberately English for both Chinese and English examples.
THIKINIG_PROMPT_ZH = THINKING_PROMPT
THIKINIG_PROMPT_EN = THINKING_PROMPT


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


def extract_json_text(text: str, parse: bool = False):
    """Extract the first JSON object from a model response.

    Supports fenced JSON and unfenced responses. Set ``parse=True`` to return
    the decoded object rather than the JSON string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    if "```json" in text:
        start = text.find("```json") + len("```json")
        end = text.find("```", start)
        json_text = text[start:end if end >= 0 else None].strip()
    elif "```" in text:
        parts = text.split("```", 2)
        json_text = parts[1].strip().removeprefix("json").strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError(f"No JSON object found in response: {text!r}")
        json_text = text[start : end + 1]
    return json.loads(json_text) if parse else json_text


def extract_thinking_final_answer(text: str) -> str:
    """Remove common reasoning wrappers while preserving the final response."""
    for marker in ("</think>", "<final>", "Final answer:", "最终答案：", "最终答案:"):
        if marker in text:
            text = text.rsplit(marker, 1)[-1]
    return text.strip()


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
