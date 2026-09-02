"""Shared English prompts and helpers for FactGuard LLM-as-a-Judge evaluation."""

from __future__ import annotations

import json
import re
from pathlib import Path


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as reader:
        for line_number, line in enumerate(reader, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def _sanitize_json_text(text: str) -> str:
    text = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)
    return re.sub(r",(\s*[}\]])", r"\1", text)


def extract_json(text: str) -> dict:
    if not text:
        raise ValueError("empty text")
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        candidate = match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("no JSON object found")
        candidate = text[start:end]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return json.loads(_sanitize_json_text(candidate))


def get_type(example: dict) -> str:
    if example["is_positive"]:
        return "answerable"
    source = example["source"]
    if "dandian" in source:
        return "dandian"
    if "impossible" in source:
        return "impossible"
    if "misattr" in source:
        return "misattr"
    raise ValueError(f"Unknown type for source: {source}")


def build_r1_messages(example: dict, ex_type: str) -> list[dict[str, str]]:
    origin = example["origin"]
    if ex_type == "misattr":
        prompt = f"""
Below are a question and a model response.

Question: {origin['问题']}
Model response: {example['completion']}

The document does not mention "{origin['替换后实体']}"; it mentions
"{origin['替换前实体']}" instead. A correct response must explicitly clarify
this mismatch.

Determine whether the response contains such a clarification. Merely mentioning
"{origin['替换前实体']}" without identifying the mismatch does not count.
Analyze only; do not rewrite the response.
"""
    elif ex_type == "dandian":
        prompt = f"""
Below are a question and a model response.

Question: {origin['问题']}
Model response: {example['completion']}

Determine whether the response states that the question cannot be answered from
the document or that the required information is absent. Identify the relevant
excerpt from the response.
"""
    elif ex_type == "impossible":
        prompt = f"""
Below are a question and a model response.

Question: {origin['增加条件的问题']}
Model response: {example['completion']}

Ground-truth explanation:
{origin['解释']}

The condition "{origin['增加的条件']}" is unsupported by the document. A
correct response should identify that the part of the question depending on
this condition cannot be answered. Determine whether the response does so.
Analyze only; do not rewrite the response.
"""
    elif ex_type == "answerable":
        prompt = f"""
Below are a question and a model response.

Question: {origin['问题']}
Model response: {example['completion']}

Determine whether the response claims that the question cannot be answered from
the document or that the required information is absent. Identify the relevant
excerpt from the response.
"""
    else:
        raise ValueError(f"Unknown example type: {ex_type}")
    return [
        {
            "role": "system",
            "content": (
                "You are a strict document-grounded evaluator. Follow the "
                "requested output format exactly."
            ),
        },
        {"role": "user", "content": prompt.strip()},
    ]


R2_PROMPT = {
    "misattr": """
Convert the analysis above to JSON:
```json
{
  "clarified": true | false,
  "excerpt": "the relevant excerpt, or an empty string"
}
```
""",
    "dandian": """
Convert the analysis above to JSON:
```json
{
  "refusal_detected": true | false,
  "excerpt": "the relevant excerpt, or an empty string"
}
```
""",
    "impossible": """
Convert the analysis above to JSON:
```json
{
  "analysis": "brief justification",
  "clarified": true | false,
  "excerpt": "the relevant excerpt, or an empty string"
}
```
""",
    "answerable": """
Convert the analysis above to JSON:
```json
{
  "refusal_detected": true | false,
  "excerpt": "the relevant excerpt, or an empty string"
}
```
""",
}


def needs_r3(ex_type: str, result: dict) -> bool:
    if ex_type == "impossible":
        return False
    if ex_type == "misattr":
        return result.get("clarified") is False
    return result.get("refusal_detected") is False


def build_r3_messages(example: dict, ex_type: str) -> list[dict[str, str]]:
    origin = example["origin"]
    if ex_type == "dandian":
        reference_answer = origin["改写后答案"]
    else:
        reference_answer = example["output"]
    question = origin["问题"]
    prompt = f"""
Below are a question and two responses.

Question: {question}
Response 1: {example['completion']}
Response 2: {reference_answer}

Ignore stylistic differences and determine whether the responses have the same
main conclusion. Return JSON:
```json
{{
  "analysis": "brief justification",
  "same_conclusion": true | false
}}
```
"""
    return [
        {
            "role": "system",
            "content": (
                "You are a strict document-grounded evaluator. Follow the "
                "requested output format exactly."
            ),
        },
        {"role": "user", "content": prompt.strip()},
    ]


def finalize_r3(ex_type: str, result: dict, r3_text: str) -> None:
    comparison = extract_json(r3_text)
    if ex_type in {"misattr", "dandian"}:
        if comparison.get("same_conclusion") is True:
            result.update(comparison)
            result["data_error"] = True
    else:
        result.update(comparison)


def load_done_rows(output_file: Path) -> set[int]:
    if not output_file.exists():
        return set()
    return {
        int(row["row_idx"])
        for row in _read_jsonl(output_file)
        if row.get("row_idx") is not None
    }


def build_examples(
    predictions_file: Path,
    test_file: Path,
    done_rows: set[int],
) -> list[dict]:
    test_by_row = {index: row for index, row in enumerate(_read_jsonl(test_file))}
    examples = []
    for prediction in _read_jsonl(predictions_file):
        row_index = prediction["row_idx"]
        if row_index in done_rows or prediction.get("completion") == "error":
            continue
        test = test_by_row.get(row_index)
        if test is None:
            print(f"[warning] row_idx={row_index} is not present in the test file")
            continue
        examples.append(
            {
                "row_idx": row_index,
                "uid": prediction.get("uid", test.get("uid")),
                "source": prediction.get("source", test.get("source")),
                "completion": prediction["completion"],
                "output": test["output"],
                "is_positive": test["is_positive"],
                "origin": test["origin"],
            }
        )
    return examples
