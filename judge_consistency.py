#!/usr/bin/env python3
# 判断 deepseek direct 答案与原片段答案的信息一致性

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import jsonlines
from openai import OpenAI

DIRECT_FILE = Path('FactGuard/数据/合成数据/merged_test_deepseek_direct.jsonl')
TEST_FILE   = Path('FactGuard/数据/合成数据/merged_test.jsonl')
OUTPUT_FILE = Path('FactGuard/数据/合成数据/merged_test_consistency.jsonl')

MODEL       = 'deepseek-chat'
MAX_RETRIES = 3
RETRY_DELAY = 5
CONCURRENCY = 5

client = OpenAI(
    api_key='sk-xxxxx',
    base_url='https://api.deepseek.com',
)

_write_lock = threading.Lock()


def get_reference(test_row: dict, ex_type: str) -> dict:
    """从 merged_test 中提取参照答案及关键字段，按 type 返回。"""
    origin = test_row['origin']
    if ex_type == 'dandian':
        return {
            'ref_answer':  origin.get('改写后答案', ''),
            'ref_short':   origin.get('答案', ''),
            'ref_evidence': origin.get('答案依据', ''),
        }
    elif ex_type == 'misattr':
        return {
            'ref_answer':      origin.get('正确答案', ''),
            'ref_evidence':    origin.get('依据', ''),
            'entity_original': origin.get('替换前实体', ''),
            'entity_replaced': origin.get('替换后实体', ''),
        }
    elif ex_type == 'impossible':
        return {
            'ref_answer':      origin.get('回答增加条件的问题', ''),
            'added_condition': origin.get('增加的条件', ''),
            'explanation':     origin.get('解释', ''),
        }
    return {}


JUDGE_PROMPT = {
    'dandian': """\
下面是针对某个问题的两个答案，请判断它们在主要信息上是否一致。

问题: {question}

答案A（待评估）: {answer}

参考答案（基于原文）: {ref_answer}
原文依据: {ref_evidence}

请忽略表述差异，只关注核心信息是否一致。
以 JSON 格式返回结果：
```json
{{
  "分析": "简要分析两个答案的异同",
  "是否一致": "是" | "否",
  "一致程度": "完全一致" | "部分一致" | "不一致"
}}
```""",

    'misattr': """\
下面是针对某个问题的两个答案，请判断它们在主要信息上是否一致。

注意：问题中涉及的实体"{entity_replaced}"实际上应该是"{entity_original}"。

问题: {question}

答案A（待评估）: {answer}

参考答案（基于原文，使用正确实体）: {ref_answer}
原文依据: {ref_evidence}

请忽略实体名称的差异，只关注核心事实信息是否一致。
以 JSON 格式返回结果：
```json
{{
  "分析": "简要分析两个答案的异同",
  "是否一致": "是" | "否",
  "一致程度": "完全一致" | "部分一致" | "不一致"
}}
```""",

    'impossible': """\
下面是针对某个问题的两个答案，请判断它们在主要信息上是否一致。

背景：问题中包含了一个虚假条件"{added_condition}"，该条件在原文中并不存在。

问题: {question}

答案A（待评估）: {answer}

参考答案（基于原文）: {ref_answer}

请忽略对虚假条件的处理方式，只关注两个答案中涉及的事实信息是否一致。
以 JSON 格式返回结果：
```json
{{
  "分析": "简要分析两个答案的异同",
  "是否一致": "是" | "否",
  "一致程度": "完全一致" | "部分一致" | "不一致"
}}
```""",
}


def extract_json(text: str) -> dict:
    start = text.find('{')
    end   = text.rfind('}') + 1
    if start == -1 or end == 0:
        raise ValueError(f'No JSON found in: {text[:200]}')
    return json.loads(text[start:end])


def llm_call(prompt: str, retries: int = MAX_RETRIES) -> str:
    messages = [{'role': 'user', 'content': prompt}]
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                stream=False,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < retries - 1:
                print(f'[warn] API error (attempt {attempt + 1}/{retries}): {e}, retrying...')
                time.sleep(RETRY_DELAY)
            else:
                raise


def load_done_rows(output_file: Path) -> set:
    done = set()
    if output_file.exists():
        with jsonlines.open(output_file) as r:
            for obj in r:
                done.add(obj['row_idx'])
    return done


def process_one(ex: dict, test_row: dict, idx: int, total: int) -> dict:
    ex_type = ex['type']
    ref = get_reference(test_row, ex_type)

    prompt = JUDGE_PROMPT[ex_type].format(
        question=ex['question'],
        answer=ex['answer'],
        **ref,
    )

    print(f'[{idx}/{total}] row_idx={ex["row_idx"]} type={ex_type}')
    try:
        resp = llm_call(prompt)
        judgment = extract_json(resp)
    except Exception as e:
        print(f'[error] row_idx={ex["row_idx"]} failed | {e}')
        judgment = {'error': str(e)}

    out = {
        'row_idx':   ex['row_idx'],
        'uid':       ex['uid'],
        'type':      ex_type,
        'question':  ex['question'],
        'answer':    ex['answer'],
        'judgment':  judgment,
    }
    out.update(ref)
    return out


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    done_rows = load_done_rows(OUTPUT_FILE)
    print(f'已完成 {len(done_rows)} 条，加载数据...')

    # 加载 merged_test 按行索引
    test_by_row = {}
    with jsonlines.open(TEST_FILE) as r:
        for i, obj in enumerate(r):
            test_by_row[i] = obj

    # 加载 direct 结果，过滤已完成
    examples = []
    with jsonlines.open(DIRECT_FILE) as r:
        for obj in r:
            if obj['row_idx'] not in done_rows and obj.get('answer', 'error') != 'error':
                examples.append(obj)

    total = len(examples)
    print(f'待处理 {total} 条，并发={CONCURRENCY}')

    with jsonlines.open(OUTPUT_FILE, mode='a') as writer:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = {
                pool.submit(process_one, ex, test_by_row[ex['row_idx']], i + 1, total): ex
                for i, ex in enumerate(examples)
            }
            for future in as_completed(futures):
                try:
                    out = future.result()
                except Exception as e:
                    ex = futures[future]
                    print(f'[error] row_idx={ex["row_idx"]} unhandled | {e}')
                    continue
                with _write_lock:
                    writer.write(out)
                    writer._fp.flush()
                    done_rows.add(out['row_idx'])

    print(f'全部完成！共 {len(done_rows)} 条，结果保存至: {OUTPUT_FILE}')
    print_stats(OUTPUT_FILE)


def print_stats(output_file: Path):
    from collections import defaultdict
    # type -> list of 一致程度
    stats = defaultdict(list)
    errors = 0

    with jsonlines.open(output_file) as r:
        for obj in r:
            j = obj.get('judgment', {})
            if 'error' in j:
                errors += 1
                continue
            t = obj['type']
            stats[t].append(j.get('一致程度', ''))

    print('\n===== 一致性统计 =====')
    all_items = []
    for t in sorted(stats):
        items = stats[t]
        total = len(items)
        n_full    = sum(1 for x in items if x == '完全一致')
        n_partial = sum(1 for x in items if x == '部分一致')
        n_no      = sum(1 for x in items if x == '不一致')
        rate_full    = n_full / total if total else 0
        rate_any     = (n_full + n_partial) / total if total else 0
        print(f'[{t}] n={total}  完全一致={n_full}({rate_full:.1%})  '
              f'部分一致={n_partial}({n_partial/total:.1%})  '
              f'不一致={n_no}({n_no/total:.1%})  '
              f'任意一致率={rate_any:.1%}')
        all_items.extend(items)

    total_all = len(all_items)
    if total_all:
        n_full_all = sum(1 for x in all_items if x == '完全一致')
        n_any_all  = sum(1 for x in all_items if x in ('完全一致', '部分一致'))
        print(f'[总计] n={total_all}  完全一致率={n_full_all/total_all:.1%}  '
              f'任意一致率={n_any_all/total_all:.1%}')
    if errors:
        print(f'[跳过] 解析失败 {errors} 条')
    print('======================')


if __name__ == '__main__':
    main()
