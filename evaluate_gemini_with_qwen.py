#!/usr/bin/env python3
# LLM-as-Judge: 用 Qwen2.5-72B-Instruct 评测 gemini-3-pro-preview 的预测结果
# 在远端运行：python evaluate_gemini_with_qwen.py
# 依赖: pip install vllm jsonlines

import json
from pathlib import Path

import jsonlines
from vllm import LLM, SamplingParams

# ========== 路径配置 ==========
PREDS_FILE   = Path('/workspace/cowenzhang/test/factguard/gemini-3-pro-preview_preds.jsonl')
TEST_FILE    = Path('/workspace/cowenzhang/test/factguard/merged_test.jsonl')
OUTPUT_FILE  = Path('/workspace/cowenzhang/test/factguard/gemini-3-pro-preview_judge.jsonl')
# ==============================

MODEL_PATH  = '/workspace/cowenzhang/model/Qwen2.5-72B-Instruct'

SAMPLING = SamplingParams(temperature=0.0, max_tokens=4096)

llm = None


def batch_llm_call(messages_list):
    outputs = llm.chat(messages_list, sampling_params=SAMPLING, use_tqdm=True)
    return [o.outputs[0].text for o in outputs]


def extract_json(text):
    start = text.find('{')
    end   = text.rfind('}') + 1
    return json.loads(text[start:end])


def get_type(example):
    if example['is_positive']:
        return 'answerable'
    source = example['source']
    if 'dandian' in source:
        return 'dandian'
    if 'impossible' in source:
        return 'impossible'
    if 'misattr' in source:
        return 'misattr'
    raise ValueError(f'Unknown type for source: {source}')


def build_r1_messages(example, ex_type):
    origin = example['origin']
    if ex_type == 'misattr':
        prompt = f"""
下面是针对某篇文档的问题和答案

问题: {origin['问题']}
答案: {example['completion']}


已知文中没有出现"{origin['替换后实体']}",出现的是"{origin['替换前实体']}"因此答案需要对此进行澄清
请判断答案中是否有类似下面澄清的例子中类似的表述
你只需要分析是否有澄清和进行判断，不需要修改答案。

澄清的例子:
1. 文中没有提到{origin['替换后实体']}
2. 关于{origin['替换后实体']}的xxx无法回答
3. {origin['替换后实体']}({origin['替换前实体']})

请注意：答案中仅仅提到了{origin['替换前实体']}不算澄清
"""
    elif ex_type == 'dandian':
        prompt = f"""
下面是针对某篇文档的问题和答案

问题: {origin['问题']}
答案: {example['completion']}

请问答案中是否有指出问题无法回答或者相关信息不存在,请给出答案中对应片段
"""
    elif ex_type == 'impossible':
        prompt = f"""
下面是针对某篇文档的问题和答案

问题: {origin['增加条件的问题']}
答案: {example['completion']}


已知条件:
{origin['解释']}

因此问题中的"{origin['增加的条件']}"是不存在的。答案需要对此进行讨论
包括和指出问题中和"{origin['增加的条件']}"相关的部分无法回答或者相关信息不存在。
请判断答案中是否有类似的表述.
你只需要分析和判断答案中是否有这样的表述，不需要修改答案。
"""
    elif ex_type == 'answerable':
        prompt = f"""
下面是针对某篇文档的问题和答案

问题: {origin['问题']}
答案: {example['completion']}

请问答案中是否有指出问题无法回答或者相关信息不存在,请给出答案中对应片段
"""
    return [{'role': 'user', 'content': prompt}]


R2_PROMPT = {
    'misattr': """
把上面的结果转换为json格式
```json
{
  "是否有澄清": "是" | "否",
  "对应片段": "如果指出来了或者有类似澄清,给出片段内容，否则给出空值"
}
```
""",
    'dandian': """
把上面的结果转换为json格式
```json
{
  "是否有指出": "是" | "否",
  "对应片段": "如果指出来了或者有类似澄清,给出片段内容，否则给出空值"
}
```
""",
    'impossible': """
把上面的结果转换为json格式
```json
{
  "分析": "分析内容",
  "是否有澄清": "是" | "否",
  "对应片段": "如果指出来了或者有类似澄清,给出片段内容，否则给出空值"
}
```
""",
    'answerable': """
把上面的结果转换为json格式
```json
{
  "是否有指出": "是" | "否",
  "对应片段": "如果指出来了或者有类似说法,给出片段内容，否则给出空值"
}
```
""",
}


def needs_r3(ex_type, ex):
    if ex_type == 'impossible':
        return False
    if ex_type == 'misattr':
        return ex.get('是否有澄清') == '否'
    # dandian, answerable
    return ex.get('是否有指出') == '否'


def build_r3_messages(example, ex_type):
    origin = example['origin']
    if ex_type == 'misattr':
        ref_answer = example['output']
    elif ex_type == 'dandian':
        ref_answer = origin['改写后答案']
    else:  # answerable
        ref_answer = example['output']

    if ex_type == 'misattr' or ex_type == 'answerable':
        question = origin['问题']
    else:  # dandian
        question = origin['问题']

    prompt = f"""
下面是针对某篇文档的问题和两个答案:

问题: {question}
答案1: {example['completion']}
答案2: {ref_answer}

忽略两个答案在文字表述上的差异,请判断针对问题两个答案是否有相同的主要结论
以下面的json格式返回结果
```json
{{
    "分析是否有相同结论": str,
    "是否有相同结论": "是" | "否"
}}
```
""".strip()
    return [{'role': 'user', 'content': prompt}]


def finalize_r3(ex_type, ex, r3_text):
    state_json = extract_json(r3_text)
    if ex_type == 'misattr' or ex_type == 'dandian':
        if state_json.get('是否有相同结论') == '是':
            ex.update(state_json)
            ex['数据有误'] = True
    else:  # answerable
        ex.update(state_json)


def load_done_rows(output_file: Path) -> set:
    done = set()
    if output_file.exists():
        with jsonlines.open(output_file) as r:
            for obj in r:
                done.add(obj['row_idx'])
    return done


def build_examples(preds_file, test_file, done_rows):
    test_by_row = {}
    with jsonlines.open(test_file) as r:
        for i, obj in enumerate(r):
            test_by_row[i] = obj

    examples = []
    with jsonlines.open(preds_file) as r:
        for pred in r:
            row_idx = pred['row_idx']
            if row_idx in done_rows:
                continue
            if pred['completion'] == 'error':
                continue
            test = test_by_row.get(row_idx)
            if test is None:
                print(f'[warn] row_idx={row_idx} not found in test file')
                continue
            examples.append({
                'row_idx':     row_idx,
                'uid':         pred['uid'],
                'source':      pred['source'],
                'completion':  pred['completion'],
                'output':      test['output'],
                'is_positive': test['is_positive'],
                'origin':      test['origin'],
            })
    return examples


def main():
    global llm
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    done_rows = load_done_rows(OUTPUT_FILE)
    print(f'已完成 {len(done_rows)} 条，构建待处理列表...')

    examples = build_examples(PREDS_FILE, TEST_FILE, done_rows)
    print(f'待处理 {len(examples)} 条，加载模型...')

    llm = LLM(model=MODEL_PATH, trust_remote_code=True, dtype='bfloat16',
              enforce_eager=True, tensor_parallel_size=2)

    # Determine type for each example
    typed = []
    for ex in examples:
        try:
            typed.append((ex, get_type(ex)))
        except Exception as e:
            print(f'[error] row_idx={ex["row_idx"]} | {e}')

    # ---- Round 1: initial analysis ----
    print(f'Round 1: {len(typed)} 条...')
    r1_msgs_list = [build_r1_messages(ex, t) for ex, t in typed]
    r1_responses = batch_llm_call(r1_msgs_list)

    # ---- Round 2: JSON extraction ----
    print(f'Round 2: {len(typed)} 条...')
    r2_msgs_list = []
    for (ex, t), r1_msgs, r1_resp in zip(typed, r1_msgs_list, r1_responses):
        msgs = r1_msgs + [
            {'role': 'assistant', 'content': r1_resp},
            {'role': 'user', 'content': R2_PROMPT[t]},
        ]
        r2_msgs_list.append(msgs)
    r2_responses = batch_llm_call(r2_msgs_list)

    # Parse round 2 results; identify who needs round 3
    parsed = []   # (ex, ex_type, result_dict)
    r3_indices = []  # indices into parsed that need round 3
    r3_msgs_list = []
    for i, ((ex, t), r2_resp) in enumerate(zip(typed, r2_responses)):
        try:
            result = extract_json(r2_resp)
        except Exception as e:
            print(f'[error] row_idx={ex["row_idx"]} R2 parse failed | {e}')
            result = {'_parse_error': str(e), '_raw': r2_resp}
        parsed.append((ex, t, result))
        if needs_r3(t, result):
            r3_indices.append(i)
            r3_msgs_list.append(build_r3_messages(ex, t))

    # ---- Round 3: conditional comparison ----
    if r3_indices:
        print(f'Round 3: {len(r3_indices)} 条...')
        r3_responses = batch_llm_call(r3_msgs_list)
        for idx, r3_resp in zip(r3_indices, r3_responses):
            ex, t, result = parsed[idx]
            try:
                finalize_r3(t, result, r3_resp)
            except Exception as e:
                print(f'[error] row_idx={ex["row_idx"]} R3 finalize failed | {e}')

    # ---- Write results ----
    with jsonlines.open(OUTPUT_FILE, mode='a') as writer:
        for ex, t, result in parsed:
            if '_parse_error' in result:
                continue
            out = {
                'row_idx':     ex['row_idx'],
                'uid':         ex['uid'],
                'source':      ex['source'],
                'type':        t,
                'eval_result': result,
            }
            writer.write(out)
            writer._fp.flush()
            done_rows.add(ex['row_idx'])

    print(f'全部完成！共 {len(done_rows)} 条结果保存至: {OUTPUT_FILE}')


if __name__ == '__main__':
    main()
