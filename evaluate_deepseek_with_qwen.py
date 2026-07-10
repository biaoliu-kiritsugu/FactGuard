#!/usr/bin/env python3
# LLM-as-Judge: 用 Qwen2.5-72B-Instruct 评测 deepseek-v4-pro 的预测结果
# 在远端运行：python evaluate_deepseek_with_qwen.py
# 远端环境: 8 x NVIDIA H200，使用 transformers + accelerate 自动多卡分片
# 依赖: pip install "transformers>=4.45" accelerate jsonlines torch

import json
import re
from pathlib import Path

import jsonlines
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ========== 路径配置 ==========
PREDS_FILE   = Path('/workspace/cowenzhang/test/factguard/deepseek-v4-pro_preds.jsonl')
TEST_FILE    = Path('/workspace/cowenzhang/test/factguard/merged_test.jsonl')
OUTPUT_FILE  = Path('/workspace/cowenzhang/test/factguard/deepseek-v4-pro_qwen-judge.jsonl')
# ==============================

MODEL_PATH = '/workspace/public/model/Qwen2.5-72B-Instruct'

# 生成参数（greedy，输出可复现）
MAX_NEW_TOKENS = 4096
DO_SAMPLE      = False

# 分块大小：每跑完一块就立刻落盘，崩溃后重启可从下一块续跑。
CHUNK_SIZE = 64

# 单次 forward 的微批大小（在 8 卡 H200 上 72B bf16 通常可以承受较大 batch；
# 显存吃紧时调小，吞吐够用时调大）。CHUNK_SIZE 应当 >= BATCH_SIZE。
BATCH_SIZE = 8

# 全局对象
llm_model = None
llm_tokenizer = None


def batch_llm_call(messages_list):
    """对一组 chat 消息做批量生成，返回每条的纯回复文本。"""
    if not messages_list:
        return []

    tokenizer = llm_tokenizer
    model = llm_model

    # 应用 chat template，得到每条样本的纯文本 prompt
    prompts = [
        tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        for msgs in messages_list
    ]

    results = [None] * len(prompts)

    # 微批切分，避免一次喂入过多导致 OOM
    for start in range(0, len(prompts), BATCH_SIZE):
        sub_prompts = prompts[start:start + BATCH_SIZE]

        model_inputs = tokenizer(
            sub_prompts,
            return_tensors='pt',
            padding=True,
            truncation=False,
        ).to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 切掉 prompt 部分（左侧 padding，所以输入长度即 prompt 末尾位置）
        input_len = model_inputs['input_ids'].shape[1]
        gen_only = generated_ids[:, input_len:]

        decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        for i, text in enumerate(decoded):
            results[start + i] = text

        # 释放本批显存碎片
        del model_inputs, generated_ids, gen_only
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def _sanitize_json_text(s: str) -> str:
    """对 LLM 生成的 JSON 文本做尽力修复，主要处理：
    1) 字符串值里出现的非法反斜杠转义（例如 \\d、\\s、\\1、Windows 路径里的孤立 \\）
       —— 把所有不属于 JSON 合法转义（" \\ / b f n r t uXXXX）的孤立反斜杠加倍。
    2) 末尾对象/数组的多余逗号。
    3) 全角引号 “ ” 替换为 ASCII 引号（仅当文本里没有 ASCII " 且明显是字符串场景时不做强制处理，
       这里只做保守的非法反斜杠修复）。
    """
    # 1) 修复非法反斜杠转义：把不在合法集合里的 \\X 替换为 \\\\X
    #    合法集合: " \\ / b f n r t u
    s = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)
    # 2) 去掉对象/数组结尾的多余逗号
    s = re.sub(r',(\s*[}\]])', r'\1', s)
    return s


def extract_json(text):
    """从 LLM 输出中提取 JSON。先去掉 ```json ... ``` 围栏，再尝试解析；
    解析失败时做一次反斜杠/末尾逗号修复后重试。"""
    if text is None:
        raise ValueError('empty text')

    # 优先取 ```json ... ``` 代码块里的内容（如果有）
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        candidate = m.group(1)
    else:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start == -1 or end <= start:
            raise ValueError('no JSON object found')
        candidate = text[start:end]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        fixed = _sanitize_json_text(candidate)
        return json.loads(fixed)


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
    return [
        {'role': 'system', 'content': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'},
        {'role': 'user', 'content': prompt},
    ]


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
    return [
        {'role': 'system', 'content': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'},
        {'role': 'user', 'content': prompt},
    ]


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


def process_chunk(chunk, writer, done_rows):
    """对一个 chunk 跑完整的 R1 → R2 → (条件) R3，并立刻落盘。
    chunk: list[(ex, ex_type)]
    """
    if not chunk:
        return 0

    # ---- Round 1 ----
    r1_msgs_list = [build_r1_messages(ex, t) for ex, t in chunk]
    r1_responses = batch_llm_call(r1_msgs_list)

    # ---- Round 2 ----
    r2_msgs_list = []
    for (ex, t), r1_msgs, r1_resp in zip(chunk, r1_msgs_list, r1_responses):
        msgs = r1_msgs + [
            {'role': 'assistant', 'content': r1_resp},
            {'role': 'user', 'content': R2_PROMPT[t]},
        ]
        r2_msgs_list.append(msgs)
    r2_responses = batch_llm_call(r2_msgs_list)

    parsed = []          # [(ex, ex_type, result_dict)]
    r3_indices = []      # indices into parsed that need round 3
    r3_msgs_list = []
    for i, ((ex, t), r2_resp) in enumerate(zip(chunk, r2_responses)):
        try:
            result = extract_json(r2_resp)
        except Exception as e:
            print(f'[error] row_idx={ex["row_idx"]} R2 parse failed | {e}')
            result = {'_parse_error': str(e), '_raw': r2_resp}
        parsed.append((ex, t, result))
        if '_parse_error' not in result and needs_r3(t, result):
            r3_indices.append(i)
            r3_msgs_list.append(build_r3_messages(ex, t))

    # ---- Round 3 (conditional) ----
    if r3_indices:
        r3_responses = batch_llm_call(r3_msgs_list)
        for idx, r3_resp in zip(r3_indices, r3_responses):
            ex, t, result = parsed[idx]
            try:
                finalize_r3(t, result, r3_resp)
            except Exception as e:
                print(f'[error] row_idx={ex["row_idx"]} R3 finalize failed | {e}')

    # ---- Persist this chunk immediately ----
    written = 0
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
        done_rows.add(ex['row_idx'])
        written += 1
    writer._fp.flush()
    return written


def load_model():
    """在 8 卡 H200 上自动分片加载 Qwen2.5-72B-Instruct。"""
    print(f'加载 tokenizer: {MODEL_PATH}')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 批量生成必须使用左侧 padding，这样 prompt 末尾在每条样本里都对齐
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f'加载模型 (torch_dtype="auto", device_map="auto")...')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype='auto',
        device_map='auto',
        trust_remote_code=True,
    )
    model.eval()

    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print(f'CUDA 可见 GPU 数量: {n_gpu}')
    return model, tokenizer


def main():
    global llm_model, llm_tokenizer
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    done_rows = load_done_rows(OUTPUT_FILE)
    print(f'已完成 {len(done_rows)} 条，构建待处理列表...')

    examples = build_examples(PREDS_FILE, TEST_FILE, done_rows)
    print(f'待处理 {len(examples)} 条，加载模型...')

    if not examples:
        print('没有需要处理的样本，结束。')
        return

    llm_model, llm_tokenizer = load_model()

    # Determine type for each example
    typed = []
    for ex in examples:
        try:
            typed.append((ex, get_type(ex)))
        except Exception as e:
            print(f'[error] row_idx={ex["row_idx"]} | {e}')

    total = len(typed)
    print(f'有效样本 {total} 条，按 chunk={CHUNK_SIZE} (micro batch={BATCH_SIZE}) 分块处理 ...')

    total_written = 0
    with jsonlines.open(OUTPUT_FILE, mode='a') as writer:
        for start in range(0, total, CHUNK_SIZE):
            chunk = typed[start:start + CHUNK_SIZE]
            print(f'>>> chunk [{start}, {start + len(chunk)}) / {total}')
            try:
                w = process_chunk(chunk, writer, done_rows)
                total_written += w
                print(f'<<< chunk done, +{w} 条已落盘 (累计 {total_written})')
            except Exception as e:
                # 该 chunk 失败：保留已落盘部分，下次重启会跳过 done 的样本
                print(f'[chunk error] start={start} | {e}')
                raise

    print(f'全部完成！本轮新增 {total_written} 条；输出文件: {OUTPUT_FILE}')


if __name__ == '__main__':
    main()
