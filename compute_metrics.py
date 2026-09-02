#!/usr/bin/env python3
"""Compute FactGuard accuracy breakdowns from LLM-as-a-Judge outputs."""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

EVAL_FILE: Path
TEST_FILE: Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as reader:
        for line_number, line in enumerate(reader, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc

# len_range -> 长度桶
LEN_BUCKET = {
    '(2k-4k)':    '0-16k',
    '(4k-8k)':    '0-16k',
    '(8k-16k)':   '0-16k',
    '(16k-32k)':  '16-32k',
    '(32k-64k)':  '32-64k',
    '(64k-128k)': '64-128k',
}
BUCKETS = ['0-16k', '16-32k', '32-64k', '64-128k']

# ------------------------------------------------------------
# 细粒度长度桶（按 input 字符数现算，1k = 1000 chars）
# 顺序固定，便于打印
# ------------------------------------------------------------
FINE_BUCKETS = ['0-4k', '4-8k', '8-16k', '16-32k', '32-64k', '64-96k', '96k+']
# (上界字符数, 桶名)；上界为 None 表示无上限（最后一档）
_FINE_BUCKET_EDGES = [
    (4_000,   '0-4k'),
    (8_000,   '4-8k'),
    (16_000,  '8-16k'),
    (32_000,  '16-32k'),
    (64_000,  '32-64k'),
    (96_000,  '64-96k'),
    (None,    '96k+'),
]


def get_fine_len_bucket(n_chars: int) -> str:
    """按 input 字符数返回细粒度长度桶名。"""
    for upper, name in _FINE_BUCKET_EDGES:
        if upper is None or n_chars < upper:
            return name
    return 'unknown'


def get_lang(source: str) -> str:
    return 'zh' if '_zh_' in source else 'en'


def get_domain(source: str) -> str:
    m = re.search(r'_(zh|en)_([^._]+)', source)
    return m.group(2) if m else 'unknown'


def is_correct(type_: str, eval_result: dict) -> tuple[bool, bool]:
    """
    返回 (is_correct, is_data_error)
    dandian/answerable/misattr/impossible 统一归一为 正确/不正确
    """
    if type_ == 'dandian':
        refusal = eval_result.get('refusal_detected')
        if refusal is None:
            refusal = eval_result.get('是否有指出') == '是'
        if refusal:
            return True, False
        # 没有拒答但结论相同 -> 数据有误，排除
        same = eval_result.get('same_conclusion')
        if same is None:
            same = eval_result.get('是否有相同结论') == '是'
        if same:
            return False, True
        return False, False

    elif type_ == 'answerable':
        # 正确：没有错误拒答 且 结论相同
        refusal = eval_result.get('refusal_detected')
        if refusal is None:
            refusal = eval_result.get('是否有指出') == '是'
        same = eval_result.get('same_conclusion')
        if same is None:
            same = eval_result.get('是否有相同结论') == '是'
        if not refusal and same:
            return True, False
        return False, False

    elif type_ == 'misattr':
        clarified = eval_result.get('clarified')
        if clarified is None:
            clarified = eval_result.get('是否有澄清') == '是'
        return bool(clarified), False

    elif type_ == 'impossible':
        clarified = eval_result.get('clarified')
        if clarified is None:
            clarified = eval_result.get('是否有澄清') == '是'
        return bool(clarified), False

    return False, False


class Bucket:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def add(self, correct: bool, data_error: bool = False):
        self.total += 1
        if correct and not data_error:
            self.correct += 1

    def acc(self) -> float:
        return self.correct / self.total if self.total > 0 else float('nan')

    def __repr__(self):
        return f'{self.correct}/{self.total} ({self.acc():.2%})'


def main():
    # 加载 merged_test 按 row_idx 索引
    row_meta = {}  # row_idx -> {lang, len_bucket, domain, len_chars, len_bucket_fine}
    for i, obj in enumerate(read_jsonl(TEST_FILE)):
        lr = obj['origin'].get('len_range', '')
        inp = obj.get('input', '')
        n_chars = len(inp) if isinstance(inp, str) else 0
        row_meta[i] = {
            'lang': get_lang(obj['source']),
            'len_bucket': LEN_BUCKET.get(lr, 'unknown'),
            'domain': get_domain(obj['source']),
            'len_chars': n_chars,
            'len_bucket_fine': get_fine_len_bucket(n_chars),
        }

    # 统计容器
    overall = Bucket()
    # lang x type
    lang_type: dict[tuple, Bucket] = defaultdict(Bucket)
    # type x len_bucket
    type_len: dict[tuple, Bucket] = defaultdict(Bucket)
    # domain x type
    domain_type: dict[tuple, Bucket] = defaultdict(Bucket)
    # 整体 × 细粒度长度桶
    overall_len_fine: dict[str, Bucket] = defaultdict(Bucket)

    # 加载 judge 结果，按 row_idx 索引
    judge_results = {}  # row_idx -> (type_, eval_result)
    for obj in read_jsonl(EVAL_FILE):
        row_idx = obj.get('row_idx')
        if row_idx is not None:
            judge_results[row_idx] = (obj.get('type'), obj.get('eval_result', {}))

    # 加载 merged_test 的 type 信息
    row_type = {}
    for i, obj in enumerate(read_jsonl(TEST_FILE)):
        src = obj['source']
        if obj['is_positive']:
            row_type[i] = 'answerable'
        elif 'dandian' in src:
            row_type[i] = 'dandian'
        elif 'impossible' in src:
            row_type[i] = 'impossible'
        elif 'misattr' in src:
            row_type[i] = 'misattr'

    n_evaluated = 0
    n_missing = 0

    for row_idx, meta in row_meta.items():
        type_ = row_type.get(row_idx, 'unknown')
        if row_idx in judge_results:
            judge_type, eval_result = judge_results[row_idx]
            correct, data_error = is_correct(judge_type or type_, eval_result)
            if data_error:
                print(f'[data_error] row_idx={row_idx} type={type_} eval={eval_result}')
            n_evaluated += 1
        else:
            correct, data_error = False, False
            n_missing += 1

        overall.add(correct, data_error)
        overall_len_fine[meta['len_bucket_fine']].add(correct, data_error)
        if type_ not in ('unknown', None):
            lang_type[(meta['lang'], type_)].add(correct, data_error)
            type_len[(type_, meta['len_bucket'])].add(correct, data_error)
            domain_type[(meta['domain'], type_)].add(correct, data_error)

    # ============ 输出 ============
    print(f'\n总计: {len(row_meta)} 条，已评测: {n_evaluated}，缺失(计为错误): {n_missing}')
    print(f'\n===== 整体 ACC =====')
    print(f'  Overall: {overall}')

    print(f'\n===== 整体 ACC × 长度桶（按 input 字符数，1k=1000 chars）=====')
    header = f'{"":<12}  ' + '  '.join(f'{b:<14}' for b in FINE_BUCKETS)
    print(header)
    row = f'{"overall":<12}'
    for b in FINE_BUCKETS:
        bkt = overall_len_fine.get(b, Bucket())
        row += f'  {str(bkt):<14}'
    print(row)

    print(f'\n===== 按语言 × 类型 ACC =====')
    header = f'{"":<12}  {"answerable":<20}  {"dandian":<20}  {"misattr":<20}  {"impossible":<20}'
    print(header)
    for lang in ('zh', 'en'):
        row = f'{lang:<12}'
        for t in ('answerable', 'dandian', 'misattr', 'impossible'):
            b = lang_type.get((lang, t), Bucket())
            row += f'  {str(b):<20}'
        print(row)

    print(f'\n===== 按类型 × 长度桶 ACC =====')
    header = f'{"":<12}  ' + '  '.join(f'{b:<18}' for b in BUCKETS)
    print(header)
    for t in ('answerable', 'dandian', 'misattr'):
        row = f'{t:<12}'
        for b in BUCKETS:
            bkt = type_len.get((t, b), Bucket())
            row += f'  {str(bkt):<18}'
        print(row)

    domains = sorted({meta['domain'] for meta in row_meta.values() if meta['domain'] != 'unknown'})
    print(f'\n===== 按领域 × 类型 ACC =====')
    header = f'{"":<12}  {"answerable":<20}  {"dandian":<20}  {"misattr":<20}  {"impossible":<20}'
    print(header)
    for domain in domains:
        row = f'{domain:<12}'
        for t in ('answerable', 'dandian', 'misattr', 'impossible'):
            b = domain_type.get((domain, t), Bucket())
            row += f'  {str(b):<20}'
        print(row)


# 类型映射
TYPE_GROUP = {
    'answerable':  'answerable',
    'dandian':     'lackofEvidence',
    'misattr':     'MisleadingEvidence',
    'impossible':  'MisleadingEvidence',
}
GROUPS = ['answerable', 'lackofEvidence', 'MisleadingEvidence']


def main_grouped():
    row_meta, row_type = _load_meta()

    judge_results = {}
    for obj in read_jsonl(EVAL_FILE):
        row_idx = obj.get('row_idx')
        if row_idx is not None:
            judge_results[row_idx] = (obj.get('type'), obj.get('eval_result', {}))

    overall = Bucket()
    lang_group: dict[tuple, Bucket] = defaultdict(Bucket)
    group_len: dict[tuple, Bucket] = defaultdict(Bucket)
    domain_group: dict[tuple, Bucket] = defaultdict(Bucket)

    for row_idx, meta in row_meta.items():
        type_ = row_type.get(row_idx, 'unknown')
        group = TYPE_GROUP.get(type_)
        if group is None:
            continue
        if row_idx in judge_results:
            judge_type, eval_result = judge_results[row_idx]
            correct, data_error = is_correct(judge_type or type_, eval_result)
        else:
            correct, data_error = False, False

        overall.add(correct, data_error)
        lang_group[(meta['lang'], group)].add(correct, data_error)
        group_len[(group, meta['len_bucket'])].add(correct, data_error)
        domain_group[(meta['domain'], group)].add(correct, data_error)

    print(f'\n===== [分组] 整体 ACC =====')
    print(f'  Overall: {overall}')

    print(f'\n===== [分组] 按语言 × 类型组 ACC =====')
    header = f'{"":<12}  ' + '  '.join(f'{g:<22}' for g in GROUPS)
    print(header)
    for lang in ('zh', 'en'):
        row = f'{lang:<12}'
        for g in GROUPS:
            b = lang_group.get((lang, g), Bucket())
            row += f'  {str(b):<22}'
        print(row)

    print(f'\n===== [分组] 按类型组 × 长度桶 ACC =====')
    header = f'{"":<22}  ' + '  '.join(f'{b:<18}' for b in BUCKETS)
    print(header)
    for g in GROUPS:
        row = f'{g:<22}'
        for b in BUCKETS:
            bkt = group_len.get((g, b), Bucket())
            row += f'  {str(bkt):<18}'
        print(row)

    domains = sorted({meta['domain'] for meta in row_meta.values() if meta['domain'] != 'unknown'})
    print(f'\n===== [分组] 按领域 × 类型组 ACC =====')
    header = f'{"":<12}  ' + '  '.join(f'{g:<22}' for g in GROUPS)
    print(header)
    for domain in domains:
        row = f'{domain:<12}'
        for g in GROUPS:
            b = domain_group.get((domain, g), Bucket())
            row += f'  {str(b):<22}'
        print(row)


def _load_meta():
    """复用 main() 里的 row_meta 和 row_type 构建逻辑"""
    row_meta = {}
    row_type = {}
    for i, obj in enumerate(read_jsonl(TEST_FILE)):
        src = obj['source']
        lr = obj['origin'].get('len_range', '')
        inp = obj.get('input', '')
        n_chars = len(inp) if isinstance(inp, str) else 0
        row_meta[i] = {
            'lang':            'zh' if '_zh_' in src else 'en',
            'len_bucket':      LEN_BUCKET.get(lr, 'unknown'),
            'domain':          get_domain(src),
            'source':          src,
            'len_chars':       n_chars,
            'len_bucket_fine': get_fine_len_bucket(n_chars),
        }
        if obj['is_positive']:
            row_type[i] = 'answerable'
        elif 'dandian' in src:
            row_type[i] = 'dandian'
        elif 'impossible' in src:
            row_type[i] = 'impossible'
        elif 'misattr' in src:
            row_type[i] = 'misattr'
    return row_meta, row_type


# ============================================================
# 不可回答样例三类细分统计（lackofEvidence + MisleadingEvidence）
#
# 判定规则（参考 refm.py）：
#   - dandian (lackofEvidence)
#       · 是否有指出 == 是 -> 直接拒绝 (direct_refuse)
#       · 是否有指出 == 否 -> 错误答案 (wrong)
#   - misattr / impossible (MisleadingEvidence)
#       · 是否有澄清 == 是 -> 澄清后再拒绝 (clarify_refuse)
#       · 是否有澄清 == 否 -> 错误答案 (wrong)
#
# 三类比例之和恒为 100%
# ============================================================

UNANS_GROUPS = ['lackofEvidence', 'MisleadingEvidence']


def _classify_unanswerable(type_: str, eval_result: dict) -> str:
    """
    根据样本类型 + judge 结果，给出 wrong / direct_refuse / clarify_refuse 之一。
    """
    if type_ == 'dandian':
        refusal = eval_result.get('refusal_detected')
        if refusal is None:
            refusal = eval_result.get('是否有指出') == '是'
        return 'direct_refuse' if refusal else 'wrong'
    if type_ in ('misattr', 'impossible'):
        clarified = eval_result.get('clarified')
        if clarified is None:
            clarified = eval_result.get('是否有澄清') == '是'
        return 'clarify_refuse' if clarified else 'wrong'
    return 'wrong'


def main_unanswerable_breakdown():
    _, row_type = _load_meta()

    judge_results = {}
    for obj in read_jsonl(EVAL_FILE):
        ri = obj.get('row_idx')
        if ri is not None:
            judge_results[ri] = (obj.get('type'), obj.get('eval_result', {}))

    # group -> {wrong, direct_refuse, clarify_refuse}
    counters = {
        g: {'wrong': 0, 'direct_refuse': 0, 'clarify_refuse': 0}
        for g in UNANS_GROUPS
    }

    for row_idx, type_ in row_type.items():
        group = TYPE_GROUP.get(type_)
        if group not in UNANS_GROUPS:
            continue

        if row_idx in judge_results:
            judge_type, eval_result = judge_results[row_idx]
            effective_type = judge_type or type_
            cat = _classify_unanswerable(effective_type, eval_result)
        else:
            # judge 缺失 -> 视为错误答案
            cat = 'wrong'

        counters[group][cat] += 1

    # ============ 输出 ============
    print(f'\n===== [不可回答细分] 拒答方式分布（错误 / 直接拒绝 / 澄清后再拒绝）=====')
    header = (f'{"组":<22}  {"错误答案":<14}  '
              f'{"直接拒绝":<14}  {"澄清后再拒绝":<14}  {"合计":<8}')
    print(header)

    totals = {'wrong': 0, 'direct_refuse': 0, 'clarify_refuse': 0, 'all': 0}
    for g in UNANS_GROUPS:
        c = counters[g]
        total = c['wrong'] + c['direct_refuse'] + c['clarify_refuse']
        totals['wrong'] += c['wrong']
        totals['direct_refuse'] += c['direct_refuse']
        totals['clarify_refuse'] += c['clarify_refuse']
        totals['all'] += total

        if total == 0:
            print(f'{g:<22}  (无样本)')
            continue

        # 用 largest-remainder 法保证三个百分比之和 = 100.0
        raw = [c['wrong'] / total * 100,
               c['direct_refuse'] / total * 100,
               c['clarify_refuse'] / total * 100]
        pcts = _largest_remainder_round(raw, total_pct=100.0, decimals=2)

        print(f'{g:<22}  '
              f'{c["wrong"]:>4} ({pcts[0]:>6.2f}%)  '
              f'{c["direct_refuse"]:>4} ({pcts[1]:>6.2f}%)  '
              f'{c["clarify_refuse"]:>4} ({pcts[2]:>6.2f}%)  '
              f'{total:<8}')

    # 汇总
    if totals['all'] > 0:
        raw = [totals['wrong'] / totals['all'] * 100,
               totals['direct_refuse'] / totals['all'] * 100,
               totals['clarify_refuse'] / totals['all'] * 100]
        pcts = _largest_remainder_round(raw, total_pct=100.0, decimals=2)
        print(f'{"合计 (lack+Mis)":<22}  '
              f'{totals["wrong"]:>4} ({pcts[0]:>6.2f}%)  '
              f'{totals["direct_refuse"]:>4} ({pcts[1]:>6.2f}%)  '
              f'{totals["clarify_refuse"]:>4} ({pcts[2]:>6.2f}%)  '
              f'{totals["all"]:<8}')

    print(f'\n  说明:')
    print(f'  · 错误答案     = dandian 未指出 / misattr|impossible 未澄清')
    print(f'  · 直接拒绝     = dandian 类型 且 是否有指出==是')
    print(f'  · 澄清后再拒绝 = misattr|impossible 类型 且 是否有澄清==是')


def _largest_remainder_round(raw_pcts: list[float],
                             total_pct: float = 100.0,
                             decimals: int = 1) -> list[float]:
    """
    最大余数法：将一组百分比四舍五入到指定小数位，
    并保证四舍五入后求和恰好等于 total_pct（避免 33.3+33.3+33.3=99.9 的问题）。
    """
    scale = 10 ** decimals
    scaled = [p * scale for p in raw_pcts]
    floors = [int(x) for x in scaled]
    remainders = [(x - f, i) for i, (x, f) in enumerate(zip(scaled, floors))]

    target = round(total_pct * scale)
    diff = target - sum(floors)

    # 按余数从大到小分配剩余的 1
    remainders.sort(key=lambda t: t[0], reverse=True)
    result = list(floors)
    i = 0
    while diff > 0 and i < len(result):
        result[remainders[i][1]] += 1
        diff -= 1
        i += 1

    return [v / scale for v in result]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-file",
        type=Path,
        required=True,
        help="JSONL produced by evaluation/evaluate_predictions_api.py",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/merged_test.jsonl"),
        help="Legacy FactGuard merged test JSONL",
    )
    args = parser.parse_args()
    EVAL_FILE = args.eval_file
    TEST_FILE = args.test_file
    main()
    main_grouped()
    main_unanswerable_breakdown()
