#!/usr/bin/env python3
"""
impossible 类型假条件词汇出现率与错误分析脚本

分析 FactGuard 评测结果中 impossible 类型的错误模式：
1. 假条件词汇在模型回答中的出现率
2. 错误 case 的原因归类
"""

import json
import jsonlines
import re
from collections import defaultdict
from pathlib import Path
import argparse
import statistics

# ------------------------------------------------------------
# 文件路径配置
# ------------------------------------------------------------
DEFAULT_JUDGE_FILE = Path('evaluation/judge_results/model_qwen_judge.jsonl')
DEFAULT_TEST_FILE = Path('data/merged_test.jsonl')
DEFAULT_PRED_FILE = Path('evaluation/predictions/model_preds.jsonl')


# ------------------------------------------------------------
# 数据加载
# ------------------------------------------------------------

def load_impossible_data(judge_file: Path, test_file: Path, pred_file: Path) -> list:
    """
    加载 impossible 类型数据，返回列表:
    [{
        'row_idx': int,
        'lang': str,
        'false_condition': str,          # 增加的条件（假条件）
        'original_question': str,        # 原始问题
        'modified_question': str,        # 增加条件后的问题
        'completion': str,               # 模型回答
        'has_clarification': bool,       # 是否有澄清
        'eval_analysis': str,            # 评判分析
    }, ...]
    """
    # 加载 judge 结果
    judge_map = {}  # row_idx -> eval_result
    with jsonlines.open(judge_file) as r:
        for obj in r:
            if obj.get('type') == 'impossible':
                ri = int(obj.get('row_idx'))
                judge_map[ri] = obj.get('eval_result', {})

    # 加载测试数据 (仅 impossible 条目)
    test_map = {}    # row_idx -> origin
    source_map = {}  # row_idx -> source
    with jsonlines.open(test_file) as r:
        for i, obj in enumerate(r):
            if 'impossible' in obj.get('source', ''):
                test_map[i] = obj.get('origin', {})
                source_map[i] = obj.get('source', '')

    # 加载预测结果
    pred_map = {}  # row_idx -> completion
    with jsonlines.open(pred_file) as r:
        for obj in r:
            ri = obj.get('row_idx')
            if ri is not None and 'impossible' in obj.get('source', ''):
                pred_map[int(ri)] = obj.get('completion', '')

    # 合并数据
    records = []
    for row_idx in sorted(judge_map.keys()):
        if row_idx not in test_map:
            continue
        origin = test_map[row_idx]
        eval_r = judge_map[row_idx]

        records.append({
            'row_idx': row_idx,
            'lang': 'zh' if '_zh_' in source_map.get(row_idx, '') else 'en',
            'false_condition': origin.get('增加的条件', ''),
            'original_question': origin.get('问题', ''),
            'modified_question': origin.get('增加条件的问题', ''),
            'completion': pred_map.get(row_idx, ''),
            'has_clarification': (
                eval_r.get('clarified')
                if isinstance(eval_r.get('clarified'), bool)
                else eval_r.get('是否有澄清') == '是'
            ),
            'eval_analysis': eval_r.get('分析', ''),
        })

    return records


# ------------------------------------------------------------
# 假条件词汇匹配
# ------------------------------------------------------------

def check_condition_mentioned(completion: str, condition: str, lang: str) -> dict:
    """
    检查假条件词汇是否出现在模型回答中
    返回:
        'exact': bool    - 精确短语匹配
        'partial': bool  - 部分词汇匹配（关键词匹配）
    """
    cond_lower = condition.lower().strip()
    comp_lower = completion.lower()

    # 精确匹配：完整短语作为子串出现
    exact = cond_lower in comp_lower

    # 部分匹配：提取关键词（去除停用词后的主要名词/动词）
    # 简单策略：按空格分词，去除短词(<=2字符)
    tokens = [t.strip() for t in cond_lower.split() if len(t.strip()) > 2]
    partial = False
    if tokens:
        matched = sum(1 for t in tokens if t in comp_lower)
        partial = matched >= max(1, len(tokens) // 2)

    return {'exact': exact, 'partial': partial}


def classify_condition_keywords(condition: str, lang: str) -> list:
    """
    从假条件中提取关键词列表
    """
    if lang == 'zh':
        # 简单按字符分，去除常用停用词
        stop_chars = set('的了是在和也与人这对上')
        tokens = [c for c in condition if c not in stop_chars and not c.isspace()]
        return tokens[:5]  # 最多5个关键词
    else:
        # 英文：按空格分，去除停用词
        stop_words = {
            'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'with',
            'by', 'is', 'was', 'were', 'are', 'be', 'been', 'being',
            'and', 'or', 'but', 'if', 'than', 'as', 'it', 'its', 'that',
            'this', 'from', 'about', 'after', 'before', 'during', 'between'
        }
        tokens = [t.lower().strip('.,!?;:"\'()[]{}')
                  for t in condition.split()
                  if len(t) > 2 and t.lower().strip('.,!?;:"\'()[]{}') not in stop_words]
        return tokens[:5]


# ------------------------------------------------------------
# 错误模式分类
# ------------------------------------------------------------

def classify_error(record: dict) -> str:
    """
    对 impossible 条目进行错误模式分类

    正确 (correct):
      - has_clarification=True

    错误 (wrong) 分类：
      - wrong_hallucinate: 假条件出现在回答中，且未澄清
        → 模型将虚假条件当作真实信息来回答（幻觉）
      - wrong_ignore: 假条件未出现在回答中，且未澄清
        → 模型完全忽略了假条件，正常回答了原问题
      - wrong_ambiguous: 假条件部分出现在回答中，且未澄清
        → 模型部分提及了假条件但未明确澄清
    """
    if record['has_clarification']:
        return 'correct'

    completion = record['completion']
    condition = record['false_condition']
    lang = record['lang']

    mention = check_condition_mentioned(completion, condition, lang)

    if mention['exact']:
        return 'wrong_hallucinate'   # 把假条件当真的幻觉
    elif mention['partial']:
        return 'wrong_ambiguous'     # 部分提及但不澄清
    else:
        return 'wrong_ignore'       # 完全忽略假条件


# ------------------------------------------------------------
# 词汇出现率统计
# ------------------------------------------------------------

def compute_vocab_stats(records: list) -> dict:
    """
    计算假条件词汇在回答中的出现率统计
    """
    total = len(records)
    if total == 0:
        return {}

    exact_mentions = 0
    partial_mentions = 0
    no_mentions = 0

    for rec in records:
        mention = check_condition_mentioned(
            rec['completion'], rec['false_condition'], rec['lang']
        )
        if mention['exact']:
            exact_mentions += 1
        elif mention['partial']:
            partial_mentions += 1
        else:
            no_mentions += 1

    return {
        'total': total,
        'exact_mentions': exact_mentions,
        'partial_mentions': partial_mentions,
        'no_mentions': no_mentions,
        'exact_rate': exact_mentions / total,
        'partial_rate': partial_mentions / total,
        'no_rate': no_mentions / total,
    }


# ------------------------------------------------------------
# 错误模式统计
# ------------------------------------------------------------

def compute_error_stats(records: list) -> dict:
    """
    按错误模式分类统计
    """
    stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'wrong': 0})

    for rec in records:
        error_type = classify_error(rec)
        stats[error_type]['total'] += 1
        if rec['has_clarification']:
            stats[error_type]['correct'] += 1
        else:
            stats[error_type]['wrong'] += 1

    return dict(stats)


def compute_error_stats_by_lang(records: list) -> dict:
    """按语言分组统计错误模式"""
    by_lang = defaultdict(lambda: defaultdict(int))
    for rec in records:
        error_type = classify_error(rec)
        by_lang[rec['lang']][error_type] += 1
    return {lang: dict(cnts) for lang, cnts in by_lang.items()}


# ------------------------------------------------------------
# 假条件词汇长度与错误率关系
# ------------------------------------------------------------

def compute_cond_length_stats(records: list) -> list:
    """
    分析假条件长度与正确率的关系
    返回 [(cond_len, has_clarification), ...]
    """
    return [
        (len(rec['false_condition']), rec['has_clarification'])
        for rec in records if rec['false_condition']
    ]


# ------------------------------------------------------------
# 输出格式化
# ------------------------------------------------------------

def fmt_vocab_stats(stats: dict) -> str:
    """格式化词汇出现率统计"""
    lines = ['\n=== 假条件词汇出现率统计 ===']
    total = stats['total']
    lines.append(f"  总数: {total}")
    lines.append(f"  精确匹配 (Exact):  {stats['exact_mentions']:>4} 条 ({stats['exact_rate']:.1%})")
    lines.append(f"  部分匹配 (Partial): {stats['partial_mentions']:>4} 条 ({stats['partial_rate']:.1%})")
    lines.append(f"  未提及 (None):     {stats['no_mentions']:>4} 条 ({stats['no_rate']:.1%})")

    # 柱状图
    lines.append('\n  精确匹配率柱状图:')
    bar = '█' * int(stats['exact_rate'] * 50)
    lines.append(f"  [{stats['exact_rate']:.1%}] {bar}")
    return '\n'.join(lines)


def fmt_error_stats(stats: dict) -> str:
    """格式化错误模式统计"""
    labels = {
        'correct':            '正确 (已澄清)',
        'wrong_hallucinate':  '错误-幻觉 (包含假条件)',
        'wrong_ambiguous':    '错误-模糊 (部分包含)',
        'wrong_ignore':       '错误-忽略 (未提及假条件)',
    }
    total = sum(v['total'] for v in stats.values())
    lines = ['\n=== 错误模式分类统计 ===']
    for key in ['correct', 'wrong_hallucinate', 'wrong_ambiguous', 'wrong_ignore']:
        info = stats.get(key, {'total': 0})
        n = info['total']
        pct = n / total * 100 if total > 0 else 0
        bar = '█' * int(pct / 2)
        label = labels.get(key, key)
        lines.append(f"  {label:<30}  {n:>4} 条 ({pct:5.1f}%)  {bar}")

    # 关键比例
    correct_n = stats.get('correct', {}).get('total', 0)
    hallucinate_n = stats.get('wrong_hallucinate', {}).get('total', 0)
    ignore_n = stats.get('wrong_ignore', {}).get('total', 0)
    ambiguous_n = stats.get('wrong_ambiguous', {}).get('total', 0)
    wrong_total = hallucinate_n + ignore_n + ambiguous_n
    lines.append(f"\n  错误总数: {wrong_total}")
    if wrong_total > 0:
        lines.append(f"  幻觉型错误占比: {hallucinate_n/wrong_total:.1%}")
        lines.append(f"  忽略型错误占比: {ignore_n/wrong_total:.1%}")
        lines.append(f"  模糊型错误占比: {ambiguous_n/wrong_total:.1%}")
    return '\n'.join(lines)


def fmt_error_by_lang(stats: dict) -> str:
    """按语言格式化错误模式"""
    labels = {
        'correct': 'correct',
        'wrong_hallucinate': 'hallucinate',
        'wrong_ambiguous': 'ambiguous',
        'wrong_ignore': 'ignore',
    }
    lines = ['\n=== 按语言错误模式分布 ===']
    header = '  {:<12}  {:>8}  {:>10}  {:>10}  {:>8}  {:>10}'.format(
        '语言', '正确', '幻觉', '忽略', '模糊', '正确率')
    lines.append(header)
    lines.append('  ' + '-' * len(header))

    for lang in ('en', 'zh'):
        cnts = stats.get(lang, {})
        total = sum(cnts.values())
        correct = cnts.get('correct', 0)
        hall = cnts.get('wrong_hallucinate', 0)
        ign = cnts.get('wrong_ignore', 0)
        amb = cnts.get('wrong_ambiguous', 0)
        acc = correct / total if total > 0 else 0
        lines.append('  {:<12}  {:>8}  {:>10}  {:>10}  {:>8}  {:>10.1%}'.format(
            lang.upper(), correct, hall, ign, amb, acc))
    return '\n'.join(lines)


def fmt_vocab_rate_by_error(stats: dict) -> str:
    """按错误类型展示词汇出现率"""
    lines = ['\n=== 错误类型的词汇出现率 ===']
    header = '  {:<30}  {:>10}  {:>10}  {:>10}'.format(
        '类型', '精确匹配', '部分匹配', '未提及')
    lines.append(header)
    lines.append('  ' + '-' * len(header))

    labels = {
        'correct':            '正确 (已澄清)',
        'wrong_hallucinate':  '错误-幻觉',
        'wrong_ambiguous':    '错误-模糊',
        'wrong_ignore':       '错误-忽略',
    }
    for key, label in labels.items():
        info = stats.get(key, {'total': 0})
        n = info['total']
        lines.append('  {:<30}  (样本不足)'.format(label))
    return '\n'.join(lines)


def print_samples_by_error_type(records: list, n: int = 5):
    """打印各错误类型的典型样本"""
    by_type = defaultdict(list)
    for rec in records:
        et = classify_error(rec)
        by_type[et].append(rec)

    labels = {
        'correct':            '正确 (已澄清)',
        'wrong_hallucinate':  '错误-幻觉 (包含假条件)',
        'wrong_ambiguous':    '错误-模糊 (部分包含)',
        'wrong_ignore':       '错误-忽略 (未提及假条件)',
    }

    print('\n=== 各错误类型典型样本 ===')
    for et in ['correct', 'wrong_hallucinate', 'wrong_ambiguous', 'wrong_ignore']:
        label = labels.get(et, et)
        samples = by_type.get(et, [])[:n]
        if not samples:
            continue
        print(f'\n--- {label} ---')
        for rec in samples:
            cond = rec['false_condition']
            comp = rec['completion']
            mention = check_condition_mentioned(comp, cond, rec['lang'])
            # 截取completion前150字
            comp_short = comp[:150].replace('\n', ' ')
            print(f'  假条件: "{cond}"')
            print(f'  精确匹配: {mention["exact"]}, 部分: {mention["partial"]}')
            print(f'  回答: {comp_short}...')
            print()


# ------------------------------------------------------------
# 假条件长度与正确率关系
# ------------------------------------------------------------

def compute_length_acc_by_bin(records: list) -> dict:
    """按假条件长度区间统计正确率"""
    bins = [(0, 10, '极短 0-10'), (10, 20, '短 10-20'),
            (20, 35, '中 20-35'), (35, 60, '长 35-60'), (60, 200, '极长 60+')]

    result = {}
    for lo, hi, label in bins:
        bin_recs = [r for r in records if lo <= len(r['false_condition']) < hi]
        total = len(bin_recs)
        correct = sum(1 for r in bin_recs if r['has_clarification'])
        result[label] = {
            'total': total,
            'correct': correct,
            'acc': correct / total if total > 0 else 0,
        }
    return result


def fmt_length_acc(stats: dict) -> str:
    """格式化长度-正确率统计"""
    lines = ['\n=== 假条件长度 vs 正确率 ===']
    header = '  {:<20}  {:>6}  {:>8}  {:>8}'.format('长度区间', '正确', '总数', '正确率')
    lines.append(header)
    lines.append('  ' + '-' * len(header))
    for label, info in stats.items():
        n = info['total']
        c = info['correct']
        acc = info['acc']
        lines.append('  {:<20}  {:>6}  {:>8}  {:>8.1%}'.format(
            label, c, n, acc))
    return '\n'.join(lines)


# ------------------------------------------------------------
# 主函数
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='impossible 类型假条件分析')
    parser.add_argument('--judge-file', type=Path, default=DEFAULT_JUDGE_FILE)
    parser.add_argument('--test-file', type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument('--pred-file', type=Path, default=DEFAULT_PRED_FILE)
    parser.add_argument('--output-json', type=str, default=None)
    parser.add_argument('--top-n', type=int, default=5, help='每类打印样本数')
    args = parser.parse_args()

    # 1. 加载数据
    print(f"加载评测结果: {args.judge_file}")
    records = load_impossible_data(args.judge_file, args.test_file, args.pred_file)
    print(f"  有效 impossible 条目: {len(records)}")

    # 2. 词汇出现率
    vocab_stats = compute_vocab_stats(records)
    print(fmt_vocab_stats(vocab_stats))

    # 3. 错误模式统计
    error_stats = compute_error_stats(records)
    print(fmt_error_stats(error_stats))

    # 4. 按语言分组
    error_by_lang = compute_error_stats_by_lang(records)
    print(fmt_error_by_lang(error_by_lang))

    # 5. 假条件长度 vs 正确率
    length_acc = compute_length_acc_by_bin(records)
    print(fmt_length_acc(length_acc))

    # 6. 各错误类型样本
    print_samples_by_error_type(records, n=args.top_n)

    # 7. JSON 输出
    if args.output_json:
        output = {
            'total': len(records),
            'vocab_stats': vocab_stats,
            'error_stats': error_stats,
            'error_by_lang': error_by_lang,
            'length_acc': length_acc,
            'records': [
                {
                    'row_idx': r['row_idx'],
                    'lang': r['lang'],
                    'false_condition': r['false_condition'],
                    'condition_len': len(r['false_condition']),
                    'mention_exact': check_condition_mentioned(
                        r['completion'], r['false_condition'], r['lang'])['exact'],
                    'has_clarification': r['has_clarification'],
                    'error_type': classify_error(r),
                    'completion_preview': r['completion'][:300],
                }
                for r in records
            ]
        }
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n[已保存] 详细结果 → {args.output_json}")


if __name__ == '__main__':
    main()
