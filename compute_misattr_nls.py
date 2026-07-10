#!/usr/bin/env python3
"""
misattr 实体 Normalized Levenshtein Similarity 分析脚本
分析 FactGuard 评测结果中 misattr 类型的实体修改幅度与错误率关系
"""

import json
import jsonlines
from collections import defaultdict
from pathlib import Path
import argparse
import statistics

# ------------------------------------------------------------
# 文件路径配置
# ------------------------------------------------------------
DEFAULT_EVAL_FILE = Path('FactGuard/数据/gemini-3-pro-preview_qwen-judge.jsonl')
DEFAULT_TEST_FILE = Path('FactGuard/数据/合成数据/merged_test.jsonl')

# ------------------------------------------------------------
# Levenshtein Similarity
# ------------------------------------------------------------

def levenshtein_distance(s1: str, s2: str) -> int:
    """计算两字符串之间的 Levenshtein 编辑距离"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    curr = [0] * (len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr[0] = i + 1
        for j, c2 in enumerate(s2):
            insertions = prev[j + 1] + 1
            deletions = curr[j] + 1
            substitutions = prev[j] + (c1 != c2)
            curr[j + 1] = min(insertions, deletions, substitutions)
        prev, curr = curr, prev
    return prev[len(s2)]


def normalized_levenshtein_similarity(s1: str, s2: str) -> float:
    """
    计算 Normalized Levenshtein Similarity (NLS)
    NLS = 1 - (编辑距离 / max(len(s1), len(s2)))
    取值范围 [0, 1]，1 表示完全相同，0 表示完全不同
    """
    if not s1 and not s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(s1, s2)
    return 1.0 - (dist / max_len)


# ------------------------------------------------------------
# 数据加载
# ------------------------------------------------------------

def load_misattr_judge_results(eval_file: Path) -> dict:
    """
    加载 judge 结果文件，返回 {row_idx: eval_result} 字典
    仅保留 type == 'misattr' 的条目
    """
    results = {}
    with jsonlines.open(eval_file) as r:
        for obj in r:
            if obj.get('type') == 'misattr':
                row_idx = obj.get('row_idx')
                if row_idx is not None:
                    results[int(row_idx)] = obj.get('eval_result', {})
    return results


def load_test_data(test_file: Path, misattr_row_indices: set) -> tuple[dict, dict]:
    """
    加载测试数据集，返回:
      - origins: {row_idx: origin_dict}
      - lang_map: {row_idx: lang}  (仅 misattr 条目)
    source 字段在每行顶层，格式如 data_misattr_en_book_v2.jsonl_test.jsonl
    """
    origins = {}
    lang_map = {}
    with jsonlines.open(test_file) as r:
        for i, obj in enumerate(r):
            origins[i] = obj.get('origin', {})
            if i in misattr_row_indices:
                source = obj.get('source', '')
                lang_map[i] = get_lang(source)
    return origins, lang_map


def get_lang(source: str) -> str:
    """根据 source 判断语言"""
    return 'zh' if '_zh_' in source else 'en'


# ------------------------------------------------------------
# 统计计算
# ------------------------------------------------------------

def compute_nls_stats(pairs: list, lang_map: dict = None) -> dict:
    """
    计算 NLS 全量统计量
    pairs: [(row_idx, orig_entity, mod_entity, eval_result), ...]
    lang_map: {row_idx: lang} 可选，用于按语言分组
    """
    if not pairs:
        return {}

    all_sims = []
    correct_sims = []
    incorrect_sims = []
    lang_sims = defaultdict(list)

    for row_idx, orig, mod, eval_r in pairs:
        nls = normalized_levenshtein_similarity(orig, mod)
        all_sims.append(nls)

        if eval_r.get('是否有澄清') == '是':
            correct_sims.append(nls)
        else:
            incorrect_sims.append(nls)

        if lang_map and row_idx in lang_map:
            lang_sims[lang_map[row_idx]].append(nls)

    def _stats(sims):
        if not sims:
            return {}
        return {
            'n': len(sims),
            'mean': statistics.mean(sims),
            'median': statistics.median(sims),
            'std': statistics.stdev(sims) if len(sims) > 1 else 0,
            'min': min(sims),
            'max': max(sims),
        }

    return {
        'overall': _stats(all_sims),
        'correct': _stats(correct_sims),
        'incorrect': _stats(incorrect_sims),
        'by_lang': {lang: _stats(sims) for lang, sims in lang_sims.items()},
    }


def compute_nls_distribution(sims: list, bins: list = None) -> dict:
    """
    计算 NLS 区间分布
    bins: [(lo, hi, label), ...]
    """
    if bins is None:
        bins = [
            (0.0, 0.2, '极低 (0-0.2)'),
            (0.2, 0.4, '低 (0.2-0.4)'),
            (0.4, 0.6, '中低 (0.4-0.6)'),
            (0.6, 0.8, '中高 (0.6-0.8)'),
            (0.8, 1.0, '高 (0.8-1.0)'),
        ]

    total = len(sims)
    dist = {}
    for lo, hi, label in bins:
        count = sum(1 for s in sims if lo <= s < hi)
        dist[label] = {
            'count': count,
            'pct': count / total * 100 if total > 0 else 0,
        }
    return dist


# ------------------------------------------------------------
# 格式化输出
# ------------------------------------------------------------

def fmt_stats(label: str, stats: dict, indent: int = 2) -> str:
    """格式化统计量输出"""
    prefix = ' ' * indent
    lines = [f"{prefix}{label} (N={stats['n']}):"]
    lines.append(f"{prefix}  均值 (Mean):   {stats['mean']:.4f}")
    lines.append(f"{prefix}  中位数 (Median): {stats['median']:.4f}")
    lines.append(f"{prefix}  标准差 (Std):  {stats['std']:.4f}")
    lines.append(f"{prefix}  最小值 (Min):  {stats['min']:.4f}")
    lines.append(f"{prefix}  最大值 (Max):  {stats['max']:.4f}")
    return '\n'.join(lines)


def fmt_distribution(dist: dict) -> str:
    """格式化区间分布输出"""
    lines = ['\nNLS 区间分布:']
    for label, info in dist.items():
        count = info['count']
        pct = info['pct']
        bar = '█' * int(pct / 2)
        lines.append(f"  {label:<16}  {count:>3} 条 ({pct:5.1f}%)  {bar}")
    return '\n'.join(lines)


# ------------------------------------------------------------
# 按 NLS 区间计算正确率
# ------------------------------------------------------------

NLS_BINS = [
    (0.0, 0.4,  '< 0.4'),
    (0.4, 0.6,  '0.4 - 0.6'),
    (0.6, 0.8,  '0.6 - 0.8'),
    (0.8, 1.01, '> 0.8'),
]


def compute_nls_bin_accuracy(pairs: list) -> dict:
    """
    按 NLS 区间计算 misattr 正确率
    pairs: [(row_idx, orig_entity, mod_entity, eval_result), ...]
    返回 {label: {'total': n, 'correct': c, 'acc': float}}
    """
    bins_data = {label: {'total': 0, 'correct': 0} for _, _, label in NLS_BINS}

    for row_idx, orig, mod, eval_r in pairs:
        nls = normalized_levenshtein_similarity(orig, mod)
        correct = eval_r.get('是否有澄清') == '是'
        for lo, hi, label in NLS_BINS:
            if lo <= nls < hi:
                bins_data[label]['total'] += 1
                if correct:
                    bins_data[label]['correct'] += 1
                break

    result = {}
    for label, data in bins_data.items():
        n = data['total']
        c = data['correct']
        result[label] = {
            'total': n,
            'correct': c,
            'acc': c / n if n > 0 else float('nan'),
        }
    return result


def fmt_nls_bin_accuracy(bin_acc: dict) -> str:
    """格式化 NLS 区间正确率输出"""
    lines = ['\n=== 按 NLS 区间 misattr 正确率 ===']
    header = f"  {'NLS 区间':<14}  {'正确':>6}  {'总数':>6}  {'正确率':>8}  分布"
    lines.append(header)
    lines.append('  ' + '-' * len(header))

    total_all = sum(d['total'] for d in bin_acc.values())
    for label, data in bin_acc.items():
        n = data['total']
        c = data['correct']
        acc = data['acc']
        pct = n / total_all * 100 if total_all > 0 else 0
        bar = '█' * int(pct / 2)
        acc_str = f'{acc:.1%}' if not (acc is None or (isinstance(acc, float) and acc != acc)) else 'N/A'
        lines.append(f"  {label:<14}  {c:>6}  {n:>6}  {acc_str:>8}  {bar}")
    return '\n'.join(lines)


# ------------------------------------------------------------
# 主函数
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='misattr 实体 NLS 分析')
    parser.add_argument('--eval-file', type=Path, default=DEFAULT_EVAL_FILE,
                        help='评测结果文件路径')
    parser.add_argument('--test-file', type=Path, default=DEFAULT_TEST_FILE,
                        help='原始测试数据文件路径')
    parser.add_argument('--top-n', type=int, default=10,
                        help='打印样本数量 (前N条)')
    parser.add_argument('--output-json', type=str, default=None,
                        help='可选: 将统计结果输出为 JSON 文件')
    args = parser.parse_args()

    # 1. 加载数据
    print(f"加载评测结果: {args.eval_file}")
    judge_results = load_misattr_judge_results(args.eval_file)
    print(f"  misattr 条目数: {len(judge_results)}")

    print(f"加载原始数据: {args.test_file}")
    origins, lang_map = load_test_data(args.test_file, set(judge_results.keys()))

    # 2. 提取 misattr 实体对
    pairs = []
    for row_idx in sorted(judge_results.keys()):
        origin = origins.get(row_idx, {})
        orig_entity = origin.get('替换前实体', '')
        mod_entity = origin.get('替换后实体', '')
        if orig_entity and mod_entity:
            pairs.append((int(row_idx), orig_entity, mod_entity, judge_results[row_idx]))
        else:
            print(f"  [警告] row_idx={row_idx} 缺少实体字段: "
                  f"orig='{orig_entity}', mod='{mod_entity}'")

    print(f"有效 misattr 对数: {len(pairs)}\n")

    # 3. 计算全量 NLS
    all_sims = []
    for row_idx, orig, mod, eval_r in pairs:
        nls = normalized_levenshtein_similarity(orig, mod)
        all_sims.append(nls)

    # 4. 打印样本
    print(f"=== 样本 (前 {args.top_n} 条) ===")
    for row_idx, orig, mod, eval_r in pairs[:args.top_n]:
        nls = normalized_levenshtein_similarity(orig, mod)
        correct = eval_r.get('是否有澄清') == '是'
        status = '✓ 已澄清' if correct else '✗ 未澄清'
        print(f"  [{row_idx}] 原始: '{orig}' → 修改: '{mod}'  NLS={nls:.4f}  {status}")

    # 5. 全量统计
    stats = compute_nls_stats(pairs, lang_map)

    print(f"\n=== 全量 NLS 统计 (N={len(pairs)}) ===")
    print(fmt_stats('整体', stats['overall']))

    # 6. 区间分布
    dist = compute_nls_distribution(all_sims)
    print(fmt_distribution(dist))

    # 6b. 按 NLS 区间计算 misattr 正确率
    bin_acc = compute_nls_bin_accuracy(pairs)
    print(fmt_nls_bin_accuracy(bin_acc))

    # 7. 按语言分组
    print(f"\n=== 按语言分组 ===")
    for lang in ('en', 'zh'):
        s = stats['by_lang'].get(lang)
        if s:
            print(fmt_stats(lang.upper(), s))

    # 8. 按是否正确澄清分组
    print(f"\n=== 按是否正确澄清分组 ===")
    print(fmt_stats('已澄清 (正确)', stats['correct']))
    print()
    print(fmt_stats('未澄清 (错误)', stats['incorrect']))
    print(f"\n  总 misattr 条目: {len(pairs)}, "
          f"已澄清: {stats['correct']['n']}, "
          f"未澄清: {stats['incorrect']['n']}")

    # 9. 可选: 输出 JSON
    if args.output_json:
        output = {
            'total_pairs': len(pairs),
            'stats': stats,
            'distribution': dist,
            'all_sims': all_sims,
            'pairs': [
                {
                    'row_idx': row_idx,
                    'orig_entity': orig,
                    'mod_entity': mod,
                    'nls': normalized_levenshtein_similarity(orig, mod),
                    'correct': eval_r.get('是否有澄清') == '是',
                }
                for row_idx, orig, mod, eval_r in pairs
            ]
        }
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n[已保存] 详细结果 → {args.output_json}")


if __name__ == '__main__':
    main()
