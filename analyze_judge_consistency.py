#!/usr/bin/env python3
"""
分析三个不同 LLM judge 在相同数据上的评判一致性。

一致性指标：
  - 两两 Cohen's Kappa
  - 两两 simple agreement
  - 三者完全一致比例
  - 分 type 细分
"""

import json
from pathlib import Path
from collections import defaultdict

try:
    from sklearn.metrics import cohen_kappa_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[warn] sklearn not found, kappa will be computed manually")

# ===== 文件路径 =====
DATA_DIR = Path('/Users/zhangqianwen/code/factguard/FactGuard/数据')
JUDGE_FILES = {
    'deepseek': DATA_DIR / 'gemini-3-pro-preview_deepseek_judge.jsonl',
    'gemini':   DATA_DIR / 'gemini-3-pro-preview_gemini_judge.jsonl',
    'qwen':     DATA_DIR / 'gemini-3-pro-preview_qwen-judge.jsonl',
}


def extract_label(record: dict):
    """
    将 eval_result 转换为二元标签：1 = 模型回答正确，0 = 错误。
    返回 None 表示无法解析。

    判断逻辑（与评测框架一致）：
      dandian / impossible / misattr（负样本）：
        模型需要拒答/澄清 → 有指出/有澄清 == '是' 为正确
      answerable（正样本）：
        模型需要正常回答 → 未拒答(是否有指出=='否') 且结论一致(是否有相同结论=='是') 为正确
        若 R3 未触发(是否有指出=='是') → 错误
    """
    ex_type = record.get('type', '')
    res = record.get('eval_result', {})

    if ex_type in ('dandian', 'impossible'):
        val = res.get('是否有指出') or res.get('是否有澄清')
        if val is None:
            return None
        return 1 if val == '是' else 0

    if ex_type == 'misattr':
        val = res.get('是否有澄清')
        if val is None:
            return None
        return 1 if val == '是' else 0

    if ex_type == 'answerable':
        pointed_out = res.get('是否有指出')
        if pointed_out == '是':
            # 模型说无法回答 → 错误
            return 0
        if pointed_out == '否':
            same = res.get('是否有相同结论')
            if same is None:
                # R3 可能失败，保守处理为 None
                return None
            return 1 if same == '是' else 0
        return None

    return None


def load_judge(path: Path):
    """以 uid 为键加载 judge 结果，返回 {uid: record}。"""
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            uid = rec.get('uid')
            if uid:
                data[uid] = rec
    return data


def cohen_kappa(a: list[int], b: list[int]) -> float:
    if HAS_SKLEARN:
        return cohen_kappa_score(a, b)
    # 手动计算
    n = len(a)
    assert n == len(b)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    p1_a = sum(a) / n; p0_a = 1 - p1_a
    p1_b = sum(b) / n; p0_b = 1 - p1_b
    p_e = p1_a * p1_b + p0_a * p0_b
    return (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0


def pct(n, total):
    return f"{n}/{total} ({100*n/total:.1f}%)" if total else "0/0"


def analyze(labels_by_judge: dict, name: str = ""):
    judge_names = list(labels_by_judge.keys())
    # 取三者都有标签的 uid
    all_uids = set.intersection(*[set(d.keys()) for d in labels_by_judge.values()])
    print(f"{'='*60}")
    if name:
        print(f"  分析范围: {name}")
    print(f"  共同样本数: {len(all_uids)}")
    if not all_uids:
        return

    # 对每个 uid 取各 judge 标签
    uid_list = sorted(all_uids)
    vecs = {j: [labels_by_judge[j][u] for u in uid_list] for j in judge_names}

    # 两两比较
    pairs = [(judge_names[i], judge_names[j])
             for i in range(len(judge_names))
             for j in range(i+1, len(judge_names))]
    print()
    print("  --- 两两一致性 ---")
    for ja, jb in pairs:
        va, vb = vecs[ja], vecs[jb]
        agree = sum(x == y for x, y in zip(va, vb))
        kappa = cohen_kappa(va, vb)
        print(f"  {ja} vs {jb}: agreement={pct(agree, len(va))}  kappa={kappa:.4f}")

    # 三者完全一致
    all_agree = sum(
        all(vecs[j][i] == vecs[judge_names[0]][i] for j in judge_names)
        for i in range(len(uid_list))
    )
    print(f"\n  三者完全一致: {pct(all_agree, len(uid_list))}")

    # 各 judge 正确率
    print()
    print("  --- 各 judge 正确率(label=1) ---")
    for j in judge_names:
        pos = sum(vecs[j])
        print(f"  {j}: {pct(pos, len(vecs[j]))}")


def analyze_pair_detail(raw: dict, labels: dict, j1: str = 'deepseek', j2: str = 'qwen'):
    """DeepSeek vs Qwen 深度对比：混淆矩阵 + 分类型分歧 + 分歧样本展示。"""
    print()
    print('=' * 60)
    print(f'  [{j1.upper()} vs {j2.upper()}] 深度分析')
    print('=' * 60)

    common = sorted(set(labels[j1]) & set(labels[j2]))
    a = [labels[j1][u] for u in common]
    b = [labels[j2][u] for u in common]

    # --- 混淆矩阵 ---
    tp = sum(x == 1 and y == 1 for x, y in zip(a, b))
    tn = sum(x == 0 and y == 0 for x, y in zip(a, b))
    fp = sum(x == 0 and y == 1 for x, y in zip(a, b))  # j1=0, j2=1
    fn = sum(x == 1 and y == 0 for x, y in zip(a, b))  # j1=1, j2=0
    n = len(common)
    print(f'\n  样本总数: {n}')
    print(f'  混淆矩阵 ({j1} \ {j2}):')
    print(f'                {j2}=1      {j2}=0')
    print(f'  {j1}=1      {tp:6d}    {fn:6d}')
    print(f'  {j1}=0      {fp:6d}    {tn:6d}')
    print(f'\n  {j1}独判正确(j1=1,j2=0): {pct(fn, n)}')
    print(f'  {j2}独判正确(j2=1,j1=0): {pct(fp, n)}')
    print(f'  两者均判正确:           {pct(tp, n)}')
    print(f'  两者均判错误:           {pct(tn, n)}')

    # --- 分类型分歧 ---
    print('\n  --- 分类型分歧率 ---')
    all_types = sorted({raw[j1][u]['type'] for u in common if u in raw[j1]})
    for t in all_types:
        uids_t = [u for u in common if raw[j1].get(u, {}).get('type') == t]
        if not uids_t:
            continue
        disagree = sum(labels[j1][u] != labels[j2][u] for u in uids_t)
        j1_only = sum(labels[j1][u] == 1 and labels[j2][u] == 0 for u in uids_t)
        j2_only = sum(labels[j1][u] == 0 and labels[j2][u] == 1 for u in uids_t)
        print(f'  {t:12s}: disagree={pct(disagree, len(uids_t))}  '
              f'{j1}独判正={j1_only}  {j2}独判正={j2_only}')

    # --- 分歧样本展示 (每个 type 最多3条) ---
    print('\n  --- 分歧样本示例 (每类最多3条) ---')
    for t in all_types:
        uids_t = [u for u in common
                  if raw[j1].get(u, {}).get('type') == t
                  and labels[j1][u] != labels[j2][u]]
        if not uids_t:
            continue
        print(f'\n  [type={t}]')
        for u in uids_t[:3]:
            r1 = raw[j1][u].get('eval_result', {})
            r2 = raw[j2][u].get('eval_result', {})
            print(f'    uid={u[:12]}...')
            print(f'      {j1}: {r1}')
            print(f'      {j2}: {r2}')


def main():
    # 加载
    raw = {name: load_judge(path) for name, path in JUDGE_FILES.items()}
    print("各文件加载量:", {k: len(v) for k, v in raw.items()})

    # 提取标签，过滤掉 None
    labels = defaultdict(dict)  # judge -> uid -> label
    for judge, records in raw.items():
        for uid, rec in records.items():
            lbl = extract_label(rec)
            if lbl is not None:
                labels[judge][uid] = lbl

    print("有效标签量:", {k: len(v) for k, v in labels.items()})

    # 全局分析
    analyze(labels, name="全部")

    # 按 type 分析
    all_types = set()
    for judge_recs in raw.values():
        for rec in judge_recs.values():
            all_types.add(rec.get('type', 'unknown'))

    for t in sorted(all_types):
        # 过滤出该 type 的 uid 集合（以任一 judge 为准）
        type_uids = set()
        for judge, records in raw.items():
            for uid, rec in records.items():
                if rec.get('type') == t and uid in labels[judge]:
                    type_uids.add(uid)

        type_labels = {
            judge: {uid: lbl for uid, lbl in labels[judge].items() if uid in type_uids}
            for judge in labels
        }
        analyze(type_labels, name=f"type={t}")

    print("="*60)

    # DeepSeek vs Qwen 深度对比
    analyze_pair_detail(raw, labels, j1='deepseek', j2='qwen')


if __name__ == '__main__':
    main()



# import json
# from pathlib import Path
# from sklearn.metrics import cohen_kappa_score
# from collections import Counter

# DATA_DIR = Path('FactGuard/数据')
# FILES = {
#     'deepseek': DATA_DIR / 'gemini-3-pro-preview_deepseek_judge.jsonl',
#     'qwen':     DATA_DIR / 'gemini-3-pro-preview_qwen-judge.jsonl',
# }

# def extract_label(record):
#     ex_type = record.get('type', '')
#     res = record.get('eval_result', {})
#     if ex_type in ('dandian', 'impossible'):
#         val = res.get('是否有指出') or res.get('是否有澄清')
#         return (1 if val == '是' else 0) if val is not None else None
#     if ex_type == 'answerable':
#         pointed = res.get('是否有指出')
#         if pointed == '是': return 0
#         if pointed == '否':
#             same = res.get('是否有相同结论')
#             return (1 if same == '是' else 0) if same is not None else None
#     return None

# raw = {}
# for name, path in FILES.items():
#     raw[name] = {}
#     with open(path) as f:
#         for line in f:
#             line = line.strip()
#             if not line: continue
#             rec = json.loads(line)
#             uid = rec.get('uid')
#             if uid:
#                 raw[name][uid] = rec

# def pct(n, total):
#     return f"{n}/{total} ({100*n/total:.1f}%)"

# for t in ['dandian', 'impossible', 'answerable']:
#     uids = sorted(
#         u for u in set(raw['deepseek']) & set(raw['qwen'])
#         if raw['deepseek'][u].get('type') == t
#     )
#     pairs = [(u, extract_label(raw['deepseek'][u]), extract_label(raw['qwen'][u])) for u in uids]
#     pairs = [(u, a, b) for u, a, b in pairs if a is not None and b is not None]
#     n = len(pairs)
#     a_vec = [p[1] for p in pairs]
#     b_vec = [p[2] for p in pairs]

#     kappa = cohen_kappa_score(a_vec, b_vec)
#     agree = sum(x == y for x, y in zip(a_vec, b_vec))

#     # 混淆矩阵
#     tp = sum(a==1 and b==1 for _,a,b in pairs)
#     tn = sum(a==0 and b==0 for _,a,b in pairs)
#     fp = sum(a==0 and b==1 for _,a,b in pairs)  # qwen独判1
#     fn = sum(a==1 and b==0 for _,a,b in pairs)  # deepseek独判1

#     print(f"\n{'='*55}")
#     print(f"  type={t}   n={n}   agree={pct(agree,n)}   κ={kappa:.4f}")
#     print(f"{'='*55}")
#     print(f"  混淆矩阵:          qwen=1   qwen=0")
#     print(f"  deepseek=1        {tp:5d}    {fn:5d}")
#     print(f"  deepseek=0        {fp:5d}    {tn:5d}")
#     print(f"\n  deepseek判正率:  {pct(sum(a_vec), n)}")
#     print(f"  qwen判正率:      {pct(sum(b_vec), n)}")
#     print(f"  deepseek独判1:   {pct(fn, n)}")
#     print(f"  qwen独判1:       {pct(fp, n)}")

#     # 分歧样本展示
#     disagree = [(u,a,b) for u,a,b in pairs if a != b]
#     print(f"\n  分歧样本 ({len(disagree)} 条，展示前3条):")
#     for u, a, b in disagree[:3]:
#         r_ds = raw['deepseek'][u].get('eval_result', {})
#         r_qw = raw['qwen'][u].get('eval_result', {})
#         src  = raw['deepseek'][u].get('source', '')
#         print(f"\n    uid={u[:14]}  source={src}")
#         print(f"    deepseek(label={a}): {json.dumps(r_ds, ensure_ascii=False)}")
#         print(f"    qwen    (label={b}): {json.dumps(r_qw, ensure_ascii=False)}")
