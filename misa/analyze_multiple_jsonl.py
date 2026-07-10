import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

def get_string_lengths(
    obj: Any, lengths: Dict[str, int], current_path: str = ""
) -> None:
    """递归遍历JSON对象，统计所有字符串字段的长度"""
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{current_path}.{key}" if current_path else key
            if isinstance(value, str):
                lengths[new_path] = len(value)
            else:
                get_string_lengths(value, lengths, new_path)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            new_path = f"{current_path}[{i}]"
            get_string_lengths(item, lengths, new_path)

def get_length_category(length: int) -> str:
    """根据长度返回对应的区间分类"""
    if length < 16 * 1024:
        return "0K-16K"
    elif length < 32 * 1024:
        return "16K-32K"
    elif length < 64 * 1024:
        return "32K-64K"
    else:
        return "64K-128K"

def analyze_jsonl_file(file_path: str) -> Dict[str, Any]:
    """分析jsonlines文件中每个JSON对象的字符串字段长度"""
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"文件不存在: {file_path}")
        return {}

    print(f"正在分析文件: {file_path}")
    print("-" * 50)

    # 用于统计长度分布
    distribution = defaultdict(int)
    total_lines = 0
    total_length_over_128k = 0
    count_over_128k = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())

                # 如果是列表，处理每个元素
                if isinstance(data, list):
                    for item in data:
                        lengths = {}
                        get_string_lengths(item, lengths)
                        if lengths:
                            max_field = max(lengths.items(), key=lambda x: x[1])
                            category = get_length_category(max_field[1])
                            distribution[category] += 1
                            total_lines += 1
                            if category == "大于128K":
                                total_length_over_128k += max_field[1]
                                count_over_128k += 1
                else:
                    lengths = {}
                    get_string_lengths(data, lengths)
                    if lengths:
                        max_field = max(lengths.items(), key=lambda x: x[1])
                        category = get_length_category(max_field[1])
                        distribution[category] += 1
                        total_lines += 1
                        if category == "大于128K":
                            total_length_over_128k += max_field[1]
                            count_over_128k += 1

            except json.JSONDecodeError:
                print(f"第 {line_num} 行JSON解析错误")
            except Exception as e:
                print(f"第 {line_num} 行处理错误: {str(e)}")

    # 计算大于128K的平均长度
    average_length_over_128k = 0
    if count_over_128k > 0:
        average_length_over_128k = total_length_over_128k / count_over_128k

    # 返回统计结果
    return {
        "distribution": distribution,
        "total_lines": total_lines,
        "average_length_over_128k": average_length_over_128k,
    }

def main(file_paths: List[str]) -> None:
    """主函数，分析多个文件并输出结果"""
    for file_path in file_paths:
        result = analyze_jsonl_file(file_path)
        if result:
            print("\n长度分布统计:")
            print("-" * 50)
            for category in ["0K-16K", "16K-32K", "32K-64K", "64K-128K", "大于128K"]:
                count = result["distribution"][category]
                percentage = (count / result["total_lines"] * 100) if result["total_lines"] > 0 else 0
                print(f"{category}: {count} 个 ({percentage:.1f}%)")

            # 打印大于128K的平均长度
            if result["average_length_over_128k"] > 0:
                print(f"\n大于128K的平均长度: {result['average_length_over_128k']:.2f} 字符")
            else:
                print("\n没有大于128K的字段")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python analyze_multiple_jsonl.py <jsonl文件路径1> <jsonl文件路径2> ...")
        sys.exit(1)

    main(sys.argv[1:]) 
