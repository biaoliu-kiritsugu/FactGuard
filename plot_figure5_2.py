import re
import json
import tqdm
import random
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局颜色方案为 colorblind
palette = "Blues"
colors = sns.color_palette(palette, 7)  # 为6个模型设置颜色
# 定义不同的hatch样式
hatch_patterns = ["", "", "", "", "", "", "", ""]  # 第一个模型无纹理，其他模型使用不同纹理

result_data = {
    "GPT-4o": [[92.71, 88, 86.71, 87.63], [48.33, 38.09, 31.85, 34.51], [39.11, 37.32, 33.64, 35.73]],
    "DeepSeek-V3-0324": [[93.74, 87.5, 86.01, 80], [47.92, 33.77, 25.18, 30.97], [44.18, 41.01, 32.6, 33.26]],
    "Gemini1.5-Pro": [[86.78, 83.33, 83.77, 86.57], [58.18, 45.21, 46.81, 57.53], [60.20, 55.06, 53.03, 53.31]],
    "Llama-3.3-70B-Instruct": [[91.14, 90, 88.81, 84.21], [49.58, 35.06, 30.37, 33.63], [40.81, 33.55, 30.16, 26.89]],
    "Mistral-Large-Instruct": [[92.71, 90, 84.61, 82.47], [51.24, 39.91, 34.81, 47.79], [40.65, 34.85, 32.36, 31.87]],
    "Qwen2.5-72B-Instruct": [[91.67, 91, 88.81, 83.16], [60.42, 57.57, 56.3, 63.72], [56.84, 53.18, 53.11, 50.46]],
    "Qwen2.5-3B-Instruct": [[78.64, 77.5, 80.42, 73.68], [49.17, 39.39, 42.22, 48.67], [39.42, 36.33, 38.84, 34.33]],
    "Qwen2.5-3B-Instruct-sft": [[80.73, 83, 81.12, 82.1], [93.8, 89.18, 88.89, 82.3], [75.13, 75.65, 72.54, 75.98]]
}



def show_length_group_results(result_data, index):
    bins = [8000, 16000, 32000, 64000, 128000]

    # 定义每个区间的标签（可选）
    labels = ["0-16k", "16k-32k", "32k-64k", "64k-128k"]

    data = np.zeros(shape=(len(labels), len(result_data)))
    for i, _ in enumerate(labels):
        for j, key in enumerate(result_data.keys()):
            data[i][j] = result_data[key][index][i]*0.01

    # 创建累积条形图
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.1

    # 使用 colorblind 调色板
    colors = sns.color_palette(palette, len(result_data))

    for i, (k, c) in enumerate(
        zip([-3, -2, -1, 0, 1, 2, 3, 4], colors)
    ):
        model = list(result_data.keys())[i]
        bars = ax.bar(
            x=np.arange(len(labels)) + k * bar_width,
            height=data[:, i],
            width=bar_width,
            label=model,
            color=c,
            edgecolor="white",
            linewidth=1,
            hatch=hatch_patterns[i],  # 添加hatch样式
        )
        # # 在每个柱子上添加数值标签
        # for bar in bars:
        #     height = bar.get_height()
        #     ax.text(
        #         bar.get_x() + bar.get_width() / 2.0,
        #         height,
        #         f"{height:.2%}",
        #         ha="center",
        #         va="bottom",
        #         fontsize=6,
        #     )

    # 添加一些文本和标题
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_xlabel("Text length groups", fontsize=14)
    # ax.set_title('Accuracy comparison on different length groups of test data')
    ax.set_xticks(np.arange(len(labels)))
    xticklabels = labels
    ax.set_xticklabels(xticklabels, fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='both', labelsize=14)  #
    ax.legend(loc='upper left', fontsize=14)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"/apdcephfs_cq8/share_2992827/shennong_4/jieeewwwang/paper/1.refuse/ICLR/data/fig/length_group_result_2.png", dpi=300)

    pass



refuse_data = {
    "GPT-4o": [62.5, 8.9, 28.6],
    "DeepSeek-V3-0324": [61.7, 8.1, 30.2],
    "Gemini1.5-Pro": [47.1, 12, 40.9],
    "Llama-3.3-70B-Instruct": [65.0, 8.6, 26.4],
    "Mistral-Large-Instruct": [62.6, 9.8, 27.6],
    # "Qwen2.5-72B-Instruct": [45, 13.2, 41.8],
    "Qwen2.5-3B-Instruct": [60.9, 9.9, 29.2],
    "Qwen2.5-7B-Instruct": [59.5, 11, 29.5],
    "Qwen2.5-14B-Instruct": [38.8, 14.5, 46.7],
    "Qwen2.5-32B-Instruct": [35.8, 15.2, 49],
    "Qwen2.5-3B-Instruct-sft": [21.9, 20, 58.1],
    "Qwen2.5-7B-Instruct-sft": [21.5, 19.1, 59.4],
    "Qwen2.5-14B-Instruct-sft": [20.8, 19.4, 59.8],
    "Qwen2.5-32B-Instruct-sft": [20.0, 20.3, 59.7]
}
def show_refuse_rate(refuse_data):
    data = np.zeros(shape=(3, len(refuse_data)))
    for i, model in enumerate(refuse_data.keys()):
        data[0][i] = refuse_data[model][0] / sum(refuse_data[model])
        data[1][i] = refuse_data[model][1] / sum(refuse_data[model])
        data[2][i] = refuse_data[model][2] / sum(refuse_data[model])

    # 修改堆叠柱状图的颜色为 colorblind 方案中的颜色
    # colors = sns.color_palette(palette)[2:5]  # 只取前三个颜色
    colors = ['#FA7F6F','#96C37D', 'gray']
    hatch_patterns = ["", "//", ""] 
    # 设置x轴的位置
    x = np.arange(len(refuse_data))

    # 创建累积条形图
    fig, ax = plt.subplots(figsize=(10, 6))
    # fig, ax = plt.subplots()

   

    # 绘制正确的数量（底部）
    p1 = ax.bar(
        x,
        data[2],
        width=0.8,
        color=colors[1],
        label="Reasoned answers",
        edgecolor="white",
        linewidth=1,
        hatch=hatch_patterns[2]
    )
    # 绘制错误的数量（堆叠在正确之上）
    p2 = ax.bar(
        x,
        data[1],
        width=0.8,
        bottom=data[2],
        color=colors[1],
        label="Direct refusals",
        edgecolor="white",
        linewidth=1,
        hatch=hatch_patterns[1],
    )
    p3 = ax.bar(
        x,
        data[0],
        width=0.8,
        bottom=data[1] + data[2],
        color=colors[0],
        label="Incorrect",
        edgecolor="white",
        linewidth=1,
        hatch=hatch_patterns[0],
    )
    # 绘制error的数量（堆叠在错误之上）
    # p3 = ax.bar(x, data[2], width=0.7, bottom=data[0]+data[1], color=colors[2], label='Error')

    # 添加一些文本和标题
    ax.set_ylabel("Rate", fontsize=16)
    # ax.set_title('Accuracy comparison of different models on all test data')
    ax.set_xticks(x)
    # xticklabels = ['Claude-3.5', 'GPT-4o', 'GPT-4o-mini', 'Gemini-1.5', 'Qwen-2.5', 'Llama-3.3']
    xticklabels = list(refuse_data.keys())
    ax.set_xticklabels(xticklabels, rotation=30, ha="right", fontsize=14)
    ax.legend(fontsize=14)

    # 设置y轴为对数尺度
    # ax.set_yscale('log')

    # 在每个柱子上添加True的比例
    for idx, prop in enumerate(data[2]):
        # 计算文本位置（顶部柱子的顶端）
        height = data[2][idx]
        ax.text(x[idx], height, f"{prop:.1%}", ha="center", va="bottom", fontsize=10)
    for idx, prop in enumerate(data[1]):
        # 计算文本位置（顶部柱子的顶端）
        height = data[2][idx] + data[1][idx]
        ax.text(x[idx], height, f"{prop:.1%}", ha="center", va="bottom", fontsize=10)
    for idx, prop in enumerate(data[0]):
        # 计算文本位置（顶部柱子的顶端）
        height = data[2][idx] + data[1][idx] + data[0][idx]
        ax.text(x[idx], height, f"{prop:.1%}", ha="center", va="bottom", fontsize=10)

    # # 在每个柱子上添加False的比例
    # for idx, prop in enumerate(false_proportions):
    #     # 计算文本位置（顶部柱子的顶端）
    #     height = data[0][idx] + 20
    #     ax.text(x[idx], height, f"{prop:.2%}", ha="center", va="bottom")

    # for idx, prop in enumerate(error_num):
    #     # 计算文本位置（顶部柱子的顶端）
    #     height = 1580
    #     ax.text(x[idx], height, f'Error: {int(prop)}', ha='center', va='bottom')

    # 显示图表
    plt.tight_layout()
    plt.savefig(f"/apdcephfs_cq8/share_2992827/shennong_4/jieeewwwang/paper/1.refuse/ICLR/data/fig/refuse_rate.png", dpi=300)


def bili_rate():
    
    answer = np.array([85.02, 84.04, 83.42, 82.57, 83.15]) / 100
    lack = np.array([54.96, 42.49, 59.58, 69.51, 85.06]) / 100 
    # mis = np.array([42.69, 72.35, 77.74, 75.13, 81.54]) / 100 
    mis = np.array([29.52, 53.76, 56.65, 55.45, 59.44]) / 100 

    


    x = np.arange(len(answer))
    colors = sns.color_palette(palette, 6)

    # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-  --   -.  :    ,
    # marker：.  ,   o   v    <    *    +    1
    plt.figure(figsize=(10, 6))
    # linestyle = "-"
    plt.grid(linestyle="-.")  # 设置背景网格线为虚线
    ax = plt.gca()
    # ax.spines['top'].set_visible(False)  # 去掉上边框
    # ax.spines['right'].set_visible(False)  # 去掉右边框

    linewidth = 3.0
    markersize = 10

    # plt.plot(x, answer, marker='o', markersize=markersize, color="blue", label="Answerable questions", linewidth=linewidth)
    # plt.plot(x, lack, marker='s', markersize=markersize, color="green", label="Lack of evidence", linewidth=linewidth)
    plt.plot(x, mis, marker='^', markersize=markersize, color="green", label="Reasoned answers", linewidth=linewidth)

    ax.set_xticks(np.arange(len(answer)))
    xticklabels = ["Base", "8:1", "4:1", "2:1", "1:1"]
    ax.set_xticklabels(xticklabels, fontsize=14)

    # group_labels = ['-', '20%', '40%', '60%', '80%']
    # plt.xticks(x, group_labels, fontsize=15)  # 默认字体大小为10
    y_ticks = [0.10, 0.15, 0.20, 0.25, 0.30]
    y_lables = ['0.10', '0.15', '0.20', '0.25', '0.30']
    # plt.yticks(np.array(y_ticks), y_lables, fontsize=15)
    # plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
    # plt.text(1, label_position, dataset,fontsize=25, fontweight='bold')
    plt.xlabel("The ratios of answerable to unanswerable data", fontsize=16)
    plt.ylabel(f"Accuracy", fontsize=16)
    plt.xlim(0, 4)  # 设置x轴的范围
    plt.ylim(0.2, 0.7)

    # plt.legend()
    # 显示各曲线的图例 loc=3 lower left
    plt.legend(loc=0, numpoints=1, ncol=2)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=14)
    # plt.setp(ltext, fontsize=25, fontweight='bold')  # 设置图例字体的大小和粗细
    plt.tight_layout()
    plt.savefig(f'/apdcephfs_cq8/share_2992827/shennong_4/jieeewwwang/paper/1.refuse/ICLR/data/fig/bili_line_reason.png', format='png')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
    # plt.show()

	# 0-4k	4-8	8-16	16-32	32-64	64-96	96-128
# qwen-72	0.8888	0.624999984	0.576165803	0.524722502	0.521167882	0.565420559	0.512605038
# ds-v3	0.666666593	0.475	0.449170124	0.391129032	0.317567567	0.284452296	0.270408162
# gpt	0.777777691	0.449999989	0.40951396	0.373615307	0.32928475	0.29964539	0.280612243
# Llama-3.3-70B	0.444444395	0.499999988	0.420289855	0.343403826	0.291497975	0.214788732	0.178571428
# gemini	0.88888879	0.56410255	0.555932203	0.468965517	0.470027247	0.457809694	0.405263156
# mis	0.999999889	0.499999988	0.4229576	0.364919354	0.309041835	0.272084805	0.248648647
# Qwen2.5-32B-Instruct	0.999999889	0.76923075	0.659067357	0.644803228	0.622254758	0.623831774	0.579831928
# Qwen2.5-14B-Instruct	0.88888879	0.724999982	0.636645962	0.600606673	0.595029239	0.616822428	0.537815122
# Qwen2.5-7B-Instruct	0.777777691	0.524999987	0.423991727	0.400201612	0.368421052	0.366197182	0.306122447
# Qwen2.5-3B-Instruct	0.88888879	0.599999985	0.408290155	0.371975806	0.392700729	0.373831775	0.344537812
# Qwen2.5-3B-Instruct-sft	0.999999889	0.824999979	0.79792746	0.785282257	0.755847952	0.771028036	0.771186434
# Qwen2.5-7B-Instruct-sft	0.999999889	0.824999979	0.804347825	0.760080644	0.752702702	0.710247349	0.70918367
# Qwen2.5-14B-Instruct-sft	0.999999889	0.820512799	0.821761657	0.764646464	0.787701317	0.786885244	0.798319321
# Qwen2.5-32B-Instruct-sft	0.999999889	0.874999978	0.832298136	0.769385699	0.790014683	0.790209788	0.840336127

llms_len_result = {
    "GPT-4o": [0.777777691, 0.449999989, 0.40951396, 0.373615307, 0.32928475, 0.29964539, 0.280612243],
    "DeepSeek-V3-0324": [0.666666593, 0.475, 0.449170124, 0.391129032, 0.317567567, 0.284452296, 0.270408162],
    "Gemini1.5-Pro": [0.88888879, 0.56410255, 0.555932203, 0.468965517, 0.470027247, 0.457809694, 0.405263156],
    "Llama-3.3-70B-Instruct": [0.80, 0.499999988, 0.420289855, 0.343403826, 0.291497975, 0.214788732, 0.178571428],
    "Mistral-Large-Instruct": [0.999999889, 0.499999988, 0.4229576, 0.364919354, 0.309041835, 0.272084805, 0.248648647],
    # "Qwen2.5-72B-Instruct": [0.8888, 0.624999984, 0.576165803, 0.524722502, 0.521167882, 0.565420559, 0.512605038],

}
qwen_len_result = {
    "Qwen2.5-32B-Instruct":	[0.999999889, 0.76923075, 0.659067357, 0.644803228, 0.622254758, 0.623831774, 0.579831928],
    "Qwen2.5-14B-Instruct":	[0.88888879, 0.724999982, 0.636645962, 0.600606673, 0.595029239, 0.616822428, 0.537815122],
    "Qwen2.5-7B-Instruct":	[0.777777691, 0.524999987, 0.423991727, 0.400201612, 0.368421052, 0.366197182, 0.306122447],
    "Qwen2.5-3B-Instruct":	[0.88888879, 0.599999985, 0.408290155, 0.371975806, 0.392700729, 0.373831775, 0.344537812],
    "Qwen2.5-32B-Instruct-sft":	[0.999999889, 0.874999978, 0.832298136, 0.769385699, 0.790014683, 0.790209788, 0.840336127],
    "Qwen2.5-14B-Instruct-sft":	[0.999999889, 0.820512799, 0.821761657, 0.764646464, 0.787701317, 0.786885244, 0.798319321],
    "Qwen2.5-7B-Instruct-sft":	[0.999999889, 0.824999979, 0.804347825, 0.760080644, 0.752702702, 0.710247349, 0.70918367],
    "Qwen2.5-3B-Instruct-sft":	[0.999999889, 0.824999979, 0.79792746, 0.785282257, 0.755847952, 0.771028036, 0.771186434],
}

result_data = {
    "GPT-4o": [[92.71, 88, 86.71, 87.63], [48.33, 38.09, 31.85, 34.51], [39.11, 37.32, 33.64, 35.73]],
    "DeepSeek-V3-0324": [[93.74, 87.5, 86.01, 80], [47.92, 33.77, 25.18, 30.97], [44.18, 41.01, 32.6, 33.26]],
    "Gemini1.5-Pro": [[86.78, 83.33, 83.77, 86.57], [58.18, 45.21, 46.81, 57.53], [60.20, 55.06, 53.03, 53.31]],
    "Llama-3.3-70B-Instruct": [[91.14, 90, 88.81, 84.21], [49.58, 35.06, 30.37, 33.63], [40.81, 33.55, 30.16, 26.89]],
    "Mistral-Large-Instruct": [[92.71, 90, 84.61, 82.47], [51.24, 39.91, 34.81, 47.79], [40.65, 34.85, 32.36, 31.87]],
    # "Qwen2.5-72B-Instruct": [[91.67, 91, 88.81, 83.16], [60.42, 57.57, 56.3, 63.72], [56.84, 53.18, 53.11, 50.46]],
    "Qwen2.5-3B-Instruct": [[78.64, 77.5, 80.42, 73.68], [49.17, 39.39, 42.22, 48.67], [39.42, 36.33, 38.84, 34.33]],
    "Qwen2.5-3B-Instruct-sft": [[80.73, 83, 81.12, 82.1], [93.8, 89.18, 88.89, 82.3], [75.13, 75.65, 72.54, 75.98]]
}
def line_show_length_group_results(llms_len_result, index):
    """折线，反应长度趋势"""
    bins = [8000, 16000, 32000, 64000, 128000]

    # 定义每个区间的标签（可选）
    labels = ["0-4k", "4k-8k", "8k-16k", "16k-32k", "32k-64k", "64k-96k", "96k-128k"]
    x = np.arange(len(labels))
    # 创建累积条形图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 使用 colorblind 调色板
    # colors = sns.color_palette(palette, len(llms_len_result))
    colors = plt.cm.tab20(np.linspace(0, 2, 12))[1:]
    linewidth = 3.0
    markersize = 10
    for i, key in enumerate(llms_len_result.keys()):
        
        

        plt.plot(x, llms_len_result[key], marker='^', markersize=markersize, color=colors[i], label=key, linewidth=linewidth)
        # # 在每个柱子上添加数值标签
        # for bar in bars:
        #     height = bar.get_height()
        #     ax.text(
        #         bar.get_x() + bar.get_width() / 2.0,
        #         height,
        #         f"{height:.2%}",
        #         ha="center",
        #         va="bottom",
        #         fontsize=6,
        #     )

    # 添加一些文本和标题
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_xlabel("Text length groups", fontsize=14)
    # ax.set_title('Accuracy comparison on different length groups of test data')
    ax.set_xticks(np.arange(len(labels)))
    xticklabels = labels
    ax.set_xticklabels(xticklabels, fontsize=14)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', labelsize=14)  #
    ax.legend(loc='lower left', fontsize=14)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"/apdcephfs_cq8/share_2992827/shennong_4/jieeewwwang/paper/1.refuse/ICLR/data/fig/line_length_group_result_2.png", dpi=300)

    pass

def qwen_line_show_length_group_results(llms_len_result, index):
    """折线，反应长度趋势"""
    bins = [8000, 16000, 32000, 64000, 128000]

    # 定义每个区间的标签（可选）
    labels = ["0-4k", "4k-8k", "8k-16k", "16k-32k", "32k-64k", "64k-96k", "96k-128k"]
    x = np.arange(len(labels))
    # 创建累积条形图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 使用 colorblind 调色板
    # colors = sns.color_palette(palette, len(llms_len_result))
    colors = plt.cm.Set2(np.linspace(0, 2, 12))
    linewidth = 3.0
    markersize = 10
    for i, key in enumerate(list(llms_len_result.keys())[:4]):
        plt.plot(x, llms_len_result[key], marker='o', markersize=markersize, color=colors[i], label=key, linewidth=linewidth)
    for i, key in enumerate(list(llms_len_result.keys())[4:]):
        plt.plot(x, llms_len_result[key], marker='^', markersize=markersize, color=colors[i], label=key, linewidth=linewidth)
        # # 在每个柱子上添加数值标签
        # for bar in bars:
        #     height = bar.get_height()
        #     ax.text(
        #         bar.get_x() + bar.get_width() / 2.0,
        #         height,
        #         f"{height:.2%}",
        #         ha="center",
        #         va="bottom",
        #         fontsize=6,
        #     )

    # 添加一些文本和标题
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_xlabel("Text length groups", fontsize=14)
    # ax.set_title('Accuracy comparison on different length groups of test data')
    ax.set_xticks(np.arange(len(labels)))
    xticklabels = labels
    ax.set_xticklabels(xticklabels, fontsize=14)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', labelsize=14)  #
    ax.legend(loc='lower left', fontsize=13)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"/apdcephfs_cq8/share_2992827/shennong_4/jieeewwwang/paper/1.refuse/ICLR/data/fig/line_length_group_result_1.png", dpi=300)

    pass

def qwen_show_length_group_results(result_data):
    bins = [8000, 16000, 32000, 64000, 128000]

    # 定义每个区间的标签（可选）
    labels = ["0-4k", "4k-8k", "8k-16k", "16k-32k", "32k-64k", "64k-96k", "96k-128k"]

    data = np.zeros(shape=(len(labels), len(result_data)))
    for i, _ in enumerate(labels):
        for j, key in enumerate(result_data.keys()):
            data[i][j] = result_data[key][i]

    # 创建累积条形图
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.1

    # 使用 colorblind 调色板
    colors = sns.color_palette(palette, len(result_data))

    for i, (k, c) in enumerate(
        zip([-3, -2, -1, 0, 1, 2, 3, 4], colors)
    ):
        model = list(result_data.keys())[i]
        bars = ax.bar(
            x=np.arange(len(labels)) + k * bar_width,
            height=data[:, i],
            width=bar_width,
            label=model,
            color=c,
            edgecolor="white",
            linewidth=1,
            hatch=hatch_patterns[i],  # 添加hatch样式
        )
        # # 在每个柱子上添加数值标签
        # for bar in bars:
        #     height = bar.get_height()
        #     ax.text(
        #         bar.get_x() + bar.get_width() / 2.0,
        #         height,
        #         f"{height:.2%}",
        #         ha="center",
        #         va="bottom",
        #         fontsize=6,
        #     )

    # 添加一些文本和标题
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_xlabel("Text length groups", fontsize=14)
    # ax.set_title('Accuracy comparison on different length groups of test data')
    ax.set_xticks(np.arange(len(labels)))
    xticklabels = labels
    ax.set_xticklabels(xticklabels, fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='both', labelsize=14)  #
    ax.legend(loc='upper left', fontsize=14)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"/apdcephfs_cq8/share_2992827/shennong_4/jieeewwwang/paper/1.refuse/ICLR/data/fig/qwen_length_group_result_1.png", dpi=300)

    pass


if __name__ == "__main__":

    # show_length_group_results(result_data, 2)
    # show_refuse_rate(refuse_data)
    # bili_rate()
    line_show_length_group_results(llms_len_result, 2)
    # qwen_line_show_length_group_results(qwen_len_result, 2)
    # print(df)
    # qwen_show_length_group_results(qwen_len_result)