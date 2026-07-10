# -*- coding: utf-8 -*-

import os
import re
import json
import tqdm
import random
import hashlib
from misattribution.generation.client import ChatApi
import warnings
from misattribution.generation.auto_rag_hy import Pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


api = ChatApi("qwen2.5")


def gen_md5(_input, _output, system_prompt=""):
    data = str(_input) + str(_output) + str(system_prompt)
    return hashlib.md5(data.encode(encoding="UTF-8")).hexdigest()


# from rag import ExampleGenerator, Pipeline
"""
text
按照窗口，句号，或者\n分割，每个窗口500-1000，分成n个片段
随机取一个片段，给7B-moe-api，得到QA，以及哪句话得到的答案
RAG-frank，输入Q以及整个text，看检索结果
"""

task = {
    "事实抽取": {
        "实体信息抽取": [
            "时间信息抽取",
            "地点信息抽取",
            "人物信息抽取",
            "原因信息抽取",
        ],
        "数值信息抽取": ["具体数值抽取", "范围数值抽取"],
        "观点信息抽取": ["情感/态度抽取", "争议点提取", "共识点提取"],
        "结构信息抽取": [
            "表格提取",
            "章节大纲提取",
            "图片提取",
            "公式提取",
            "URL提取",
            "文案抽取",
            "其他自定义元素抽取",
        ],
        "内容抽取": ["术语解释", "列举内容抽取", "原文片段抽取"],
    }
}

# ['实体信息抽取', '数值信息抽取', '观点信息抽取', '结构信息抽取', '内容抽取'] 示例
# https://doc.weixin.qq.com/sheet/e3_AUsAXgbzAKUFrsawxeLTZ00A8SiI0?scode=AJEAIQdfAAoXRQDVqRAUsAXgbzAKU&tab=BB08J2
f = open(
    "/Users/fangli/Code/python/misattribution/data/misc/任务分类器query.json",
    "r",
    encoding="utf-8",
).read()
example_dict = json.loads(f)


def add_prompt(text):
    # if random.random() > 0.5:
    #     use_fewshot = True
    # else:
    #     use_fewshot = False
    use_fewshot = True
    prompt = f"""你是一个语义信息识别专家，请根据下面提供的文本，生成一个问答对。\n"""

    if not use_fewshot:
        prompt += """请一步步思考清楚，并在最后一行以json格式输出，参考格式: {"分类": "xx", "Q": "xx", "A": "xx", "答案依据": "xxx"}\n\n"""
        prompt += f"文本：\n{text}\n"
    else:
        prompt += f"文本：\n{text}\n"
        class2 = random.choice(list(example_dict.keys()))
        questions = example_dict[class2]
        question = random.choice(questions)
        prompt += (
            '问答对信息输出示例: {"分类": "'
            + class2
            + '", "问题": "'
            + question
            + '", "答案": "xxx", "答案依据": "xxx"} \n请一步步思考清楚，要求：1. 按示例的json格式输出。2. 问答对的分类限制在【实体信息抽取、数值信息抽取、观点信息抽取、结构信息抽取、内容抽取】之中。3. 答案应该清晰、明确。 4. 答案依据来自文本中的连续内容片段。\n\n'
        )

    return prompt






def add_prompt_step23(data):
    prompt = (
        "根据文档、问题、答案，\n1. 请判断问题答案质量是否合格，如果问题和答案没有问题判断为“合格”，有任何问题，都判断为“错误”。比如：答案出现无法回答、无明确答案或拒答的情况，判断“错误”。答案都是链接的情况，判断“错误”。问题答案不合理、不连贯等，判断“错误”。\n2. 请对答案进行优化改写：先给出明确答案，再根据文档的信息给出推理思考的过程，最后将两者顺序融合。注意：文档只是长文的一个中间片段，新答案不要出现段落位置等表述。\n 已知信息如下：\n文档："
        + data["meta"]["select_segment"]
        + "\n问题："
        + data["meta"]["question"]
        + " \n答案："
        + data["output"]
        + '\n\n要求json格式输出，输出示例：{"问题答案质量": "xxx", "明确答案": "xxx", "推理思考的过程": "xxx", "融合": "xxx"} \n'
    )
    return prompt


def add_prompt_step4(data):
    prompt = (
        "已知原始问题："
        + data["meta"]["question"]
        + '\n\n要求1. 改写原始问题：问题语义不变，问题表达上融合【基于文档回答，文档中，本文，请严格按照文章分析，根据上文，参考文档】等等相似含义内容。 2. 基于新问题，假设提供了文档、但文档无答案的情况下，给出合理、呼应问题的拒答回复语。3. 只输出一种改写结果即可。要求json格式输出，输出示例：{"新问题": "xxx", "拒答回复语": "xxx"} \n'
    )
    return prompt


def add_prompt_final(data, con):
    prompt = "根据以下文档回答问题，如果文档中没有答案，回复以”对不起“开头。\n已知文档：{} \n 基于以上文档回答问题：{}\n答案：".format(
        con, data["meta"]["question"]
    )
    return prompt


def split_text_into_segments(text, min_length=500, max_length=1000):
    # 按照换行符分割文本
    lines = text.split("\n")
    segments = []
    current_segment = ""
    for line in lines:
        # 如果当前片段加上新行的长度超过最大长度，并且当前片段不为空则将当前片段加入片段列表
        if len(current_segment) + len(line) + 1 > max_length and current_segment:
            segments.append(current_segment)
            current_segment = line
        else:
            # 如果当前片段不为空，则添加换行符
            if current_segment:
                current_segment += "\n"
            current_segment += line

        # 如果当前片段的长度已经超过最小长度，则将其加入片段列表
        if len(current_segment) >= min_length:
            segments.append(current_segment)
            current_segment = ""

    # 添加最后一个片段
    if current_segment:
        segments.append(current_segment)
    new_segment = []
    for item in segments:
        if len(item) > 5000:
            # print( "异常切片，保证切片控制在5k...")
            # return [], False
            continue
        if len(item) > min_length:
            new_segment.append(item)
    if new_segment:
        return new_segment, True
    else:
        return [], False


def json_strip(res_txt):
    if "```" in res_txt:
        res_txt = res_txt.split("```")[1].strip()
        if res_txt.startswith("json"):
            res_txt = res_txt[4:]
    return res_txt


def process_line(line, outfile, outfile_del):
    try:
        data = json.loads(line.strip())
        contents = []
        # elements = ['title', 'author', 'translator', 'intro', 'publisher']
        # sampled_elements = random.sample(elements, random.randint(0, 5) )
        # sampled_elements.append('markdown')

        for item in ["title", "text", "doc", "context"]:
            if item not in data:
                continue
            if len(data[item]) > 0:
                contents.append(data[item])
        content = "\n".join(contents)
        ##  跳过一些文档
        if (
            content.find("【解析】") > -1
            or len(content) > 256000
            or len(content) < 3000
        ):
            return

        segments, stat = split_text_into_segments(content)
        if not stat:
            return
        ## 请求片段构建QA对
        select_segments = random.sample(segments, min(3, len(segments)))
        for select in select_segments:
            prompt = add_prompt(select)
            ##  生成用主模型
            model_answer = api.chat(prompt)
            res_txt = model_answer.content
            try:
                res_txt = json_strip(res_txt)

                res_json = json.loads(res_txt)
            except Exception:
                continue
            if (
                "问题" not in res_json.keys()
                or "答案" not in res_json.keys()
                or "答案依据" not in res_json.keys()
            ):
                continue
            question = str(res_json["问题"])
            answer = str(res_json["答案"])
            new_data = {}
            new_data["md5"] = ""
            new_data["status"] = "update"
            new_data["input"] = ""
            new_data["output"] = answer
            new_data["system_prompt"] = ""
            refuse_doc = content.replace(select, "")
            assert len(refuse_doc) < len(content)
            new_data["meta"] = {
                "question": question,
                "select_segment": select,
                "small_segment": res_json["答案依据"],  # keyerror
                "doc": content,
                "7B-MoE-SFT-256K-V2_answer": str(model_answer),
            }
            new_data["md5"] = ""
            new_data["id"] = "mrc_" + new_data["md5"]
            new_data["一级任务"] = "事实抽取"
            new_data["二级任务"] = res_json["分类"]

            ## 合并为一次请求
            prompt23 = add_prompt_step23(new_data)
            model_answer = api.chat(prompt23)
            res_txt = model_answer.content
            try:
                res_txt = json_strip(res_txt)
                res_json = json.loads(res_txt)
            except Exception as e:
                print(e)
                continue
            if "融合" not in res_json.keys() or "问题答案质量" not in res_json.keys():
                continue
            if res_json["问题答案质量"] != "合格":
                continue
            new_data["meta"]["answer1"] = new_data["output"]
            new_data["meta"]["answer2"] = res_txt
            new_data["output"] = res_json["融合"]

            # RAG
            pipeline = Pipeline(new_data)
            retrails = pipeline.run_item()
            new_data["meta"]["retrails"] = retrails
            # retrails过滤
            is_ok = True
            for item in new_data["meta"]["retrails"]:
                prompt = add_prompt_final(new_data, item)
                model_answer = api.chat(prompt)
                res_txt = model_answer.content
                ## 只要有一个找到答案，就舍弃
                if res_txt.find("对不起") == -1:
                    new = {}
                    new["question"] = new_data["meta"]["question"]
                    new["output"] = new_data["output"]
                    new["select_segment"] = new_data["meta"]["select_segment"]
                    new["find_rest_retrail"] = item
                    new["find_rest_ans"] = res_txt
                    outfile_del.write(json.dumps(new, ensure_ascii=False) + "\n")
                    outfile_del.flush()
                    is_ok = False
                    break
            if is_ok:
                prompt4 = add_prompt_step4(new_data)
                model_answer = api.chat(prompt4)
                res_txt = model_answer.content
                try:
                    res_txt = json_strip(res_txt)
                    res_json = json.loads(res_txt)
                    if (
                        "新问题" not in res_json.keys()
                        or "拒答回复语" not in res_json.keys()
                    ):
                        new_data["meta"]["refuse_state"] = "error"
                        new_data["meta"]["question_r"] = ""
                        new_data["meta"]["answer_r"] = ""
                    else:
                        new_data["meta"]["refuse_state"] = "ok"
                        new_data["meta"]["question_r"] = res_json["新问题"]
                        new_data["meta"]["answer_r"] = res_json["拒答回复语"]
                except:
                    new_data["meta"]["refuse_state"] = "error"
                    new_data["meta"]["question_r"] = ""
                    new_data["meta"]["answer_r"] = ""
                # new_data["input"] = random.choice(["文档：","文档：\n","文档:","文档:\n","","",""])+ new_data["meta"]["doc"] +"\n"+random.choice(["问题：","问题：\n","问题:","问题:\n","回答以下问题：\n","基于文档，回答问题：\n","问题:\n","问题:"]) + new_data["meta"]["question"]
                # new_data["input"] = ""
                # new_data["md5"] = ""
                json_dict_str = json.dumps(new_data, ensure_ascii=False)
                outfile.write("%s\n" % json_dict_str)
                outfile.flush()
    except Exception as e:
        print(e)
        return


def process_create_item(input_file_path, output_file_path, executor, start_line):
    try:
        with open(output_file_path, "a", encoding="utf-8") as outfile, open(
            output_file_path + ".del.jsonl", "a", encoding="utf-8"
        ) as outfile_del, open(
            output_file_path + ".txt", "a", encoding="utf-8"
        ) as fnum_txt:
            infile = open(input_file_path, "r", encoding="utf-8")
            i = 0
            inner_line = 30
            for line in tqdm.tqdm(
                infile, desc="Processing [{}] ".format(input_file_path.split("/")[-1])
            ):
                i += 1
                if start_line != 0:
                    if i < start_line + inner_line:
                        continue

                process_line(line, outfile, outfile_del)
                # 等待这一批任务完成
                fnum_txt.write("已到行数: {}\n".format(i))
                fnum_txt.flush()

    except IOError as e:
        print(f"IOError: {e}")


def process_create(key, path_list, output_path, executor):
    is_continue = True
    start_line = 0
    for input_file_path in path_list:
        output_file_path = os.path.join(
            output_path, input_file_path.split("/")[-1] + ".output.jsonl"
        )
        if is_continue:
            #     pre_dir = "/apdcephfs_cq8/share_2992827/shennong_4/jieeewwwang/projects/hunyaun_sn/dandian_pipline/re_data/wangjie/flk_hy"
            #     txt_file = os.path.join(
            #         pre_dir,
            #         input_file_path.split("/")[-1].split(".json")[0],
            #         input_file_path.split("/")[-1] + ".output.jsonl.txt",
            #     )
            start_line = read_txt_get_line(output_file_path + ".txt")
        process_create_item(input_file_path, output_file_path, executor, start_line)
        # if is_continue:
        #     pre_dir = open("/apdcephfs_cq8/share_2992827/shennong_4/jieeewwwang/projects/hunyaun_sn/dandian_pipline/re_data/jtyb_wj.json", 'r', encoding='utf-8')
        #     num_dict = json.load(pre_dir)
        #     start_line = num_dict[input_file_path.split("/")[-1].split(".")[0]]
        # process_create_item(input_file_path, output_file_path, executor, start_line)


def create_output_directory(key):
    output_path = os.path.join(output_root, key)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return output_path


def worker(key, path_list, executor):
    output_path = create_output_directory(key)
    process_create(key, path_list, output_path, executor)


def read_txt_get_line(txt_file):
    if os.path.isfile(txt_file):
        with open(txt_file, "r", encoding="utf-8") as file:
            last_line = None
            for line in file:
                last_line = line
            if last_line is None:
                return 0
            else:
                match = re.findall(r"已到行数:\s*(\d+)", last_line)
                print("文件：{}\n恢复到行数：{}".format(txt_file, match[0]))
                if match:
                    return int(match[0])
                else:
                    return 0
    else:
        print("不存在文件：{}".format(txt_file))
        return 0


output_root = "/Users/fangli/Code/python/misattribution/data/generation/dandian"


def main():
    files_path = [
        "/Users/fangli/Code/python/misattribution/data/docs/LongBench/data/dureader.jsonl",
        "/Users/fangli/Code/python/misattribution/data/docs/LongData-Corpus/LongData_zh/万卷-新闻-16k-2490条.json",
        "/Users/fangli/Code/python/misattribution/data/docs/LongData-Corpus/LongData_zh/万卷-专利-16k-16715条.json",
        "/Users/fangli/Code/python/misattribution/data/docs/LongData-Corpus/LongData_en/RedPajamaArxiv_train_32k.json",
        "/Users/fangli/Code/python/misattribution/data/docs/LongData-Corpus/LongData_en/RedPajamaWikipedia-16k-6900条.json",
    ]
    # 还有两个中文论文文本可以利用
    path_dict = {}
    for file_file in files_path:
        name_ = file_file.split("/")[-1].split(".json")[0]
        path_dict[name_] = [file_file]
    # path_dict = {
    #     "wechat0":["/apdcephfs_cq8/share_2992827/shennong_4/jieeewwwang/projects/hunyaun_sn/0808_data/webook_3.json"]}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for key, path_list in path_dict.items():
            # print(key)
            # print(path_list)
            futures.append(executor.submit(worker, key, path_list, executor))

        for future in futures:
            future.result()


if __name__ == "__main__":
    main()


# jq -r '"Question: \(.meta.question)\nAnswer: \(.output)"' mrc_synthesis_v0805_step2.jsonl
# directory_path="/apdcephfs_cq8/share_2992827/shennong_4/jieeewwwang/projects/hunyaun_sn/0805_数据集自动化构建/jieeewwwang/taiji/data"
# find "$directory_path" -type f -exec realpath {} \; | sort -r | awk '{print "\"" $0 "\","}'
# nohup python -u auto_create_v2.py > log.log 2>&1 &
# nohup python -u auto_create_v2moe.py > logmoe.log 2>&1 &

# nohup python -u auto_create_v2.py > log0809_1.log 2>&1 &
# [1] 27993
