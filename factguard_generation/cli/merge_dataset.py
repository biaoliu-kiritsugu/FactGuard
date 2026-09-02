import jsonlines
from factguard_generation.env import FACTGUARD_DATA_DIR


def merge_files(name_pattern, output_file, is_train):
    output_file = FACTGUARD_DATA_DIR.joinpath("split", output_file)
    writer = jsonlines.open(output_file, "w")
    for filename in FACTGUARD_DATA_DIR.joinpath("split").glob(name_pattern):
        print(filename.name)

        def make_example(example, doc, q, a):
            language_instruction = (
                "Answer in English."
                if "_en" in filename.name
                else "Answer in Chinese."
            )
            question = f"""Document:
{doc}

Answer the question using only the document.
Question: {q}
{language_instruction}"""

            out = {
                "source": filename.name,
                "uid": example["uid"],
                "input": question,
                "output": a,
                "origin": example,
            }
            return out

        with jsonlines.open(filename, "r") as reader:
            for i, example in enumerate(reader):
                if "dandian" in filename.name:
                    if is_train and i % 2 == 0 or (not is_train and i % 5 == 0):
                        doc = example["doc"]
                        question = example["问题"]
                        answer = example["改写后答案"]
                        is_positive = True
                    else:
                        doc = example["refuse_doc"]
                        question = example["问题"]
                        answer = example["拒答回复语"]
                        is_positive = False
                elif "impossible" in filename.name:
                    doc = example["doc"]
                    question = example["增加条件的问题"]
                    answer = example["回答增加条件的问题"]
                    is_positive = False
                elif "misattr" in filename.name:
                    if (is_train and i % 2 == 0) or (not is_train and i % 5 == 0):
                        doc = example["doc"]
                        question = example["原始问题"]
                        answer = example["正确答案"]
                        is_positive = True
                    else:
                        doc = example["doc"]
                        question = example["问题"]
                        answer = example["final"]
                        is_positive = False
                new_ex = make_example(example, doc, question, answer)
                new_ex["is_positive"] = is_positive
                writer.write(new_ex)
    writer.close()


merge_files("*jsonl_test.jsonl", "merged_test.jsonl", False)
merge_files("*jsonl_train.jsonl", "merged_train.jsonl", True)
merge_files("*jsonl_dev.jsonl", "merged_dev.jsonl", False)
