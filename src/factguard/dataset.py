import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from hashlib import md5
from typing import Any, Dict, Iterable, Literal, Union
import jsonlines

from factguard import env


@dataclass
class Example:
    document: str

    @cached_property
    def unique_id(self):
        return md5(self.document.encode("utf8")).hexdigest()

    @cached_property
    def len_range(self):
        k = len(self.document)
        length = 4
        while length * 1024 < k:
            length *= 2
        return f"({length//2}k-{length}k)"


@dataclass
class Meta:
    lang: Union[Literal["en"], Literal["zh"]]
    domain: Union[Literal["common"], Literal["paper"], Literal["pattent"]]


class Dataset(ABC):
    def __init__(self, max_length=128 * 1024, *, min_length=2048) -> None:
        self.max_length = max_length
        self.min_length = min_length

    @abstractmethod
    def examples(self) -> Iterable[Example]: ...

    def __iter__(self):
        def is_valid(ex: Example):
            return self.min_length <= len(ex.document) <= self.max_length

        for example in filter(is_valid, self.examples()):
            yield example

    @classmethod
    def metadata(cls):
        return {"lang": cls.meta.lang, "domain": cls.meta.domain}


class LongAlpaca(Dataset):
    meta = Meta("en", "paper")

    def __init__(self) -> None:
        super().__init__()
        self.ids = set()

    def examples(self) -> Iterable[Dict[str, Any]]:
        def parse_doc(text: str) -> str:
            try:
                text = text.split("\n", 1)[1].rsplit("paper ends", 2)[0]
                if "paper begins" not in text[:100]:
                    return ""
                return text.split("\n", 1)[1].strip() + " paper ends."
            except Exception:
                return ""

        def transform(raw):
            document = parse_doc(raw["instruction"])
            return Example(document)

        paths = [
            env.MISA_DATA_DIR.joinpath("docs", "LongAlpaca-12k", "LongAlpaca-12k.json"),
            env.MISA_DATA_DIR.joinpath(
                "docs", "LongAlpaca-16k-length", "LongAlpaca-16k-length.json"
            ),
        ]
        for path in paths:
            with path.open("r", encoding="utf8") as fp:
                for raw in json.load(fp):
                    example = transform(raw)
                    if example.unique_id in self.ids:
                        continue
                    self.ids.add(example.unique_id)
                    yield example


class AncientBook(Dataset):
    meta = Meta("zh", "book")

    def __init__(self, max_length=32 * 1024, * , min_length=2048) -> None:
        super().__init__(max_length, min_length=min_length)

    def examples(self) -> Iterable[Example]:
        for example in jsonlines.open(
            env.MISA_DATA_DIR / "docs" / "chinese_book.jsonl", "r"
        ):
            yield Example(example["text"])
        for filepath in env.MISA_DATA_DIR.joinpath("docs", "shu", "books").glob(
            "**/*.txt"
        ):
            content = filepath.read_text()
            yield Example(content)



class Gutenberg(Dataset):
    meta = Meta("en", "book")

    def __init__(self, max_length=128 * 1024, *, min_length=2048):
        super().__init__(max_length, min_length=min_length)

    def examples(self):
        data_dir = env.MISA_DATA_DIR.joinpath("docs", "gutenberg_english")
        for example in jsonlines.open(data_dir / "train.jsonl", "r"):
            yield Example(example["text"])


class PileOfLaw(Dataset):
    meta = Meta("en", "law")

    def __init__(self, max_length=128 * 1024, *, min_length=2048):
        super().__init__(max_length, min_length=min_length)

    def examples(self):
        data_dir = env.MISA_DATA_DIR.joinpath("docs", "pile-of-law")
        for example in jsonlines.open(data_dir / "train.jsonl", "r"):
            yield Example(example["text"])


class ChineseLaw(Dataset):
    meta = Meta("zh", "law")

    def __init__(self, max_length=128 * 1024, *, min_length=2048):
        super().__init__(max_length, min_length=min_length)

    def examples(self):
        data_dir = env.MISA_DATA_DIR.joinpath("docs")
        for example in jsonlines.open(
            data_dir / "tigerbot-laws-plugin-merge.json", "r"
        ):
            yield Example(example["content"])
        for example in jsonlines.open(data_dir / "merge_over_8k.jsonl", "r"):
            yield Example(example["content"])
        for example in jsonlines.open(
            data_dir / "law_unsuper_longtext_train_1_of_1_casenum_6473.json", "r"
        ):
            yield Example(example["content"])


class LongDataSkyPile(Dataset):
    meta = Meta("zh", "common")

    def __init__(self, max_length=32 * 1024) -> None:
        super().__init__(max_length)

    def examples(self) -> Iterable[Example]:
        data_dir = env.MISA_DATA_DIR.joinpath("docs", "LongData-Corpus", "LongData_zh")
        for example in jsonlines.open(
            data_dir.joinpath("SkyPile_大于16k字_9720条.json"), "r"
        ):
            yield Example(example["text"])


class LongDataPattent(Dataset):
    meta = Meta("zh", "pattent")

    def __init__(self, max_length=32 * 1024) -> None:
        super().__init__(max_length)

    def examples(self) -> Iterable[Example]:
        data_dir = env.MISA_DATA_DIR.joinpath("docs", "LongData-Corpus", "LongData_zh")
        for example in jsonlines.open(
            data_dir.joinpath("万卷-专利-16k-16715条.json"), "r"
        ):
            yield Example(example["text"])


class LongDataEnWiki(Dataset):
    meta = Meta("en", "wiki")

    def __init__(self, max_length=32 * 1024) -> None:
        super().__init__(max_length)

    def examples(self) -> Iterable[Example]:
        data_dir = env.MISA_DATA_DIR.joinpath("docs", "LongData-Corpus", "LongData_en")
        for example in jsonlines.open(
            data_dir.joinpath("RedPajamaWikipedia-16k-6900条.json"), "r"
        ):
            yield Example(example["text"])


class LongBench(Dataset):
    meta = Meta("zh", "long_bench")

    def __init__(self, max_length=32 * 1024) -> None:
        super().__init__(max_length)

    def examples(self) -> Iterable[Example]:
        data_dir = env.MISA_DATA_DIR.joinpath("docs", "LongBench", "data")
        for example in jsonlines.open(data_dir.joinpath("dureader.jsonl"), "r"):
            yield Example(example["context"])


if __name__ == "__main__":
    from collections import Counter

    dataset = LongDataEnWiki(128 * 1024)
    c = Counter()
    for example in dataset:
        c[example.len_range] += 1
    print(c)
