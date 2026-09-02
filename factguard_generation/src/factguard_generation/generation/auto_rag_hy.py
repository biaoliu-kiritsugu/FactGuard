import json
import jsonlines
import pathlib
import re
from typing import Any, Callable, Dict, Optional, Union, List, Mapping

import jsonlines
import tqdm
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


import logging

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra
from functools import cached_property
from itertools import islice
import base64
import numpy as np

import uuid
import os

logger = logging.getLogger(__name__)

TAIJI_EMBEDDING_URL = os.environ.get(
    "EMBEDDING_BASE_URL", "http://127.0.0.1:8081"
)


def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

from typing import Any, List, Mapping
import json

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra
from functools import cached_property


class HunyuanEmbPrivate(BaseModel, Embeddings):
    addr: str = TAIJI_EMBEDDING_URL
    show_progress: bool = False
    """Whether to show a tqdm progress bar. Must have `tqdm` installed."""

    @cached_property
    def _model_id(self):
        return "hunyuan-embedding-public"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"addr": self.addr, "model_id": self._model_id}

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using an Ollama deployed embedding model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        assert isinstance(texts, List)
        if self.show_progress:
            import tqdm

            iter = tqdm.tqdm(texts, total=len(texts), desc="embed")
        else:
            iter = texts

        embeddings = []
        for batch in iter:
            batch = "Create a retrieval embedding for the following passage: " + batch
            rsp = requests.post(
                self.addr,
                json={
                    "text": batch,
                },
                headers={
                    "Authorization": f"Bearer {os.environ.get('EMBEDDING_API_KEY', '')}",
                    "Variant": "retrieval-v1-offline",
                },
            )
            try:
                rsp = rsp.json()
            except json.decoder.JSONDecodeError:
                raise ValueError(rsp.content.decode("utf8"))
            embeddings.extend([rsp["result"]["embed"]])

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using a Ollama deployed embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        rsp = requests.post(
            self.addr,
            json={
                "text": "Create a retrieval embedding for the following query: " + text,
            },
            headers={
                "Authorization": f"Bearer {os.environ.get('EMBEDDING_API_KEY', '')}",
                "Variant": "retrieval-v1-offline",
            },
        )
        try:
            rsp = rsp.json()
        except json.decoder.JSONDecodeError:
            raise ValueError(rsp.content.decode("utf8"))
        embed = rsp["result"]["embed"]
        return embed


# hunyuan_embeddigns = HunyuanEmbPrivate()


class HunyuanEmbeddings(BaseModel, Embeddings):
    addr: str = TAIJI_EMBEDDING_URL
    show_progress: bool = False
    """Whether to show a tqdm progress bar. Must have `tqdm` installed."""

    @cached_property
    def _model_id(self):
        return "hunyuan-embedding-public"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"addr": self.addr, "model_id": self._model_id}

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using an Ollama deployed embedding model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        if self.show_progress:
            import tqdm

            iter = tqdm.tqdm(texts, total=len(texts), desc="embed")
        else:
            iter = texts

        embeddings = []
        for batch in batched(iter, 1):
            batch = [
                "Create a retrieval embedding for the following passage: " + item
                for item in batch
            ]
            rsp = requests.post(
                self.addr,
                json={
                    "query_id": str(uuid.uuid4()),
                    "model": self._model_id,
                    "input": batch[0],
                },
                headers={
                    "Authorization": f"Bearer {os.environ.get('EMBEDDING_API_KEY', '')}",
                },
            )
            try:
                rsp = rsp.json()
            except json.decoder.JSONDecodeError:
                raise ValueError(rsp.content.decode("utf8"))
            embeddings.extend([emb["embedding"] for emb in rsp["data"]])

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using a Ollama deployed embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        rsp = requests.post(
            self.addr,
            json={
                "query_id": str(uuid.uuid4()),
                "model": self._model_id,
                "input": "Create a retrieval embedding for the following query: " + text,
            },
            headers={
                "Authorization": f"Bearer {os.environ.get('EMBEDDING_API_KEY', '')}",
            },
        )
        try:
            rsp = rsp.json()
        except json.decoder.JSONDecodeError:
            raise ValueError(rsp.content.decode("utf8"))
        embed = rsp["data"][0]["embedding"]
        return embed


class TaijiEmbeddings(BaseModel, Embeddings):
    addr: str

    show_progress: bool = False
    """Whether to show a tqdm progress bar. Must have `tqdm` installed."""

    @cached_property
    def _model_id(self):
        model_id = requests.get(self.addr + "/model_id").json()
        return model_id

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"addr": self.addr, "model_id": self._model_id}

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using an Ollama deployed embedding model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        if self.show_progress:
            import tqdm

            iter = tqdm.tqdm(texts, total=len(texts), desc="embed")
        else:
            iter = texts

        embeddings = []
        for batch in batched(iter,32):
            rsp = requests.post(self.addr + "/embed_docs", json={"docs": batch}).json()
            embed = (
                np.frombuffer(
                    base64.b64decode(rsp["embedding"].encode("ascii")),
                    dtype=np.float32,
                )
                .reshape(rsp["shape"])
                .tolist()
            )
            embeddings.extend(embed)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using a Ollama deployed embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        rsp = requests.post(self.addr + "/embed_query", json={"query": text}).json()
        embed = (
            np.frombuffer(
                base64.b64decode(rsp["embedding"].encode("ascii")),
                dtype=np.float32,
            )
            .reshape(rsp["shape"])
            .tolist()
        )

        return embed


# taiji_embeddings = TaijiEmbeddings(addr=TAIJI_EMBEDDING_URL, show_progress=True)
# taiji_embeddings.show_progress = False
taiji_embeddings = HunyuanEmbPrivate()


class SearchDB:
    def __init__(self, faiss: FAISS) -> None:
        self.faiss = faiss

    @classmethod
    def from_binary(cls, path):
        faiss = FAISS.load_local(
            path, taiji_embeddings, allow_dangerous_deserialization=True
        )
        return cls(faiss)

    @classmethod
    def from_texts(cls, texts):
        faiss = FAISS.from_texts(texts, taiji_embeddings)
        return cls(faiss)

    @classmethod
    def from_documents(cls, documents):
        faiss = FAISS.from_documents(documents, taiji_embeddings)
        return cls(faiss)

    def save(self, path):
        self.faiss.save_local(path)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Union[Callable, Dict[str, Any]]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ):
        return self.faiss.similarity_search(
            query=query, k=k, filter=filter, fetch_k=fetch_k, kwargs=kwargs
        )

    def similarity_search_with_relevance_scores(self, query: str, k: int = 4):
        return self.faiss.similarity_search_with_relevance_scores(query, k)


class ExampleGenerator:
    def __init__(self, original_file) -> None:
        self.original_file = original_file
    def transform(self, example):
        return example

    def __iter__(self):
        def counting(reader):
            for i, example in enumerate(reader, 1):
                if i % 100 == 0:
                    print(f"{i} examples passed")
                yield example

        with jsonlines.open(self.original_file, "r") as reader:
            for example in map(self.transform, counting(reader)):
                yield example


class Pipeline:
    K_IN_TEXT = 2 * 1024
    LENGTH_LIMIT = 256 * K_IN_TEXT * 10

    PROMPT = """system:You are a helpful, honest, and harmless assistant. Answer
only from the supplied context and use the language of the user's question.</s>
user:Answer the question using only the context below. If the context does not
provide sufficient evidence, decline to answer and explain why.
Context:
{context}
Question:
{question}
</s>
assistant:"""

    def __init__(self, example_generator) -> None:
        self.example_generator = example_generator

    def create_prompt_for_rag(self, example, doc):
        prompt = self.PROMPT.format(
            context=doc.page_content, question=example["question"]
        )
        return prompt

    def judge_answerable(self, prompts, completions):
        for prompt, completion in zip(prompts, completions):
            try:
                item = json.loads(completion)
                if item.get("answerable") is True:
                    return prompt, completion
            except json.decoder.JSONDecodeError:
                if '"answerable":true' in completion.replace(" ", "").lower():
                    return prompt, completion
        return False, False

    def run(self, output_path):
            ftxt = open(output_path, 'w', encoding='utf-8')
            i=0
            for example in tqdm.tqdm(self.example_generator, desc="proc"):
                i+=1
                # if i<=85:
                #     continue
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1024,
                    chunk_overlap=512,
                    length_function=len,
                    is_separator_regex=False,
                )
                refuse_doc = example["meta"]["doc"].replace(example["meta"]["select_segment"],"")
                texts = text_splitter.create_documents([refuse_doc])
                searcher = SearchDB.from_documents(texts)
                
                docs = searcher.similarity_search_with_relevance_scores(
                    example["meta"]["question"], k=5
                )
                retrails = [doc[0].page_content for doc in docs]
                example["meta"]["retrails"] = retrails
                json_dict_str= json.dumps(example, ensure_ascii=False)
                ftxt.write("%s\n"%json_dict_str)
                ftxt.flush()
            ftxt.close()


    def run_item(self):
                example = self.example_generator
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2048,
                    chunk_overlap=512,
                    length_function=len,
                    is_separator_regex=False,
                )
                refuse_doc = example["meta"]["doc"].replace(example["meta"]["select_segment"],"")
                texts = text_splitter.create_documents([refuse_doc])
                searcher = SearchDB.from_documents(texts)
                
                docs = searcher.similarity_search_with_relevance_scores(
                    example["meta"]["question"], k=5
                )
                retrails = [doc[0].page_content for doc in docs]
                return retrails



# Example usage should provide local input/output paths through the caller.
# file_in = jsonlines.open(input_file, "r")
# # open(input_file, 'r', encoding='utf-8').readlines()

# pipeline = Pipeline(file_in)

# pipeline.run(out_file)
