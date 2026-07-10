import json
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from typing import List, Literal, Union

import aiohttp
import requests


@dataclass
class Message:
    role: Union[Literal["system"], Literal["user"], Literal["assistant"]]
    content: str


@dataclass
class ChatMessages:
    messages: List[Message] = field(default_factory=list)

    def append(self, message):
        if isinstance(message, str):
            self.messages.append(Message("user", message))
        elif isinstance(message, Message):
            self.messages.append(message)
        else:
            raise TypeError(f"message type not support :{type(message)}")

    @staticmethod
    def from_types(messages: Union[Message, str, List[Message]]):
        if isinstance(messages, ChatMessages):
            return messages
        if isinstance(messages, (Message, str)):
            chat_message = ChatMessages()
            chat_message.append(messages)
            return chat_message
        if isinstance(messages, list) and all(
            isinstance(message) for message in messages
        ):
            chat_message = ChatMessages(messages)
            return chat_message
        raise TypeError(f"messages type {type(messages)}")


class HunyuanUniversalApi:
    VALUES = [
        {
            "url": "YOUR_CHAT_URL",
            "wsid": "YOUR_WSID",
            "auth": "YOUR_EMBEDDING_TOKEN",
        },
        {
            "url": "YOUR_CHAT_URL",
            "wsid": "",
            "auth": "lG2tL6fitGU1J8hTjye4ID5JGrjn3Doc",
        },
    ]
    CONFIGS = {
        "hy_large_dpo_exp59_step585_new": VALUES[0],
        "hy-t2t-mmdl-30b-turbo-fp8-hpc-online": VALUES[0],
        "hunyuan": VALUES[1],
        "hunyuan-turbo": VALUES[1],
    }

    def __init__(
        self,
        model="hy-t2t-mmdl-30b-turbo-fp8-hpc-online",
        retry=3,
    ) -> None:
        self.model = model
        self.config = self.CONFIGS[model]
        self.retry = retry

    def prepare_payload(self, messages, stops):
        chat_message = ChatMessages.from_types(messages)
        kwargs = {
            "model": self.model,
            "output_seq_len": 4096,
            "max_input_seq_len": 28672,
            "stream": False,
            "random_seed": 9969,
        }
        if stops:
            kwargs["stop"] = stops

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个由腾讯开发的有用的人工智能助手，你的名字是“腾讯元宝”，简称“元宝”，你的英文名是“Tencent Yuanbao”，你乐于帮助大家解答问题。",
                },
                *asdict(chat_message)["messages"],
            ],
            **kwargs,
        }
        payload["query_id"] = "test_query_id_" + str(uuid.uuid4())
        return payload

    def stream_chat(self, messages, stops=None):
        content = ""
        for content in self.iter_chat(messages, stops):
            continue
        message = Message(role="assistant", content=content)
        return message

    def iter_chat(self, messages, stops=None):
        payload = self.prepare_payload(messages, stops)
        payload["stream"] = True
        rsp = requests.post(
            self.config["url"],
            json=payload,
            headers={
                "Authorization": f"Bearer {self.config['auth']}",
                "Wsid": self.config["wsid"],
            },
            stream=True,
        )
        content = ""
        for chunk in rsp.iter_lines(delimiter=b"\n", chunk_size=4096):
            if chunk:
                data = json.loads(chunk.decode("utf8")[6:])
                if "error" in data:
                    raise ValueError(json.dumps(data, indent=2, ensure_ascii=False))
                if "finish_reason" in data["choices"][0]:
                    finish_reason = data["choices"][0]["finish_reason"]
                    if finish_reason:
                        if finish_reason != "stop":
                            raise RuntimeError(f"finish with {finish_reason}")
                        else:
                            break
                delta = data["choices"][0]["delta"]
                if "role" in delta:
                    assert delta["role"] == "assistant"
                if "content" in delta:
                    content += delta["content"]
                    yield content

    def chat(self, messages, raw_json=False, stops=None):
        payload = self.prepare_payload(messages, stops)
        error = None
        for retry in range(self.retry):
            try:
                payload["query_id"] = "test_query_id_" + str(uuid.uuid4())
                rsp = requests.post(
                    self.config["url"],
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.config['auth']}",
                        "Wsid": self.config["wsid"],
                    },
                )
                if rsp.status_code != 200:
                    raise ValueError(rsp.content.decode("utf8"))
                rsp = rsp.json()
                if "error" in rsp:
                    raise ValueError(f"error: {rsp}")
                else:
                    error = None
                    break
            except Exception as e:
                if retry == self.retry - 1:
                    error = e
            if error:
                raise error

        finish_reason = rsp["choices"][0]["finish_reason"]
        if finish_reason != "stop":
            warnings.warn(f"message finish reason : {finish_reason}")
        if raw_json:
            return rsp
        else:
            message = Message(
                role="assistant", content=rsp["choices"][0]["message"]["content"]
            )
            return message

    async def async_chat(self, messages, raw_json=False, stops=None):
        payload = self.prepare_payload(messages, stops=stops)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config["url"],
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.config['auth']}",
                    "Wsid": self.config["wsid"],
                },
            ) as rsp:
                if rsp.status != 200:
                    raise ValueError(rsp.content.decode("utf8"))
                rsp = await rsp.json()
        if "error" in rsp:
            raise ValueError(f"error: {rsp}")

        finish_reason = rsp["choices"][0]["finish_reason"]
        if finish_reason != "stop":
            warnings.warn(f"message finish reason : {finish_reason}")
        if raw_json:
            return rsp
        else:
            message = Message(
                role="assistant", content=rsp["choices"][0]["message"]["content"]
            )
            return message

    async def async_iter_chat(self, messages, raw_json=False, stops=None):
        payload = self.prepare_payload(messages, stops=stops)
        payload["stream"] = True
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config["url"],
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.config['auth']}",
                    "Wsid": self.config["wsid"],
                },
            ) as rsp:
                content = ""
                async for chunk in rsp.content:
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    data = json.loads(chunk.decode("utf8")[6:])
                    if "error" in data:
                        raise ValueError(json.dumps(data, indent=2, ensure_ascii=False))
                    if "finish_reason" in data["choices"][0]:
                        finish_reason = data["choices"][0]["finish_reason"]
                        if finish_reason:
                            if finish_reason != "stop":
                                raise RuntimeError(f"finish with {finish_reason}")
                            else:
                                break
                    delta = data["choices"][0]["delta"]
                    if "role" in delta:
                        assert delta["role"] == "assistant"
                    if "content" in delta:
                        content += delta["content"]
                        yield content


if __name__ == "__main__":
    api = HunyuanUniversalApi()
    print(api.chat("你好"))
