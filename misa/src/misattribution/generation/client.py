import json
import warnings
from dataclasses import asdict, dataclass, field
from typing import List, Literal, NamedTuple, Optional, Union

import aiohttp
import requests


class Env(NamedTuple):
    VLLM_API_KEY: str
    VLLM_API_URL: str


global_env = Env(
    VLLM_API_KEY="YOUR_API_KEY",
    # VLLM_API_URL="http://11.214.111.184:8081/v1/chat/completions",
    VLLM_API_URL="http://9.91.78.106:8081/v1/chat/completions",
)


@dataclass
class Message:
    role: Union[Literal["system"], Literal["user"], Literal["assistant"]]
    content: str
    finish_reason: Union[Literal["stop"], Literal["length"]] = "stop"


@dataclass
class ChatMessages:
    messages: List[Message] = field(default_factory=list)

    def append(self, message):
        if isinstance(message, str):
            self.messages.append(Message(role="user", content=message))
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
            isinstance(message, (str, Message)) for message in messages
        ):
            chat_message = ChatMessages()
            for message in messages:
                chat_message.append(message)
            return chat_message
        raise TypeError(f"messages type {type(messages)}")

    def as_dict(self):
        return asdict(self)["messages"]


class ChatApi:
    """a simple client, not fully functional"""

    def __init__(self, model, retry=3, env=None) -> None:
        self.model = model
        self.retry = retry
        self.env = env if env else global_env

    def prepare_payload(self, messages, stops: Optional[List[str]], extra_body=None):
        chat_message = ChatMessages.from_types(messages)
        kwargs = {
            "model": self.model,
            "stream": False,
            "seed": 42,
            "max_tokens": 8192,
        }
        if extra_body is None:
            extra_body = {}
        payload = {"messages": chat_message.as_dict(), **kwargs, **extra_body}
        for message in payload["messages"]:
            message.pop("finish_reason")
        return payload

    def iter_chat(self, messages, stops=None):
        payload = self.prepare_payload(messages, stops)
        payload["stream"] = True
        rsp = requests.post(
            self.env.VLLM_API_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.env.VLLM_API_KEY}",
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
                            content += data["choices"][0]["delta"]["content"]
                            yield content
                            break
                delta = data["choices"][0]["delta"]
                if "role" in delta:
                    assert delta["role"] == "assistant"
                if "content" in delta:
                    content += delta["content"]
                    yield content

    def get_headers(self):
        if self.env.VLLM_API_KEY:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.env.VLLM_API_KEY}",
            }
        else:
            headers = {"Content-Type": "application/json"}
        return headers

    def chat(self, messages, raw_json=False, stops=None, extra_body=None):
        payload = self.prepare_payload(messages, stops, extra_body=extra_body)
        error = None
        for retry in range(self.retry):
            try:
                rsp = requests.post(
                    self.env.VLLM_API_URL, json=payload, headers=self.get_headers()
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
                self.env.VLLM_API_URL,
                json=payload,
                headers=self.get_headers(),
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
                role="assistant",
                content=rsp["choices"][0]["message"]["content"],
                finish_reason=finish_reason,
            )
            return message

    async def async_iter_chat(self, messages, raw_json=False, stops=None):
        payload = self.prepare_payload(messages, stops=stops)
        payload["stream"] = True
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.env.VLLM_API_URL,
                json=payload,
                headers=self.get_headers(),
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
  api = ChatApi(
      "qwen2.5",
      env=Env(
          "YOUR_API_KEY",
          "YOUR_VLLM_BASE_URL/chat/completions",
      ),
  )
  rsp = api.chat("你好")
  print(rsp.content)

  
