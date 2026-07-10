import warnings
from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional, Union

import requests

URL = "http://9.206.39.72:8081/v1/chat/completions"


@dataclass
class Message:
    role: Union[Literal["system"], Literal["user"], Literal["assistant"]]
    content: str


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
            isinstance(message) for message in messages
        ):
            chat_message = ChatMessages(messages)
            return chat_message
        raise TypeError(f"messages type {type(messages)}")

    def as_dict(self):
        return asdict(self)["messages"]


class ChatApi:
    """a simple client, not fully functional"""

    def __init__(self, model, retry=3) -> None:
        self.model = model
        self.retry = retry

    def prepare_payload(self, messages, stops: Optional[List[str]]):
        chat_message = ChatMessages.from_types(messages)
        kwargs = {
            "model": self.model,
            "stream": False,
            "seed": 42,
        }
        if stops:
            kwargs["stop"] = stops

        payload = {
            "messages": chat_message.as_dict(),
            **kwargs,
        }
        return payload

    def get_headers(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_API_KEY",
        }
        return headers

    def chat(self, messages, raw_json=False, stops=None):
        payload = self.prepare_payload(messages, stops)
        error = None
        for retry in range(self.retry):
            try:
                rsp = requests.post(URL, json=payload, headers=self.get_headers())
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


if __name__ == "__main__":
    api = ChatApi("qwen2.5")
    print(api.chat("你好"))
