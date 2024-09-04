import typing

import pydantic
import rich


Roles = typing.Literal["user", "assistant", "system"]

Models = typing.Literal[
    # LLama (MetaAI)
    "llama3.1:8b-instruct-q6_K",
    "llama2:70b-chat-q6_K",
    "llama3:70b-instruct-q6_K",
    "llama3.1:70b-instruct-q6_K",
    # Mi(s/x)tral (Mistral AI)
    "mistral:7b-instruct-v0.2-q6_K",
    "mixtral:8x22b-instruct-v0.1-q6_K",
    "mixtral:8x7b-instruct-v0.1-q6_K",
    # Phi (Mircosoft)
    "phi3:14b-medium-128k-instruct-q6_K",
    # Gemma (Google)
    "gemma:7b-instruct-q6_K",
    "gemma2:27b-instruct-q6_K",
    # QWEN (Alibaba)
    "qwen:72b-chat-v1.5-q6_K",
    "qwen2:72b-instruct-q6_K",
]


class Message(pydantic.BaseModel):
    role: Roles
    content: str

    def pprint(self) -> None:
        rich.print(self)

    def format_content(self, **args) -> "Message":
        return Message(role=self.role, content=self.content.format(**args))


class Chat(pydantic.BaseModel):
    messages: typing.List[Message]

    def __getitem__(self, index: int):
        return self.messages[index]

    def __iter__(self):
        return iter(self.messages)

    def add_message(self, message: Message) -> "Chat":
        return Chat(messages=[*self, message])

    def remove_message(self, index: int) -> "Chat":
        return Chat(messages=[self[:index] + self[index + 1 :]])  # type: ignore[index]

    def to_json(self, path: str) -> None:
        open(path, "w").write(self.model_dump_json(indent=4))

    def pprint(self) -> None:
        rich.print(self)
