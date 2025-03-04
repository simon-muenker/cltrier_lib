import typing

import pydantic
import rich


ROLES = typing.Literal["user", "assistant", "system"]

MODELS = typing.Literal[
    # LLama (MetaAI)
    "llama3.1:8b-instruct-q6_K",
    "llama3.3:70b-instruct-q6_K",
    # Mi(s/x)tral (Mistral AI)
    "mistral:7b-instruct-v0.2-q6_K",
    "mixtral:8x7b-instruct-v0.1-q6_K",
    # Phi (Mircosoft)
    "phi4:latest",
    # Gemma (Google)
    "gemma2:27b-instruct-q6_K",
    # Deepseek
    "deepseek-r1:70b",
]


class Message(pydantic.BaseModel):
    role: ROLES
    content: str

    def pprint(self) -> None:
        rich.print(self)

    def format_content(self, **args) -> "Message":
        return Message(role=self.role, content=self.content.format(**args))


class Chat(pydantic.BaseModel):
    messages: typing.List[Message]

    def __getitem__(self, index: int) -> Message:
        return self.messages[index]

    def __iter__(self) -> typing.Iterator[Message]:
        return iter(self.messages)

    def add_message(self, message: Message) -> "Chat":
        return Chat(messages=[*self, message])

    def remove_message(self, index: int) -> "Chat":
        return Chat(messages=[self[:index] + self[index + 1 :]])  # type: ignore[index]

    def to_json(self, path: str) -> None:
        open(path, "w").write(self.model_dump_json(indent=4))

    def pprint(self) -> None:
        rich.print(self)


class Options(pydantic.BaseModel):
    seed: int = 42
    temperature: float = 0.8
    num_predict: int = 128
