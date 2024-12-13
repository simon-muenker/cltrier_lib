import typing

import pydantic
import rich

CATEGORIES = typing.Literal[
    "topics",
    "emotions",
    "sentiment",
    "irony",
    "offensive",
    "hate",
]


class Classification(pydantic.BaseModel):
    sample: str
    results: typing.Dict[str, typing.Dict[str, float]]

    def pprint(self) -> None:
        rich.print(self)

    def filter(self, categories: typing.List[CATEGORIES]) -> "Classification":
        return Classification(
            sample=self.sample,
            results={category: self.results[category] for category in categories},
        )
