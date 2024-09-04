import logging
import typing

import pydantic
import requests
import tqdm

from cltrier_lib.inference import schemas


class Pipeline(pydantic.BaseModel):
    model: schemas.Models = typing.get_args(schemas.Models)[0]
    endpoint: str = "https://inf.cl.uni-trier.de/chat/"

    def __call__(self, chat: schemas.Chat) -> schemas.Chat:
        response: str = ""

        try:
            response = requests.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "messages": chat.model_dump()["messages"],
                },
            ).json()["response"]

        except Exception as e:
            logging.warning(e)

        finally:
            return chat.add_message(schemas.Message(role="assistant", content=response))

    def batch_process(
        self, chats: typing.List[schemas.Chat]
    ) -> typing.List[schemas.Chat]:
        return [self(chat) for chat in tqdm.tqdm(chats)]


__all__ = ["Pipeline", "schemas"]
