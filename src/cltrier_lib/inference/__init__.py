import logging
import typing

import pydantic
import requests
import tqdm

from cltrier_lib.inference import schemas


class Pipeline(pydantic.BaseModel):
    model: schemas.MODELS = typing.get_args(schemas.MODELS)[0]
    endpoint: str = "https://inf.cl.uni-trier.de/chat/"

    def __call__(
        self, chat: schemas.Chat, options: schemas.Options = schemas.Options()
    ) -> schemas.Chat:
        response: str = ""

        try:
            response = requests.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "messages": chat.model_dump()["messages"],
                    "options": options.model_dump(),
                },
            ).json()["response"]

        except Exception as e:
            logging.warning(e)

        finally:
            return chat.add_message(schemas.Message(role="assistant", content=response))

    def batch_process(
        self,
        chats: typing.List[schemas.Chat],
        options: schemas.Options = schemas.Options(),
    ) -> typing.List[schemas.Chat]:
        return [self(chat, options) for chat in tqdm.tqdm(chats)]


__all__ = ["Pipeline", "schemas"]
