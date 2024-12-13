import logging
import typing

import pydantic
import requests

from cltrier_lib.classify import schemas


class Pipeline(pydantic.BaseModel):
    endpoint: str = "https://metrics.twon.uni-trier.de/"
    categories: typing.List[schemas.CATEGORIES] = list(
        typing.get_args(schemas.CATEGORIES)
    )

    def __call__(
        self, samples: typing.List[str], threshold: float = 0.5
    ) -> typing.List[schemas.Classification]:
        response: typing.List[typing.Dict] = []

        try:
            response = requests.post(
                self.endpoint,
                json={
                    "samples": samples,
                    "threshold": threshold,
                },
            ).json()["predictions"]

        except Exception as e:
            logging.warning(e)

        finally:
            return [
                schemas.Classification(**prediction).filter(self.categories)
                for prediction in response
            ]


__all__ = ["Pipeline", "schemas"]
