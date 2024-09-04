import logging
import typing

import pydantic
import requests


class Pipeline(pydantic.BaseModel):
    endpoint: str = "https://metrics.twon.uni-trier.de/"

    def __call__(
        self, samples: typing.List[str], theta: float = 0.5
    ) -> typing.List[typing.Dict[str, str | float]]:
        response: typing.List[typing.Dict[str, str | float]] = []

        try:
            response = requests.post(
                self.endpoint,
                json={
                    "samples": samples,
                    "theta": theta,
                },
            ).json()["predictions"]

        except Exception as e:
            logging.warning(e)

        finally:
            return response


__all__ = ["Pipeline"]
