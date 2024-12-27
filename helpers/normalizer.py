import json
from dataclasses import dataclass
from typing import List

import requests


@dataclass()
class ExpansionItem:
    nsw: str  # non standard word
    expansion: str
    start: int
    end: int

@dataclass()
class NormalizedText:
    normalized_text: str
    expansion_info: List[ExpansionItem]

    @classmethod
    def from_dict(cls, data: dict):
        expansion_info = []
        for it in data["expansion_info"]:
            expansion_info.append(
                ExpansionItem(
                    nsw=it["nsw"],
                    expansion=it["expansion"],
                    start=it["start"],
                    end=it["end"],
                )
            )
        return cls(
            normalized_text=data["normalised_text"],
            expansion_info=expansion_info,
        )

class Normalizer:
    """TEXT NORMALIZER FOR TEXT PROCESSING"""

    def __init__(
            self,
            endpoint
    ) -> None:
        self.endpoint = endpoint
        self.timeout = 30

    async def process(
        self,
        text: str,
    ):
        """Normalize text"""

    async def __call__(self, text, pauses, client_session):
        """NORMALIZE TEXT ASYNC"""

    def normalize(self, text: str) -> NormalizedText:
        """
        ..warning::
            This sync method is depricated and will be removed
            in the future.
        """
        payload = {"text": text}
        payload_encoded = json.dumps(payload).encode("utf-8")
        response = requests.post(
            self.endpoint,
            data=payload_encoded,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        output = response.json()

        return NormalizedText.from_dict(data=output)
