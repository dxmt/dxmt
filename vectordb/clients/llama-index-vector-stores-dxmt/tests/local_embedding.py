"""A deterministic embedding model, so the integration tests need no API key.

The integration suite asked for `OpenAIEmbedding`, which needs a key and a network call, so it
skipped itself everywhere -- including CI. The live path of this package was therefore never
run: every green result came from unit tests against a mocked client.

This is a hashed bag of words. Each word lands in one bucket, so two texts sharing words share
a direction and cosine similarity tracks word overlap. That is enough for the assertions the
suite actually makes ("the query about neural networks retrieves the deep-learning document"),
and it makes them deterministic rather than dependent on a model that can be retrained under
them. It is not a language model and is not meant to look like one: `llama_index`'s own
`MockEmbedding` returns a constant vector, which would make every retrieval assertion vacuous.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, List

from llama_index.core.base.embeddings.base import BaseEmbedding

DIMENSION = 256

_WORD = re.compile(r"[a-z0-9]+")


def embed_text(text: str, dimension: int = DIMENSION) -> List[float]:
    vector = [0.0] * dimension
    for word in _WORD.findall(text.lower()):
        bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % dimension
        vector[bucket] += 1.0
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        # A zero vector is refused by the server, and rightly: it has no direction to compare.
        vector[0] = 1.0
        return vector
    return [v / norm for v in vector]


class LocalEmbedding(BaseEmbedding):
    """`BaseEmbedding` over `embed_text`, usable anywhere an embedding model is."""

    def __init__(self, dimension: int = DIMENSION, **kwargs: Any) -> None:
        super().__init__(model_name=f"local-bag-of-words-{dimension}", **kwargs)
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_query_embedding(self, query: str) -> List[float]:
        return embed_text(query, self._dimension)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return embed_text(query, self._dimension)

    def _get_text_embedding(self, text: str) -> List[float]:
        return embed_text(text, self._dimension)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [embed_text(text, self._dimension) for text in texts]
