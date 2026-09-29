"""A deterministic embedding model, so the live tests need no API key.

Fifty tests in this package -- including LangChain's own standard conformance suite -- were
gated on three things at once: `langchain-openai` installed, `OPENAI_API_KEY` set, and that
key's provider actually serving an embedding model. CI satisfies none of them, so none of
those tests had ever run anywhere. A skipped test and an absent one are the same thing.

This is a hashed bag of words. Each word lands in one bucket, so two texts sharing words share
a direction and cosine similarity tracks word overlap. That is enough for what the suite
actually asserts ("the query about cats retrieves the cat document"), and it makes the
assertions deterministic rather than dependent on a model that can be retrained under them.
It is not a language model and is not meant to look like one -- but it is also not
`FakeEmbeddings`, which returns random or constant vectors and would make every retrieval
assertion vacuous.

`tests/local_embedding.py` in the LlamaIndex package is the same idea against that framework's
interface; the two are deliberately the same algorithm so a result can be compared across them.

Set `DXMT_TEST_EMBEDDING=openai` to run the same tests against a real model instead, which
needs `langchain-openai`, a key, and a provider that serves embeddings -- `DXMT_TEST_EMBEDDING_MODEL`
picks which one, because `text-embedding-ada-002` is `OpenAIEmbeddings`' default and many
gateways no longer carry it.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import List

from langchain_core.embeddings import Embeddings

DIMENSION = 256

_WORD = re.compile(r"[a-z0-9]+")


def embed_text(text: str, dimension: int = DIMENSION) -> List[float]:
    vector = [0.0] * dimension
    for word in _WORD.findall(text.lower()):
        bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % dimension
        vector[bucket] += 1.0
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        # A zero vector is refused under cosine, and rightly: it has no direction to compare.
        vector[0] = 1.0
        return vector
    return [v / norm for v in vector]


class LocalEmbedding(Embeddings):
    """`Embeddings` over `embed_text`, usable anywhere a LangChain embedding model is."""

    def __init__(self, dimension: int = DIMENSION) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [embed_text(t, self.dimension) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return embed_text(text, self.dimension)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


def make_embeddings() -> Embeddings:
    """What the fixtures use: local by default, a real model when asked for one."""
    if os.getenv("DXMT_TEST_EMBEDDING", "local").lower() != "openai":
        return LocalEmbedding()
    from langchain_openai import OpenAIEmbeddings

    model = os.getenv("DXMT_TEST_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)


def embedding_dimension() -> int:
    """The width the table has to be created with, for whichever model is in use."""
    if os.getenv("DXMT_TEST_EMBEDDING", "local").lower() != "openai":
        return DIMENSION
    return len(make_embeddings().embed_query("probe"))
