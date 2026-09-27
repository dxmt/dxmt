"""The embedding model the examples use, and the dimension that goes with it.

Every example here opened with `OpenAIEmbedding()`, so running any of them needed an API key
and a network call -- for examples whose subject is a vector database, not an embedding
provider. This picks the best model available and says which one it picked:

1. `OpenAIEmbedding` when `OPENAI_API_KEY` is set, because that is what a real application
   uses and its 1,536 dimensions are what the tables in these examples were written for;
2. `sentence-transformers` (all-MiniLM-L6-v2, 384 dimensions) when it is installed -- a real
   model, locally, no key;
3. otherwise a hashed bag of words, which is not a language model and is not pretending to be
   one: it maps each word to a bucket, so cosine similarity tracks word overlap. Enough to
   watch retrieval work end to end, and deterministic, so an example prints the same thing
   twice.

A table's dimension is fixed when it is created, so use `EMBED_DIMENSION` for it -- and note
that switching models means a new table name, or a table that refuses the new vectors.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any, List

from llama_index.core.base.embeddings.base import BaseEmbedding

_WORD = re.compile(r"[a-z0-9]+")
_LOCAL_DIMENSION = 256


class _BagOfWords(BaseEmbedding):
    """A deterministic, dependency-free stand-in for an embedding model."""

    def __init__(self, dimension: int = _LOCAL_DIMENSION, **kwargs: Any) -> None:
        super().__init__(model_name=f"bag-of-words-{dimension}", **kwargs)
        self._dimension = dimension

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self._dimension
        for word in _WORD.findall(text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self._dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            # The server refuses a zero vector: it has no direction to compare against.
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed(text)


def default_embed_model() -> BaseEmbedding:
    """The best embedding model available here, printing which one that is."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding

            print("embeddings: OpenAI text-embedding-ada-002 (1536d)")
            return OpenAIEmbedding(model="text-embedding-ada-002")
        except ImportError:
            print("OPENAI_API_KEY is set but llama-index-embeddings-openai is not installed")
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        print("embeddings: sentence-transformers all-MiniLM-L6-v2 (384d), locally")
        return HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except ImportError:
        pass
    print(f"embeddings: a hashed bag of words ({_LOCAL_DIMENSION}d) -- no model, no key. "
          "Set OPENAI_API_KEY or pip install llama-index-embeddings-huggingface for a real one.")
    return _BagOfWords()


def embed_dimension(model: BaseEmbedding) -> int:
    """The width of the vectors `model` produces, which a table has to be created with."""
    if isinstance(model, _BagOfWords):
        return model._dimension
    # Ask the model rather than assume: this is the number a wrong guess turns into a
    # dimension-mismatch error on the first insert.
    return len(model.get_text_embedding("dimension probe"))


def has_llm() -> bool:
    """Whether an example that has to *generate* text can run here."""
    return bool(os.getenv("OPENAI_API_KEY"))
