"""The embedding model these examples use, and the dimension that goes with it.

Fifteen of the seventeen examples here opened with `OpenAIEmbeddings()`, so the first thing
anyone evaluating DXMT through LangChain met was a missing API key -- for examples whose subject
is a vector database, not an embedding provider. This picks the best model available and says
which one it picked:

1. `OpenAIEmbeddings` when `OPENAI_API_KEY` is set, because that is what a real application uses
   and its 1,536 dimensions are what these examples were written for;
2. `sentence-transformers` (all-MiniLM-L6-v2, 384 dimensions) when it is installed -- a real
   model, locally, no key, no network after the first download;
3. otherwise a hashed bag of words, which is not a language model and does not pretend to be
   one: it maps each word to a bucket, so cosine similarity tracks word overlap. Enough to watch
   retrieval work end to end, and deterministic, so an example prints the same thing twice.

A table's dimension is fixed when it is created, so pass `embed_dimension(model)` to it -- and
note that switching models means a new table name, or a table that refuses the new vectors.

The same file exists for the LlamaIndex package, with the same three tiers, for the same reason.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import List

from langchain_core.embeddings import Embeddings

_WORD = re.compile(r"[a-z0-9]+")
_LOCAL_DIMENSION = 256


class BagOfWords(Embeddings):
    """A deterministic, dependency-free stand-in for an embedding model."""

    def __init__(self, dimension: int = _LOCAL_DIMENSION) -> None:
        self.dimension = dimension

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        for word in _WORD.findall(text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            # The server refuses a zero vector: it has no direction to compare against.
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def default_embeddings() -> Embeddings:
    """The best embedding model available here, printing which one that is."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import OpenAIEmbeddings

            print("embeddings: OpenAI text-embedding-ada-002 (1536d)")
            return OpenAIEmbeddings(model="text-embedding-ada-002")
        except ImportError:
            print("OPENAI_API_KEY is set but langchain-openai is not installed")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        print("embeddings: sentence-transformers all-MiniLM-L6-v2 (384d), locally")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except ImportError:
        pass
    try:
        from sentence_transformers import SentenceTransformer

        print("embeddings: sentence-transformers all-MiniLM-L6-v2 (384d), locally")
        return _SentenceTransformers(SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"))
    except ImportError:
        pass
    print(
        f"embeddings: a hashed bag of words ({_LOCAL_DIMENSION}d) -- no model, no key. Set "
        "OPENAI_API_KEY, or pip install sentence-transformers, for a real one."
    )
    return BagOfWords()


class _SentenceTransformers(Embeddings):
    """`sentence-transformers` without the `langchain-huggingface` wrapper.

    That wrapper is a separate package and is often not installed even where the model is. The
    model is the part that matters, so use it directly when only it is there.
    """

    def __init__(self, model) -> None:
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(list(texts), normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


def embed_dimension(model: Embeddings) -> int:
    """The width of the vectors `model` produces, which a table has to be created with.

    Asked of the model rather than assumed: a wrong guess here is a dimension-mismatch error on
    the first insert, several steps away from the line that caused it.
    """
    if isinstance(model, BagOfWords):
        return model.dimension
    return len(model.embed_query("dimension probe"))


def needs_an_llm(step: str) -> bool:
    """Whether the generation half of an example can run, with one line if it cannot.

    Retrieval is what these examples are about and it runs without a key; generation needs a
    model that writes text. Saying so once, here, keeps every example from re-deciding what to
    print when the key is missing.
    """
    if os.getenv("OPENAI_API_KEY"):
        return True
    print(f"\n[skipped] {step} needs an LLM: set OPENAI_API_KEY to run it.")
    return False
