"""DXMT as CrewAI knowledge storage.

``Knowledge`` hands this class chunks of text (``save``) and asks it text questions
(``search``), so the embedding is this layer's business. Two ways to get it:

* ``embedding={"model": ...}`` has the **server** embed: the table is created with that model
  (any OpenAI-compatible ``/embeddings`` endpoint through ``base_url``), chunks go in as
  ``text`` and queries as ``text``. ``embedding="server"`` is the same with the model the
  server itself was started with (``DXMT_EMBEDDING_MODEL``). No embedding model runs in the
  agent's process.
* ``embedder=callable`` embeds **here**: a function from a list of strings to a list of
  vectors, which is what CrewAI's own ``build_embedder`` returns.

With ``enable_full_text_search`` a query is answered by hybrid search (vector + BM25 fused on
the server) rather than vector search alone.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any, Callable, Optional, Union

from crewai.knowledge.storage.base_knowledge_storage import BaseKnowledgeStorage
from crewai.rag.types import SearchResult
from dxmt import DXMTClient
from dxmt.exceptions import DXMTError
from pydantic import ConfigDict, Field, PrivateAttr


def _cond(key: str, type_: str, value: Any) -> dict[str, Any]:
    return {"key": key, "operation": {"type": type_, "value": value}}


class DXMTKnowledgeStorage(BaseKnowledgeStorage):
    """A CrewAI ``BaseKnowledgeStorage`` over one DXMT table."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    table_name: str = Field(default="crewai_knowledge")
    host: str = "localhost"
    port: int = 9999
    ssl: bool = False
    api_key: Optional[str] = Field(default=None, exclude=True, repr=False)
    embedding: Optional[Union[dict[str, Any], str]] = Field(
        default=None,
        description='Server-side embedding: {"model": ..., "base_url": ...} for the table, or "server" for the model the server was started with',
    )
    embedder: Optional[Callable[[list[str]], list[list[float]]]] = Field(
        default=None, exclude=True, description="Client-side embedding function"
    )
    dimension: Optional[int] = None
    distance_metric: str = "cosine"
    enable_full_text_search: bool = False
    text_analyzer: str = "standard"
    _client: Optional[DXMTClient] = PrivateAttr(default=None)
    _ready: bool = PrivateAttr(default=False)

    def __init__(self, client: Optional[DXMTClient] = None, **data: Any) -> None:
        super().__init__(**data)
        if self.embedding is None and self.embedder is None:
            raise ValueError(
                "DXMTKnowledgeStorage needs a way to embed: embedding={'model': ...} for the "
                "server, or embedder=fn for this process"
            )
        self._client = client or DXMTClient(host=self.host, port=self.port, ssl=self.ssl, api_key=self.api_key)

    @property
    def client(self) -> DXMTClient:
        assert self._client is not None
        return self._client

    def _exists(self) -> bool:
        try:
            self.client.get_table(self.table_name)
            return True
        except DXMTError as e:
            if "TABLE_NOT_FOUND" in str(e):
                return False
            raise

    def _ensure(self, dim: Optional[int]) -> None:
        if self._ready or self._exists():
            self._ready = True
            return
        text_fields = None
        if self.enable_full_text_search:
            text_fields = [{"name": "content", "source": {"metadata_field": "content"},
                            "analyzer": self.text_analyzer, "boost": 1.0}]
        self.client.create_table(
            name=self.table_name,
            dimension=None if self.embedding else (dim or self.dimension),
            distance_metric=self.distance_metric,
            enable_full_text_search=self.enable_full_text_search,
            text_fields=text_fields,
            embedding=self.embedding if isinstance(self.embedding, dict) else None,
        )
        self._ready = True

    # ---- protocol ------------------------------------------------------------------------

    def save(self, documents: list[str]) -> None:
        docs = [d for d in documents if d and d.strip()]
        if not docs:
            return
        rows: list[dict[str, Any]] = []
        if self.embedder is not None:
            vectors = self.embedder(docs)
            self._ensure(len(vectors[0]) if vectors else None)
            for d, v in zip(docs, vectors):
                rows.append({"id": self._id(d), "vector": list(v), "payload": {"content": d}})
        else:
            self._ensure(None)
            # The server stores the text in the payload field `text`; `content` keeps the
            # same shape as the embedder path so search reads one field.
            rows = [{"id": self._id(d), "text": d, "payload": {"content": d}} for d in docs]
        self.client.insert_records(self.table_name, rows)

    @staticmethod
    def _id(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def search(
        self,
        query: list[str],
        limit: int = 5,
        metadata_filter: Optional[dict[str, Any]] = None,
        score_threshold: float = 0.6,
    ) -> list[SearchResult]:
        if not self._exists():
            return []
        f = None
        if metadata_filter:
            f = {"must": [_cond(k, "Equal", v) for k, v in metadata_filter.items()]}
        best: dict[str, SearchResult] = {}
        queries = [q for q in query if q and q.strip()]
        vectors = self.embedder(queries) if self.embedder is not None and queries else None
        for i, q in enumerate(queries):
            if self.enable_full_text_search:
                hits = self.client.hybrid_search(
                    self.table_name, vector=list(vectors[i]) if vectors else None, text=q, k=limit, filter=f
                )
            elif vectors:
                hits = self.client.search(self.table_name, vector=list(vectors[i]), k=limit, filter=f)
            else:
                hits = self.client.search(self.table_name, text=q, k=limit, filter=f)
            for h in hits:
                score = float(h.get("score", 0.0))
                if score < score_threshold:
                    continue
                p = h.get("payload") or {}
                content = p.get("content") or p.get("text") or ""
                meta = {k: v for k, v in p.items() if k not in ("content", "text")}
                prev = best.get(h["id"])
                if prev is None or prev["score"] < score:
                    best[h["id"]] = SearchResult(id=h["id"], content=content, metadata=meta, score=score)
        return sorted(best.values(), key=lambda r: r["score"], reverse=True)[:limit]

    def reset(self) -> None:
        try:
            self.client.delete_table(self.table_name)
        except DXMTError as e:
            if "TABLE_NOT_FOUND" not in str(e):
                raise
        self._ready = False

    async def asave(self, documents: list[str]) -> None:
        await asyncio.to_thread(self.save, documents)

    async def asearch(
        self,
        query: list[str],
        limit: int = 5,
        metadata_filter: Optional[dict[str, Any]] = None,
        score_threshold: float = 0.6,
    ) -> list[SearchResult]:
        return await asyncio.to_thread(self.search, query, limit, metadata_filter, score_threshold)

    async def areset(self) -> None:
        await asyncio.to_thread(self.reset)
