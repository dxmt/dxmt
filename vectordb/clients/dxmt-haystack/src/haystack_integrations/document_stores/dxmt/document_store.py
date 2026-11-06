"""DXMT as a Haystack ``DocumentStore``.

One Haystack document is one DXMT record: the document's ``id`` is the record id, its
``embedding`` the vector, its ``sparse_embedding`` the sparse vector, and the payload holds
``content`` and ``meta`` as they are -- ``meta`` nested, so a Haystack filter on ``meta.page``
is the payload path ``meta.page`` on the server and nothing has to be flattened or renamed.

A document without an embedding is still a document here, as it is in every Haystack store.
The server needs a vector per record, so such a document is written with a fixed unit vector
and ``__has_embedding: false`` in its payload; it comes back with ``embedding=None``, and the
embedding retriever excludes it, so the stand-in vector is never ranked. (A table whose server
side has an embedding model is the other way to take text without vectors; see
``server_embedding``.)

Filters are translated by ``filters.translate`` when the server has a flat form for them and
evaluated here otherwise -- see that module for the line between the two.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from dxmt import DXMTClient
from dxmt.exceptions import DXMTError
from haystack import default_from_dict, default_to_dict
from haystack.dataclasses import Document, SparseEmbedding
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.filters import document_matches_filter

from .filters import translate, validate

logger = logging.getLogger(__name__)

HAS_EMBEDDING = "__has_embedding"


class DXMTDocumentStore:
    """A Haystack document store over a DXMT table.

    Args:
        table_name: The table; made on first use if it does not exist.
        embedding_dim: Vector width, needed only to create the table.
        host, port, ssl, api_key: Where the server is; ignored when ``client`` is given.
        client: A ``DXMTClient`` to use instead of building one.
        distance_metric: ``cosine`` (default), ``euclidean``, ``dot``.
        enable_full_text_search: Index ``content`` with BM25 so ``DXMTBM25Retriever`` and the
            hybrid retriever can answer; ``text_analyzer`` is ``standard`` or ``chinese``.
        server_embedding: ``{"model": ..., "base_url": ..., "api_key_env": ...}`` to have the
            server embed ``content`` on write and queries on read; then documents need no
            ``embedding`` and the embedding retriever takes text.
        recreate_table: Drop the table at construction. For tests and scratch work.
    """

    def __init__(
        self,
        table_name: str = "haystack",
        embedding_dim: Optional[int] = 768,
        *,
        host: str = "localhost",
        port: int = 9999,
        ssl: bool = False,
        api_key: Optional[str] = None,
        client: Optional[DXMTClient] = None,
        distance_metric: str = "cosine",
        enable_full_text_search: bool = False,
        text_analyzer: str = "standard",
        server_embedding: Optional[dict[str, Any]] = None,
        recreate_table: bool = False,
    ) -> None:
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.host, self.port, self.ssl = host, port, ssl
        self.distance_metric = distance_metric
        self.enable_full_text_search = enable_full_text_search
        self.text_analyzer = text_analyzer
        self.server_embedding = server_embedding
        self._client = client or DXMTClient(host=host, port=port, ssl=ssl, api_key=api_key)
        self._table_ready = False
        if recreate_table:
            try:
                self._client.delete_table(table_name)
            except DXMTError as e:
                if "TABLE_NOT_FOUND" not in str(e):
                    raise

    # ---- Haystack serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            table_name=self.table_name,
            embedding_dim=self.embedding_dim,
            host=self.host,
            port=self.port,
            ssl=self.ssl,
            distance_metric=self.distance_metric,
            enable_full_text_search=self.enable_full_text_search,
            text_analyzer=self.text_analyzer,
            server_embedding=self.server_embedding,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DXMTDocumentStore":
        return default_from_dict(cls, data)

    # ---- table -----------------------------------------------------------------------

    @property
    def client(self) -> DXMTClient:
        return self._client

    def _ensure_table(self) -> None:
        if self._table_ready:
            return
        try:
            self._client.get_table(self.table_name)
        except DXMTError as e:
            if "TABLE_NOT_FOUND" not in str(e):
                raise
            text_fields = None
            if self.enable_full_text_search:
                text_fields = [
                    {
                        "name": "content",
                        "source": {"metadata_field": "content"},
                        "analyzer": self.text_analyzer,
                        "boost": 1.0,
                    }
                ]
            self._client.create_table(
                name=self.table_name,
                dimension=None if self.server_embedding else self.embedding_dim,
                distance_metric=self.distance_metric,
                enable_full_text_search=self.enable_full_text_search,
                text_fields=text_fields,
                embedding=self.server_embedding,
            )
        self._table_ready = True

    def _exists(self) -> bool:
        try:
            self._client.get_table(self.table_name)
            return True
        except DXMTError as e:
            if "TABLE_NOT_FOUND" in str(e):
                return False
            raise

    # ---- protocol --------------------------------------------------------------------

    def count_documents(self) -> int:
        if not self._exists():
            return 0
        return self._client.count(self.table_name)

    def filter_documents(self, filters: Optional[dict[str, Any]] = None) -> list[Document]:
        if filters:
            validate(filters)
        if not self._exists():
            return []
        server_filter = translate(filters)
        rows = self._client.scroll(
            self.table_name, filter=server_filter.to_dict() if server_filter is not None else None, with_vector=True
        )
        docs = [self._to_document(row) for row in rows]
        if filters and server_filter is None:
            # No flat server form: Haystack's own evaluator decides, over every row.
            docs = [d for d in docs if document_matches_filter(filters, d)]
        return docs

    def write_documents(
        self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
    ) -> int:
        if not isinstance(documents, list) or any(not isinstance(d, Document) for d in documents):
            raise ValueError("write_documents takes a list of haystack Document objects")
        if not documents:
            return 0
        self._ensure_table()
        if policy in (DuplicatePolicy.SKIP, DuplicatePolicy.FAIL):
            present = {r["id"] for r in self._client.get_records(self.table_name, [d.id for d in documents])}
            if present and policy == DuplicatePolicy.FAIL:
                raise DuplicateDocumentError(f"ids already in {self.table_name!r}: {sorted(present)}")
            documents = [d for d in documents if d.id not in present]
            if not documents:
                return 0
        # NONE and OVERWRITE: the server's write is an upsert, so a repeated id replaces the row.
        self._client.insert_records(self.table_name, [self._to_record(d) for d in documents])
        return len(documents)

    def delete_documents(self, document_ids: list[str]) -> None:
        if not document_ids or not self._exists():
            return
        self._client.delete_records(self.table_name, document_ids)

    # ---- retrieval, used by the retriever components ---------------------------------

    def _server_filter_or_refuse(self, filters: Optional[dict[str, Any]], what: str):
        if not filters:
            return None
        validate(filters)
        server_filter = translate(filters)
        if server_filter is None:
            raise DocumentStoreError(
                f"{what} cannot apply this filter: the server has no flat form for it "
                f"(nested AND/OR, NOT over several conditions, comparisons against dates, "
                f"strings or None are evaluated client-side by filter_documents only). Filter: {filters}"
            )
        return server_filter.to_dict()

    def embedding_retrieval(
        self,
        query_embedding: Optional[list[float]] = None,
        *,
        query_text: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10,
        return_embedding: bool = False,
    ) -> list[Document]:
        if query_embedding is None and query_text is None:
            raise ValueError("embedding_retrieval needs a query_embedding, or query_text on a table with server_embedding")
        if not self._exists():
            return []
        server_filter = self._server_filter_or_refuse(filters, "embedding retrieval")
        # Documents written without an embedding carry a stand-in vector and must not rank.
        only_embedded = {"key": HAS_EMBEDDING, "operation": {"type": "Equal", "value": True}}
        merged = dict(server_filter or {})
        merged["must"] = list(merged.get("must") or []) + [only_embedded]
        hits = self._client.search(
            self.table_name,
            vector=query_embedding,
            text=query_text if query_embedding is None else None,
            k=top_k,
            filter=merged,
        )
        return [self._to_document(h, with_embedding=return_embedding, score=h.get("score")) for h in hits]

    def bm25_retrieval(
        self, query: str, *, filters: Optional[dict[str, Any]] = None, top_k: int = 10
    ) -> list[Document]:
        if not self._exists():
            return []
        server_filter = self._server_filter_or_refuse(filters, "BM25 retrieval")
        hits = self._client.full_text_search(self.table_name, query=query, k=top_k, filter=server_filter)
        return [self._to_document(h, score=h.get("score")) for h in hits]

    def hybrid_retrieval(
        self,
        query: str,
        query_embedding: Optional[list[float]] = None,
        *,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10,
        vector_weight: Optional[float] = None,
    ) -> list[Document]:
        if not self._exists():
            return []
        server_filter = self._server_filter_or_refuse(filters, "hybrid retrieval")
        hits = self._client.hybrid_search(
            self.table_name,
            vector=query_embedding,
            text=query,
            k=top_k,
            vector_weight=vector_weight,
            filter=server_filter,
        )
        return [self._to_document(h, score=h.get("score")) for h in hits]

    # ---- conversion ------------------------------------------------------------------

    def _to_record(self, doc: Document) -> dict[str, Any]:
        payload: dict[str, Any] = {"content": doc.content, "meta": dict(doc.meta or {})}
        if doc.blob is not None:
            logger.warning("document %s: blob is not stored; DXMT payloads are JSON", doc.id)
        record: dict[str, Any] = {"id": doc.id, "payload": payload}
        if doc.embedding is not None:
            record["vector"] = list(doc.embedding)
            payload[HAS_EMBEDDING] = True
        elif self.server_embedding and doc.content:
            record["text"] = doc.content
            payload[HAS_EMBEDDING] = True
        else:
            if not self.embedding_dim:
                raise DocumentStoreError(
                    f"document {doc.id} has no embedding and the store has no embedding_dim to make a stand-in"
                )
            record["vector"] = [1.0] + [0.0] * (self.embedding_dim - 1)
            payload[HAS_EMBEDDING] = False
        if doc.sparse_embedding is not None:
            record["sparse_vector"] = {
                "indices": list(doc.sparse_embedding.indices),
                "values": list(doc.sparse_embedding.values),
            }
        return record

    @staticmethod
    def _to_document(
        row: dict[str, Any], *, with_embedding: bool = True, score: Optional[float] = None
    ) -> Document:
        payload = row.get("payload") or row.get("metadata") or {}
        if isinstance(payload, str):
            import json

            payload = json.loads(payload)
        has_embedding = payload.get(HAS_EMBEDDING, True)
        vector = row.get("vector") if (with_embedding and has_embedding) else None
        sparse = row.get("sparse_vector")
        return Document(
            id=row["id"],
            content=payload.get("content"),
            meta=dict(payload.get("meta") or {}),
            embedding=list(vector) if vector else None,
            sparse_embedding=SparseEmbedding(indices=list(sparse["indices"]), values=list(sparse["values"]))
            if sparse
            else None,
            score=score,
        )
