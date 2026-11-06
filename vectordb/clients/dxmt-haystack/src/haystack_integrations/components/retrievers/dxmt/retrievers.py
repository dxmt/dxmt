"""Retriever components over ``DXMTDocumentStore``.

Three components, one per kind of query the server answers: a vector (or, on a table with
``server_embedding``, the query text) for ``DXMTEmbeddingRetriever``; a BM25 query for
``DXMTBM25Retriever``; both at once, fused on the server, for ``DXMTHybridRetriever``.
``filter_policy`` is Haystack's: ``REPLACE`` (default) uses the runtime filter alone, ``MERGE``
ANDs it with the one given at construction.
"""

from __future__ import annotations

from typing import Any, Optional

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.dxmt import DXMTDocumentStore


def _store_from_dict(data: dict[str, Any]) -> None:
    init = data["init_parameters"]
    init["document_store"] = DXMTDocumentStore.from_dict(init["document_store"])
    if isinstance(init.get("filter_policy"), str):
        init["filter_policy"] = FilterPolicy.from_str(init["filter_policy"])


@component
class DXMTEmbeddingRetriever:
    """Nearest documents to a query embedding, or to query text when the table embeds on the server."""

    def __init__(
        self,
        document_store: DXMTDocumentStore,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10,
        return_embedding: bool = False,
        filter_policy: FilterPolicy = FilterPolicy.REPLACE,
    ) -> None:
        if not isinstance(document_store, DXMTDocumentStore):
            raise ValueError("document_store must be a DXMTDocumentStore")
        self.document_store = document_store
        self.filters = filters
        self.top_k = top_k
        self.return_embedding = return_embedding
        self.filter_policy = filter_policy

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            document_store=self.document_store.to_dict(),
            filters=self.filters,
            top_k=self.top_k,
            return_embedding=self.return_embedding,
            filter_policy=self.filter_policy.value,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DXMTEmbeddingRetriever":
        _store_from_dict(data)
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: Optional[list[float]] = None,
        query_text: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
        return_embedding: Optional[bool] = None,
    ) -> dict[str, list[Document]]:
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        docs = self.document_store.embedding_retrieval(
            query_embedding,
            query_text=query_text,
            filters=filters,
            top_k=top_k if top_k is not None else self.top_k,
            return_embedding=return_embedding if return_embedding is not None else self.return_embedding,
        )
        return {"documents": docs}


@component
class DXMTBM25Retriever:
    """Documents ranked by BM25 over ``content``; the store needs ``enable_full_text_search``."""

    def __init__(
        self,
        document_store: DXMTDocumentStore,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10,
        filter_policy: FilterPolicy = FilterPolicy.REPLACE,
    ) -> None:
        if not isinstance(document_store, DXMTDocumentStore):
            raise ValueError("document_store must be a DXMTDocumentStore")
        self.document_store = document_store
        self.filters = filters
        self.top_k = top_k
        self.filter_policy = filter_policy

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            document_store=self.document_store.to_dict(),
            filters=self.filters,
            top_k=self.top_k,
            filter_policy=self.filter_policy.value,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DXMTBM25Retriever":
        _store_from_dict(data)
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self, query: str, filters: Optional[dict[str, Any]] = None, top_k: Optional[int] = None
    ) -> dict[str, list[Document]]:
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        docs = self.document_store.bm25_retrieval(
            query, filters=filters, top_k=top_k if top_k is not None else self.top_k
        )
        return {"documents": docs}


@component
class DXMTHybridRetriever:
    """Vector and BM25 rankings fused on the server; ``vector_weight`` is the vector side's share."""

    def __init__(
        self,
        document_store: DXMTDocumentStore,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10,
        vector_weight: Optional[float] = None,
        filter_policy: FilterPolicy = FilterPolicy.REPLACE,
    ) -> None:
        if not isinstance(document_store, DXMTDocumentStore):
            raise ValueError("document_store must be a DXMTDocumentStore")
        self.document_store = document_store
        self.filters = filters
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.filter_policy = filter_policy

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            document_store=self.document_store.to_dict(),
            filters=self.filters,
            top_k=self.top_k,
            vector_weight=self.vector_weight,
            filter_policy=self.filter_policy.value,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DXMTHybridRetriever":
        _store_from_dict(data)
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str,
        query_embedding: Optional[list[float]] = None,
        filters: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> dict[str, list[Document]]:
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        docs = self.document_store.hybrid_retrieval(
            query,
            query_embedding,
            filters=filters,
            top_k=top_k if top_k is not None else self.top_k,
            vector_weight=self.vector_weight,
        )
        return {"documents": docs}
