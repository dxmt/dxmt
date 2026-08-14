"""DXMT VectorDB integration for LangChain.

This module provides LangChain VectorStore implementation for DXMT VectorDB.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from dxmt import Client as DXMTClient

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)
DEFAULT_K = 4


def _extract_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Extract payload from DXMT record.

    DXMT server may return payload as:
    - 'payload': dict (if present in search results)
    - 'payload_json': JSON string (from get_records)

    Args:
        record: DXMT record.

    Returns:
        Payload dictionary.
    """
    # Try payload field first (may be present in search results)
    if "payload" in record and record["payload"] is not None:
        return record["payload"]

    # Try payload_json field (from get_records)
    if "payload_json" in record:
        payload_json = record["payload_json"]
        if payload_json and payload_json != "null":
            try:
                return json.loads(payload_json)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Failed to parse payload_json: {payload_json}")
                return {}

    return {}


class DXMTVectorDB(VectorStore):
    """DXMT VectorDB vector store integration.

    DXMT is a lightweight, horizontally scalable vector database with:
    - Multiple index types (HNSW, IVF, IVF-PQ, LSH)
    - Full-text search with BM25 and Chinese support (Jieba)
    - Hybrid search (vector + text)
    - Multiple distance metrics
    - Storage native architecture

    Setup:
        Install `langchain-dxmt` package:

        ```bash
        pip install -qU langchain-dxmt
        ```

    Key init args — indexing params:
        table_name:
            Name of the table (collection).
        embedding:
            Embedding function to use.
        dimension:
            Vector dimension (required for table creation).

    Key init args — client params:
        client:
            DXMT client to use. If not provided, creates new client.
        host:
            Hostname of DXMT server. Default is 'localhost'.
        port:
            Port of DXMT server. Default is 9999.
        ssl:
            Whether to use HTTPS. Default is False.

    Key init args — advanced params:
        index_type:
            Index type ('hnsw', 'ivf', 'ivf_pq', 'lsh', 'brute_force').
        distance_metric:
            Distance metric ('cosine', 'euclidean', 'manhattan', 'dot').
        enable_full_text_search:
            Enable full-text search capabilities. Default is False.
        insert_mode:
            Batch insert mode ('standard', 'lock_free', 'ultimate', etc.).

    Instantiate:
        ```python
        from langchain_dxmt import DXMTVectorDB
        from langchain_openai import OpenAIEmbeddings

        vector_store = DXMTVectorDB(
            table_name="my_documents",
            embedding=OpenAIEmbeddings(),
            dimension=1536,
            host="localhost",
            port=9999,
        )
        ```

    Add Documents:
        ```python
        from langchain_core.documents import Document

        document_1 = Document(
            page_content="foo", metadata={"source": "doc1"}
        )
        document_2 = Document(
            page_content="bar", metadata={"source": "doc2"}
        )

        documents = [document_1, document_2]
        ids = vector_store.add_documents(documents=documents)
        ```

    Search:
        ```python
        results = vector_store.similarity_search("query text", k=3)
        for doc in results:
            print(f"* {doc.page_content} [{doc.metadata}]")
        ```

    Search with score:
        ```python
        results = vector_store.similarity_search_with_score(
            "query text", k=3
        )
        for doc, score in results:
            print(
                f"* [SIM={score:.3f}] {doc.page_content} [{doc.metadata}]"
            )
        ```

    Search with filter:
        ```python
        results = vector_store.similarity_search(
            "query text", k=3, filter={"source": "doc1"}
        )
        ```

    Hybrid Search (DXMT-specific):
        ```python
        # Requires enable_full_text_search=True during initialization
        results = vector_store.hybrid_search(
            query="machine learning",
            k=5,
            vector_weight=0.7,  # 70% vector, 30% text
        )
        ```

    Async:
        ```python
        # add documents
        # await vector_store.aadd_documents(documents=documents)

        # search
        # results = await vector_store.asimilarity_search("query text", k=3)

        # search with score
        results = await vector_store.asimilarity_search_with_score(
            "query", k=3
        )
        for doc, score in results:
            print(f"* [SIM={score:.3f}] {doc.page_content}")
        ```

    Use as Retriever:
        ```python
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )
        docs = retriever.invoke("query text")
        ```
    """

    _LANGCHAIN_DEFAULT_TABLE_NAME = "langchain"

    def __init__(
        self,
        table_name: str = _LANGCHAIN_DEFAULT_TABLE_NAME,
        embedding: Optional[Embeddings] = None,
        dimension: Optional[int] = None,
        *,
        client: Optional[DXMTClient] = None,
        host: str = "localhost",
        port: int = 9999,
        ssl: bool = False,
        headers: Optional[dict[str, str]] = None,
        index_type: str = "hnsw",
        distance_metric: str = "cosine",
        enable_full_text_search: bool = False,
        text_fields: Optional[list[dict[str, Any]]] = None,
        insert_mode: str = "standard",
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        create_table_if_not_exists: bool = True,
    ) -> None:
        """Initialize DXMT VectorDB.

        Args:
            table_name: Name of the table. Default is 'langchain'.
            embedding: Embedding function. Required for adding/searching documents.
            dimension: Vector dimension. Required if create_table_if_not_exists is True.
            client: DXMT client instance. If not provided, creates new client.
            host: DXMT server hostname. Default is 'localhost'.
            port: DXMT server port. Default is 9999.
            ssl: Whether to use HTTPS. Default is False.
            headers: Additional HTTP headers.
            index_type: Index type ('hnsw', 'ivf', 'ivf_pq', 'lsh', 'brute_force').
            distance_metric: Distance metric ('cosine', 'euclidean', 'manhattan', 'dot').
            enable_full_text_search: Enable full-text search. Default is False.
            text_fields: Text field configurations for full-text search.
            insert_mode: Batch insert mode. Default is 'standard'.
            relevance_score_fn: Custom relevance score function.
            create_table_if_not_exists: Whether to create table if it doesn't exist.
        """
        # Validate parameters
        if create_table_if_not_exists and dimension is None:
            msg = "dimension must be provided when create_table_if_not_exists is True"
            raise ValueError(msg)

        # Initialize client
        if client is not None:
            if not isinstance(client, DXMTClient):
                msg = f"client must be DXMTClient instance, got {type(client)}"
                raise TypeError(msg)
            self._client = client
        else:
            self._client = DXMTClient(
                host=host,
                port=port,
                ssl=ssl,
                headers=headers,
            )

        # Store configuration
        self._table_name = table_name
        self._embedding = embedding
        self._dimension = dimension
        self._index_type = index_type
        self._distance_metric = distance_metric
        self._enable_full_text_search = enable_full_text_search
        self._text_fields = text_fields
        self._insert_mode = insert_mode
        self.override_relevance_score_fn = relevance_score_fn

        # Create or verify table
        if create_table_if_not_exists:
            self.__ensure_table()
        else:
            # Verify table exists
            try:
                table_info = self._client.get_table(table_name)
                table_data = table_info.get("data", {}).get("table", {})
                self._dimension = table_data.get("dimension")
            except Exception as e:
                msg = f"Table {table_name} does not exist: {e}"
                raise ValueError(msg) from e

    def __ensure_table(self) -> None:
        """Ensure table exists or create it."""
        try:
            self._client.get_table(self._table_name)
            logger.info("Table %s already exists", self._table_name)
        except Exception:
            # Table doesn't exist, create it
            logger.info("Creating table %s", self._table_name)

            # If full-text search is enabled but no text_fields configured,
            # use default configuration to index the 'text' field in payload
            text_fields = self._text_fields
            if self._enable_full_text_search and text_fields is None:
                text_fields = [
                    {
                        "name": "text",
                        "source": {"metadata_field": "text"},
                        "analyzer": "standard",
                        "boost": 1.0,
                    }
                ]
                logger.info("Using default text_fields configuration for full-text search")

            self._client.create_table(
                name=self._table_name,
                dimension=self._dimension,  # type: ignore[arg-type]
                distance_metric=self._distance_metric,
                index_type=self._index_type,
                enable_full_text_search=self._enable_full_text_search,
                text_fields=text_fields,
            )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Access the embedding function."""
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts to the vector store.

        Args:
            texts: Texts to add.
            metadatas: Optional metadata for each text.
            ids: Optional IDs for each text.
            **kwargs: Additional keyword arguments (e.g., insert_mode).

        Returns:
            List of IDs of added texts.
        """
        if self._embedding is None:
            msg = "embedding function is required for add_texts"
            raise ValueError(msg)

        # Convert texts to list
        texts_: list[str] = list(texts) if not isinstance(texts, list) else texts

        # Validate metadatas
        if metadatas and len(metadatas) != len(texts_):
            msg = (
                f"Number of metadatas ({len(metadatas)}) must match "
                f"number of texts ({len(texts_)})"
            )
            raise ValueError(msg)

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts_]
        else:
            # Fill in missing IDs
            ids = [id_ if id_ is not None else str(uuid.uuid4()) for id_ in ids]

        # Validate IDs count
        if len(ids) != len(texts_):
            msg = (
                f"Number of IDs ({len(ids)}) must match number of texts ({len(texts_)})"
            )
            raise ValueError(msg)

        # Embed texts
        embeddings = self._embedding.embed_documents(texts_)

        # Prepare records
        metadatas_ = metadatas or [{} for _ in texts_]
        records = [
            {
                "id": id_,
                "vector": embedding,
                "payload": {  # DXMT server expects 'payload' not 'metadata'
                    **metadata,
                    "text": text,  # Store text in payload for retrieval
                },
            }
            for id_, text, embedding, metadata in zip(
                ids, texts_, embeddings, metadatas_
            )
        ]

        # Insert records
        insert_mode = kwargs.get("insert_mode", self._insert_mode)
        self._client.insert_records(
            table_name=self._table_name,
            records=records,
            mode=insert_mode,
        )

        return ids

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Async add texts to the vector store."""
        if self._embedding is None:
            msg = "embedding function is required for aadd_texts"
            raise ValueError(msg)

        # Convert texts to list
        texts_: list[str] = list(texts) if not isinstance(texts, list) else texts

        # Validate metadatas
        if metadatas and len(metadatas) != len(texts_):
            msg = (
                f"Number of metadatas ({len(metadatas)}) must match "
                f"number of texts ({len(texts_)})"
            )
            raise ValueError(msg)

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts_]
        else:
            # Fill in missing IDs
            ids = [id_ if id_ is not None else str(uuid.uuid4()) for id_ in ids]

        # Validate IDs count
        if len(ids) != len(texts_):
            msg = (
                f"Number of IDs ({len(ids)}) must match number of texts ({len(texts_)})"
            )
            raise ValueError(msg)

        # Embed texts (async)
        embeddings = await self._embedding.aembed_documents(texts_)

        # Prepare records
        metadatas_ = metadatas or [{} for _ in texts_]
        records = [
            {
                "id": id_,
                "vector": embedding,
                "payload": {  # DXMT server expects 'payload' not 'metadata'
                    **metadata,
                    "text": text,
                },
            }
            for id_, text, embedding, metadata in zip(
                ids, texts_, embeddings, metadatas_
            )
        ]

        # Insert records (async)
        insert_mode = kwargs.get("insert_mode", self._insert_mode)
        await self._client.ainsert_records(
            table_name=self._table_name,
            records=records,
            mode=insert_mode,
        )

        return ids

    def delete(
        self,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete records by IDs.

        Args:
            ids: IDs to delete.
            **kwargs: Additional keyword arguments.

        Returns:
            True if deletion was successful.
        """
        if not ids:
            logger.warning("No IDs provided for deletion")
            return False

        try:
            self._client.delete_records(
                table_name=self._table_name,
                ids=ids,
            )
            return True
        except Exception as e:
            logger.error("Error deleting records: %s", e)
            return False

    async def adelete(
        self,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Async delete records by IDs."""
        if not ids:
            logger.warning("No IDs provided for deletion")
            return False

        try:
            await self._client.adelete_records(
                table_name=self._table_name,
                ids=ids,
            )
            return True
        except Exception as e:
            logger.error("Error deleting records: %s", e)
            return False

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by IDs.

        Args:
            ids: List of IDs to retrieve.

        Returns:
            List of Documents.
        """
        records = self._client.get_records(
            table_name=self._table_name,
            ids=list(ids),
        )

        return [
            Document(
                id=record.get("id"),
                page_content=_extract_payload(record).get("text", ""),
                metadata={
                    k: v for k, v in _extract_payload(record).items() if k != "text"
                },
            )
            for record in records
        ]

    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Async get documents by IDs."""
        records = await self._client.aget_records(
            table_name=self._table_name,
            ids=list(ids),
        )

        return [
            Document(
                id=record.get("id"),
                page_content=_extract_payload(record).get("text", ""),
                metadata={
                    k: v for k, v in _extract_payload(record).items() if k != "text"
                },
            )
            for record in records
        ]

    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to query.

        Args:
            query: Query text.
            k: Number of documents to return.
            filter: Optional metadata filter.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents most similar to the query.
        """
        docs_and_scores = self.similarity_search_with_score(
            query, k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    async def asimilarity_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Async return docs most similar to query."""
        docs_and_scores = await self.asimilarity_search_with_score(
            query, k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and scores most similar to query.

        Args:
            query: Query text.
            k: Number of documents to return.
            filter: Optional metadata filter.
            **kwargs: Additional keyword arguments.

        Returns:
            List of (Document, score) tuples.
        """
        if self._embedding is None:
            msg = "embedding function is required for similarity_search"
            raise ValueError(msg)

        # Embed query
        query_embedding = self._embedding.embed_query(query)

        return self.similarity_search_with_score_by_vector(
            query_embedding, k, filter=filter, **kwargs
        )

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Async return docs and scores most similar to query."""
        if self._embedding is None:
            msg = "embedding function is required for asimilarity_search"
            raise ValueError(msg)

        # Embed query (async)
        query_embedding = await self._embedding.aembed_query(query)

        return await self.asimilarity_search_with_score_by_vector(
            query_embedding, k, filter=filter, **kwargs
        )

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = DEFAULT_K,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding vector.
            k: Number of documents to return.
            filter: Optional metadata filter.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents most similar to the embedding.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding, k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    async def asimilarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = DEFAULT_K,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Async return docs most similar to embedding vector."""
        docs_and_scores = await self.asimilarity_search_with_score_by_vector(
            embedding, k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = DEFAULT_K,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and scores most similar to embedding vector.

        Args:
            embedding: Embedding vector.
            k: Number of documents to return.
            filter: Optional metadata filter.
            **kwargs: Additional keyword arguments.

        Returns:
            List of (Document, score) tuples.
        """
        # Perform search
        results = self._client.search(
            table_name=self._table_name,
            vector=embedding,
            k=k,
            filter=filter,
        )

        # Convert to documents
        return self._results_to_docs_and_scores(results)

    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = DEFAULT_K,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Async return docs and scores most similar to embedding vector."""
        # Perform search (async)
        results = await self._client.asearch(
            table_name=self._table_name,
            vector=embedding,
            k=k,
            filter=filter,
        )

        # Convert to documents
        return self._results_to_docs_and_scores(results)

    def _results_to_docs_and_scores(
        self, results: list[dict[str, Any]]
    ) -> list[tuple[Document, float]]:
        """Convert search results to (Document, score) tuples."""
        docs_and_scores = []
        for result in results:
            payload = _extract_payload(result).copy()
            text = payload.pop("text", "")
            doc = Document(
                id=result.get("id"),
                page_content=text,
                metadata=payload,
            )
            score = result.get("score", 0.0)

            # Apply relevance score function if provided
            if self.override_relevance_score_fn:
                score = self.override_relevance_score_fn(score)

            docs_and_scores.append((doc, score))

        return docs_and_scores

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Select relevance score function based on distance metric."""
        if self.override_relevance_score_fn:
            return self.override_relevance_score_fn

        # DXMT returns similarity scores (higher is better)
        # For cosine: score is already in [0, 1] range
        if self._distance_metric in ("cosine", "dot"):
            return self._cosine_relevance_score_fn
        elif self._distance_metric == "euclidean":
            return self._euclidean_relevance_score_fn
        else:
            # Default to cosine
            return self._cosine_relevance_score_fn

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Document objects to return.
            fetch_k: Number of Document objects to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
            filter: Optional metadata filter.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Document objects selected by maximal marginal relevance.
        """
        if self._embedding is None:
            msg = "embedding function is required for MMR search"
            raise ValueError(msg)

        # Embed query
        query_embedding = self._embedding.embed_query(query)

        # Fetch candidate documents
        return self.max_marginal_relevance_search_by_vector(
            query_embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = DEFAULT_K,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using maximal marginal relevance.

        Args:
            embedding: Embedding vector.
            k: Number of Document objects to return.
            fetch_k: Number of Document objects to fetch.
            lambda_mult: Diversity parameter (0=max diversity, 1=min diversity).
            filter: Optional metadata filter.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Document objects selected by maximal marginal relevance.
        """
        # Fetch more candidates than needed
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding, k=fetch_k, filter=filter, **kwargs
        )

        if not docs_and_scores:
            return []

        # Extract documents and embeddings
        # We need to get embeddings for MMR calculation
        # For now, re-embed the documents (not optimal but works)
        if self._embedding is None:
            msg = "embedding function is required for MMR search"
            raise ValueError(msg)

        # Get embeddings for all candidate documents
        candidate_docs = [doc for doc, _ in docs_and_scores]
        candidate_texts = [doc.page_content for doc in candidate_docs]

        # Batch embed all candidates
        candidate_embeddings = self._embedding.embed_documents(candidate_texts)

        # Perform MMR selection
        selected_indices = self._mmr_selection(
            embedding,
            candidate_embeddings,
            k=min(k, len(candidate_docs)),
            lambda_mult=lambda_mult,
        )

        return [candidate_docs[i] for i in selected_indices]

    @staticmethod
    def _mmr_selection(
        query_embedding: list[float],
        candidate_embeddings: list[list[float]],
        k: int,
        lambda_mult: float,
    ) -> list[int]:
        """Perform MMR selection.

        Args:
            query_embedding: Query embedding vector.
            candidate_embeddings: List of candidate embedding vectors.
            k: Number of documents to select.
            lambda_mult: Diversity parameter.

        Returns:
            List of selected indices.
        """
        import numpy as np

        # Convert to numpy arrays
        query_emb = np.array(query_embedding)
        candidates = np.array(candidate_embeddings)

        # Calculate similarities to query
        query_sims = np.dot(candidates, query_emb) / (
            np.linalg.norm(candidates, axis=1) * np.linalg.norm(query_emb)
        )

        # Initialize selected indices
        selected = []
        remaining = list(range(len(candidate_embeddings)))

        # Select first document (most similar to query)
        first_idx = int(np.argmax(query_sims))
        selected.append(first_idx)
        remaining.remove(first_idx)

        # Select remaining k-1 documents
        while len(selected) < k and remaining:
            best_score = float("-inf")
            best_idx = None

            for idx in remaining:
                # Calculate similarity to query
                query_sim = query_sims[idx]

                # Calculate maximum similarity to already selected documents
                selected_embs = candidates[selected]
                candidate_emb = candidates[idx]

                sims_to_selected = np.dot(selected_embs, candidate_emb) / (
                    np.linalg.norm(selected_embs, axis=1) * np.linalg.norm(candidate_emb)
                )
                max_sim_to_selected = np.max(sims_to_selected)

                # Calculate MMR score
                mmr_score = lambda_mult * query_sim - (1 - lambda_mult) * max_sim_to_selected

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return selected

    # DXMT-specific methods

    def hybrid_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        vector_weight: float = 0.7,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Hybrid search combining vector similarity and full-text search.

        Note: Requires enable_full_text_search=True during initialization.

        Args:
            query: Query text.
            k: Number of documents to return.
            vector_weight: Weight for vector similarity (0-1). Default is 0.7.
                          Text weight will be (1 - vector_weight).
            filter: Optional metadata filter.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents.
        """
        if not self._enable_full_text_search:
            msg = "Full-text search not enabled. Set enable_full_text_search=True"
            raise ValueError(msg)

        if self._embedding is None:
            msg = "embedding function is required for hybrid_search"
            raise ValueError(msg)

        # Embed query
        query_embedding = self._embedding.embed_query(query)

        # Perform hybrid search
        results = self._client.hybrid_search(
            table_name=self._table_name,
            vector=query_embedding,
            text=query,
            vector_weight=vector_weight,
            k=k,
            filter=filter,
        )

        # Convert to documents
        docs_and_scores = self._results_to_docs_and_scores(results)
        return [doc for doc, _ in docs_and_scores]

    async def ahybrid_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        vector_weight: float = 0.7,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Async hybrid search."""
        if not self._enable_full_text_search:
            msg = "Full-text search not enabled. Set enable_full_text_search=True"
            raise ValueError(msg)

        if self._embedding is None:
            msg = "embedding function is required for ahybrid_search"
            raise ValueError(msg)

        # Embed query (async)
        query_embedding = await self._embedding.aembed_query(query)

        # Perform hybrid search (async)
        results = await self._client.ahybrid_search(
            table_name=self._table_name,
            vector=query_embedding,
            text=query,
            vector_weight=vector_weight,
            k=k,
            filter=filter,
        )

        # Convert to documents
        docs_and_scores = self._results_to_docs_and_scores(results)
        return [doc for doc, _ in docs_and_scores]

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        table_name: str = _LANGCHAIN_DEFAULT_TABLE_NAME,
        dimension: Optional[int] = None,
        host: str = "localhost",
        port: int = 9999,
        ssl: bool = False,
        index_type: str = "hnsw",
        distance_metric: str = "cosine",
        enable_full_text_search: bool = False,
        insert_mode: str = "standard",
        **kwargs: Any,
    ) -> DXMTVectorDB:
        """Create DXMTVectorDB from texts.

        Args:
            texts: List of texts.
            embedding: Embedding function.
            metadatas: Optional list of metadatas.
            ids: Optional list of IDs.
            table_name: Table name.
            dimension: Vector dimension. If not provided, inferred from first embedding.
            host: DXMT server hostname.
            port: DXMT server port.
            ssl: Whether to use HTTPS.
            index_type: Index type.
            distance_metric: Distance metric.
            enable_full_text_search: Enable full-text search.
            insert_mode: Batch insert mode.
            **kwargs: Additional keyword arguments.

        Returns:
            DXMTVectorDB instance.
        """
        # Infer dimension if not provided
        if dimension is None:
            sample_embedding = embedding.embed_query(texts[0])
            dimension = len(sample_embedding)

        # Create instance
        instance = cls(
            table_name=table_name,
            embedding=embedding,
            dimension=dimension,
            host=host,
            port=port,
            ssl=ssl,
            index_type=index_type,
            distance_metric=distance_metric,
            enable_full_text_search=enable_full_text_search,
            insert_mode=insert_mode,
            create_table_if_not_exists=True,
            **kwargs,
        )

        # Add texts
        instance.add_texts(texts, metadatas, ids=ids)

        return instance

    @classmethod
    async def afrom_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        table_name: str = _LANGCHAIN_DEFAULT_TABLE_NAME,
        dimension: Optional[int] = None,
        host: str = "localhost",
        port: int = 9999,
        ssl: bool = False,
        index_type: str = "hnsw",
        distance_metric: str = "cosine",
        enable_full_text_search: bool = False,
        insert_mode: str = "standard",
        **kwargs: Any,
    ) -> DXMTVectorDB:
        """Async create DXMTVectorDB from texts."""
        # Infer dimension if not provided
        if dimension is None:
            sample_embedding = await embedding.aembed_query(texts[0])
            dimension = len(sample_embedding)

        # Create instance
        instance = cls(
            table_name=table_name,
            embedding=embedding,
            dimension=dimension,
            host=host,
            port=port,
            ssl=ssl,
            index_type=index_type,
            distance_metric=distance_metric,
            enable_full_text_search=enable_full_text_search,
            insert_mode=insert_mode,
            create_table_if_not_exists=True,
            **kwargs,
        )

        # Add texts (async)
        await instance.aadd_texts(texts, metadatas, ids=ids)

        return instance

    @classmethod
    def from_existing_table(
        cls,
        table_name: str,
        embedding: Embeddings,
        *,
        host: str = "localhost",
        port: int = 9999,
        ssl: bool = False,
        **kwargs: Any,
    ) -> DXMTVectorDB:
        """Connect to existing DXMT table.

        Args:
            table_name: Name of existing table.
            embedding: Embedding function.
            host: DXMT server hostname.
            port: DXMT server port.
            ssl: Whether to use HTTPS.
            **kwargs: Additional keyword arguments.

        Returns:
            DXMTVectorDB instance connected to existing table.
        """
        return cls(
            table_name=table_name,
            embedding=embedding,
            host=host,
            port=port,
            ssl=ssl,
            create_table_if_not_exists=False,
            **kwargs,
        )
