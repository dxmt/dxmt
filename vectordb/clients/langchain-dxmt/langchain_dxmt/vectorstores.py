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
from dxmt.exceptions import DXMTConnectionError, DXMTError

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)
DEFAULT_K = 4

#: What the server's `DistanceMetric` enum accepts.
_METRICS = frozenset({"cosine", "euclidean", "manhattan", "dot"})
#: Spellings other vector stores use, mapped to this one's.
_METRIC_ALIASES = {
    "dot_product": "dot",
    "dotproduct": "dot",
    "inner_product": "dot",
    "ip": "dot",
    "l2": "euclidean",
    "l1": "manhattan",
}


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
        api_key: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        index_type: str = "hnsw",
        distance_metric: str = "cosine",
        enable_full_text_search: bool = False,
        text_fields: Optional[list[dict[str, Any]]] = None,
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
            api_key: The server's API key, if it was started with ``DXMT_API_KEY``. Ignored
                when ``client`` is given — configure that client instead.
            headers: Additional HTTP headers.
            index_type: Index type ('hnsw', 'ivf', 'ivf_pq', 'lsh', 'brute_force').
            distance_metric: Distance metric ('cosine', 'euclidean', 'manhattan', 'dot').
            enable_full_text_search: Enable full-text search. Default is False.
            text_fields: Text field configurations for full-text search.
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
                api_key=api_key,
                headers=headers,
            )

        # Checked here rather than on the wire. Both of these used to travel to the server
        # and come back as a `DXMTTableError` wrapping a JSON body, which is a needless round
        # trip and a worse message: what the caller got wrong is visible from the argument.
        if not table_name or not table_name.strip():
            msg = "table_name cannot be empty"
            raise ValueError(msg)

        # `dot_product` is what LangChain users write (Pinecone spells it `dotproduct`,
        # Qdrant `Dot`); this server's enum says `dot`. Accepting the spellings people
        # actually reach for is not the same as accepting anything: an unrecognised metric
        # still fails, now naming what is accepted instead of quoting a server error.
        metric = _METRIC_ALIASES.get(distance_metric.strip().lower(), distance_metric)
        if metric not in _METRICS:
            msg = (
                f"Unknown distance_metric {distance_metric!r}. "
                f"Accepted: {', '.join(sorted(_METRICS))} "
                f"(aliases: {', '.join(sorted(_METRIC_ALIASES))})."
            )
            raise ValueError(msg)

        # Store configuration
        self._table_name = table_name
        self._embedding = embedding
        self._dimension = dimension
        self._index_type = index_type
        self._distance_metric = metric
        self._enable_full_text_search = enable_full_text_search
        self._text_fields = text_fields
        self.override_relevance_score_fn = relevance_score_fn

        # Create or verify table
        if create_table_if_not_exists:
            self.__ensure_table()
        else:
            # Verify table exists.
            #
            # A missing table and an unreachable server are different problems and used to
            # get the same sentence. `except Exception` turned a DNS failure, a refused
            # connection and a timeout all into "Table X does not exist", which sends
            # whoever mistyped a host off to look for a table that is sitting there fine.
            # Only the server's own "not found" is a missing table; everything else is
            # re-raised as itself.
            try:
                table_info = self._client.get_table(table_name)
                table_data = table_info.get("data", {}).get("table", {})
                self._dimension = table_data.get("dimension")
            except DXMTConnectionError:
                raise
            except DXMTError as e:
                if "TABLE_NOT_FOUND" not in str(e):
                    raise
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
            **kwargs: Additional keyword arguments.

        Returns:
            List of IDs of added texts.
        """
        if self._embedding is None:
            msg = "embedding function is required for add_texts"
            raise ValueError(msg)

        # Convert texts to list
        texts_: list[str] = list(texts) if not isinstance(texts, list) else texts

        # Nothing to add is not an error. LangChain's contract is that adding no texts
        # returns `[]`, and any caller that filters or pages its input reaches that case on
        # ordinary data. Without this the request goes out and the server refuses it with
        # `VALIDATION_FAILED: Batch cannot be empty` -- correct of the server, and wrong
        # here: the empty batch should never have been sent.
        if not texts_:
            return []

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
        self._client.insert_records(
            table_name=self._table_name,
            records=records,
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

        # Nothing to add is not an error. LangChain's contract is that adding no texts
        # returns `[]`, and any caller that filters or pages its input reaches that case on
        # ordinary data. Without this the request goes out and the server refuses it with
        # `VALIDATION_FAILED: Batch cannot be empty` -- correct of the server, and wrong
        # here: the empty batch should never have been sent.
        if not texts_:
            return []

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
        await self._client.ainsert_records(
            table_name=self._table_name,
            records=records,
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
        # Asking for no rows is not an error either. `k=0` falls out of arithmetic -- a
        # budget that has been spent, a `top_k` computed from a page that turned out empty --
        # and LangChain's answer is an empty list. The server refuses `top_k: 0` with
        # `INVALID_INPUT`, which is right for a hand-written request and wrong for this one.
        if k <= 0:
            return []

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
        if k <= 0:  # same reason as the sync path above
            return []

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
            # The score the server computed, unconverted.
            #
            # `override_relevance_score_fn` used to be applied here as well as in
            # `_select_relevance_score_fn`, and LangChain's
            # `_similarity_search_with_relevance_scores` calls the second one over the output
            # of the first -- so a caller who supplied a normaliser got it applied **twice**.
            # `similarity_search_with_score` is documented to return the store's own score;
            # normalising is the other method's job and belongs in one place.
            score = result.get("score", 0.0)

            docs_and_scores.append((doc, score))

        return docs_and_scores

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Turn what the server returns into LangChain's [0, 1] relevance.

        **This used to invert the ranking.** The comment said "DXMT returns similarity scores
        (higher is better), for cosine the score is already in [0, 1]" and the next line
        returned `_cosine_relevance_score_fn`, which is LangChain's `1.0 - distance` and
        expects a *distance*. A similarity of 0.89 came out as 0.11, so
        `similarity_score_threshold` kept the least relevant documents and dropped the most
        relevant ones. Measured on three documents and the query they were written for:

            相似度        relevance(旧)   文档
            0.8944        0.1056          neural networks and machine learning
            0.8944        0.1056          machine learning neural networks deep
            0.1890        0.8110          a recipe for cooking pasta with tomato

        `score_threshold=0.7` returned the pasta recipe and nothing else.

        What the server actually returns, per `similarity_from_distance`:

            cosine                 `1 - distance`, in [-1, 1]
            euclidean / manhattan  `1 / (1 + distance)`, in (0, 1]
            dot                    `-distance`, the raw inner product -- unbounded

        The first two are already relevances in the right direction; the clamp only handles
        a cosine similarity below zero, which means "pointing away" and is no relevance at
        all. The third cannot be mapped onto [0, 1] without knowing the scale of the
        embeddings, so it says so rather than inventing a number -- the caller who knows
        their model passes `relevance_score_fn=`.
        """
        if self.override_relevance_score_fn:
            return self.override_relevance_score_fn

        if self._distance_metric in ("cosine", "euclidean", "manhattan"):
            return lambda score: max(0.0, min(1.0, score))

        if self._distance_metric == "dot":
            msg = (
                "A dot-product table returns an unbounded inner product, which cannot be "
                "normalised to LangChain's [0, 1] relevance without knowing the scale of "
                "your embeddings. Pass relevance_score_fn= to DXMTVectorDB, or use "
                "similarity_search_with_score, which returns the score as the server "
                "computed it."
            )
            raise ValueError(msg)

        msg = f"No relevance score function for distance metric {self._distance_metric!r}"
        raise ValueError(msg)

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

        The reranking happens **on the server**, which already holds the vectors. This used to
        fetch `fetch_k` documents and re-embed their text to get vectors to diversify with:
        an embedding call per search, paid in latency and in money, and -- worse -- diversity
        computed over vectors that are not the ones in the index. A document whose stored vector
        came from different text, or from an earlier version of the model, was diversified
        against something the database had never seen.

        Args:
            embedding: Embedding vector.
            k: Number of Document objects to return.
            fetch_k: Number of candidates the server reranks. Diversity can only come from
                candidates that were fetched, so this is the cost knob.
            lambda_mult: 0 is maximum diversity, 1 is the plain ranking.
            filter: Optional metadata filter.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Document objects selected by maximal marginal relevance.
        """
        results = self._client.search(
            table_name=self._table_name,
            vector=embedding,
            k=k,
            filter=filter,
            mmr={"lambda_mult": lambda_mult, "fetch_k": fetch_k},
            **kwargs,
        )
        return [doc for doc, _ in self._results_to_docs_and_scores(results)]

    def hybrid_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        vector_weight: Optional[float] = None,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Hybrid search combining vector similarity and full-text search.

        Note: Requires enable_full_text_search=True during initialization.

        Args:
            query: Query text.
            k: Number of documents to return.
            vector_weight: Weight for vector similarity (0-1); the text side gets the rest.
                          Setting one selects weighted fusion, which is the method that reads
                          it. Left unset, the server fuses by distribution (`dbsf`), which
                          needs no weight -- this defaulted to 0.7, a number that was sent and
                          then ignored.
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
        vector_weight: Optional[float] = None,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Async hybrid search. See `hybrid_search` for what `vector_weight` selects."""
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
