"""DXMT VectorStore for LangChain.

This module provides a LangChain VectorStore implementation for DXMT VectorDB,
enabling integration with LangChain-based frameworks like Bisheng.

Example:
    ```python
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.schema import Document
    from dxmt.langchain_vectorstore import DXMTVectorStore

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create DXMT vector store
    vector_store = DXMTVectorStore(
        embedding=embeddings,
        collection_name="documents",
        host="localhost",
        port=9999
    )

    # Add documents
    documents = [Document(page_content="Sample document", metadata={"source": "test"})]
    vector_store.add_documents(documents)

    # Search
    results = vector_store.similarity_search("query", k=5)
    ```
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Iterable, List, Optional, Tuple

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Define dummy classes
    Document = Any  # type: ignore
    Embeddings = Any  # type: ignore
    VectorStore = object  # type: ignore

from .client import DXMTClient
from .exceptions import DXMTError
from .filters import convert_simple_filter_to_dxmt

logger = logging.getLogger(__name__)


class DXMTVectorStore(VectorStore):
    """DXMT VectorStore for LangChain.

    Implements LangChain's VectorStore interface to enable DXMT VectorDB
    integration with LangChain and Bisheng.

    Args:
        embedding: Embeddings instance for encoding text.
        collection_name: Name of the DXMT table/collection.
        host: DXMT server host.
        port: DXMT server port.
        dimension: Vector dimension (auto-detected from embeddings if not provided).
        distance_metric: Distance metric (cosine, euclidean, dot, manhattan).
        client: Optional DXMTClient instance (created if not provided).
        **kwargs: Additional arguments.

    Attributes:
        embeddings: The embeddings instance.
        collection_name: Collection name in DXMT.
        client: DXMT client instance.

    Example:
        ```python
        from langchain.embeddings import OpenAIEmbeddings
        from dxmt.langchain_vectorstore import DXMTVectorStore

        embeddings = OpenAIEmbeddings()
        vector_store = DXMTVectorStore(
            embedding=embeddings,
            collection_name="my_docs",
            host="localhost",
            port=9999
        )

        # Add documents
        vector_store.add_texts(
            texts=["Document 1", "Document 2"],
            metadatas=[{"source": "doc1"}, {"source": "doc2"}]
        )

        # Search
        results = vector_store.similarity_search("query text", k=3)
        ```
    """

    def __init__(
        self,
        embedding: Embeddings,
        collection_name: str = "langchain_collection",
        host: str = "localhost",
        port: int = 9999,
        dimension: Optional[int] = None,
        distance_metric: str = "cosine",
        client: Optional[DXMTClient] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize DXMT vector store. ``api_key`` is ignored when ``client`` is given."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for DXMTVectorStore. "
                "Install with: pip install langchain langchain-core"
            )

        self._embeddings = embedding
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self._dimension = dimension

        # Create or use provided client
        if client is None:
            self.client = DXMTClient(host=host, port=port, api_key=api_key)
        else:
            self.client = client

        # Ensure table exists
        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Ensure the DXMT collection/table exists."""
        try:
            # Try to get existing table
            self.client.get_table(self.collection_name)
            logger.info(f"Collection {self.collection_name} already exists")
        except DXMTError:
            # Table doesn't exist, create it
            # Determine dimension from embeddings if not provided
            if self._dimension is None:
                # Get dimension by embedding a test string
                test_embedding = self._embeddings.embed_query("test")
                self._dimension = len(test_embedding)

            logger.info(f"Creating collection {self.collection_name} with dimension {self._dimension}")
            self.client.create_table(
                name=self.collection_name,
                dimension=self._dimension,
                distance_metric=self.distance_metric,
                description=f"LangChain collection (dimension={self._dimension})"
            )

    @property
    def embeddings(self) -> Embeddings:
        """Get the embeddings instance."""
        return self._embeddings

    @property
    def embedding_function(self) -> Embeddings:
        """Get the embeddings function."""
        return self._embeddings

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vector store.

        Args:
            texts: Texts to add to the vector store.
            metadatas: Optional list of metadata dicts for each text.
            ids: Optional list of IDs for each text.
            **kwargs: Additional arguments.

        Returns:
            List of IDs for the added texts.
        """
        texts_list = list(texts)

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts_list]

        # Prepare metadatas
        if metadatas is None:
            metadatas = [{} for _ in texts_list]
        elif len(metadatas) != len(texts_list):
            raise ValueError("Number of metadatas must match number of texts")

        # Generate embeddings
        embeddings = self.embeddings.embed_documents(texts_list)

        # Prepare records for DXMT
        records = []
        for i, (text, embedding, metadata) in enumerate(zip(texts_list, embeddings, metadatas)):
            # Add text to metadata
            metadata_with_text = metadata.copy()
            metadata_with_text["text"] = text

            record = {
                "id": ids[i],
                "vector": embedding,
                "metadata": metadata_with_text
            }
            records.append(record)

        # Insert into DXMT
        try:
            self.client.insert_records(
                table_name=self.collection_name,
                records=records
            )
            logger.info(f"Added {len(records)} texts to {self.collection_name}")
            return ids
        except Exception as e:
            logger.error(f"Error adding texts: {e}")
            raise DXMTError(f"Failed to add texts: {e}")

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for similar documents.

        Args:
            query: Query text.
            k: Number of results to return.
            filter: Optional metadata filter.
            **kwargs: Additional arguments.

        Returns:
            List of similar documents.
        """
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)

        # Convert filter to DXMT format
        dxmt_filter = self._convert_filter(filter) if filter is not None else None

        # Search DXMT
        try:
            results = self.client.search(
                table_name=self.collection_name,
                vector=query_embedding,
                k=k,
                filter=dxmt_filter
            )
        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise DXMTError(f"Failed to search: {e}")

        # Convert results to Documents
        documents = []
        for result in results:
            metadata = result.get("metadata", {})
            text = metadata.pop("text", "")

            doc = Document(
                page_content=text,
                metadata=metadata
            )
            documents.append(doc)

        return documents

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores.

        Args:
            query: Query text.
            k: Number of results to return.
            filter: Optional metadata filter.
            **kwargs: Additional arguments.

        Returns:
            List of tuples (document, score).
        """
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)

        # Convert filter to DXMT format
        dxmt_filter = self._convert_filter(filter) if filter is not None else None

        # Search DXMT
        try:
            results = self.client.search(
                table_name=self.collection_name,
                vector=query_embedding,
                k=k,
                filter=dxmt_filter
            )
        except Exception as e:
            logger.error(f"Error searching with score: {e}")
            raise DXMTError(f"Failed to search: {e}")

        # Convert results to Documents with scores
        documents_with_scores = []
        for result in results:
            metadata = result.get("metadata", {})
            text = metadata.pop("text", "")
            score = result.get("score", 0.0)

            doc = Document(
                page_content=text,
                metadata=metadata
            )
            documents_with_scores.append((doc, score))

        return documents_with_scores

    def _convert_filter(self, filter: dict) -> Optional[dict]:
        """Convert LangChain filter to DXMT format.

        Supports simple equality dicts and MongoDB-style operators
        ($eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $contains, $startswith,
        $endswith) as well as logical operators ($and, $or, $not).

        Args:
            filter: LangChain filter dict.

        Returns:
            DXMT filter dict.
        """
        return convert_simple_filter_to_dxmt(filter)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = "langchain_collection",
        host: str = "localhost",
        port: int = 9999,
        **kwargs: Any,
    ) -> "DXMTVectorStore":
        """Create a DXMT vector store from texts.

        Args:
            texts: List of texts to add.
            embedding: Embeddings instance.
            metadatas: Optional list of metadata dicts.
            ids: Optional list of IDs.
            collection_name: Collection name.
            host: DXMT server host.
            port: DXMT server port.
            **kwargs: Additional arguments.

        Returns:
            DXMTVectorStore instance.
        """
        vector_store = cls(
            embedding=embedding,
            collection_name=collection_name,
            host=host,
            port=port,
            **kwargs,
        )
        vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return vector_store

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        ids: Optional[List[str]] = None,
        collection_name: str = "langchain_collection",
        host: str = "localhost",
        port: int = 9999,
        **kwargs: Any,
    ) -> "DXMTVectorStore":
        """Create a DXMT vector store from documents.

        Args:
            documents: List of Document objects.
            embedding: Embeddings instance.
            ids: Optional list of IDs.
            collection_name: Collection name.
            host: DXMT server host.
            port: DXMT server port.
            **kwargs: Additional arguments.

        Returns:
            DXMTVectorStore instance.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            host=host,
            port=port,
            **kwargs,
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search using a pre-computed embedding vector.

        Args:
            embedding: Query embedding vector.
            k: Number of results to return.
            filter: Optional metadata filter.
            **kwargs: Additional arguments.

        Returns:
            List of similar documents.
        """
        dxmt_filter = self._convert_filter(filter) if filter is not None else None
        try:
            results = self.client.search(
                table_name=self.collection_name,
                vector=embedding,
                k=k,
                filter=dxmt_filter,
            )
        except Exception as e:
            logger.error(f"Error searching by vector: {e}")
            raise DXMTError(f"Failed to search: {e}")

        documents = []
        for result in results:
            metadata = result.get("metadata", {})
            text = metadata.pop("text", "")
            documents.append(Document(page_content=text, metadata=metadata))
        return documents

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by IDs.

        Args:
            ids: List of IDs to delete.
            **kwargs: Additional arguments.

        Returns:
            True if successful.
        """
        if ids is None:
            return None

        try:
            self.client.delete_records(
                table_name=self.collection_name,
                ids=ids,
            )
            logger.info(f"Deleted {len(ids)} documents from {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise DXMTError(f"Failed to delete: {e}")

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Async add texts — delegates to sync add_texts via thread executor."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.add_texts(texts, metadatas=metadatas, ids=ids, **kwargs)
        )

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Async similarity search."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.similarity_search(query, k=k, filter=filter, **kwargs)
        )

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Async similarity search with scores."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.similarity_search_with_score(query, k=k, filter=filter, **kwargs),
        )
