"""DXMT Retriever for MetaGPT.

This module provides a MetaGPT-compatible retriever implementation for DXMT VectorDB.

Example:
    ```python
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.core.schema import Document
    from dxmt import DXMTClient
    from dxmt.metagpt_vector_store import DXMTVectorStore
    from dxmt.metagpt_retriever import DXMTRetriever

    # Create documents
    documents = [Document(text="DXMT is a cost-efficient vector database")]

    # Create DXMT vector store
    client = DXMTClient(host="localhost", port=9999)
    vector_store = DXMTVectorStore(
        client=client,
        table_name="metagpt_docs",
        dimension=1536
    )

    # Create index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )

    # Create DXMT retriever
    retriever = DXMTRetriever(index=index, similarity_top_k=5)

    # Use in MetaGPT
    nodes = retriever.retrieve("What is DXMT?")
    ```
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import BaseNode, NodeWithScore

from .metagpt_vector_store import DXMTVectorStore

logger = logging.getLogger(__name__)


class DXMTRetriever(VectorIndexRetriever):
    """DXMT Retriever for MetaGPT.

    Implements MetaGPT's retriever interface for DXMT VectorDB,
    providing vector similarity search with additional utility methods.

    This retriever extends LlamaIndex's VectorIndexRetriever with:
    - add_nodes: Add new documents to the index
    - persist: Save the index to disk
    - query_total_count: Get total number of documents
    - clear: Remove all documents

    Args:
        index: VectorStoreIndex using DXMTVectorStore.
        similarity_top_k: Number of top results to return.
        **kwargs: Additional arguments passed to VectorIndexRetriever.

    Example:
        ```python
        from llama_index.core import VectorStoreIndex
        from dxmt.metagpt_retriever import DXMTRetriever

        # Assuming index is already created with DXMTVectorStore
        retriever = DXMTRetriever(
            index=index,
            similarity_top_k=5
        )

        # Retrieve documents
        results = retriever.retrieve("search query")

        # Add new documents
        from llama_index.core.schema import Document
        new_docs = [Document(text="New document")]
        retriever.add_documents(new_docs)

        # Get total count
        count = retriever.query_total_count()
        ```
    """

    def __init__(
        self,
        index: Any,
        similarity_top_k: int = 5,
        **kwargs: Any,
    ) -> None:
        """Initialize DXMT retriever."""
        super().__init__(
            index=index,
            similarity_top_k=similarity_top_k,
            **kwargs,
        )

    @property
    def vector_store(self) -> DXMTVectorStore:
        """Get the DXMT vector store.

        Returns:
            DXMTVectorStore instance.

        Raises:
            TypeError: If vector store is not DXMTVectorStore.
        """
        vector_store = self._index.vector_store
        if not isinstance(vector_store, DXMTVectorStore):
            raise TypeError(
                f"Expected DXMTVectorStore, got {type(vector_store).__name__}"
            )
        return vector_store

    def add_nodes(self, nodes: List[BaseNode], **kwargs: Any) -> None:
        """Add nodes to the index.

        This method is required by MetaGPT's ModifiableRAGRetriever interface.

        Args:
            nodes: List of nodes to add.
            **kwargs: Additional arguments.

        Example:
            ```python
            from llama_index.core.schema import TextNode

            nodes = [
                TextNode(text="Document 1", embedding=[...]),
                TextNode(text="Document 2", embedding=[...]),
            ]
            retriever.add_nodes(nodes)
            ```
        """
        self._index.insert_nodes(nodes, **kwargs)
        logger.info(f"Added {len(nodes)} nodes to DXMT index")

    def add_documents(self, documents: List[Any], **kwargs: Any) -> None:
        """Add documents to the index.

        Convenience method for adding raw documents.

        Args:
            documents: List of Document objects.
            **kwargs: Additional arguments.

        Example:
            ```python
            from llama_index.core.schema import Document

            docs = [
                Document(text="Document 1"),
                Document(text="Document 2"),
            ]
            retriever.add_documents(docs)
            ```
        """
        # Convert documents to nodes and insert
        for doc in documents:
            self._index.insert(doc, **kwargs)
        logger.info(f"Added {len(documents)} documents to DXMT index")

    def persist(self, persist_dir: str, **kwargs: Any) -> None:
        """Persist the index.

        This method is required by MetaGPT's PersistableRAGRetriever interface.

        Note: DXMT automatically persists data to its backend storage
        (tiered: Memory → SSD → S3), so this method primarily saves the
        index metadata.

        Args:
            persist_dir: Directory to persist to.
            **kwargs: Additional arguments.

        Example:
            ```python
            retriever.persist("./storage")
            ```
        """
        # DXMT automatically persists vector data
        # We only need to persist the index metadata
        self._index.storage_context.persist(persist_dir)
        logger.info(f"Persisted DXMT index metadata to {persist_dir}")

    def query_total_count(self) -> int:
        """Query total number of documents in the index.

        This method is required by MetaGPT's QueryableRAGRetriever interface.

        Returns:
            Total number of documents.

        Example:
            ```python
            count = retriever.query_total_count()
            print(f"Total documents: {count}")
            ```
        """
        try:
            # Get table info from DXMT
            table_info = self.vector_store.client.get_table(
                self.vector_store.table_name
            )
            # Try to get record count from table info
            count = table_info.get("record_count", 0)
            if count == 0:
                # Fallback: count documents in docstore
                count = len(self._index.docstore.docs)
            return count
        except Exception as e:
            logger.warning(f"Error getting total count: {e}, falling back to docstore")
            # Fallback to docstore count
            return len(self._index.docstore.docs)

    def clear(self, **kwargs: Any) -> None:
        """Clear all documents from the index.

        This method is required by MetaGPT's DeletableRAGRetriever interface.

        Warning: This operation is irreversible and will delete all data
        in the DXMT table.

        Args:
            **kwargs: Additional arguments.

        Example:
            ```python
            retriever.clear()  # Deletes all documents
            ```
        """
        self.vector_store.clear()
        logger.info("Cleared all documents from DXMT index")

    async def _aretrieve(self, query_str: str) -> List[NodeWithScore]:
        """Async retrieve nodes.

        Required by MetaGPT's RAGRetriever interface.

        Args:
            query_str: Query string.

        Returns:
            List of nodes with scores.
        """
        # Use the sync retrieve method
        # LlamaIndex VectorIndexRetriever doesn't have async by default
        return self.retrieve(query_str)

    def _retrieve(self, query_str: str) -> List[NodeWithScore]:
        """Sync retrieve nodes.

        Args:
            query_str: Query string.

        Returns:
            List of nodes with scores.
        """
        # Delegate to parent's retrieve method
        return self.retrieve(query_str)


# Convenience function for creating DXMT retriever
def create_dxmt_retriever(
    documents: List[Any],
    table_name: str = "metagpt_docs",
    dimension: int = 1536,
    similarity_top_k: int = 5,
    host: str = "localhost",
    port: int = 9999,
    embed_model: Optional[Any] = None,
    **kwargs: Any,
) -> DXMTRetriever:
    """Create a DXMT retriever from documents.

    Convenience function that creates the full pipeline:
    1. Creates DXMTVectorStore
    2. Creates VectorStoreIndex
    3. Creates DXMTRetriever

    Args:
        documents: List of documents to index.
        table_name: DXMT table name.
        dimension: Vector dimension.
        similarity_top_k: Number of results to return.
        host: DXMT server host.
        port: DXMT server port.
        embed_model: Embedding model (optional).
        **kwargs: Additional arguments.

    Returns:
        Configured DXMTRetriever instance.

    Example:
        ```python
        from llama_index.core.schema import Document
        from dxmt.metagpt_retriever import create_dxmt_retriever

        documents = [
            Document(text="DXMT is a vector database"),
            Document(text="MetaGPT is an AI agent framework"),
        ]

        retriever = create_dxmt_retriever(
            documents=documents,
            table_name="my_docs",
            similarity_top_k=3
        )

        # Use the retriever
        results = retriever.retrieve("What is DXMT?")
        ```
    """
    from llama_index.core import StorageContext, VectorStoreIndex

    from .client import DXMTClient
    from .metagpt_vector_store import DXMTVectorStore

    # Create DXMT client and vector store
    client = DXMTClient(host=host, port=port)
    vector_store = DXMTVectorStore(
        client=client,
        table_name=table_name,
        dimension=dimension,
        create_if_not_exists=True,
    )

    # Create storage context and index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index_kwargs = {
        "storage_context": storage_context,
        "show_progress": True,
    }
    if embed_model is not None:
        index_kwargs["embed_model"] = embed_model

    index = VectorStoreIndex.from_documents(documents, **index_kwargs)

    # Create and return retriever
    retriever = DXMTRetriever(
        index=index,
        similarity_top_k=similarity_top_k,
        **kwargs,
    )

    logger.info(
        f"Created DXMT retriever with {len(documents)} documents "
        f"in table '{table_name}'"
    )

    return retriever
