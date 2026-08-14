"""DXMT + Bisheng Integration Module.

This module provides the initialization function for DXMT VectorStore
to be used with Bisheng's vector store system.

Bisheng Integration Pattern:
    Bisheng uses a factory pattern with initialization functions for each
    vector store. This module provides the initialization function that
    Bisheng can call to create DXMT vector store instances.

Usage in Bisheng:
    Add to bisheng/interface/initialize/vector_store.py:

    ```python
    from dxmt.bisheng_integration import initialize_dxmt

    vecstore_initializer = {
        # ... other vector stores ...
        'DXMT': initialize_dxmt,
    }
    ```

Example:
    ```python
    from langchain.embeddings import OpenAIEmbeddings
    from dxmt.bisheng_integration import initialize_dxmt
    from dxmt.langchain_vectorstore import DXMTVectorStore

    # Bisheng-style params
    params = {
        'embedding': OpenAIEmbeddings(),
        'collection_name': 'test_docs',
        'host': 'localhost',
        'port': 9999,
        'documents': [...],  # Optional initial documents
    }

    # Initialize via Bisheng pattern
    vector_store = initialize_dxmt(DXMTVectorStore, params, {})
    ```
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Type

try:
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    Document = Any  # type: ignore

from .langchain_vectorstore import DXMTVectorStore

logger = logging.getLogger(__name__)


def docs_in_params(params: dict) -> bool:
    """Check if params has documents OR texts.

    This is a helper function following Bisheng's pattern to determine
    if we should create a new index with documents or return an existing index.

    Args:
        params: Parameters dict.

    Returns:
        True if documents or texts are present and not empty.
    """
    return ('documents' in params and params['documents']) or ('texts' in params and params['texts'])


def initialize_dxmt(
    class_object: Type[DXMTVectorStore],
    params: Dict[str, Any],
    search_kwargs: Dict[str, Any]
) -> DXMTVectorStore:
    """Initialize DXMT vector store for Bisheng.

    This function follows Bisheng's initialization pattern for vector stores.
    It handles both creating a new DXMT collection with documents and
    connecting to an existing collection.

    Args:
        class_object: The DXMTVectorStore class (passed by Bisheng).
        params: Initialization parameters from Bisheng, including:
            - embedding: Embeddings instance (required)
            - collection_name: Name of the DXMT collection (default: "bisheng_collection")
            - host: DXMT server host (default: "localhost")
            - port: DXMT server port (default: 9999)
            - dimension: Vector dimension (optional, auto-detected)
            - distance_metric: Distance metric (default: "cosine")
            - documents: Optional list of Document objects to add
            - texts: Optional list of text strings to add
            - metadatas: Optional list of metadata dicts (for texts)
            - collection_id: Optional Bisheng knowledge base ID
        search_kwargs: Additional search parameters (currently unused).

    Returns:
        Initialized DXMTVectorStore instance.

    Raises:
        ValueError: If required parameters are missing.
        DXMTError: If initialization fails.

    Example:
        ```python
        from langchain.embeddings import OpenAIEmbeddings
        from dxmt.bisheng_integration import initialize_dxmt
        from dxmt.langchain_vectorstore import DXMTVectorStore

        params = {
            'embedding': OpenAIEmbeddings(),
            'collection_name': 'my_docs',
            'host': 'localhost',
            'port': 9999,
        }

        # Without initial documents (connect to existing)
        vector_store = initialize_dxmt(DXMTVectorStore, params, {})

        # With initial documents (create new index)
        params['documents'] = [Document(page_content="doc1", metadata={})]
        vector_store = initialize_dxmt(DXMTVectorStore, params, {})
        ```
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is required for Bisheng integration. "
            "Install with: pip install langchain langchain-core"
        )

    # Validate required parameters
    if 'embedding' not in params:
        raise ValueError('Embedding must be provided in params')

    # Extract DXMT-specific parameters
    collection_name = params.pop('collection_name', 'bisheng_collection')
    host = params.pop('host', 'localhost')
    port = params.pop('port', 9999)
    dimension = params.pop('dimension', None)
    distance_metric = params.pop('distance_metric', 'cosine')
    collection_id = params.pop('collection_id', None)  # Bisheng knowledge base ID

    logger.info(
        f"Initializing DXMT vector store: collection={collection_name}, "
        f"host={host}, port={port}, dimension={dimension}"
    )

    # Handle Bisheng knowledge base ID mapping
    if collection_id:
        logger.info(f"Using Bisheng knowledge base ID: {collection_id}")
        # Bisheng knowledge bases use collection_id to identify collections
        # You could implement custom logic here to map collection_id to collection_name

    # Check if we have initial documents to add
    if not docs_in_params(params):
        # No documents - connect to existing collection
        params.pop('documents', None)
        params.pop('texts', None)
        params.pop('metadatas', None)

        logger.info(f"Connecting to existing DXMT collection: {collection_name}")

        # Create vector store instance (connects to existing table)
        vector_store = class_object(
            embedding=params['embedding'],
            collection_name=collection_name,
            host=host,
            port=port,
            dimension=dimension,
            distance_metric=distance_metric,
        )

        return vector_store

    # Has documents - create new index with documents
    logger.info(f"Creating new DXMT collection with documents: {collection_name}")

    # Handle both 'documents' and 'texts' formats
    if 'texts' in params:
        # Convert texts to documents
        texts = params.pop('texts')
        metadatas = params.pop('metadatas', None)

        # Use from_texts class method
        vector_store = class_object.from_texts(
            texts=texts,
            embedding=params.pop('embedding'),
            metadatas=metadatas,
            collection_name=collection_name,
            host=host,
            port=port,
            dimension=dimension,
            distance_metric=distance_metric,
        )
    else:
        # Use documents directly
        documents = params.pop('documents')

        # Use from_documents class method
        vector_store = class_object.from_documents(
            documents=documents,
            embedding=params.pop('embedding'),
            collection_name=collection_name,
            host=host,
            port=port,
            dimension=dimension,
            distance_metric=distance_metric,
        )

    logger.info(f"Successfully created DXMT vector store: {collection_name}")
    return vector_store


# Alias for consistency with Bisheng naming
initial_dxmt = initialize_dxmt


__all__ = [
    'initialize_dxmt',
    'initial_dxmt',
    'docs_in_params',
]
