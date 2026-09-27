"""
DXMT Haystack Retrievers

Retriever components for using DXMT in Haystack pipelines.
Provides both BM25 (full-text) and embedding (vector) retrievers.
"""

from typing import Any, Dict, List, Optional

try:
    from haystack import Document, component, default_from_dict, default_to_dict
    from haystack.document_stores.types import FilterPolicy
    HAYSTACK_AVAILABLE = True
except ImportError:
    HAYSTACK_AVAILABLE = False
    # Minimal stubs
    Document = Any

    class FilterPolicy:
        REPLACE = "replace"
        MERGE = "merge"

        @classmethod
        def from_str(cls, value):
            return value

    def component(cls):
        return cls

    def default_to_dict(*args, **kwargs):
        return {}

    def default_from_dict(cls, data):
        return cls(**data.get("init_parameters", {}))

from dxmt.haystack import DXMTDocumentStore


@component
class DXMTEmbeddingRetriever:
    """
    Retriever for semantic search using DXMT vector similarity.

    This retriever performs approximate nearest neighbor search on document embeddings
    stored in DXMT. It leverages DXMT's tiered storage and adaptive index selection
    for cost-efficient, high-performance vector search.

    Usage example:
        ```python
        from haystack import Pipeline
        from haystack.components.embedders import SentenceTransformersTextEmbedder
        from dxmt import Client as DXMTClient
        from dxmt.haystack import DXMTDocumentStore
        from dxmt.haystack_retrievers import DXMTEmbeddingRetriever

        # Initialize
        client = DXMTClient(host="localhost", port=9999)
        doc_store = DXMTDocumentStore(client=client, embedding_dim=384)
        retriever = DXMTEmbeddingRetriever(document_store=doc_store, top_k=10)

        # Use in pipeline
        pipeline = Pipeline()
        pipeline.add_component("embedder", SentenceTransformersTextEmbedder())
        pipeline.add_component("retriever", retriever)
        pipeline.connect("embedder.embedding", "retriever.query_embedding")

        result = pipeline.run({"embedder": {"text": "What is vector search?"}})
        ```
    """

    def __init__(
        self,
        document_store: DXMTDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = False,
        return_embedding: bool = False,
        filter_policy: "FilterPolicy" = None,
    ):
        """
        Initialize the DXMT Embedding Retriever.

        Args:
            document_store: DXMTDocumentStore instance
            filters: Default filters to apply
            top_k: Number of documents to retrieve (default: 10)
            scale_score: Scale scores to [0, 1] range (default: False)
            return_embedding: Return embeddings with documents (default: False)
            filter_policy: How to merge runtime and init filters (default: REPLACE)

        Raises:
            ValueError: If document_store is not a DXMTDocumentStore instance
            ImportError: If haystack-ai is not installed (only when used in real pipelines)

        Note:
            While haystack-ai is recommended, this works without it for testing purposes.
        """
        # Note: We don't enforce HAYSTACK_AVAILABLE to allow mock testing

        if not isinstance(document_store, DXMTDocumentStore):
            raise ValueError("document_store must be an instance of DXMTDocumentStore")

        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, got {top_k}")

        self.document_store = document_store
        self.filters = filters
        self.top_k = top_k
        self.scale_score = scale_score
        self.return_embedding = return_embedding
        self.filter_policy = filter_policy or FilterPolicy.REPLACE

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the retriever to a dictionary."""
        return default_to_dict(
            self,
            document_store=self.document_store,
            filters=self.filters,
            top_k=self.top_k,
            scale_score=self.scale_score,
            return_embedding=self.return_embedding,
            filter_policy=self.filter_policy.value if hasattr(self.filter_policy, "value") else str(self.filter_policy),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DXMTEmbeddingRetriever":
        """Deserialize the retriever from a dictionary."""
        init_params = data.get("init_parameters", {})
        if "filter_policy" in init_params and HAYSTACK_AVAILABLE:
            init_params["filter_policy"] = FilterPolicy.from_str(init_params["filter_policy"])
        return default_from_dict(cls, data)

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """Telemetry data for usage analytics."""
        return {"document_store": type(self.document_store).__name__}

    @component.output_types(documents=List[Document]) if HAYSTACK_AVAILABLE else lambda x: x
    def run(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
        return_embedding: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Run embedding retrieval.

        Args:
            query_embedding: Query embedding vector
            filters: Runtime filters to apply
            top_k: Override default top_k
            scale_score: Override default scale_score
            return_embedding: Override default return_embedding

        Returns:
            Dictionary with "documents" key containing retrieved documents
        """
        # Merge filters based on policy
        if self.filter_policy == FilterPolicy.MERGE and filters:
            filters = {**(self.filters or {}), **filters}
        else:
            filters = filters or self.filters

        # Use provided values or defaults
        top_k = top_k if top_k is not None else self.top_k
        scale_score = scale_score if scale_score is not None else self.scale_score
        return_embedding = return_embedding if return_embedding is not None else self.return_embedding

        # Perform retrieval
        docs = self.document_store.embedding_retrieval(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
            scale_score=scale_score,
            return_embedding=return_embedding,
        )

        return {"documents": docs}

    @component.output_types(documents=List[Document]) if HAYSTACK_AVAILABLE else lambda x: x
    async def run_async(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
        return_embedding: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Async version of run."""
        # Merge filters
        if self.filter_policy == FilterPolicy.MERGE and filters:
            filters = {**(self.filters or {}), **filters}
        else:
            filters = filters or self.filters

        # Use provided values or defaults
        top_k = top_k if top_k is not None else self.top_k
        scale_score = scale_score if scale_score is not None else self.scale_score
        return_embedding = return_embedding if return_embedding is not None else self.return_embedding

        # Perform async retrieval
        docs = await self.document_store.embedding_retrieval_async(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
            scale_score=scale_score,
            return_embedding=return_embedding,
        )

        return {"documents": docs}


@component
class DXMTBM25Retriever:
    """
    Retriever for full-text search using BM25 ranking in DXMT.

    This retriever performs BM25-ranked full-text search on document content
    stored in DXMT. It uses DXMT's built-in full-text search capabilities
    powered by Tantivy.

    Usage example:
        ```python
        from haystack import Pipeline
        from dxmt import Client as DXMTClient
        from dxmt.haystack import DXMTDocumentStore
        from dxmt.haystack_retrievers import DXMTBM25Retriever

        # Initialize
        client = DXMTClient(host="localhost", port=9999)
        doc_store = DXMTDocumentStore(client=client)
        retriever = DXMTBM25Retriever(document_store=doc_store, top_k=10)

        # Use in pipeline
        pipeline = Pipeline()
        pipeline.add_component("retriever", retriever)

        result = pipeline.run({"retriever": {"query": "machine learning"}})
        ```
    """

    def __init__(
        self,
        document_store: DXMTDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = False,
        filter_policy: "FilterPolicy" = None,
    ):
        """
        Initialize the DXMT BM25 Retriever.

        Args:
            document_store: DXMTDocumentStore instance
            filters: Default filters to apply
            top_k: Number of documents to retrieve (default: 10)
            scale_score: Scale scores to [0, 1] range (default: False)
            filter_policy: How to merge runtime and init filters (default: REPLACE)

        Raises:
            ValueError: If document_store is not a DXMTDocumentStore instance
            ImportError: If haystack-ai is not installed (only when used in real pipelines)

        Note:
            While haystack-ai is recommended, this works without it for testing purposes.
        """
        # Note: We don't enforce HAYSTACK_AVAILABLE to allow mock testing

        if not isinstance(document_store, DXMTDocumentStore):
            raise ValueError("document_store must be an instance of DXMTDocumentStore")

        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, got {top_k}")

        self.document_store = document_store
        self.filters = filters
        self.top_k = top_k
        self.scale_score = scale_score
        self.filter_policy = filter_policy or FilterPolicy.REPLACE

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the retriever to a dictionary."""
        return default_to_dict(
            self,
            document_store=self.document_store,
            filters=self.filters,
            top_k=self.top_k,
            scale_score=self.scale_score,
            filter_policy=self.filter_policy.value if hasattr(self.filter_policy, "value") else str(self.filter_policy),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DXMTBM25Retriever":
        """Deserialize the retriever from a dictionary."""
        init_params = data.get("init_parameters", {})
        if "filter_policy" in init_params and HAYSTACK_AVAILABLE:
            init_params["filter_policy"] = FilterPolicy.from_str(init_params["filter_policy"])
        return default_from_dict(cls, data)

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """Telemetry data for usage analytics."""
        return {"document_store": type(self.document_store).__name__}

    @component.output_types(documents=List[Document]) if HAYSTACK_AVAILABLE else lambda x: x
    def run(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Run BM25 retrieval.

        Args:
            query: Search query string
            filters: Runtime filters to apply
            top_k: Override default top_k
            scale_score: Override default scale_score

        Returns:
            Dictionary with "documents" key containing retrieved documents
        """
        # Merge filters based on policy
        if self.filter_policy == FilterPolicy.MERGE and filters:
            filters = {**(self.filters or {}), **filters}
        else:
            filters = filters or self.filters

        # Use provided values or defaults
        top_k = top_k if top_k is not None else self.top_k
        scale_score = scale_score if scale_score is not None else self.scale_score

        # Perform retrieval
        docs = self.document_store.bm25_retrieval(
            query=query,
            filters=filters,
            top_k=top_k,
            scale_score=scale_score,
        )

        return {"documents": docs}

    @component.output_types(documents=List[Document]) if HAYSTACK_AVAILABLE else lambda x: x
    async def run_async(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Async version of run."""
        # Merge filters
        if self.filter_policy == FilterPolicy.MERGE and filters:
            filters = {**(self.filters or {}), **filters}
        else:
            filters = filters or self.filters

        # Use provided values or defaults
        top_k = top_k if top_k is not None else self.top_k
        scale_score = scale_score if scale_score is not None else self.scale_score

        # Perform async retrieval
        docs = await self.document_store.bm25_retrieval_async(
            query=query,
            filters=filters,
            top_k=top_k,
            scale_score=scale_score,
        )

        return {"documents": docs}
