"""DXMT Vector Store implementation for LlamaIndex."""

from typing import Any, Dict, List, Optional, Sequence, Callable
from pydantic import PrivateAttr
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
    MetadataFilters,
    VectorStoreQueryMode,
)
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.utils import (
    node_to_metadata_dict,
    metadata_dict_to_node,
)

try:
    from dxmt import Client as DXMTClient
    from dxmt.exceptions import DXMTError
except ImportError:
    raise ImportError(
        "DXMT SDK is required for this vector store. "
        "Install it with: pip install dxmt"
    )

from .utils import (
    convert_llamaindex_filters_to_dxmt,
    node_to_dxmt_record,
    dxmt_hit_to_node,
)


class DXMTVectorStore(BasePydanticVectorStore):
    """DXMT VectorDB vector store implementation for LlamaIndex.

    This vector store provides integration with DXMT VectorDB, supporting:
    - Vector similarity search with multiple distance metrics
    - Hybrid search combining vector and full-text search
    - Multiple index types (HNSW, IVF, IVF-PQ, LSH, BruteForce)
    - Metadata filtering with complex filter expressions
    - Batch operations with optimized insert modes
    - Async operations

    Examples:
        Basic usage:

        >>> from llama_index.vector_stores.dxmt import DXMTVectorStore
        >>> vector_store = DXMTVectorStore(
        ...     table_name="my_docs",
        ...     dimension=1536,
        ...     host="localhost",
        ...     port=9999,
        ... )

        With existing client:

        >>> from dxmt import Client
        >>> client = Client(host="localhost", port=9999)
        >>> vector_store = DXMTVectorStore(
        ...     table_name="my_docs",
        ...     client=client,
        ...     dimension=1536,
        ... )

        From existing table:

        >>> vector_store = DXMTVectorStore.from_existing_table(
        ...     table_name="my_docs",
        ...     host="localhost",
        ...     port=9999,
        ... )
    """

    stores_text: bool = True
    is_embedding_query: bool = True
    flat_metadata: bool = False

    # Configuration
    table_name: str
    host: str = "localhost"
    port: int = 9999
    ssl: bool = False
    dimension: Optional[int] = None
    distance_metric: str = "cosine"
    index_type: str = "hnsw"
    enable_full_text_search: bool = False
    text_fields: Optional[List[Dict[str, Any]]] = None
    insert_mode: str = "standard"
    batch_size: int = 100

    # Private attributes
    _client: Optional[DXMTClient] = PrivateAttr(default=None)
    _table_created: bool = PrivateAttr(default=False)

    def __init__(
        self,
        table_name: str = "llamaindex",
        client: Optional[DXMTClient] = None,
        host: str = "localhost",
        port: int = 9999,
        ssl: bool = False,
        dimension: Optional[int] = None,
        distance_metric: str = "cosine",
        index_type: str = "hnsw",
        enable_full_text_search: bool = False,
        text_fields: Optional[List[Dict[str, Any]]] = None,
        insert_mode: str = "standard",
        batch_size: int = 100,
        **kwargs: Any,
    ):
        """Initialize DXMT vector store.

        Args:
            table_name: Name of the table in DXMT. Defaults to "llamaindex".
            client: Optional existing DXMT client. If not provided, a new client will be created.
            host: DXMT server host. Defaults to "localhost".
            port: DXMT server port. Defaults to 9999.
            ssl: Whether to use SSL for connection. Defaults to False.
            dimension: Vector dimension. Required for table creation.
            distance_metric: Distance metric for similarity search.
                Options: "cosine", "euclidean", "manhattan", "dot". Defaults to "cosine".
            index_type: Index type for vector search.
                Options: "hnsw", "ivf", "ivf_pq", "lsh", "brute_force". Defaults to "hnsw".
            enable_full_text_search: Enable full-text search capability. Defaults to False.
            text_fields: Configuration for text fields if full-text search is enabled.
            insert_mode: Insert mode for batch operations.
                Options: "standard", "lock_free", "streaming", "deferred_index",
                "ultimate", "batch", etc. Defaults to "standard".
            batch_size: Batch size for bulk operations. Defaults to 100.
            **kwargs: Additional arguments passed to DXMTClient.
        """
        # Call super().__init__ with all the model fields
        super().__init__(
            table_name=table_name,
            host=host,
            port=port,
            ssl=ssl,
            dimension=dimension,
            distance_metric=distance_metric,
            index_type=index_type,
            enable_full_text_search=enable_full_text_search,
            text_fields=text_fields,
            insert_mode=insert_mode,
            batch_size=batch_size,
            **kwargs
        )

        # Create or use provided client
        if client is not None:
            self._client = client
        else:
            self._client = DXMTClient(
                host=host,
                port=port,
                ssl=ssl,
            )

        self._table_created = False

    @property
    def client(self) -> DXMTClient:
        """Return the underlying DXMT client."""
        if self._client is None:
            raise ValueError("Client not initialized")
        return self._client

    def _ensure_table_exists(self) -> None:
        """Create table if it doesn't exist."""
        if self._table_created:
            return

        try:
            self.client.get_table(self.table_name)
            self._table_created = True
        except DXMTError:
            # Table doesn't exist, create it
            if self.dimension is None:
                raise ValueError(
                    "dimension must be specified for table creation. "
                    "Provide dimension parameter when initializing DXMTVectorStore."
                )

            table_config = {
                "name": self.table_name,
                "dimension": self.dimension,
                "distance_metric": self.distance_metric,
                "index_type": self.index_type,
                "enable_full_text_search": self.enable_full_text_search,
            }

            if self.text_fields is not None:
                table_config["text_fields"] = self.text_fields

            self.client.create_table(**table_config)
            self._table_created = True

    def add(
        self,
        nodes: Sequence[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to vector store.

        Args:
            nodes: Sequence of nodes to add. Each node should have an embedding.
            **add_kwargs: Additional keyword arguments:
                - sparse_encoder: Optional callable for encoding sparse vectors.

        Returns:
            List of node IDs that were added.

        Raises:
            ValueError: If dimension is not set and table doesn't exist.
        """
        self._ensure_table_exists()

        # Convert nodes to DXMT records
        records = []
        sparse_encoder = add_kwargs.get("sparse_encoder")

        for node in nodes:
            metadata = node_to_metadata_dict(
                node,
                remove_text=False,
                flat_metadata=self.flat_metadata
            )

            # Store node content in payload
            payload = metadata.copy()
            if hasattr(node, "get_content"):
                payload["_node_content"] = node.get_content()
            elif hasattr(node, "text"):
                payload["_node_content"] = node.text

            record = {
                "id": node.node_id,
                "vector": node.get_embedding(),
                "payload": payload,
            }

            # Add sparse vector if encoder provided
            if sparse_encoder is not None and hasattr(node, "get_content"):
                sparse_vec = sparse_encoder(node.get_content())
                record["sparse_vector"] = sparse_vec

            records.append(record)

        # Batch insert with specified insert mode
        batch_size = add_kwargs.get("batch_size", self.batch_size)

        for batch_start in range(0, len(records), batch_size):
            batch = records[batch_start : batch_start + batch_size]
            self.client.insert_records(
                self.table_name,
                batch,
                mode=self.insert_mode
            )

        return [node.node_id for node in nodes]

    async def async_add(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """Async version of add.

        Args:
            nodes: Sequence of nodes to add.
            **kwargs: Additional keyword arguments passed to add().

        Returns:
            List of node IDs that were added.
        """
        # For now, use synchronous version
        # TODO: Implement with async client when needed
        return self.add(nodes, **kwargs)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete nodes by ref_doc_id.

        This method deletes all nodes that have the specified ref_doc_id
        in their metadata.

        Args:
            ref_doc_id: Reference document ID to delete.
            **delete_kwargs: Additional delete options.
        """
        # Use metadata filter to find matching records
        # We need to get records with this ref_doc_id
        # Since DXMT doesn't have a direct "delete by filter" endpoint,
        # we need to query first, then delete by IDs

        try:
            # Create a dummy query vector
            dummy_vector = [0.0] * (self.dimension or 1536)

            # Search with filter to find matching records
            filter_dict = {"ref_doc_id": ref_doc_id}

            results = self.client.search(
                self.table_name,
                vector=dummy_vector,
                k=10000,  # Large number to get all matches
                filter=filter_dict,
            )

            # Extract IDs
            ids = [hit["id"] for hit in results.get("hits", [])]

            # Delete by IDs if any found
            if ids:
                self.client.delete_records(self.table_name, ids)

        except DXMTError:
            # Table might not exist or other error
            pass

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete nodes by IDs or filters.

        Args:
            node_ids: Optional list of node IDs to delete.
            filters: Optional metadata filters for selecting nodes to delete.
            **delete_kwargs: Additional delete options.
        """
        if node_ids is not None:
            # Direct deletion by IDs
            self.client.delete_records(self.table_name, node_ids)

        elif filters is not None:
            # Delete by filter - query first, then delete
            try:
                dxmt_filter = convert_llamaindex_filters_to_dxmt(filters)
                dummy_vector = [0.0] * (self.dimension or 1536)

                results = self.client.search(
                    self.table_name,
                    vector=dummy_vector,
                    k=10000,
                    filter=dxmt_filter,
                )

                ids = [hit["id"] for hit in results.get("hits", [])]
                if ids:
                    self.client.delete_records(self.table_name, ids)

            except DXMTError:
                pass

    def clear(self) -> None:
        """Clear all nodes from table.

        This deletes the entire table.
        """
        try:
            self.client.delete_table(self.table_name)
            self._table_created = False
        except DXMTError:
            # Table might not exist
            pass

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query vector store.

        Args:
            query: Vector store query object containing:
                - query_embedding: Query vector for similarity search
                - query_str: Query string for text search
                - similarity_top_k: Number of top results to return
                - filters: Optional metadata filters
                - mode: Query mode (DEFAULT, HYBRID, TEXT_SEARCH, etc.)
                - alpha: Hybrid search weight (0=text only, 1=vector only)
            **kwargs: Additional query options.

        Returns:
            VectorStoreQueryResult containing nodes, similarities, and IDs.
        """
        # Convert filters if present
        dxmt_filter = None
        if query.filters:
            dxmt_filter = convert_llamaindex_filters_to_dxmt(query.filters)

        # Determine query mode and execute appropriate search
        results = None

        if query.mode == VectorStoreQueryMode.HYBRID:
            # Hybrid search (vector + text)
            if not query.query_embedding:
                raise ValueError("query_embedding required for hybrid search")
            if not query.query_str:
                raise ValueError("query_str required for hybrid search")

            alpha = query.alpha if query.alpha is not None else 0.7

            results = self.client.hybrid_search(
                self.table_name,
                vector=query.query_embedding,
                text=query.query_str,
                k=query.similarity_top_k,
                vector_weight=alpha,
                filter=dxmt_filter,
            )

        elif query.mode == VectorStoreQueryMode.TEXT_SEARCH:
            # Full-text search only
            if not query.query_str:
                raise ValueError("query_str required for text search")

            results = self.client.full_text_search(
                self.table_name,
                query=query.query_str,
                k=query.similarity_top_k,
                filter=dxmt_filter,
            )

        else:
            # Standard vector search (DEFAULT mode)
            if not query.query_embedding:
                raise ValueError("query_embedding required for vector search")

            results = self.client.search(
                self.table_name,
                vector=query.query_embedding,
                k=query.similarity_top_k,
                filter=dxmt_filter,
            )

        # Convert results to nodes
        # Note: DXMT SDK returns a list of hits directly, not a dict
        nodes = []
        similarities = []
        ids = []

        for hit in results:
            # Reconstruct node from metadata
            node = dxmt_hit_to_node(hit)

            nodes.append(node)
            similarities.append(hit.get("score", 0.0))
            ids.append(hit["id"])

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )

    async def aquery(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Async version of query.

        Args:
            query: Vector store query object.
            **kwargs: Additional query options.

        Returns:
            VectorStoreQueryResult containing nodes, similarities, and IDs.
        """
        # For now, use synchronous version
        # TODO: Implement with async client when needed
        return self.query(query, **kwargs)

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Get nodes by IDs or filters.

        Args:
            node_ids: Optional list of node IDs to retrieve.
            filters: Optional metadata filters.

        Returns:
            List of matching nodes.
        """
        if node_ids is not None:
            # Get by IDs directly
            try:
                records = self.client.get_records(self.table_name, node_ids)
                return [dxmt_hit_to_node(record) for record in records]
            except DXMTError:
                return []

        elif filters is not None:
            # Get by filter - use search with dummy vector
            try:
                dxmt_filter = convert_llamaindex_filters_to_dxmt(filters)
                dummy_vector = [0.0] * (self.dimension or 1536)

                results = self.client.search(
                    self.table_name,
                    vector=dummy_vector,
                    k=10000,
                    filter=dxmt_filter,
                )

                # Note: DXMT SDK returns a list of hits directly
                return [dxmt_hit_to_node(hit) for hit in results]
            except DXMTError:
                return []

        return []

    @classmethod
    def from_existing_table(
        cls,
        table_name: str,
        client: Optional[DXMTClient] = None,
        host: str = "localhost",
        port: int = 9999,
        ssl: bool = False,
        **kwargs: Any,
    ) -> "DXMTVectorStore":
        """Create DXMTVectorStore from an existing table.

        This method queries the table metadata to automatically configure
        the vector store with the correct dimension and distance metric.

        Args:
            table_name: Name of existing table.
            client: Optional existing DXMT client.
            host: DXMT server host if client not provided.
            port: DXMT server port if client not provided.
            ssl: Whether to use SSL if client not provided.
            **kwargs: Additional arguments passed to DXMTVectorStore.

        Returns:
            Configured DXMTVectorStore instance.

        Raises:
            DXMTError: If table doesn't exist or can't be accessed.
        """
        if client is None:
            client = DXMTClient(host=host, port=port, ssl=ssl)

        # Get table info
        table_info = client.get_table(table_name)

        return cls(
            table_name=table_name,
            client=client,
            dimension=table_info.get("dimension"),
            distance_metric=table_info.get("distance_metric", "cosine"),
            index_type=table_info.get("index_type", "hnsw"),
            enable_full_text_search=table_info.get("enable_full_text_search", False),
            **kwargs
        )

    @classmethod
    def class_name(cls) -> str:
        """Return class name for serialization."""
        return "DXMTVectorStore"
