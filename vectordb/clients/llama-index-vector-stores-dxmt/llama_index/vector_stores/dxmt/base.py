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
    - MMR, reranked on the server rather than by a client-side postprocessor
    - Sparse-vector search, when a ``sparse_encoder`` is configured
    - Hybrid search combining vector and full-text search (server needs ``text-bm25``)
    - Metadata filtering with complex filter expressions
    - Batch operations

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
    batch_size: int = 100

    # Private attributes
    _client: Optional[DXMTClient] = PrivateAttr(default=None)
    _table_created: bool = PrivateAttr(default=False)
    #: Text -> `{"indices": [...], "values": [...], "dimension": n}`. A private attribute and
    #: not a field, for the reason `api_key` is not one: a pydantic field comes back out of
    #: `to_dict()` and a serialised index, and a callable does not survive that trip.
    _sparse_encoder: Optional[Callable[[str], Dict[str, Any]]] = PrivateAttr(default=None)

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
        batch_size: int = 100,
        api_key: Optional[str] = None,
        sparse_encoder: Optional[Callable[[str], Dict[str, Any]]] = None,
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
                Options: "standard", "lock_free", "streaming", "deferred_index",
                "ultimate", "batch", etc. Defaults to "standard".
            batch_size: Batch size for bulk operations. Defaults to 100.
            sparse_encoder: Turns a string into ``{"indices": [...], "values": [...],
                "dimension": n}``. Set it and records carry a sparse vector on the way in and
                ``VectorStoreQueryMode.SPARSE`` becomes answerable; leave it unset and sparse
                mode is refused rather than quietly answered with a dense search.
                ``LlamaIndex``'s ``VectorStoreQuery`` has no field for a sparse vector, so the
                encoder has to live on the store -- the same shape the Pinecone and Qdrant
                integrations use.
            api_key: The server's API key, if it was started with ``DXMT_API_KEY``. Handed to
                the client and not kept on this model: every other argument here is a pydantic
                field and comes back out of ``to_dict()``, ``repr()`` and a serialised index,
                which is exactly where a credential must not go. Ignored when ``client`` is
                given — configure that client instead.
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
                api_key=api_key,
            )

        self._table_created = False
        self._sparse_encoder = sparse_encoder

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
        sparse_encoder = add_kwargs.get("sparse_encoder") or self._sparse_encoder

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
            self.client.insert_records(self.table_name, batch)

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

    def _ids_matching(self, dxmt_filter: dict) -> List[str]:
        """Every record id the filter selects.

        Three things this replaces, each of which was a defect in both callers:

        **It reads the answer correctly.** The old code did `results.get("hits", [])` on the
        return of `client.search`, which is a `list`. `query()` twenty lines below has the
        comment "the DXMT SDK returns a list of hits directly, not a dict" -- the same file
        knew. So `delete()` and `delete_nodes(filters=...)` raised
        `AttributeError: 'list' object has no attribute 'get'` and deleted nothing, on every
        call there has ever been. The unit tests mock the client, and a `MagicMock` does have
        `.get()`; the integration tests cover `delete_nodes(node_ids=...)`, which takes the
        other branch. Neither could see it.

        **It scrolls instead of searching with a zero vector.** Selecting rows by filter is
        not a nearest-neighbour question, and the old code faked one with `[0.0] * dimension`
        -- a vector with no direction, which under this store's default cosine metric orders
        the results arbitrarily. That did not matter while the whole match set came back, and
        it mattered a great deal at the `k=10000` cap: past ten thousand rows the cap silently
        decided *which* records were deleted, by an arbitrary ordering. `scroll` pages with a
        cursor, so there is no cap and no vector.

        **It does not swallow the error.** Both callers ended in `except DXMTError: pass`, so
        a delete that failed for any reason at all reported success. A missing table is the
        one case where doing nothing is right -- deleting from a table that is not there has
        already achieved what the caller asked for -- and that case is now the only one caught.
        """
        try:
            return [row["id"] for row in self.client.scroll(self.table_name, filter=dxmt_filter)]
        except DXMTError as e:
            if "TABLE_NOT_FOUND" in str(e):
                return []
            raise

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete nodes by ref_doc_id.

        This method deletes all nodes that have the specified ref_doc_id
        in their metadata.

        Args:
            ref_doc_id: Reference document ID to delete.
            **delete_kwargs: Additional delete options.
        """
        # There is no delete-by-filter endpoint, so this selects the ids first and then
        # deletes them.
        ids = self._ids_matching({"ref_doc_id": ref_doc_id})
        if ids:
            self.client.delete_records(self.table_name, ids)

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
            ids = self._ids_matching(convert_llamaindex_filters_to_dxmt(filters))
            if ids:
                self.client.delete_records(self.table_name, ids)

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
                - mode: DEFAULT, MMR, SPARSE, HYBRID or TEXT_SEARCH. Anything else is
                  refused by name rather than answered with a dense search.
                - mmr_threshold: MMR's lambda, 0 = most diverse, 1 = plain ranking
                - sparse_top_k: page size for SPARSE, falling back to similarity_top_k
                - alpha: Hybrid search weight (0=text only, 1=vector only)
            **kwargs: Additional query options. `fetch_k` sets how many candidates MMR
                reranks; diversity can only come from candidates that were fetched.

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

            # `alpha` selects weighted fusion, where it means what its name says: the share
            # of the score that comes from the vector side. Passing it unconditionally would
            # force that method on every caller, including the ones who never set it -- so
            # when it is unset the server stays on its own default fusion (`dbsf`), which
            # normalises each side by its distribution and needs no weight. Passing
            # `query.alpha` straight through is what implements that: it defaults to `None`,
            # and the SDK's `_fusion_body` omits both fields for `None` rather than inventing
            # a weight. Verified rather than assumed -- `_fusion_body(None, None, None)`
            # returns `{}`.
            results = self.client.hybrid_search(
                self.table_name,
                vector=query.query_embedding,
                text=query.query_str,
                k=query.similarity_top_k,
                vector_weight=query.alpha,
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

        elif query.mode == VectorStoreQueryMode.DEFAULT:
            if not query.query_embedding:
                raise ValueError("query_embedding required for vector search")

            results = self.client.search(
                self.table_name,
                vector=query.query_embedding,
                k=query.similarity_top_k,
                filter=dxmt_filter,
            )

        elif query.mode == VectorStoreQueryMode.MMR:
            # Diversified ranking, reranked **on the server**. LlamaIndex's own MMR is a
            # postprocessor over whatever the store returned, which can only diversify within
            # `similarity_top_k` rows and needs their vectors on the client; DXMT reranks
            # inside `assemble_hits`, where the candidates' vectors are already to hand.
            #
            # `mmr_threshold` and `lambda_mult` are the same λ, checked rather than assumed:
            # both score `λ·sim(q,d) - (1-λ)·max sim(d,d')`, both default to 0.5, and in both
            # 0 is maximum diversity and 1 is the plain ranking. Passing it straight through
            # would be a silent wrong answer if either had been the other way round.
            if not query.query_embedding:
                raise ValueError("query_embedding required for MMR search")

            # Diversity can only come from candidates that were fetched, so `fetch_k` is the
            # real knob and LlamaIndex has nowhere to put it. Five times the page, floored at
            # the server's own default of 20 -- the ratio LangChain uses for its 4/20 pair --
            # and `fetch_k=` overrides it.
            fetch_k = kwargs.get("fetch_k") or max(query.similarity_top_k * 5, 20)
            mmr = {"fetch_k": fetch_k}
            if query.mmr_threshold is not None:
                mmr["lambda_mult"] = query.mmr_threshold

            results = self.client.search(
                self.table_name,
                vector=query.query_embedding,
                k=query.similarity_top_k,
                filter=dxmt_filter,
                mmr=mmr,
            )

        elif query.mode == VectorStoreQueryMode.SPARSE:
            # `VectorStoreQuery` carries no sparse vector -- only `sparse_top_k`, a size for a
            # thing it has no field for -- so the encoder lives on the store and runs here.
            if self._sparse_encoder is None:
                raise ValueError(
                    "VectorStoreQueryMode.SPARSE needs a sparse encoder. Pass "
                    "DXMTVectorStore(..., sparse_encoder=fn), where fn maps a string to "
                    '{"indices": [...], "values": [...], "dimension": n}. The same encoder '
                    "is used on the way in, so the two halves cannot disagree."
                )
            if not query.query_str:
                raise ValueError("query_str required for sparse search")

            results = self.client.sparse_search(
                self.table_name,
                sparse_vector=self._sparse_encoder(query.query_str),
                limit=query.sparse_top_k or query.similarity_top_k,
                filter=dxmt_filter,
            )

        else:
            # `VectorStoreQueryMode` has nine values and this store serves five. The other
            # four used to land here when this was a bare `else` reading "standard vector
            # search (DEFAULT mode)", so asking for SEMANTIC_HYBRID or any of the regression
            # modes returned a plain dense page with no warning and no error -- which a caller
            # cannot tell from a result that honoured the mode, the worst shape a wrong answer
            # can take.
            raise NotImplementedError(
                f"DXMTVectorStore does not serve VectorStoreQueryMode.{query.mode.name}. "
                f"Supported: DEFAULT, MMR, SPARSE, HYBRID, TEXT_SEARCH (the last two need a "
                f"server built with --features text-bm25; SPARSE needs a sparse_encoder)."
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
        api_key: Optional[str] = None,
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
            api_key: The server's API key, if client not provided. It has to be a parameter
                here rather than in ``kwargs``: the table is read before the store is built.
            **kwargs: Additional arguments passed to DXMTVectorStore.

        Returns:
            Configured DXMTVectorStore instance.

        Raises:
            DXMTError: If table doesn't exist or can't be accessed.
        """
        if client is None:
            client = DXMTClient(host=host, port=port, ssl=ssl, api_key=api_key)

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
