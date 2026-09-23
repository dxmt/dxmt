"""DXMT VectorStore for LlamaIndex.

This module provides a LlamaIndex VectorStore implementation for DXMT VectorDB,
enabling integration with LlamaIndex-based frameworks like MetaGPT.

Example:
    ```python
    from llama_index.core import VectorStoreIndex, StorageContext
    from dxmt import DXMTClient
    from dxmt.metagpt_vector_store import DXMTVectorStore

    # Create DXMT vector store
    client = DXMTClient(host="localhost", port=9999)
    vector_store = DXMTVectorStore(
        client=client,
        table_name="documents",
        dimension=1536
    )

    # Create storage context and index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )
    ```
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, List, Optional

from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from pydantic import ConfigDict, PrivateAttr

from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

from .client import DXMTClient
from .exceptions import DXMTError
from .filters import convert_simple_filter_to_dxmt

logger = logging.getLogger(__name__)


class DXMTVectorStore(BasePydanticVectorStore):
    """DXMT Vector Store for LlamaIndex.

    Implements LlamaIndex's BasePydanticVectorStore interface to enable
    DXMT VectorDB integration with LlamaIndex and MetaGPT.

    Args:
        client: DXMTClient instance. If None, creates with default settings.
        table_name: Name of the DXMT table to use.
        dimension: Vector dimension. Required if table doesn't exist.
        distance_metric: Distance metric (cosine, euclidean, dot, manhattan).
        create_if_not_exists: Create table if it doesn't exist.
        host: DXMT server host (used if client is None).
        port: DXMT server port (used if client is None).

    Attributes:
        stores_text: Always True - DXMT stores text in metadata.
        is_embedding_query: Always True - queries use embeddings.
    """

    stores_text: bool = True
    is_embedding_query: bool = True

    # Pydantic fields. `client` is not one of them: `BasePydanticVectorStore` declares it as an
    # abstract property, and a field of the same name shadows it -- pydantic said so in a
    # warning printed on every `import dxmt` once llama-index was installed. It is a private
    # attribute with the property the interface asks for, which is what the standalone
    # `llama-index-vector-stores-dxmt` package has always done.
    _client: Optional[DXMTClient] = PrivateAttr(default=None)
    table_name: str = "documents"
    dimension: int = 1536
    distance_metric: str = "cosine"
    create_if_not_exists: bool = True
    host: str = "localhost"
    port: int = 9999

    _initialized: bool = PrivateAttr(default=False)

    # A class-based `Config` is the pydantic v1 spelling, deprecated since v2 and removed in
    # v3; it warned on every import of this module.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        client: Optional[DXMTClient] = None,
        table_name: str = "documents",
        dimension: int = 1536,
        distance_metric: str = "cosine",
        create_if_not_exists: bool = True,
        host: str = "localhost",
        port: int = 9999,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize DXMT vector store.

        ``api_key`` goes to the client and is not kept as a field: the other arguments are
        pydantic fields and come back out of ``repr()`` and serialisation.
        """
        super().__init__(
            table_name=table_name,
            dimension=dimension,
            distance_metric=distance_metric,
            create_if_not_exists=create_if_not_exists,
            host=host,
            port=port,
            **kwargs,
        )

        # Create client if not provided
        self._client = client or DXMTClient(host=self.host, port=self.port, api_key=api_key)

        # Initialize table
        if create_if_not_exists and not self._initialized:
            self._ensure_table_exists()
            self._initialized = True

    def _ensure_table_exists(self) -> None:
        """Ensure the DXMT table exists."""
        try:
            # Try to get table
            self.client.get_table(self.table_name)
            logger.info(f"Table {self.table_name} already exists")
        except DXMTError:
            # Table doesn't exist, create it
            logger.info(f"Creating table {self.table_name}")
            self.client.create_table(
                name=self.table_name,
                dimension=self.dimension,
                distance_metric=self.distance_metric,
                description=f"DXMT vector store for LlamaIndex (dimension={self.dimension})"
            )

    @property
    def client(self) -> DXMTClient:
        """The underlying DXMT client, as `BasePydanticVectorStore` requires."""
        assert self._client is not None  # set in __init__, which is the only way in
        return self._client

    @property
    def client_instance(self) -> DXMTClient:
        """The client, under the name this class used before `client` became a property."""
        return self.client

    def add(
        self,
        nodes: List[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """Add nodes to DXMT.

        Args:
            nodes: List of nodes to add.
            **kwargs: Additional arguments.

        Returns:
            List of node IDs.
        """
        if not nodes:
            return []

        # Convert nodes to DXMT records
        records = []
        for node in nodes:
            # Get embedding
            embedding = node.get_embedding()
            if embedding is None:
                logger.warning(f"Node {node.node_id} has no embedding, skipping")
                continue

            # Convert node to metadata dict
            metadata = node_to_metadata_dict(node, remove_text=False, flat_metadata=False)

            # Add text content explicitly
            if hasattr(node, 'text') and node.text:
                metadata['text'] = node.text

            # Create record
            record = {
                "id": node.node_id,
                "vector": embedding,
                "metadata": metadata
            }
            records.append(record)

        if not records:
            return []

        # Insert into DXMT
        try:
            self.client.insert_records(
                table_name=self.table_name,
                records=records
            )
            logger.info(f"Added {len(records)} nodes to {self.table_name}")
            return [r["id"] for r in records]
        except Exception as e:
            logger.error(f"Error adding nodes: {e}")
            raise DXMTError(f"Failed to add nodes: {e}")

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Delete nodes by reference document ID.

        Args:
            ref_doc_id: Reference document ID.
            **kwargs: Additional arguments.
        """
        try:
            # Search for all nodes with this ref_doc_id using a filter-only scan.
            # A dummy zero vector is used because DXMT has no filter-scan endpoint;
            # results are restricted by the ref_doc_id filter, not by proximity.
            results = self.client.search(
                table_name=self.table_name,
                vector=[0.0] * self.dimension,
                k=10000,
                filter={
                    "must": [
                        {
                            "key": "ref_doc_id",
                            "operation": {"type": "Equal", "value": ref_doc_id},
                        }
                    ]
                },
            )

            node_ids = [r.get("id") for r in results if r.get("id")]
            if node_ids:
                self.client.delete_records(
                    table_name=self.table_name,
                    ids=node_ids,
                )

            logger.info(f"Deleted {len(node_ids)} nodes with ref_doc_id={ref_doc_id}")
        except Exception as e:
            logger.error(f"Error deleting nodes: {e}")
            raise DXMTError(f"Failed to delete nodes: {e}")

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query DXMT vector store.

        Respects VectorStoreQuery.mode:
            - DEFAULT / None     → dense vector search
            - HYBRID             → hybrid_search (vector + text)
            - SEMANTIC_HYBRID    → the same; `alpha`, when the query carries one, is the
              vector side's share and selects weighted fusion
            - SPARSE / TEXT_SEARCH → full_text_search (BM25 only)

        Args:
            query: Vector store query.
            **kwargs: Additional arguments.

        Returns:
            Query results with nodes and similarities.
        """
        try:
            from llama_index.core.vector_stores.types import VectorStoreQueryMode
            _MODE = VectorStoreQueryMode
        except ImportError:
            _MODE = None

        dxmt_filter = self._convert_filters(query.filters) if query.filters else None
        top_k = query.similarity_top_k or 5
        mode = getattr(query, "mode", None)

        try:
            # --- HYBRID modes ---
            if _MODE and mode in (_MODE.HYBRID, _MODE.SEMANTIC_HYBRID):
                # Pass the query's alpha through rather than inventing one: a weight selects
                # weighted fusion, and a caller who set none should get the server's default
                # (`dbsf`), which needs no weight.
                results = self.client.hybrid_search(
                    table_name=self.table_name,
                    vector=query.query_embedding,
                    text=getattr(query, "query_str", None),
                    alpha=getattr(query, "alpha", None),
                    k=top_k,
                    filter=dxmt_filter,
                )
            # --- SPARSE / TEXT_SEARCH modes ---
            elif _MODE and mode in (_MODE.SPARSE, _MODE.TEXT_SEARCH):
                query_str = getattr(query, "query_str", None)
                if not query_str:
                    raise ValueError("query_str is required for SPARSE/TEXT_SEARCH mode")
                results = self.client.full_text_search(
                    table_name=self.table_name,
                    query=query_str,
                    k=top_k,
                    filter=dxmt_filter,
                )
            # --- DEFAULT: dense vector search ---
            else:
                if query.query_embedding is None:
                    raise ValueError("query_embedding is required for DEFAULT mode")
                results = self.client.search(
                    table_name=self.table_name,
                    vector=query.query_embedding,
                    k=top_k,
                    filter=dxmt_filter,
                )
        except (ValueError, AttributeError):
            raise
        except Exception as e:
            logger.error(f"Error querying DXMT: {e}")
            raise DXMTError(f"Failed to query: {e}")

        # Convert results to LlamaIndex format
        nodes = []
        similarities = []
        ids = []

        for result in results:
            # Extract metadata
            metadata = result.get("metadata", {})

            # Reconstruct node from metadata
            try:
                node = metadata_dict_to_node(metadata)
                node.id_ = result.get("id")
            except Exception:
                # Fallback: create TextNode manually
                text = metadata.get("text", "")
                node = TextNode(
                    text=text,
                    id_=result.get("id"),
                    metadata=metadata,
                    embedding=result.get("vector")
                )

            nodes.append(node)

            # Get similarity score
            score = result.get("score", 0.0)
            similarities.append(score)

            # Get ID
            ids.append(result.get("id"))

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )

    def persist(
        self,
        persist_path: str,
        fs: Optional[Any] = None,
    ) -> None:
        """Persist the vector store.

        DXMT automatically persists data, so this is a no-op.

        Args:
            persist_path: Path to persist to (unused).
            fs: Filesystem (unused).
        """
        logger.info("DXMT automatically persists data")

    def _convert_filters(self, filters: Any) -> Optional[dict]:
        """Convert LlamaIndex MetadataFilters (or plain dict) to DXMT format.

        Handles:
            - LlamaIndex MetadataFilters object (with .filters list of
              MetadataFilter, each having .key / .value / .operator)
            - Plain dict (passed through convert_simple_filter_to_dxmt which
              supports MongoDB-style operators: $eq, $ne, $gt, $gte, $lt,
              $lte, $in, $nin, $contains, $startswith, $endswith, $and, $or)

        Args:
            filters: LlamaIndex filters or plain dict.

        Returns:
            DXMT filter dict, or None.
        """
        if filters is None:
            return None

        # LlamaIndex MetadataFilters object
        if hasattr(filters, "filters") and hasattr(filters, "condition"):
            from llama_index.core.vector_stores.types import FilterOperator, FilterCondition
            _OP_MAP = {
                FilterOperator.EQ: "Equal",
                FilterOperator.NE: "NotEqual",
                FilterOperator.GT: "GreaterThan",
                FilterOperator.GTE: "GreaterThanOrEqual",
                FilterOperator.LT: "LessThan",
                FilterOperator.LTE: "LessThanOrEqual",
                FilterOperator.IN: "In",
                FilterOperator.NIN: "NotIn",
                FilterOperator.CONTAINS: "Contains",
            }
            conditions = []
            for f in filters.filters:
                op_type = _OP_MAP.get(f.operator, "Equal")
                conditions.append({
                    "key": f.key,
                    "operation": {"type": op_type, "value": f.value},
                })
            if not conditions:
                return None
            cond_key = "must"
            if hasattr(FilterCondition, "OR") and filters.condition == FilterCondition.OR:
                cond_key = "should"
            return {cond_key: conditions}

        # Plain dict — delegate to shared filter utility
        if isinstance(filters, dict):
            return convert_simple_filter_to_dxmt(filters)

        return None

    def clear(self) -> None:
        """Clear all nodes from the table.

        Deletes and recreates the table.
        """
        try:
            self.client.delete_table(self.table_name)
            logger.info(f"Deleted table {self.table_name}")

            # Recreate table
            self._ensure_table_exists()
            logger.info(f"Recreated table {self.table_name}")
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            raise DXMTError(f"Failed to clear: {e}")

    async def async_add(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        """Async add nodes."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.add(nodes, **kwargs))

    async def aquery(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Async query."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.query(query, **kwargs))

    async def adelete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Async delete by ref_doc_id."""
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self.delete(ref_doc_id, **kwargs))
