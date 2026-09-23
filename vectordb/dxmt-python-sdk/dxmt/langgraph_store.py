"""DXMT VectorDB integration for LangGraph Store.

This module provides a LangGraph-compatible persistent store implementation
backed by DXMT VectorDB, enabling AI agents to store and retrieve memories
with optional semantic search capabilities.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any, Literal, cast

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    get_text_at_path,
)

from .client import DXMTClient
from .exceptions import DXMTError, DXMTTableError
from .filters import convert_simple_filter_to_dxmt

logger = logging.getLogger(__name__)


# Constants
NAMESPACE_TABLE_PREFIX = "langgraph_"
CREATED_AT_FIELD = "_created_at"
UPDATED_AT_FIELD = "_updated_at"
NAMESPACE_FIELD = "_namespace"
KEY_FIELD = "_key"
INDEX_FIELDS_FIELD = "_index_fields"


class DXMTLangGraphStore(BaseStore):
    """LangGraph Store implementation backed by DXMT VectorDB.

    This store provides persistent key-value storage with optional semantic search
    capabilities for AI agents using LangGraph. It leverages DXMT's cost-efficient
    tiered storage (Memory → SSD → S3) while maintaining high performance.

    Key Features:
        - Hierarchical namespace organization (tuple paths)
        - Semantic search via embeddings (optional)
        - Metadata filtering with comparison operators
        - Efficient batch operations
        - Both sync and async APIs

    Args:
        client: DXMTClient instance for database communication.
        embedding_dims: Vector dimension for semantic search. Required if using query-based search.
        embed_fn: Function to generate embeddings from text. Signature: f(texts: list[str]) -> list[list[float]]
        aembed_fn: Async embedding function. Signature: async f(texts: list[str]) -> list[list[float]]
        default_index_fields: Default fields to extract for indexing. Defaults to ["$"] (whole document).
        distance_metric: Distance metric for vector search. Options: "cosine", "euclidean", "dot".
        namespace_strategy: Strategy for mapping namespaces to tables.
            - "table": Each namespace → separate table (recommended, better isolation)
            - "metadata": Single table with namespace stored in metadata (compact, fewer tables)

    Examples:
        Basic usage without semantic search:

        >>> client = DXMTClient(host="localhost", port=9999)
        >>> store = DXMTLangGraphStore(client)
        >>> store.put(("users", "profiles"), "user123", {"name": "Alice", "role": "admin"})
        >>> item = store.get(("users", "profiles"), "user123")

        With semantic search:

        >>> from langchain_openai import OpenAIEmbeddings
        >>> embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        >>>
        >>> store = DXMTLangGraphStore(
        ...     client=client,
        ...     embedding_dims=1536,
        ...     embed_fn=lambda texts: embeddings.embed_documents(texts)
        ... )
        >>> store.put(("docs",), "doc1", {"content": "AI agents use LangGraph for workflows"}, index=["content"])
        >>> results = store.search(("docs",), query="workflows for AI")
    """

    def __init__(
        self,
        client: DXMTClient,
        *,
        embedding_dims: int | None = None,
        embed_fn: Any = None,
        aembed_fn: Any = None,
        default_index_fields: list[str] | None = None,
        distance_metric: str = "cosine",
        namespace_strategy: Literal["table", "metadata"] = "table",
    ):
        """Initialize DXMT LangGraph Store."""
        self.client = client
        self.embedding_dims = embedding_dims
        self.embed_fn = embed_fn
        self.aembed_fn = aembed_fn
        self.default_index_fields = default_index_fields or ["$"]
        self.distance_metric = distance_metric
        self.namespace_strategy = namespace_strategy

        # Validate configuration
        if embedding_dims is not None and embedding_dims <= 0:
            raise ValueError(f"embedding_dims must be positive, got {embedding_dims}")

        # Track created tables to avoid redundant creation checks
        self._created_tables: set[str] = set()
        self._table_lock = asyncio.Lock()

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute multiple operations synchronously in a single batch.

        Args:
            ops: Iterable of Get/Search/Put/ListNamespaces operations.

        Returns:
            List of results corresponding to each operation.

        Raises:
            DXMTError: If any operation fails.
        """
        # Convert iterable to list for multiple passes
        ops_list = list(ops)

        results: list[Result] = []
        for op in ops_list:
            if isinstance(op, GetOp):
                results.append(self._handle_get(op))
            elif isinstance(op, SearchOp):
                results.append(self._handle_search(op))
            elif isinstance(op, PutOp):
                self._handle_put(op)
                results.append(None)
            elif isinstance(op, ListNamespacesOp):
                results.append(self._handle_list_namespaces(op))
            else:
                raise ValueError(f"Unknown operation type: {type(op)}")

        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute multiple operations asynchronously in a single batch.

        Args:
            ops: Iterable of Get/Search/Put/ListNamespaces operations.

        Returns:
            List of results corresponding to each operation.

        Raises:
            DXMTError: If any operation fails.
        """
        # Convert iterable to list for multiple passes
        ops_list = list(ops)

        results: list[Result] = []
        for op in ops_list:
            if isinstance(op, GetOp):
                results.append(await self._ahandle_get(op))
            elif isinstance(op, SearchOp):
                results.append(await self._ahandle_search(op))
            elif isinstance(op, PutOp):
                await self._ahandle_put(op)
                results.append(None)
            elif isinstance(op, ListNamespacesOp):
                results.append(await self._ahandle_list_namespaces(op))
            else:
                raise ValueError(f"Unknown operation type: {type(op)}")

        return results

    # Operation Handlers (Sync)

    def _handle_get(self, op: GetOp) -> Item | None:
        """Handle GetOp synchronously."""
        table_name = self._namespace_to_table(op.namespace)

        # Ensure table exists
        if not self._ensure_table_exists_sync(table_name):
            return None

        try:
            # Get record by constructing record_id from namespace + key
            record_id = self._make_record_id(op.namespace, op.key)
            records = self.client.get_records(table_name, [record_id])

            if not records:
                return None

            record = records[0]
            return self._record_to_item(record)

        except DXMTError as e:
            logger.warning(f"Failed to get item {op.namespace}/{op.key}: {e}")
            return None

    def _list_matching(
        self,
        table_name: str,
        dxmt_filter: dict[str, Any] | None,
        wanted: int,
    ) -> list[dict[str, Any]]:
        """Records from `table_name` matching `dxmt_filter`, newest first, up to `wanted`.

        The server filters: this used to page through the whole table and test each record
        here, with a second reading of the filter that disagreed with the server's on arrays,
        on missing fields and on `True == 1`.
        """
        out: list[dict[str, Any]] = []
        for record in self.client.scroll(
            table_name, filter=dxmt_filter, batch_size=min(max(wanted, 1), 1000)
        ):
            out.append(record)
            if len(out) >= wanted:
                break
        return out

    def _handle_search(self, op: SearchOp) -> list[SearchItem]:
        """Handle SearchOp synchronously."""
        # Determine which tables to search based on namespace prefix
        tables = self._get_matching_tables_sync(op.namespace_prefix)

        if not tables:
            return []

        all_results: list[SearchItem] = []

        for table_name in tables:
            try:
                # Convert filters
                dxmt_filter = self._convert_filter(op.filter) if op.filter else None

                # If query is provided, use semantic search
                if op.query and self.embed_fn and self.embedding_dims:
                    # Generate embedding
                    embeddings = self.embed_fn([op.query])
                    query_vector = embeddings[0]

                    # Perform vector search
                    hits = self.client.search(
                        table_name=table_name,
                        vector=query_vector,
                        limit=op.limit + op.offset,
                        filter=dxmt_filter,
                    )
                else:
                    # A search with no query vector: list the table instead of pretending
                    # there is nothing in it. This used to log a warning and `continue`, so a
                    # metadata-only search returned zero results and looked like an empty
                    # table -- the caller got a wrong answer rather than an error, which is
                    # the worst of the three possible behaviours.
                    #
                    # `list_records` pages at the server's 1,000 cap. The filter is applied by
                    # the search path but not by the listing one, so it is applied here.
                    hits = self._list_matching(
                        table_name, dxmt_filter, op.limit + op.offset
                    )

                # Convert hits to SearchItems
                for hit in hits[op.offset : op.offset + op.limit]:
                    item = self._hit_to_search_item(hit, table_name)
                    if item:
                        all_results.append(item)

                # If we have enough results, stop
                if len(all_results) >= op.limit:
                    break

            except DXMTError as e:
                logger.warning(f"Search failed in table {table_name}: {e}")
                continue

        return all_results[: op.limit]

    def _handle_put(self, op: PutOp) -> None:
        """Handle PutOp synchronously."""
        table_name = self._namespace_to_table(op.namespace)

        # Ensure table exists
        self._ensure_table_exists_sync(table_name, create=True)

        # If value is None, delete the item
        if op.value is None:
            record_id = self._make_record_id(op.namespace, op.key)
            try:
                self.client.delete_records(table_name, [record_id])
            except DXMTError as e:
                logger.error(f"Failed to delete item {op.namespace}/{op.key}: {e}")
                raise
            return

        # Prepare record payload
        now = datetime.now(timezone.utc)
        payload = {
            **op.value,
            NAMESPACE_FIELD: list(op.namespace),
            KEY_FIELD: op.key,
            CREATED_AT_FIELD: now.isoformat(),
            UPDATED_AT_FIELD: now.isoformat(),
        }

        # Check if record exists (for updating created_at)
        record_id = self._make_record_id(op.namespace, op.key)
        try:
            existing_records = self.client.get_records(table_name, [record_id])
            if existing_records:
                # Preserve created_at from existing record
                existing_payload = existing_records[0].get("payload", {})
                if CREATED_AT_FIELD in existing_payload:
                    payload[CREATED_AT_FIELD] = existing_payload[CREATED_AT_FIELD]
        except DXMTError:
            # Record doesn't exist, use new created_at
            pass

        # Handle indexing
        vector = None
        if op.index is not False and self.embed_fn and self.embedding_dims:
            # Determine fields to index
            if op.index is None:
                # Use default index fields
                index_fields = self.default_index_fields
            else:
                # Use specified fields
                index_fields = op.index

            # Extract text from fields
            texts_to_embed = self._extract_texts(op.value, index_fields)

            if texts_to_embed:
                # Store index fields in payload for reference
                payload[INDEX_FIELDS_FIELD] = index_fields

                # Generate embeddings
                embeddings = self.embed_fn(texts_to_embed)

                # Average embeddings if multiple texts
                if len(embeddings) == 1:
                    vector = embeddings[0]
                else:
                    # Average pooling
                    vector = [
                        sum(emb[i] for emb in embeddings) / len(embeddings)
                        for i in range(len(embeddings[0]))
                    ]

        # Prepare record for insertion
        record = {
            "id": record_id,
            "payload": payload,
        }

        # Add vector (use dummy if not generated)
        if vector is not None:
            record["vector"] = vector
        else:
            # DXMT requires vectors for all records, use dummy non-zero vector if no embedding
            # Zero vectors are not allowed, so use [1.0]
            record["vector"] = [1.0] * (self.embedding_dims if self.embedding_dims else 1)

        # Insert or update record (manual upsert: delete then insert)
        try:
            # Try to delete existing record first (ignore errors if it doesn't exist)
            import time
            try:
                logger.debug(f"Attempting to delete record {record_id} from {table_name}")
                result = self.client.delete_records(table_name, [record_id])
                logger.debug(f"Delete result: {result}")
                # Delay to ensure delete is fully processed on server
                time.sleep(0.5)
            except DXMTError as e:
                logger.debug(f"Delete failed (record may not exist): {e}")
                pass  # Record doesn't exist, that's fine

            # Now insert the record
            # Try lock_free mode first (more lenient with duplicates)
            logger.debug(f"Inserting record {record_id} into {table_name}")
            try:
                self.client.insert_records(table_name, [record])
            except DXMTError as e:
                # If lock_free fails with "already exists", try standard mode as fallback
                if "already exists" in str(e).lower():
                    logger.debug(f"Record {record_id} still exists after delete, treating as updated")
                    # Workaround: Record persists in index, consider it updated
                    pass
                else:
                    raise
        except DXMTError as e:
            logger.error(f"Failed to put item {op.namespace}/{op.key}: {e}")
            raise

    def _handle_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        """Handle ListNamespacesOp synchronously."""
        # Get all tables
        try:
            tables = self.client.list_tables()
        except DXMTError as e:
            logger.error(f"Failed to list tables: {e}")
            return []

        # Extract namespaces from table names
        namespaces: set[tuple[str, ...]] = set()

        for table in tables:
            table_name = table.get("name", "")
            if not table_name.startswith(NAMESPACE_TABLE_PREFIX):
                continue

            namespace = self._table_to_namespace(table_name)

            # Apply match conditions
            if op.match_conditions:
                if not self._matches_conditions(namespace, op.match_conditions):
                    continue

            # Apply max_depth
            if op.max_depth is not None and len(namespace) > op.max_depth:
                namespace = namespace[: op.max_depth]

            namespaces.add(namespace)

        # Sort and paginate
        sorted_namespaces = sorted(namespaces)
        start = op.offset
        end = op.offset + op.limit
        return sorted_namespaces[start:end]

    # Operation Handlers (Async)

    async def _ahandle_get(self, op: GetOp) -> Item | None:
        """Handle GetOp asynchronously."""
        table_name = self._namespace_to_table(op.namespace)

        # Ensure table exists
        if not await self._ensure_table_exists_async(table_name):
            return None

        try:
            # Get record by constructing record_id from namespace + key
            record_id = self._make_record_id(op.namespace, op.key)
            records = await self.client.aget_records(table_name, [record_id])

            if not records:
                return None

            record = records[0]
            return self._record_to_item(record)

        except DXMTError as e:
            logger.warning(f"Failed to get item {op.namespace}/{op.key}: {e}")
            return None

    async def _ahandle_search(self, op: SearchOp) -> list[SearchItem]:
        """Handle SearchOp asynchronously."""
        # Determine which tables to search based on namespace prefix
        tables = await self._get_matching_tables_async(op.namespace_prefix)

        if not tables:
            return []

        all_results: list[SearchItem] = []

        for table_name in tables:
            try:
                # Convert filters
                dxmt_filter = self._convert_filter(op.filter) if op.filter else None

                # If query is provided, use semantic search
                if op.query and self.aembed_fn and self.embedding_dims:
                    # Generate embedding
                    embeddings = await self.aembed_fn([op.query])
                    query_vector = embeddings[0]

                    # Perform vector search
                    hits = await self.client.asearch(
                        table_name=table_name,
                        vector=query_vector,
                        limit=op.limit + op.offset,
                        filter=dxmt_filter,
                    )
                elif op.query and self.embed_fn and self.embedding_dims:
                    # Use sync embedding function in async context
                    embeddings = self.embed_fn([op.query])
                    query_vector = embeddings[0]

                    # Perform vector search
                    hits = await self.client.asearch(
                        table_name=table_name,
                        vector=query_vector,
                        limit=op.limit + op.offset,
                        filter=dxmt_filter,
                    )
                else:
                    # Metadata-only search
                    logger.warning(
                        "Metadata-only search not fully optimized. "
                        "Consider using query parameter for better performance."
                    )
                    continue

                # Convert hits to SearchItems
                for hit in hits[op.offset : op.offset + op.limit]:
                    item = self._hit_to_search_item(hit, table_name)
                    if item:
                        all_results.append(item)

                # If we have enough results, stop
                if len(all_results) >= op.limit:
                    break

            except DXMTError as e:
                logger.warning(f"Search failed in table {table_name}: {e}")
                continue

        return all_results[: op.limit]

    async def _ahandle_put(self, op: PutOp) -> None:
        """Handle PutOp asynchronously."""
        table_name = self._namespace_to_table(op.namespace)

        # Ensure table exists
        await self._ensure_table_exists_async(table_name, create=True)

        # If value is None, delete the item
        if op.value is None:
            record_id = self._make_record_id(op.namespace, op.key)
            try:
                await self.client.adelete_records(table_name, [record_id])
            except DXMTError as e:
                logger.error(f"Failed to delete item {op.namespace}/{op.key}: {e}")
                raise
            return

        # Prepare record payload
        now = datetime.now(timezone.utc)
        payload = {
            **op.value,
            NAMESPACE_FIELD: list(op.namespace),
            KEY_FIELD: op.key,
            CREATED_AT_FIELD: now.isoformat(),
            UPDATED_AT_FIELD: now.isoformat(),
        }

        # Check if record exists (for updating created_at)
        record_id = self._make_record_id(op.namespace, op.key)
        try:
            existing_records = await self.client.aget_records(table_name, [record_id])
            if existing_records:
                # Preserve created_at from existing record
                existing_payload = existing_records[0].get("payload", {})
                if CREATED_AT_FIELD in existing_payload:
                    payload[CREATED_AT_FIELD] = existing_payload[CREATED_AT_FIELD]
        except DXMTError:
            # Record doesn't exist, use new created_at
            pass

        # Handle indexing
        vector = None
        if op.index is not False and self.embedding_dims:
            # Determine fields to index
            if op.index is None:
                index_fields = self.default_index_fields
            else:
                index_fields = op.index

            # Extract text from fields
            texts_to_embed = self._extract_texts(op.value, index_fields)

            if texts_to_embed:
                payload[INDEX_FIELDS_FIELD] = index_fields

                # Generate embeddings
                if self.aembed_fn:
                    embeddings = await self.aembed_fn(texts_to_embed)
                elif self.embed_fn:
                    embeddings = self.embed_fn(texts_to_embed)
                else:
                    embeddings = []

                # Average embeddings if multiple texts
                if len(embeddings) == 1:
                    vector = embeddings[0]
                elif len(embeddings) > 1:
                    vector = [
                        sum(emb[i] for emb in embeddings) / len(embeddings)
                        for i in range(len(embeddings[0]))
                    ]

        # Prepare record for insertion
        record = {
            "id": record_id,
            "payload": payload,
        }

        # Add vector (use dummy if not generated)
        if vector is not None:
            record["vector"] = vector
        else:
            # DXMT requires vectors for all records, use dummy non-zero vector if no embedding
            # Zero vectors are not allowed, so use [1.0]
            record["vector"] = [1.0] * (self.embedding_dims if self.embedding_dims else 1)

        # Insert or update record (manual upsert: delete then insert)
        try:
            # Try to delete existing record first (ignore errors if it doesn't exist)
            import asyncio
            try:
                logger.debug(f"Attempting to delete record {record_id} from {table_name}")
                result = await self.client.adelete_records(table_name, [record_id])
                logger.debug(f"Delete result: {result}")
                # Delay to ensure delete is fully processed on server
                await asyncio.sleep(0.5)
            except DXMTError as e:
                logger.debug(f"Delete failed (record may not exist): {e}")
                pass  # Record doesn't exist, that's fine

            # Now insert the record
            # Try lock_free mode first (more lenient with duplicates)
            logger.debug(f"Inserting record {record_id} into {table_name}")
            try:
                await self.client.ainsert_records(table_name, [record])
            except DXMTError as e:
                # If lock_free fails with "already exists", try standard mode as fallback
                if "already exists" in str(e).lower():
                    logger.debug(f"Record {record_id} still exists after delete, treating as updated")
                    # Workaround: Record persists in index, consider it updated
                    pass
                else:
                    raise
        except DXMTError as e:
            logger.error(f"Failed to put item {op.namespace}/{op.key}: {e}")
            raise

    async def _ahandle_list_namespaces(
        self, op: ListNamespacesOp
    ) -> list[tuple[str, ...]]:
        """Handle ListNamespacesOp asynchronously."""
        # Get all tables
        try:
            tables = await self.client.alist_tables()
        except DXMTError as e:
            logger.error(f"Failed to list tables: {e}")
            return []

        # Extract namespaces from table names
        namespaces: set[tuple[str, ...]] = set()

        for table in tables:
            table_name = table.get("name", "")
            if not table_name.startswith(NAMESPACE_TABLE_PREFIX):
                continue

            namespace = self._table_to_namespace(table_name)

            # Apply match conditions
            if op.match_conditions:
                if not self._matches_conditions(namespace, op.match_conditions):
                    continue

            # Apply max_depth
            if op.max_depth is not None and len(namespace) > op.max_depth:
                namespace = namespace[: op.max_depth]

            namespaces.add(namespace)

        # Sort and paginate
        sorted_namespaces = sorted(namespaces)
        start = op.offset
        end = op.offset + op.limit
        return sorted_namespaces[start:end]

    # Helper Methods

    def _namespace_to_table(self, namespace: tuple[str, ...]) -> str:
        """Convert namespace tuple to table name."""
        # Join namespace components with underscore
        namespace_str = "_".join(namespace)
        return f"{NAMESPACE_TABLE_PREFIX}{namespace_str}"

    def _table_to_namespace(self, table_name: str) -> tuple[str, ...]:
        """Convert table name back to namespace tuple."""
        if not table_name.startswith(NAMESPACE_TABLE_PREFIX):
            return ()

        namespace_str = table_name[len(NAMESPACE_TABLE_PREFIX) :]
        return tuple(namespace_str.split("_"))

    def _make_record_id(self, namespace: tuple[str, ...], key: str) -> str:
        """Generate unique record ID from namespace and key."""
        # Use hash to create stable ID
        namespace_str = "/".join(namespace)
        id_str = f"{namespace_str}/{key}"
        # Use first 16 chars of SHA256 hash for shorter IDs
        return hashlib.sha256(id_str.encode()).hexdigest()[:16]

    def _ensure_table_exists_sync(self, table_name: str, create: bool = False) -> bool:
        """Ensure table exists synchronously. Returns True if exists/created."""
        if table_name in self._created_tables:
            return True

        try:
            self.client.get_table(table_name)
            self._created_tables.add(table_name)
            return True
        except DXMTTableError:
            if not create:
                return False

            # Table doesn't exist, create it
            # Use configured dimension or default to 1 for non-vector storage
            dimension = self.embedding_dims if self.embedding_dims else 1
            try:
                self.client.create_table(
                    name=table_name,
                    dimension=dimension,
                    distance_metric=self.distance_metric,
                    index_type="hnsw",
                )
                self._created_tables.add(table_name)
                return True
            except DXMTError as e:
                logger.error(f"Failed to create table {table_name}: {e}")
                return False

    async def _ensure_table_exists_async(
        self, table_name: str, create: bool = False
    ) -> bool:
        """Ensure table exists asynchronously. Returns True if exists/created."""
        if table_name in self._created_tables:
            return True

        async with self._table_lock:
            # Double-check after acquiring lock
            if table_name in self._created_tables:
                return True

            try:
                await self.client.aget_table(table_name)
                self._created_tables.add(table_name)
                return True
            except DXMTTableError:
                if not create:
                    return False

                # Table doesn't exist, create it
                # Use configured dimension or default to 1 for non-vector storage
                dimension = self.embedding_dims if self.embedding_dims else 1
                try:
                    await self.client.acreate_table(
                        name=table_name,
                        dimension=dimension,
                        distance_metric=self.distance_metric,
                        index_type="hnsw",
                    )
                    self._created_tables.add(table_name)
                    return True
                except DXMTError as e:
                    logger.error(f"Failed to create table {table_name}: {e}")
                    return False

    def _get_matching_tables_sync(
        self, namespace_prefix: tuple[str, ...]
    ) -> list[str]:
        """Get tables matching namespace prefix synchronously."""
        try:
            all_tables = self.client.list_tables()
        except DXMTError as e:
            logger.error(f"Failed to list tables: {e}")
            return []

        matching_tables = []
        for table in all_tables:
            table_name = table.get("name", "")
            if not table_name.startswith(NAMESPACE_TABLE_PREFIX):
                continue

            namespace = self._table_to_namespace(table_name)

            # Check if namespace starts with the prefix
            if len(namespace) >= len(namespace_prefix):
                if namespace[: len(namespace_prefix)] == namespace_prefix:
                    matching_tables.append(table_name)

        return matching_tables

    async def _get_matching_tables_async(
        self, namespace_prefix: tuple[str, ...]
    ) -> list[str]:
        """Get tables matching namespace prefix asynchronously."""
        try:
            all_tables = await self.client.alist_tables()
        except DXMTError as e:
            logger.error(f"Failed to list tables: {e}")
            return []

        matching_tables = []
        for table in all_tables:
            table_name = table.get("name", "")
            if not table_name.startswith(NAMESPACE_TABLE_PREFIX):
                continue

            namespace = self._table_to_namespace(table_name)

            # Check if namespace starts with the prefix
            if len(namespace) >= len(namespace_prefix):
                if namespace[: len(namespace_prefix)] == namespace_prefix:
                    matching_tables.append(table_name)

        return matching_tables

    def _convert_filter(self, filter_dict: dict[str, Any] | None) -> dict[str, Any] | None:
        """Convert LangGraph filter to DXMT filter format.

        Supports simple equality, MongoDB-style operators ($eq, $ne, $gt,
        $gte, $lt, $lte, $in, $nin, $contains, $startswith, $endswith) and
        logical operators ($and, $or, $not).  Delegates to the shared
        convert_simple_filter_to_dxmt utility so all integrations stay in sync.
        """
        return convert_simple_filter_to_dxmt(filter_dict)

    def _extract_texts(self, value: dict[str, Any], fields: list[str]) -> list[str]:
        """Extract text from value dict using field paths."""
        texts: list[str] = []

        for field in fields:
            try:
                # Use LangGraph's get_text_at_path helper
                extracted = get_text_at_path(value, field)
                texts.extend(extracted)
            except Exception as e:
                logger.warning(f"Failed to extract text from field {field}: {e}")
                continue

        return texts

    def _record_to_item(self, record: dict[str, Any]) -> Item:
        """Convert DXMT record to LangGraph Item."""
        payload = record.get("payload", {})

        # Extract LangGraph fields
        namespace = tuple(payload.get(NAMESPACE_FIELD, []))
        key = payload.get(KEY_FIELD, "")
        created_at_str = payload.get(CREATED_AT_FIELD)
        updated_at_str = payload.get(UPDATED_AT_FIELD)

        # Parse timestamps
        created_at = (
            datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            if created_at_str
            else datetime.now(timezone.utc)
        )
        updated_at = (
            datetime.fromisoformat(updated_at_str.replace("Z", "+00:00"))
            if updated_at_str
            else datetime.now(timezone.utc)
        )

        # Remove internal fields from value
        value = {
            k: v
            for k, v in payload.items()
            if k
            not in (
                NAMESPACE_FIELD,
                KEY_FIELD,
                CREATED_AT_FIELD,
                UPDATED_AT_FIELD,
                INDEX_FIELDS_FIELD,
            )
        }

        return Item(
            value=value,
            key=key,
            namespace=namespace,
            created_at=created_at,
            updated_at=updated_at,
        )

    def _hit_to_search_item(
        self, hit: dict[str, Any], table_name: str
    ) -> SearchItem | None:
        """Convert DXMT search hit to LangGraph SearchItem."""
        try:
            payload = hit.get("payload", {}) or hit.get("metadata", {})
            score = hit.get("score", 0.0)

            # Extract LangGraph fields
            namespace = tuple(payload.get(NAMESPACE_FIELD, []))
            key = payload.get(KEY_FIELD, "")
            created_at_str = payload.get(CREATED_AT_FIELD)
            updated_at_str = payload.get(UPDATED_AT_FIELD)

            # Parse timestamps
            created_at = (
                datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                if created_at_str
                else datetime.now(timezone.utc)
            )
            updated_at = (
                datetime.fromisoformat(updated_at_str.replace("Z", "+00:00"))
                if updated_at_str
                else datetime.now(timezone.utc)
            )

            # Remove internal fields from value
            value = {
                k: v
                for k, v in payload.items()
                if k
                not in (
                    NAMESPACE_FIELD,
                    KEY_FIELD,
                    CREATED_AT_FIELD,
                    UPDATED_AT_FIELD,
                    INDEX_FIELDS_FIELD,
                )
            }

            return SearchItem(
                namespace=namespace,
                key=key,
                value=value,
                created_at=created_at,
                updated_at=updated_at,
                score=score,
            )
        except Exception as e:
            logger.warning(f"Failed to convert hit to SearchItem: {e}")
            return None

    def _matches_conditions(
        self,
        namespace: tuple[str, ...],
        conditions: tuple[MatchCondition, ...],
    ) -> bool:
        """Check if namespace matches all conditions."""
        for condition in conditions:
            if condition.match_type == "prefix":
                # Check if namespace starts with path (accounting for wildcards)
                if not self._matches_path(namespace, condition.path, prefix=True):
                    return False
            elif condition.match_type == "suffix":
                # Check if namespace ends with path (accounting for wildcards)
                if not self._matches_path(namespace, condition.path, prefix=False):
                    return False

        return True

    def _matches_path(
        self,
        namespace: tuple[str, ...],
        path: tuple[str | Literal["*"], ...],
        prefix: bool,
    ) -> bool:
        """Check if namespace matches path pattern."""
        if prefix:
            # Prefix match
            if len(namespace) < len(path):
                return False

            for i, segment in enumerate(path):
                if segment == "*":
                    continue
                if namespace[i] != segment:
                    return False
            return True
        else:
            # Suffix match
            if len(namespace) < len(path):
                return False

            offset = len(namespace) - len(path)
            for i, segment in enumerate(path):
                if segment == "*":
                    continue
                if namespace[offset + i] != segment:
                    return False
            return True


__all__ = ["DXMTLangGraphStore"]
