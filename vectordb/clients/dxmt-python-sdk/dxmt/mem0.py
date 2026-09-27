"""DXMT mem0 VectorStore Adapter.

This module provides a mem0-compatible VectorStore implementation backed by DXMT VectorDB.
mem0 (https://mem0.ai) is an intelligent memory layer for AI assistants and agents.

Example:
    >>> from mem0 import Memory
    >>> from dxmt import Client
    >>> from dxmt.mem0 import DXMTMemoryStore
    >>>
    >>> # Initialize DXMT memory store
    >>> client = Client(host="localhost", port=9999)
    >>> vector_store = DXMTMemoryStore(
    ...     collection_name="user_memories",
    ...     client=client,
    ...     embedding_model_dims=768
    ... )
    >>>
    >>> # Configure mem0
    >>> config = {
    ...     "vector_store": {
    ...         "provider": "custom",
    ...         "config": {"custom_vector_store": vector_store}
    ...     }
    ... }
    >>> m = Memory.from_config(config)
    >>>
    >>> # Store memories
    >>> m.add("I love Python programming", user_id="user_123")
    >>>
    >>> # Search memories
    >>> results = m.search("programming preferences", user_id="user_123")
"""

from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from mem0.vector_stores.base import VectorStoreBase
    MEM0_AVAILABLE = True
except ImportError:
    # mem0 not installed, define dummy base class
    MEM0_AVAILABLE = False

    class VectorStoreBase:  # type: ignore
        """Dummy base class when mem0 is not installed."""
        pass

from .client import DXMTClient
from .exceptions import DXMTError, DXMTTableError
from .filters import convert_simple_filter_to_dxmt

logger = logging.getLogger(__name__)


class OutputData:
    """What mem0 reads off a result: `.id`, `.score`, `.payload`.

    mem0 indexes results by attribute (`memory.id`, `memory.score`) and every backend in its
    tree returns a small model with these three fields. This adapter returned dicts, which
    worked while mem0 went through `provider: "custom"` and stopped when it did not:
    `'dict' object has no attribute 'id'`. A plain class rather than a pydantic model, so
    the SDK does not gain a dependency for three attributes.
    """

    __slots__ = ("id", "score", "payload")

    def __init__(self, id, score, payload):
        self.id = id
        self.score = score
        self.payload = payload

    # Subscripting kept working for anything that still reads these as dicts.
    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __repr__(self):
        return f"OutputData(id={self.id!r}, score={self.score!r}, payload={self.payload!r})"


class DXMTMemoryStore(VectorStoreBase):
    """mem0-compatible VectorStore implementation using DXMT VectorDB.

    Implements mem0's VectorStoreBase interface with DXMT backend.
    Supports multi-scope memory storage (user_id, agent_id, run_id) and
    efficient semantic search with filtering.

    Args:
        collection_name (str): Name of the memory collection (table).
        client (DXMTClient): DXMT client instance.
        embedding_model_dims (int): Dimension of embedding vectors.
        distance_metric (str): Distance metric (cosine, euclidean, manhattan, dot).
            Defaults to "cosine".

    Example:
        >>> client = DXMTClient(host="localhost", port=9999)
        >>> store = DXMTMemoryStore(
        ...     collection_name="user_memories",
        ...     client=client,
        ...     embedding_model_dims=768
        ... )
        >>>
        >>> # Insert memories
        >>> store.insert(
        ...     vectors=[[0.1, 0.2, ...]],
        ...     payloads=[{"data": "User likes Python", "user_id": "user_1"}],
        ...     ids=["mem_1"]
        ... )
        >>>
        >>> # Search memories
        >>> results = store.search(
        ...     query="",
        ...     vectors=[[0.1, 0.2, ...]],
        ...     limit=5,
        ...     filters={"user_id": "user_1"}
        ... )
    """

    def __init__(
        self,
        collection_name: str,
        client: DXMTClient,
        embedding_model_dims: int,
        distance_metric: str = "cosine",
    ):
        """Initialize DXMT memory store."""
        if not MEM0_AVAILABLE:
            logger.warning(
                "mem0 is not installed. Install with: pip install mem0ai"
            )

        self.collection_name = collection_name
        self.client = client
        self.embedding_model_dims = embedding_model_dims
        self.distance_metric = distance_metric

        # Create collection (table) if not exists
        self.create_col(collection_name, embedding_model_dims, distance_metric)

    def create_col(
        self, name: str, vector_size: int, distance: str = "cosine"
    ) -> None:
        """Create a new collection (table).

        Args:
            name (str): Collection name.
            vector_size (int): Vector embedding dimension.
            distance (str): Distance metric (cosine, euclidean, manhattan, dot).

        Raises:
            DXMTTableError: If table creation fails.
        """
        try:
            # Check if table exists
            self.client.get_table(name)
            logger.info(f"Collection {name} already exists, skipping creation")
        except DXMTError:
            # Table doesn't exist, create it
            logger.info(f"Creating collection {name} with dimension {vector_size}")
            self.client.create_table(
                name=name,
                dimension=vector_size,
                distance_metric=distance,
                index_type="hnsw",  # Use HNSW for good recall
                description=f"mem0 memory collection: {name}",
            )

    def insert(
        self,
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Insert vectors with payloads into the collection.

        Args:
            vectors (List[List[float]]): List of embedding vectors.
            payloads (Optional[List[Dict[str, Any]]]): List of metadata payloads.
            ids (Optional[List[str]]): List of vector IDs. Generated if not provided.

        Raises:
            ValueError: If vector dimensions don't match.
            DXMTError: If insertion fails.
        """
        if not vectors:
            logger.warning("insert called with empty vectors list")
            return

        # Validate dimensions
        for i, vector in enumerate(vectors):
            if len(vector) != self.embedding_model_dims:
                raise ValueError(
                    f"Vector {i} dimension mismatch: expected {self.embedding_model_dims}, "
                    f"got {len(vector)}"
                )

        # Generate defaults
        if payloads is None:
            payloads = [{}] * len(vectors)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]

        # Build records for batch insert
        records = []
        for i, (vector, payload, record_id) in enumerate(zip(vectors, payloads, ids)):
            # Add timestamps if not present
            if "created_at" not in payload:
                payload["created_at"] = datetime.utcnow().isoformat() + "Z"
            if "updated_at" not in payload:
                payload["updated_at"] = payload["created_at"]

            records.append({
                "id": record_id,
                "vector": vector,
                "payload": payload,
            })

        # Batch insert
        logger.debug(f"Inserting {len(records)} records into {self.collection_name}")
        self.client.insert_records(self.collection_name, records)

    def search(
        self,
        query: str,  # Not used, kept for mem0 interface compatibility
        vectors: List[List[float]],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.

        Args:
            query (str): Query string (not used, vectors are used instead).
            vectors (List[List[float]]): Query vector(s). Only first vector is used.
            limit (int): Maximum number of results to return.
            filters (Optional[Dict[str, Any]]): Metadata filters (user_id, agent_id, etc.).

        Returns:
            List[Dict[str, Any]]: Search results with id, score, and payload.

        Raises:
            ValueError: If no vectors provided.
            DXMTError: If search fails.
        """
        if not vectors or len(vectors) == 0:
            raise ValueError("At least one query vector is required")

        # `VectorStoreBase.search` spells this `top_k`; this adapter called it `limit` and so
        # raised `unexpected keyword argument 'top_k'` the moment mem0 called it. Both are
        # accepted, `limit` winning when given, so a caller written against either signature
        # keeps working.
        if limit is not None:
            top_k = limit

        # `vectors` is one query embedding, flat, in current mem0 -- `search(vectors=
        # query_embedding, ...)` in `Memory._add_to_vector_store`. It used to be a list of
        # them, and this adapter took `vectors[0]`, which on a flat list is a float and
        # produced `object of type 'float' has no len()` two lines later. Both shapes are
        # read here rather than one being chosen, because the caller is not ours to pin.
        query_vector = vectors if vectors and isinstance(vectors[0], (int, float)) else vectors[0]

        # Validate dimension
        if len(query_vector) != self.embedding_model_dims:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self.embedding_model_dims}, "
                f"got {len(query_vector)}"
            )

        # Convert mem0 filters to DXMT format
        dxmt_filter = self._convert_filters(filters) if filters else None

        # Perform search
        logger.debug(
            f"Searching {self.collection_name} with top_k={top_k}, filters={filters}"
        )
        results = self.client.search(
            table_name=self.collection_name,
            vector=query_vector,
            k=top_k,
            filter=dxmt_filter,
        )

        # Normalize scores and format output
        output = []
        for hit in results:
            output.append(OutputData(
                id=hit["id"],
                score=self._normalize_score(hit["score"]),
                payload=hit.get("metadata", hit.get("payload", {})),
            ))

        return output

    def delete(self, vector_id: str) -> None:
        """Delete a vector by ID.

        Args:
            vector_id (str): Vector ID to delete.

        Raises:
            DXMTError: If deletion fails.
        """
        logger.debug(f"Deleting vector {vector_id} from {self.collection_name}")
        self.client.delete_records(self.collection_name, [vector_id])

    def update(
        self,
        vector_id: str,
        vector: Optional[List[float]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update a vector and its payload.

        Args:
            vector_id (str): Vector ID to update.
            vector (Optional[List[float]]): New vector. If None, vector is not updated.
            payload (Optional[Dict[str, Any]]): New payload. Merged with existing.

        Raises:
            ValueError: If vector dimension doesn't match.
            DXMTError: If update fails.
        """
        # Validate dimension if vector provided
        if vector is not None and len(vector) != self.embedding_model_dims:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.embedding_model_dims}, "
                f"got {len(vector)}"
            )

        # Add updated_at timestamp
        if payload is not None:
            payload["updated_at"] = datetime.utcnow().isoformat() + "Z"

        logger.debug(f"Updating vector {vector_id} in {self.collection_name}")
        self.client.update_record(
            table_name=self.collection_name,
            record_id=vector_id,
            vector=vector,
            payload=payload,
            merge_mode="merge",  # Preserve existing fields
        )

    def get(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a vector by ID.

        Args:
            vector_id (str): Vector ID to retrieve.

        Returns:
            Optional[Dict[str, Any]]: Record with id, score (None), and payload.
                Returns None if not found.

        Raises:
            DXMTError: If retrieval fails.
        """
        try:
            records = self.client.get_records(self.collection_name, [vector_id])
            if not records:
                return None

            record = records[0]
            return OutputData(
                id=record["id"],
                score=None,  # No score for direct retrieval
                payload=record.get("metadata", record.get("payload", {})),
            )
        except DXMTError as e:
            logger.error(f"Error getting vector {vector_id}: {e}")
            return None

    def list_cols(self) -> List[str]:
        """List all collections (tables).

        Returns:
            List[str]: List of collection names.

        Raises:
            DXMTError: If listing fails.
        """
        tables = self.client.list_tables()
        return [table["name"] for table in tables]

    def delete_col(self) -> None:
        """Delete the collection (table).

        Raises:
            DXMTTableError: If deletion fails.
        """
        logger.info(f"Deleting collection {self.collection_name}")
        self.client.delete_table(self.collection_name)

    def col_info(self) -> Dict[str, Any]:
        """Get collection (table) information.

        Returns:
            Dict[str, Any]: Collection metadata including name, dimension, count, etc.

        Raises:
            DXMTError: If retrieval fails.
        """
        return self.client.get_table(self.collection_name)

    def list(
        self,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[List["OutputData"]]:
        """List all vectors in the collection.

        Args:
            filters (Optional[Dict[str, Any]]): Metadata filters to apply.
            limit (int): Maximum number of results to return.

        Returns:
            List[Dict[str, Any]]: List of records with id, score (None), and payload.

        Raises:
            DXMTError: If listing fails.

        Note:
            DXMT doesn't have a direct "list all" endpoint, so we use search
            with a dummy vector. For large collections, this may be inefficient.
        """
        # Create a dummy vector for search
        dummy_vector = [0.0] * self.embedding_model_dims

        # Convert filters
        dxmt_filter = self._convert_filters(filters) if filters else None

        # Search with dummy vector to get records
        # `VectorStoreBase.list` spells it `top_k`, as `search` does. Both accepted, and
        # neither given means the old default of 100 rather than everything.
        top_k = limit if limit is not None else (top_k if top_k is not None else 100)
        logger.debug(f"Listing records from {self.collection_name} with top_k={top_k}")
        results = self.client.search(
            table_name=self.collection_name,
            vector=dummy_vector,
            k=top_k,
            filter=dxmt_filter,
        )

        # Format output
        output = []
        for hit in results:
            output.append(OutputData(
                id=hit["id"],
                score=None,  # Not relevant for listing
                payload=hit.get("metadata", hit.get("payload", {})),
            ))

        # Wrapped in a list, which looks wrong and is the contract. mem0 reads this as
        # `self.vector_store.list(...)[0]` -- see `Memory.delete_all` -- and every backend in
        # its own tree ends with `return [self._parse_output(results)]`. Returning the flat
        # list made `delete_all` raise `IndexError: list index out of range` on an empty
        # result and iterate a single `OutputData` on a non-empty one.
        return [output]

    def reset(self) -> None:
        """Reset the collection by deleting and recreating it.

        Raises:
            DXMTError: If reset fails.
        """
        logger.info(f"Resetting collection {self.collection_name}")
        try:
            self.delete_col()
        except DXMTError as e:
            logger.warning(f"Error deleting collection during reset: {e}")

        # Recreate collection
        self.create_col(
            self.collection_name, self.embedding_model_dims, self.distance_metric
        )

    # Helper methods

    def _convert_filters(self, mem0_filters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert mem0 filters to DXMT filter format.

        mem0 filters support both simple equality and operator-based syntax:
            {"user_id": "user_1"}
            {"user_id": "user_1", "agent_id": "agent_1"}
            {"created_at": {"gte": "2024-01-01T00:00:00Z", "lte": "2024-12-31T23:59:59Z"}}

        DXMT expects structured Filter format:
            {"must": [{"key": "user_id", "operation": {"type": "Equal", "value": "user_1"}}]}

        Args:
            mem0_filters (Dict[str, Any]): mem0 filter dictionary.

        Returns:
            Dict[str, Any]: DXMT filter format.

        Raises:
            ValueError: If filter format is invalid.
        """
        if not mem0_filters:
            return None

        conditions = []

        for key, value in mem0_filters.items():
            if isinstance(value, dict):
                # Handle operators: {"created_at": {"gte": "...", "lte": "..."}}
                for op, op_value in value.items():
                    if op == "eq":
                        conditions.append({
                            "key": key,
                            "operation": {"type": "Equal", "value": op_value}
                        })
                    elif op == "ne":
                        conditions.append({
                            "key": key,
                            "operation": {"type": "NotEqual", "value": op_value}
                        })
                    elif op in ["gt", "gte", "lt", "lte"]:
                        # Map mem0 operators to DXMT operators
                        op_map = {
                            "gt": "GreaterThan",
                            "gte": "GreaterThanOrEqual",
                            "lt": "LessThan",
                            "lte": "LessThanOrEqual",
                        }
                        conditions.append({
                            "key": key,
                            "operation": {"type": op_map[op], "value": op_value}
                        })
                    elif op == "in":
                        conditions.append({
                            "key": key,
                            "operation": {"type": "In", "value": op_value}
                        })
                    elif op == "nin":
                        conditions.append({
                            "key": key,
                            "operation": {"type": "NotIn", "value": op_value}
                        })
                    else:
                        logger.warning(f"Unsupported operator: {op}, treating as equality")
                        conditions.append({
                            "key": key,
                            "operation": {"type": "Equal", "value": value}
                        })
                        break
            else:
                # Simple equality: {"user_id": "user_1"}
                conditions.append({
                    "key": key,
                    "operation": {"type": "Equal", "value": value}
                })

        return {"must": conditions} if conditions else None

    def _normalize_score(self, score: float) -> float:
        """Pass through DXMT similarity score to mem0.

        DXMT already returns a similarity score in [0, 1] for cosine and dot
        metrics (higher = more similar), and a non-negative distance for
        euclidean.  mem0 expects similarity where 1.0 is most similar.

        Args:
            score (float): DXMT score from search result.

        Returns:
            float: Similarity score suitable for mem0 (higher = more similar).
        """
        if self.distance_metric in ("cosine", "dot"):
            # DXMT returns cosine/dot similarity directly in [0, 1]; pass through.
            return max(0.0, min(1.0, score))

        elif self.distance_metric == "euclidean":
            # DXMT returns L2 distance (0 = identical); convert to similarity.
            return math.exp(-score)

        elif self.distance_metric == "manhattan":
            return max(0.0, min(1.0, 1.0 / (1.0 + abs(score))))

        else:
            logger.warning(f"Unknown distance metric {self.distance_metric}, returning raw score")
            return score


def register_with_mem0(provider: str = "dxmt") -> str:
    """Make `{"vector_store": {"provider": "dxmt", ...}}` a thing mem0 accepts.

    mem0 used to take any `VectorStoreBase` through `provider: "custom"` and a
    `custom_vector_store` in the config. That door is closed: current mem0 validates the
    provider against a fixed table of twenty-five named backends, and a config for an
    unlisted one fails in pydantic before any of this code runs --

        ValidationError: Unsupported vector store provider: custom

    -- which is what every DXMT + mem0 example in this repository hit. The adapter below was
    never the problem; the way it was plugged in stopped existing.

    There are two tables to join, because mem0 validates the config and builds the store in
    different places, and the validator reaches the config class by importing
    `mem0.configs.vector_stores.<provider>`. So this registers a module under that name as
    well as the two dictionary entries. Idempotent, and it touches nothing but the entries it
    adds.

    Returns the provider name, so a caller can write

        config = {"vector_store": {"provider": register_with_mem0(), "config": {...}}}

    and not repeat the string. `config` then takes this class's own arguments --
    `collection_name`, `client`, `embedding_model_dims`, `distance_metric` -- rather than a
    prebuilt instance, because mem0 constructs the store itself.
    """
    import sys
    import types

    from pydantic import BaseModel, Field

    from mem0.utils.factory import VectorStoreFactory
    from mem0.vector_stores.configs import VectorStoreConfig

    VectorStoreFactory.provider_to_class[provider] = "dxmt.mem0.DXMTMemoryStore"

    module_name = f"mem0.configs.vector_stores.{provider}"
    if module_name not in sys.modules:
        class DXMTVectorStoreConfig(BaseModel):
            collection_name: str = Field("mem0", description="table name")
            client: Any = Field(None, description="a dxmt.Client")
            embedding_model_dims: int = Field(1536, description="vector width")
            distance_metric: str = Field("cosine", description="cosine|euclidean|manhattan|dot")

            model_config = {"arbitrary_types_allowed": True}

        module = types.ModuleType(module_name)
        module.DXMTVectorStoreConfig = DXMTVectorStoreConfig
        sys.modules[module_name] = module

    # `_provider_configs` is a pydantic private attribute, so reading it off the class gives
    # the descriptor rather than the dict. The dict is its default, and is the object the
    # validator actually consults.
    VectorStoreConfig.__private_attributes__["_provider_configs"].default[provider] = (
        "DXMTVectorStoreConfig"
    )
    return provider


__all__ = ["DXMTMemoryStore", "OutputData", "register_with_mem0"]
