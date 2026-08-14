"""DXMT Python SDK HTTP Client.

This module provides sync and async HTTP clients for interacting with
the DXMT VectorDB REST API.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from .exceptions import (
    DXMTConnectionError,
    DXMTError,
    DXMTSearchError,
    DXMTTableError,
)
from .filters import convert_simple_filter_to_dxmt

logger = logging.getLogger(__name__)


class DXMTClient:
    """HTTP client for DXMT VectorDB API.

    Handles all communication with DXMT server via REST API.
    Supports both sync and async operations.

    Args:
        host: DXMT server hostname. Default is 'localhost'.
        port: DXMT server port. Default is 9999.
        ssl: Whether to use HTTPS. Default is False.
        headers: Additional HTTP headers.
        timeout: Request timeout in seconds. Default is 30.0.
        **kwargs: Additional httpx client parameters.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9999,
        *,
        ssl: bool = False,
        headers: Optional[dict[str, str]] = None,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        """Initialize DXMT client."""
        protocol = "https" if ssl else "http"
        self.base_url = f"{protocol}://{host}:{port}/api/v1"
        self._headers = headers or {}
        self._timeout = timeout
        self._client_kwargs = kwargs

        # Create sync client
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=self._headers,
            timeout=self._timeout,
            **self._client_kwargs,
        )

        # Async client is created lazily
        self._async_client: Optional[httpx.AsyncClient] = None

    def __enter__(self) -> DXMTClient:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()

    async def __aenter__(self) -> DXMTClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.aclose()

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    async def aclose(self) -> None:
        """Async close the HTTP clients."""
        self._client.close()
        if self._async_client:
            await self._async_client.aclose()

    @property
    def async_client(self) -> httpx.AsyncClient:
        """Get or create async client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._headers,
                timeout=self._timeout,
                **self._client_kwargs,
            )
        return self._async_client

    # Table Management

    def create_table(
        self,
        name: str,
        dimension: int,
        distance_metric: str = "cosine",
        index_type: str = "hnsw",
        description: Optional[str] = None,
        enable_full_text_search: bool = False,
        text_fields: Optional[list[dict[str, Any]]] = None,
        memory_optimization: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Create a new table (collection).

        Args:
            name: Table name.
            dimension: Vector dimension.
            distance_metric: Distance metric (cosine, euclidean, manhattan, dot).
            index_type: Index type (hnsw, ivf, ivf_pq, lsh, brute_force).
            description: Optional table description.
            enable_full_text_search: Enable full-text search.
            text_fields: Text field configurations.
            memory_optimization: Enable memory optimization (optional).
                - None: Use global config (DB_MEMORY_OPTIMIZATION)
                - True: Enable memory optimization (Phase 4 Hybrid), saves 85-90% memory, 87-95% recall
                - False: High accuracy mode (HNSW/Phase 3), maintains 95-99% recall

        Returns:
            Response data from server.

        Raises:
            DXMTTableError: If table creation fails.
        """
        payload = {
            "name": name,
            "dimension": dimension,
            "distance_metric": distance_metric,
            "index_type": index_type,
            "description": description,
            "enable_full_text_search": enable_full_text_search,
            "text_fields": text_fields,
            "memory_optimization": memory_optimization,
        }
        try:
            response = self._client.post("/tables", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            msg = f"Failed to create table {name}: {e.response.text}"
            raise DXMTTableError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error creating table {name}: {e}"
            raise DXMTConnectionError(msg) from e

    async def acreate_table(
        self,
        name: str,
        dimension: int,
        distance_metric: str = "cosine",
        index_type: str = "hnsw",
        description: Optional[str] = None,
        enable_full_text_search: bool = False,
        text_fields: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Async create a new table."""
        payload = {
            "name": name,
            "dimension": dimension,
            "distance_metric": distance_metric,
            "index_type": index_type,
            "description": description,
            "enable_full_text_search": enable_full_text_search,
            "text_fields": text_fields,
        }
        try:
            response = await self.async_client.post("/tables", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            msg = f"Failed to create table {name}: {e.response.text}"
            raise DXMTTableError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error creating table {name}: {e}"
            raise DXMTConnectionError(msg) from e

    def get_table(self, name: str) -> dict[str, Any]:
        """Get table information.

        Args:
            name: Table name.

        Returns:
            Table information.

        Raises:
            DXMTTableError: If table not found or request fails.
        """
        try:
            response = self._client.get(f"/tables/{name}")
            response.raise_for_status()
            data = response.json()
            # API returns {"data": {"table": {...}, "record_count": N}}
            return data.get("data", {}).get("table", {})
        except httpx.HTTPStatusError as e:
            msg = f"Failed to get table {name}: {e.response.text}"
            raise DXMTTableError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error getting table {name}: {e}"
            raise DXMTConnectionError(msg) from e

    async def aget_table(self, name: str) -> dict[str, Any]:
        """Async get table information."""
        try:
            response = await self.async_client.get(f"/tables/{name}")
            response.raise_for_status()
            data = response.json()
            # API returns {"data": {"table": {...}, "record_count": N}}
            return data.get("data", {}).get("table", {})
        except httpx.HTTPStatusError as e:
            msg = f"Failed to get table {name}: {e.response.text}"
            raise DXMTTableError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error getting table {name}: {e}"
            raise DXMTConnectionError(msg) from e

    def delete_table(self, name: str) -> dict[str, Any]:
        """Delete a table.

        Args:
            name: Table name.

        Returns:
            Response data from server.

        Raises:
            DXMTTableError: If deletion fails.
        """
        try:
            response = self._client.delete(f"/tables/{name}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            msg = f"Failed to delete table {name}: {e.response.text}"
            raise DXMTTableError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error deleting table {name}: {e}"
            raise DXMTConnectionError(msg) from e

    async def adelete_table(self, name: str) -> dict[str, Any]:
        """Async delete a table."""
        try:
            response = await self.async_client.delete(f"/tables/{name}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            msg = f"Failed to delete table {name}: {e.response.text}"
            raise DXMTTableError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error deleting table {name}: {e}"
            raise DXMTConnectionError(msg) from e

    def list_tables(self) -> list[dict[str, Any]]:
        """List all tables.

        Returns:
            List of table information dicts.

        Raises:
            DXMTTableError: If listing fails.
        """
        try:
            response = self._client.get("/tables")
            response.raise_for_status()
            data = response.json()
            # API returns {"data": {"tables": [...], "total": N, ...}}
            return data.get("data", {}).get("tables", [])
        except httpx.HTTPStatusError as e:
            msg = f"Failed to list tables: {e.response.text}"
            raise DXMTTableError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error listing tables: {e}"
            raise DXMTConnectionError(msg) from e

    async def alist_tables(self) -> list[dict[str, Any]]:
        """Async list all tables."""
        try:
            response = await self.async_client.get("/tables")
            response.raise_for_status()
            data = response.json()
            # API returns {"data": {"tables": [...], "total": N, ...}}
            return data.get("data", {}).get("tables", [])
        except httpx.HTTPStatusError as e:
            msg = f"Failed to list tables: {e.response.text}"
            raise DXMTTableError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error listing tables: {e}"
            raise DXMTConnectionError(msg) from e

    # Document Operations

    def insert_records(
        self,
        table_name: str,
        records: list[dict[str, Any]],
        mode: str = "standard",
    ) -> dict[str, Any]:
        """Insert records into a table.

        Args:
            table_name: Name of the table.
            records: List of records with id, vector, and metadata.
            mode: Insert mode (standard, lock_free, streaming, ultimate, etc.).

        Returns:
            Response data from server.

        Raises:
            DXMTError: If insertion fails.
        """
        # Normalize records: map 'metadata' to 'payload' if present
        normalized_records = []
        for record in records:
            normalized = record.copy()
            if "metadata" in normalized and "payload" not in normalized:
                normalized["payload"] = normalized.pop("metadata")
            normalized_records.append(normalized)

        payload = {"records": normalized_records, "mode": mode}
        try:
            response = self._client.post(
                f"/tables/{table_name}/records",
                json=payload,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            msg = f"Failed to insert records: {e.response.text}"
            raise DXMTError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error inserting records: {e}"
            raise DXMTConnectionError(msg) from e

    async def ainsert_records(
        self,
        table_name: str,
        records: list[dict[str, Any]],
        mode: str = "standard",
    ) -> dict[str, Any]:
        """Async insert records into a table."""
        # Normalize records: map 'metadata' to 'payload' if present
        normalized_records = []
        for record in records:
            normalized = record.copy()
            if "metadata" in normalized and "payload" not in normalized:
                normalized["payload"] = normalized.pop("metadata")
            normalized_records.append(normalized)

        payload = {"records": normalized_records, "mode": mode}
        try:
            response = await self.async_client.post(
                f"/tables/{table_name}/records",
                json=payload,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            msg = f"Failed to insert records: {e.response.text}"
            raise DXMTError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error inserting records: {e}"
            raise DXMTConnectionError(msg) from e

    def update_record(
        self,
        table_name: str,
        record_id: str,
        vector: Optional[list[float]] = None,
        payload: Optional[dict[str, Any]] = None,
        merge_mode: str = "replace",
        upsert: bool = False,
    ) -> dict[str, Any]:
        """Update a record in a table.

        Args:
            table_name: Name of the table.
            record_id: ID of the record to update.
            vector: New vector for the record (optional).
            payload: New metadata/payload for the record (optional).
            merge_mode: "replace" (default) or "merge".
                        - "replace": Replace entire payload (may lose fields)
                        - "merge": Merge with existing payload (preserves fields)
            upsert: If True, create record if it doesn't exist. Default False.

        Returns:
            Response data from server.

        Raises:
            DXMTError: If update fails.

        Examples:
            # Replace mode (default) - replaces entire payload
            client.update_record("docs", "doc1", payload={"views": 100})
            # Result: payload = {"views": 100} only

            # Merge mode - preserves existing fields
            client.update_record("docs", "doc1",
                               payload={"views": 150},
                               merge_mode="merge")
            # Result: payload = {...existing fields..., "views": 150}

            # Upsert - create if not exists
            client.update_record("docs", "new_doc",
                               vector=embedding,
                               payload={"content": "..."},
                               upsert=True)
        """
        request_payload = {}
        if vector is not None:
            request_payload["vector"] = vector
        if payload is not None:
            request_payload["payload"] = payload
        request_payload["merge_mode"] = merge_mode
        request_payload["upsert"] = upsert

        try:
            response = self._client.put(
                f"/tables/{table_name}/records/{record_id}",
                json=request_payload,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            msg = f"Failed to update record {record_id}: {e.response.text}"
            raise DXMTError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error updating record {record_id}: {e}"
            raise DXMTConnectionError(msg) from e

    async def aupdate_record(
        self,
        table_name: str,
        record_id: str,
        vector: Optional[list[float]] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Async update a record in a table.

        Args:
            table_name: Name of the table.
            record_id: ID of the record to update.
            vector: New vector for the record (optional).
            payload: New metadata/payload for the record (optional).

        Returns:
            Response data from server.

        Raises:
            DXMTError: If update fails.
        """
        request_payload = {}
        if vector is not None:
            request_payload["vector"] = vector
        if payload is not None:
            request_payload["payload"] = payload

        try:
            response = await self.async_client.put(
                f"/tables/{table_name}/records/{record_id}",
                json=request_payload,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            msg = f"Failed to update record {record_id}: {e.response.text}"
            raise DXMTError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error updating record {record_id}: {e}"
            raise DXMTConnectionError(msg) from e

    def delete_records(
        self,
        table_name: str,
        ids: list[str],
    ) -> dict[str, Any]:
        """Delete records by IDs.

        Args:
            table_name: Name of the table.
            ids: List of record IDs to delete.

        Returns:
            Response data from server.

        Raises:
            DXMTError: If deletion fails.
        """
        payload = {"ids": ids}
        try:
            response = self._client.request(
                "DELETE",
                f"/tables/{table_name}/records",
                json=payload,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            msg = f"Failed to delete records: {e.response.text}"
            raise DXMTError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error deleting records: {e}"
            raise DXMTConnectionError(msg) from e

    async def adelete_records(
        self,
        table_name: str,
        ids: list[str],
    ) -> dict[str, Any]:
        """Async delete records by IDs."""
        payload = {"ids": ids}
        try:
            response = await self.async_client.request(
                "DELETE",
                f"/tables/{table_name}/records",
                json=payload,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            msg = f"Failed to delete records: {e.response.text}"
            raise DXMTError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error deleting records: {e}"
            raise DXMTConnectionError(msg) from e

    def get_records(
        self,
        table_name: str,
        ids: list[str],
    ) -> list[dict[str, Any]]:
        """Get records by IDs.

        Args:
            table_name: Name of the table.
            ids: List of record IDs to retrieve.

        Returns:
            List of records.

        Raises:
            DXMTError: If retrieval fails.
        """
        import json
        # Note: No batch get endpoint, so we fetch individually
        records = []
        for record_id in ids:
            try:
                response = self._client.get(
                    f"/tables/{table_name}/records/{record_id}"
                )
                response.raise_for_status()
                data = response.json()
                # Extract record from response envelope
                record = data.get("data", {})
                if record:
                    # Parse payload_json if present
                    if "payload_json" in record and record["payload_json"]:
                        try:
                            record["payload"] = json.loads(record["payload_json"])
                        except (json.JSONDecodeError, TypeError):
                            record["payload"] = {}
                    records.append(record)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    # Skip not found records
                    continue
                msg = f"Failed to get record {record_id}: {e.response.text}"
                raise DXMTError(msg) from e
            except httpx.RequestError as e:
                msg = f"Connection error getting record {record_id}: {e}"
                raise DXMTConnectionError(msg) from e
        return records

    async def aget_records(
        self,
        table_name: str,
        ids: list[str],
    ) -> list[dict[str, Any]]:
        """Async get records by IDs."""
        import json
        # Note: No batch get endpoint, so we fetch individually
        records = []
        for record_id in ids:
            try:
                response = await self.async_client.get(
                    f"/tables/{table_name}/records/{record_id}"
                )
                response.raise_for_status()
                data = response.json()
                # Extract record from response envelope
                record = data.get("data", {})
                if record:
                    # Parse payload_json if present
                    if "payload_json" in record and record["payload_json"]:
                        try:
                            record["payload"] = json.loads(record["payload_json"])
                        except (json.JSONDecodeError, TypeError):
                            record["payload"] = {}
                    records.append(record)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    # Skip not found records
                    continue
                msg = f"Failed to get record {record_id}: {e.response.text}"
                raise DXMTError(msg) from e
            except httpx.RequestError as e:
                msg = f"Connection error getting record {record_id}: {e}"
                raise DXMTConnectionError(msg) from e
        return records

    # Search Operations

    def search(
        self,
        table_name: str,
        vector: list[float],
        k: Optional[int] = None,
        limit: Optional[int] = None,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Vector similarity search.

        Args:
            table_name: Name of the table.
            vector: Query vector.
            k: Number of results to return (preferred parameter name).
            limit: Number of results to return (alias for k, for compatibility).
            filter: Optional metadata filter.

        Returns:
            List of search results.

        Raises:
            DXMTSearchError: If search fails.
            ValueError: If neither k nor limit is provided.
        """
        # Support both k and limit parameters (limit is an alias for k)
        if k is None and limit is None:
            k = 4  # Default value
        elif k is None:
            k = limit
        elif limit is not None and limit != k:
            # Both provided but different - prefer k
            pass

        payload = {
            "vector": vector,
            "limit": k,
            "filter": convert_simple_filter_to_dxmt(filter),
        }
        try:
            response = self._client.post(
                f"/tables/{table_name}/search",
                json=payload,
            )
            response.raise_for_status()
            import json
            data = response.json()
            # DXMT server returns "hits" not "results"
            hits = data.get("data", {}).get("hits", [])

            # Parse payload_json if present
            for hit in hits:
                if "payload_json" in hit and hit["payload_json"]:
                    try:
                        hit["payload"] = json.loads(hit["payload_json"])
                    except (json.JSONDecodeError, TypeError):
                        hit["payload"] = {}
                # Map 'payload' to 'metadata' for user convenience
                if "payload" in hit and "metadata" not in hit:
                    hit["metadata"] = hit["payload"]

            return hits
        except httpx.HTTPStatusError as e:
            msg = f"Search failed: {e.response.text}"
            raise DXMTSearchError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error during search: {e}"
            raise DXMTConnectionError(msg) from e

    async def asearch(
        self,
        table_name: str,
        vector: list[float],
        k: Optional[int] = None,
        limit: Optional[int] = None,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Async vector similarity search.

        Args:
            table_name: Name of the table.
            vector: Query vector.
            k: Number of results to return (preferred parameter name).
            limit: Number of results to return (alias for k, for compatibility).
            filter: Optional metadata filter.

        Returns:
            List of search results.
        """
        # Support both k and limit parameters
        if k is None and limit is None:
            k = 4
        elif k is None:
            k = limit

        payload = {
            "vector": vector,
            "limit": k,
            "filter": convert_simple_filter_to_dxmt(filter),
        }
        try:
            response = await self.async_client.post(
                f"/tables/{table_name}/search",
                json=payload,
            )
            response.raise_for_status()
            import json
            data = response.json()
            # DXMT server returns "hits" not "results"
            hits = data.get("data", {}).get("hits", [])

            # Parse payload_json if present
            for hit in hits:
                if "payload_json" in hit and hit["payload_json"]:
                    try:
                        hit["payload"] = json.loads(hit["payload_json"])
                    except (json.JSONDecodeError, TypeError):
                        hit["payload"] = {}
                # Map 'payload' to 'metadata' for user convenience
                if "payload" in hit and "metadata" not in hit:
                    hit["metadata"] = hit["payload"]

            return hits
        except httpx.HTTPStatusError as e:
            msg = f"Search failed: {e.response.text}"
            raise DXMTSearchError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error during search: {e}"
            raise DXMTConnectionError(msg) from e

    def hybrid_search(
        self,
        table_name: str,
        vector: Optional[list[float]] = None,
        text: Optional[str] = None,
        text_query: Optional[str] = None,  # Alias for text
        vector_weight: Optional[float] = None,
        alpha: Optional[float] = None,  # Alias for vector_weight
        k: Optional[int] = None,
        limit: Optional[int] = None,  # Alias for k
        filter: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search (vector + text).

        Args:
            table_name: Name of the table.
            vector: Query vector (optional if text provided).
            text: Query text (optional if vector provided).
            text_query: Alias for text parameter (for compatibility).
            vector_weight: Weight for vector similarity (0-1). Default 0.7.
            alpha: Alias for vector_weight (for compatibility).
            k: Number of results to return (preferred parameter name).
            limit: Number of results to return (alias for k).
            filter: Optional metadata filter.

        Returns:
            List of search results.

        Raises:
            DXMTSearchError: If search fails.
        """
        # Support text_query as alias for text
        if text is None and text_query is not None:
            text = text_query

        # Support alpha as alias for vector_weight
        if vector_weight is None and alpha is not None:
            vector_weight = alpha
        elif vector_weight is None:
            vector_weight = 0.7

        # Support limit as alias for k
        if k is None and limit is None:
            k = 4
        elif k is None:
            k = limit

        payload = {
            "vector": vector,
            "text": text,
            "vector_weight": vector_weight,
            "limit": k,
            "filter": convert_simple_filter_to_dxmt(filter),
        }
        try:
            response = self._client.post(
                f"/tables/{table_name}/search/hybrid",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", {}).get("hits", [])
        except httpx.HTTPStatusError as e:
            msg = f"Hybrid search failed: {e.response.text}"
            raise DXMTSearchError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error during hybrid search: {e}"
            raise DXMTConnectionError(msg) from e

    async def ahybrid_search(
        self,
        table_name: str,
        vector: Optional[list[float]] = None,
        text: Optional[str] = None,
        text_query: Optional[str] = None,  # Alias for text
        vector_weight: Optional[float] = None,
        alpha: Optional[float] = None,  # Alias for vector_weight
        k: Optional[int] = None,
        limit: Optional[int] = None,  # Alias for k
        filter: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Async hybrid search.

        Args:
            table_name: Name of the table.
            vector: Query vector (optional if text provided).
            text: Query text (optional if vector provided).
            text_query: Alias for text parameter (for compatibility).
            vector_weight: Weight for vector similarity (0-1). Default 0.7.
            alpha: Alias for vector_weight (for compatibility).
            k: Number of results to return (preferred parameter name).
            limit: Number of results to return (alias for k).
            filter: Optional metadata filter.

        Returns:
            List of search results.
        """
        # Support text_query as alias for text
        if text is None and text_query is not None:
            text = text_query

        # Support alpha as alias for vector_weight
        if vector_weight is None and alpha is not None:
            vector_weight = alpha
        elif vector_weight is None:
            vector_weight = 0.7

        # Support limit as alias for k
        if k is None and limit is None:
            k = 4
        elif k is None:
            k = limit

        payload = {
            "vector": vector,
            "text": text,
            "vector_weight": vector_weight,
            "limit": k,
            "filter": convert_simple_filter_to_dxmt(filter),
        }
        try:
            response = await self.async_client.post(
                f"/tables/{table_name}/search/hybrid",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", {}).get("hits", [])
        except httpx.HTTPStatusError as e:
            msg = f"Hybrid search failed: {e.response.text}"
            raise DXMTSearchError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error during hybrid search: {e}"
            raise DXMTConnectionError(msg) from e

    def full_text_search(
        self,
        table_name: str,
        query: Optional[str] = None,
        text_query: Optional[str] = None,  # Alias for query
        k: Optional[int] = None,
        limit: Optional[int] = None,  # Alias for k
        fields: Optional[list[str]] = None,
        highlight: bool = False,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Full-text search.

        Args:
            table_name: Name of the table.
            query: Search query text (preferred parameter name).
            text_query: Alias for query parameter (for compatibility).
            k: Number of results to return (preferred parameter name).
            limit: Number of results to return (alias for k).
            fields: Optional list of fields to search.
            highlight: Whether to include highlights.
            filter: Optional metadata filter conditions.

        Returns:
            List of search results.

        Raises:
            DXMTSearchError: If search fails.
            ValueError: If neither query nor text_query is provided.
        """
        # Support text_query as alias for query
        if query is None and text_query is not None:
            query = text_query
        elif query is None:
            msg = "Either 'query' or 'text_query' parameter is required"
            raise ValueError(msg)

        # Support limit as alias for k
        if k is None and limit is None:
            k = 4
        elif k is None:
            k = limit

        payload = {
            "query": query,
            "limit": k,
            "fields": fields,
            "highlight": highlight,
            "filter": filter,
        }
        try:
            response = self._client.post(
                f"/tables/{table_name}/search/text",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            hits = data.get("data", {}).get("hits", [])

            # Map 'payload' to 'metadata' for user convenience
            for hit in hits:
                if "payload" in hit and "metadata" not in hit:
                    hit["metadata"] = hit["payload"]

            return hits
        except httpx.HTTPStatusError as e:
            msg = f"Full-text search failed: {e.response.text}"
            raise DXMTSearchError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error during full-text search: {e}"
            raise DXMTConnectionError(msg) from e

    def query(
        self,
        sql: str,
    ) -> dict[str, Any]:
        """Execute a DXQL query.

        Args:
            sql: DXQL query string (SELECT, INSERT, UPDATE, DELETE, CREATE TABLE, DROP TABLE).

        Returns:
            Dict with 'records' list and 'metadata' dict.

        Raises:
            DXMTError: If query fails.

        Example:
            client.query("SELECT * FROM my_table WHERE category = 'AI' LIMIT 10")
        """
        try:
            response = self._client.post(
                "/query",
                json={"query": sql},
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", {})
        except httpx.HTTPStatusError as e:
            msg = f"DXQL query failed: {e.response.text}"
            raise DXMTError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error during DXQL query: {e}"
            raise DXMTConnectionError(msg) from e

    async def aquery(
        self,
        sql: str,
    ) -> dict[str, Any]:
        """Async execute a DXQL query."""
        try:
            response = await self.async_client.post(
                "/query",
                json={"query": sql},
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", {})
        except httpx.HTTPStatusError as e:
            msg = f"DXQL query failed: {e.response.text}"
            raise DXMTError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error during DXQL query: {e}"
            raise DXMTConnectionError(msg) from e

    async def afull_text_search(
        self,
        table_name: str,
        query: Optional[str] = None,
        text_query: Optional[str] = None,  # Alias for query
        k: Optional[int] = None,
        limit: Optional[int] = None,  # Alias for k
        fields: Optional[list[str]] = None,
        highlight: bool = False,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Async full-text search.

        Args:
            table_name: Name of the table.
            query: Search query text (preferred parameter name).
            text_query: Alias for query parameter (for compatibility).
            k: Number of results to return (preferred parameter name).
            limit: Number of results to return (alias for k).
            fields: Optional list of fields to search.
            highlight: Whether to include highlights.
            filter: Optional metadata filter conditions.

        Returns:
            List of search results.
        """
        # Support text_query as alias for query
        if query is None and text_query is not None:
            query = text_query
        elif query is None:
            msg = "Either 'query' or 'text_query' parameter is required"
            raise ValueError(msg)

        # Support limit as alias for k
        if k is None and limit is None:
            k = 4
        elif k is None:
            k = limit

        payload = {
            "query": query,
            "limit": k,
            "fields": fields,
            "highlight": highlight,
            "filter": filter,
        }
        try:
            response = await self.async_client.post(
                f"/tables/{table_name}/search/text",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            hits = data.get("data", {}).get("hits", [])

            # Map 'payload' to 'metadata' for user convenience
            for hit in hits:
                if "payload" in hit and "metadata" not in hit:
                    hit["metadata"] = hit["payload"]

            return hits
        except httpx.HTTPStatusError as e:
            msg = f"Full-text search failed: {e.response.text}"
            raise DXMTSearchError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error during full-text search: {e}"
            raise DXMTConnectionError(msg) from e
