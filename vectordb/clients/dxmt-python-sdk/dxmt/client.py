"""DXMT Python SDK HTTP Client.

This module provides sync and async HTTP clients for interacting with
the DXMT VectorDB REST API.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterable, Iterator, Optional

from urllib.parse import quote

import httpx

from .exceptions import (
    DXMTConnectionError,
    DXMTError,
    DXMTRecordError,
    DXMTSearchError,
    DXMTTableError,
)
from .filters import convert_simple_filter_to_dxmt
from .types import Filter

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised by whichever branch is installed
    import orjson

    _HAS_ORJSON = True
except ImportError:  # pragma: no cover
    orjson = None
    _HAS_ORJSON = False

# Content-Type has to be set by hand when a body is passed as bytes rather than through
# httpx's `json=`. httpx merges per-request headers over the client's, so auth survives.
_JSON_HEADERS = {"Content-Type": "application/json"}


def _encode_json(payload: Any) -> bytes:
    """Serialise a request body, preferring orjson when it is installed.

    An insert is mostly floats -- 100 records of 768 dimensions is 76,800 of them per
    request -- and `json.dumps` spends that entirely in Python holding the GIL, so
    concurrent writer threads serialise one at a time no matter how many there are.
    Measured on this SDK's own insert path, that capped a four-thread load at about a
    third of what the server could absorb.

    Measured here on a 100 x 768 body: the standard library encodes 1,215 rows/s on one
    thread and *1,124 on four* -- it does not scale at all -- while orjson does about
    7,900 either way, 7.3x faster. The four-thread figure is the one that matters: it is
    within noise of the 1,262 rows/s a four-worker VectorDBBench load measured against
    this SDK, which is to say that benchmark was measuring this function.

    orjson is optional and the fallback is what httpx itself would have done. The two
    bodies are not byte-identical -- the standard library writes small magnitudes in
    scientific notation (8.859986183940016e-05) where orjson writes them out
    (0.00008859986183940016) -- but they parse to the same values, so this is a
    formatting difference and not a protocol change. One real behavioural difference:
    `allow_nan=False` makes the fallback raise on a NaN or infinity, while orjson writes
    `null` and lets the server reject it. A NaN in a vector is a bug either way; it just
    surfaces one hop later.
    """
    if _HAS_ORJSON:
        return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")



def _seg(value: str) -> str:
    """Percent-encode one URL path segment.

    A table name or a record id goes into the path, and ids carry characters the path reserves.
    `doc.md#3` stops at the `#` -- a fragment delimiter -- so the request reaches
    `/records/doc.md` and the server answers about a record nobody asked for; `s3://bucket/key`
    turns one segment into three. Both shapes are routine in a RAG corpus. Characters that need
    no encoding pass through unchanged, so this is safe on every id.
    """
    return quote(str(value), safe="")


#: Statuses a retry can fix: the server is busy (429, 503) or momentarily wrong (5xx). A 4xx
#: other than 429 says the request itself is wrong, and sending it again changes nothing.
_RETRYABLE = frozenset({429, 500, 502, 503, 504})


@dataclass
class UploadFailure:
    """One batch that did not go in, after every retry it was worth."""

    ids: list[str]
    error: str
    status: Optional[int] = None


@dataclass
class UploadResult:
    """What `upload` did. `failed` is empty when everything went in."""

    uploaded: int = 0
    batches: int = 0
    retries: int = 0
    failed: list[UploadFailure] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed


def _payload_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """`metadata` renamed to `payload`, copying only the records that need it."""
    out = []
    for record in records:
        if "metadata" in record and "payload" not in record:
            record = dict(record)
            record["payload"] = record.pop("metadata")
        out.append(record)
    return out


def _batches(records: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    """Chunks of `size`, drawn lazily — so a generator of ten million records is never held."""
    batch: list[dict[str, Any]] = []
    for record in records:
        batch.append(record)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


class DXMTClient:
    """HTTP client for DXMT VectorDB API.

    Handles all communication with DXMT server via REST API.
    Supports both sync and async operations.

    Args:
        host: DXMT server hostname. Default is 'localhost'.
        port: DXMT server port. Default is 9999.
        ssl: Whether to use HTTPS. Default is False.
        api_key: The server's API key, when it was started with ``DXMT_API_KEY``. Sent as the
            ``api-key`` header on every request. Default is None: the server's key is off by
            default, and without one the client sends nothing.
        headers: Additional HTTP headers.
        timeout: Request timeout in seconds. Default is 30.0.
        **kwargs: Additional httpx client parameters.
    """

    #: Hosts where sending a key over plain HTTP does not cross a network.
    _LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9999,
        *,
        ssl: bool = False,
        api_key: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        """Initialize DXMT client."""
        protocol = "https" if ssl else "http"
        self.base_url = f"{protocol}://{host}:{port}/api/v1"
        # Copied, so adding the key does not write into a dict the caller still holds.
        self._headers = dict(headers or {})
        if api_key:
            # The explicit argument wins over an `api-key` the caller also put in `headers`, the
            # way qdrant-client resolves the same collision.
            self._headers["api-key"] = api_key
            if not ssl and host not in self._LOCAL_HOSTS:
                # A key over plain HTTP reaches every hop between here and the server, which is
                # worse than sending none. qdrant-client switches to HTTPS when given a key; that
                # would break a server that has no TLS in front of it, so this only says so.
                warnings.warn(
                    f"sending an API key to {host} over plain HTTP; pass ssl=True, or put TLS "
                    "in front of the server",
                    stacklevel=2,
                )
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

    async def ahealth_check(self) -> dict[str, Any]:
        """Async variant of :meth:`health_check`."""
        try:
            response = await self.async_client.get("/health")
            response.raise_for_status()
            data = self._response_data(response.json())
            return data if isinstance(data, dict) else {}
        except httpx.HTTPStatusError as e:
            raise DXMTConnectionError(f"Health check failed: {e.response.text}") from e
        except httpx.RequestError as e:
            raise DXMTConnectionError(f"Connection error during health check: {e}") from e

    async def ahealth_metrics(self) -> dict[str, Any]:
        """Async variant of :meth:`health_metrics`."""
        try:
            response = await self.async_client.get("/health/metrics")
            response.raise_for_status()
            data = self._response_data(response.json())
            return data if isinstance(data, dict) else {}
        except httpx.HTTPStatusError as e:
            raise DXMTConnectionError(f"Health metrics failed: {e.response.text}") from e
        except httpx.RequestError as e:
            raise DXMTConnectionError(f"Connection error during health metrics: {e}") from e

    def health_check(self) -> dict[str, Any]:
        """Return the server's readiness and durability health snapshot."""
        try:
            response = self._client.get("/health")
            response.raise_for_status()
            data = self._response_data(response.json())
            return data if isinstance(data, dict) else {}
        except httpx.HTTPStatusError as e:
            raise DXMTConnectionError(f"Health check failed: {e.response.text}") from e
        except httpx.RequestError as e:
            raise DXMTConnectionError(f"Connection error during health check: {e}") from e

    def health_metrics(self) -> dict[str, Any]:
        """Return detailed server and storage health metrics."""
        try:
            response = self._client.get("/health/metrics")
            response.raise_for_status()
            data = self._response_data(response.json())
            return data if isinstance(data, dict) else {}
        except httpx.HTTPStatusError as e:
            raise DXMTConnectionError(f"Health metrics failed: {e.response.text}") from e
        except httpx.RequestError as e:
            raise DXMTConnectionError(f"Connection error during health metrics: {e}") from e

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

    @staticmethod
    def _response_data(response: Any) -> Any:
        """Return the payload from a DXMT response envelope.

        The API normally returns ``{"success": true, "data": ...}``, while
        a couple of compatibility endpoints return the payload directly.  SDK
        methods should not need to know which form a route used.
        """
        if (
            isinstance(response, dict)
            and "data" in response
            and (response.get("success") is True or set(response) == {"data"})
        ):
            return response.get("data")
        return response

    @classmethod
    def _normalize_record(cls, record: Any) -> dict[str, Any]:
        """Normalize Rust record/search-hit fields to the public SDK shape."""
        if not isinstance(record, dict):
            return {}
        normalized = dict(record)

        # ``Record`` uses payload_json on the wire to avoid bincode issues;
        # search hits already expose a decoded payload object.
        payload = normalized.get("payload")
        if payload is None and normalized.get("payload_json"):
            try:
                payload = json.loads(normalized["payload_json"])
            except (json.JSONDecodeError, TypeError):
                payload = {}
            normalized["payload"] = payload

        if "payload" in normalized and "metadata" not in normalized:
            normalized["metadata"] = normalized["payload"]
        return normalized

    @classmethod
    def _normalize_hits(cls, response: Any) -> list[dict[str, Any]]:
        """Extract and normalize a search response's hits."""
        data = cls._response_data(response)
        if not isinstance(data, dict):
            return []
        hits = data.get("hits", [])
        if not isinstance(hits, list):
            return []
        return [cls._normalize_record(hit) for hit in hits]

    # Table Management

    def create_table(
        self,
        name: str,
        dimension: Optional[int] = None,
        distance_metric: str = "cosine",
        index_type: str = "hnsw",
        description: Optional[str] = None,
        enable_full_text_search: bool = False,
        text_fields: Optional[list[dict[str, Any]]] = None,
        memory_optimization: Optional[bool] = None,
        hnsw: Optional[dict[str, int]] = None,
        quantization: Optional[dict[str, int]] = None,
        payload_index: Optional[dict[str, list[str]]] = None,
        embedding: Optional[dict[str, Any]] = None,
        rerank: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Create a new table (collection).

        Args:
            name: Table name.
            dimension: Vector dimension, fixed for the life of the table. Optional only
                when `embedding` is given: the server then asks the model how wide its vectors
                are, which is the number most often got wrong (1536 for text-embedding-3-small,
                1024 for bge-m3, 384 for all-MiniLM-L6-v2) and whose cost is a table that
                accepts no writes.
            distance_metric: Distance metric (cosine, euclidean, manhattan, dot).
            index_type: Index type (hnsw, ivf, ivf_pq, lsh, brute_force).
            description: Optional table description.
            hnsw: Graph parameters for this table: `m`, `ef_construction`, `ef_search`. They
                decide the graph and cannot be recovered afterwards -- searching harder does
                not add edges the build never made -- and they are per table because one
                server holds a small lookup table beside a large corpus.
            quantization: `num_subvectors`, `num_centroids`, `quantize_at_rows`,
                `fast_scan`. Setting `quantize_at_rows` above the table's eventual size
                keeps it exact; `fast_scan: False` leaves out the 4-bit block-scan codes
                that speed up filtered search over large match sets (default on).
            payload_index: Which payload fields to index for filtering: `{"only": [...]}` or
                `{"except": [...]}`. Every field is indexed by default, which is what makes a
                filter fast without declaring anything; a field nobody filters on pays memory
                and rebuild time for nothing. A filter on an excluded field is refused, not
                answered empty.
            embedding: Server-side embedding for this table, e.g.
                `{"model": "text-embedding-3-small"}` or
                `{"model": "BAAI/bge-m3", "base_url": "http://10.0.0.7:8080/v1"}`. Records may
                then be inserted with `text` in place of `vector`, and `search(text=...)`
                embeds the query with this same model. The key is read from the server's
                environment (`api_key_env`, default `OPENAI_API_KEY`), never sent from here.
            enable_full_text_search: Enable full-text search.
            text_fields: Text field configurations.
            memory_optimization: Enable memory optimization (optional).
                - None: Use global config (DXMT_MEMORY_OPTIMIZATION)
                - True: Enable memory optimization (Phase 4 Hybrid), saves 85-90% memory, 87-95% recall
                - False: High accuracy mode (HNSW/Phase 3), maintains 95-99% recall

        Returns:
            Response data from server.

        Raises:
            DXMTTableError: If table creation fails.
        """
        payload = {
            "name": name,
            "distance_metric": distance_metric,
            "index_type": index_type,
            "description": description,
            "enable_full_text_search": enable_full_text_search,
            "text_fields": text_fields,
            "memory_optimization": memory_optimization,
        }
        # Omitted rather than sent as null: the server refuses fields it does not have, and
        # would have to decide what a null means where None already means "the default".
        if hnsw is not None:
            payload["hnsw"] = hnsw
        if quantization is not None:
            payload["quantization"] = quantization
        # Omitted rather than null when the model is to supply it: the server reads its
        # absence as "ask the model how wide its vectors are", and a null would be a value.
        if dimension is not None:
            payload["dimension"] = dimension
        if payload_index is not None:
            payload["payload_index"] = payload_index
        # The model the server embeds text with: `{"model": "text-embedding-3-small"}`, plus
        # an optional `base_url` for anything else speaking the OpenAI /embeddings protocol.
        # With it, records and searches may carry `text` instead of a vector. The API key is
        # not in here -- `api_key_env` names an environment variable of the *server*.
        if embedding is not None:
            payload["embedding"] = embedding
        if rerank is not None:
            # A cross-encoder the server calls when a search asks for `rerank=...`:
            # `{"model": ..., "base_url": ..., "format": "cohere" | "tei", "api_key_env": ...}`.
            # The key is read from the server's environment, never sent from here.
            payload["rerank"] = rerank
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
        dimension: Optional[int] = None,
        distance_metric: str = "cosine",
        index_type: str = "hnsw",
        description: Optional[str] = None,
        enable_full_text_search: bool = False,
        text_fields: Optional[list[dict[str, Any]]] = None,
        memory_optimization: Optional[bool] = None,
        hnsw: Optional[dict[str, int]] = None,
        quantization: Optional[dict[str, int]] = None,
        payload_index: Optional[dict[str, list[str]]] = None,
        embedding: Optional[dict[str, Any]] = None,
        rerank: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Async create a new table. See `create_table` for what each argument does."""
        payload = {
            "name": name,
            "distance_metric": distance_metric,
            "index_type": index_type,
            "description": description,
            "enable_full_text_search": enable_full_text_search,
            "text_fields": text_fields,
            "memory_optimization": memory_optimization,
        }
        if hnsw is not None:
            payload["hnsw"] = hnsw
        if quantization is not None:
            payload["quantization"] = quantization
        # Omitted rather than null when the model is to supply it: the server reads its
        # absence as "ask the model how wide its vectors are", and a null would be a value.
        if dimension is not None:
            payload["dimension"] = dimension
        if payload_index is not None:
            payload["payload_index"] = payload_index
        # The model the server embeds text with: `{"model": "text-embedding-3-small"}`, plus
        # an optional `base_url` for anything else speaking the OpenAI /embeddings protocol.
        # With it, records and searches may carry `text` instead of a vector. The API key is
        # not in here -- `api_key_env` names an environment variable of the *server*.
        if embedding is not None:
            payload["embedding"] = embedding
        if rerank is not None:
            # A cross-encoder the server calls when a search asks for `rerank=...`:
            # `{"model": ..., "base_url": ..., "format": "cohere" | "tei", "api_key_env": ...}`.
            # The key is read from the server's environment, never sent from here.
            payload["rerank"] = rerank
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
            data = self._response_data(response.json())
            # API returns {"data": {"table": {...}, "record_count": N}}
            if isinstance(data, dict) and isinstance(data.get("table"), dict):
                table = dict(data["table"])
                if "record_count" in data:
                    table["record_count"] = data["record_count"]
                return table
            return data if isinstance(data, dict) else {}
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
            data = self._response_data(response.json())
            # API returns {"data": {"table": {...}, "record_count": N}}
            if isinstance(data, dict) and isinstance(data.get("table"), dict):
                table = dict(data["table"])
                if "record_count" in data:
                    table["record_count"] = data["record_count"]
                return table
            return data if isinstance(data, dict) else {}
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
            data = self._response_data(response.json())
            # API returns {"data": {"tables": [...], "total": N, ...}}
            return data.get("tables", []) if isinstance(data, dict) else []
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
            data = self._response_data(response.json())
            # API returns {"data": {"tables": [...], "total": N, ...}}
            return data.get("tables", []) if isinstance(data, dict) else []
        except httpx.HTTPStatusError as e:
            msg = f"Failed to list tables: {e.response.text}"
            raise DXMTTableError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error listing tables: {e}"
            raise DXMTConnectionError(msg) from e

    # Document Operations

    def scroll(
        self,
        table_name: str,
        *,
        filter: Optional[Filter] = None,
        batch_size: int = 100,
        with_vector: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Every record of a table, newest first, fetched a page at a time.

        Pages by cursor, not offset: every record present for the whole scroll comes back
        exactly once however the table is written to meanwhile, where an offset shifts with each
        insert and repeats or skips records. For exporting a table, re-indexing it elsewhere, or
        anything else that has to see all of it.

        Args:
            table_name: Name of the table.
            filter: Only records matching this filter.
            batch_size: Records per request, at most 1,000.
            with_vector: Include each record's vector.

        Yields:
            Records, normalised the way `list_records` returns them.

        Raises:
            DXMTRecordError: If a page fails, or the server does not page by cursor.
            DXMTConnectionError: On a connection error.
        """
        cursor: Optional[str] = None
        while True:
            body = self._scroll_body(filter, batch_size, with_vector, cursor)
            try:
                response = self._client.post(f"/tables/{_seg(table_name)}/records/list", json=body)
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise DXMTRecordError(f"Failed to scroll '{table_name}': {e.response.text}") from e
            except httpx.RequestError as e:
                raise DXMTConnectionError(f"Connection error while scrolling: {e}") from e
            records, cursor = self._scroll_page(response.json(), batch_size)
            yield from records
            if cursor is None:
                return

    async def ascroll(
        self,
        table_name: str,
        *,
        filter: Optional[Filter] = None,
        batch_size: int = 100,
        with_vector: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async version of :meth:`scroll`."""
        cursor: Optional[str] = None
        while True:
            body = self._scroll_body(filter, batch_size, with_vector, cursor)
            try:
                response = await self.async_client.post(f"/tables/{_seg(table_name)}/records/list", json=body)
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise DXMTRecordError(f"Failed to scroll '{table_name}': {e.response.text}") from e
            except httpx.RequestError as e:
                raise DXMTConnectionError(f"Connection error while scrolling: {e}") from e
            records, cursor = self._scroll_page(response.json(), batch_size)
            for record in records:
                yield record
            if cursor is None:
                return

    @staticmethod
    def _scroll_body(filter, batch_size, with_vector, cursor) -> dict[str, Any]:
        body: dict[str, Any] = {"limit": batch_size, "with_vector": with_vector}
        converted = convert_simple_filter_to_dxmt(filter)
        if converted is not None:
            body["filter"] = converted
        if cursor is not None:
            body["cursor"] = cursor
        return body

    def _scroll_page(self, raw: Any, batch_size: int) -> tuple[list[dict[str, Any]], Optional[str]]:
        data = self._response_data(raw)
        records = data.get("records") or []
        # A server from before cursors answers without the field. Taking that as "no next page"
        # would end every scroll after one page, silently; a full page without it is an error.
        if "next_cursor" not in data and len(records) >= batch_size:
            raise DXMTRecordError("this server does not page by cursor; upgrade it to scroll")
        return [self._normalize_record(r) for r in records], data.get("next_cursor")

    def upload(
        self,
        table_name: str,
        records: Iterable[dict[str, Any]],
        *,
        batch_size: int = 256,
        parallel: int = 1,
        max_retries: int = 3,
        backoff: float = 0.5,
    ) -> UploadResult:
        """Insert many records, in batches, retrying what a retry can fix.

        `records` may be any iterable, a generator included; it is drawn a batch at a time, so a
        load larger than memory never has to be held at once. A batch that fails for a reason a
        retry can fix — a dropped connection, a timeout, 429, a 5xx — is sent again after an
        exponential backoff; one that fails because the request is wrong (a dimension mismatch,
        a missing table) is not, and is reported. Inserts are upserts, so a retried batch that
        had in fact landed does not duplicate anything.

        It does not raise for a failed batch: every batch is attempted, and `result.failed`
        says which ids did not go in and why, so one bad record does not abandon the rest.

        Args:
            table_name: Name of the table.
            records: Records with `id`, `vector` and optional `payload` (or `metadata`).
            batch_size: Records per request.
            parallel: Requests in flight at once.
            max_retries: Further attempts per batch after the first.
            backoff: Seconds before the first retry, doubling after each.

        Returns:
            How many records went in, how many batches and retries it took, and what failed.
        """
        result = UploadResult()
        path = f"/tables/{_seg(table_name)}/records"

        def send(batch: list[dict[str, Any]]) -> tuple[int, Optional[UploadFailure], int]:
            body = _encode_json({"records": _payload_records(batch)})
            retries = 0
            while True:
                status: Optional[int] = None
                try:
                    response = self._client.post(path, content=body, headers=_JSON_HEADERS)
                    if response.is_success:
                        return len(batch), None, retries
                    status, error = response.status_code, response.text
                except httpx.RequestError as e:
                    error = f"connection error: {e}"
                retryable = status is None or status in _RETRYABLE
                if not retryable or retries >= max_retries:
                    ids = [str(r.get("id")) for r in batch]
                    return 0, UploadFailure(ids=ids, error=error[:500], status=status), retries
                time.sleep(backoff * (2 ** retries))
                retries += 1

        def record(outcome: tuple[int, Optional[UploadFailure], int]) -> None:
            uploaded, failure, retries = outcome
            result.uploaded += uploaded
            result.batches += 1
            result.retries += retries
            if failure is not None:
                result.failed.append(failure)

        if parallel <= 1:
            for batch in _batches(records, batch_size):
                record(send(batch))
            return result

        # Bounded in flight, so a generator is still drawn only as fast as batches are sent.
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            pending: deque = deque()
            for batch in _batches(records, batch_size):
                pending.append(pool.submit(send, batch))
                if len(pending) >= parallel * 2:
                    record(pending.popleft().result())
            while pending:
                record(pending.popleft().result())
        return result

    def insert_records(
        self,
        table_name: str,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Insert records into a table.

        Args:
            table_name: Name of the table.
            records: List of records with id, vector, and metadata.

        Returns:
            Response data from server.

        Raises:
            DXMTError: If insertion fails.
        """
        # Normalize records: map 'metadata' to 'payload' if present. Only the records that
        # actually need remapping are copied -- this used to copy every record on every
        # insert, which on a bulk load is a dict allocation per row for nothing.
        normalized_records = []
        for record in records:
            if "metadata" in record and "payload" not in record:
                normalized = dict(record)
                normalized["payload"] = normalized.pop("metadata")
                normalized_records.append(normalized)
            else:
                normalized_records.append(record)

        payload = {"records": normalized_records}
        try:
            response = self._client.post(
                f"/tables/{_seg(table_name)}/records",
                content=_encode_json(payload),
                headers=_JSON_HEADERS,
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
    ) -> dict[str, Any]:
        """Async insert records into a table."""
        # Normalize records: map 'metadata' to 'payload' if present
        normalized_records = []
        for record in records:
            normalized = record.copy()
            if "metadata" in normalized and "payload" not in normalized:
                normalized["payload"] = normalized.pop("metadata")
            normalized_records.append(normalized)

        payload = {"records": normalized_records}
        try:
            response = await self.async_client.post(
                f"/tables/{_seg(table_name)}/records",
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
                f"/tables/{_seg(table_name)}/records/{_seg(record_id)}",
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
        merge_mode: str = "replace",
        upsert: bool = False,
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
        request_payload["merge_mode"] = merge_mode
        request_payload["upsert"] = upsert

        try:
            response = await self.async_client.put(
                f"/tables/{_seg(table_name)}/records/{_seg(record_id)}",
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

    def sparse_search(
        self,
        table_name: str,
        sparse_vector: dict[str, Any],
        *,
        limit: int = 10,
        filter: Optional[Filter] = None,
        with_payload: bool = True,
        with_vector: bool = False,
        payload_fields: Optional[list[str]] = None,
        score_threshold: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve by sparse vector alone, ranked by dot product.

        Sparse vectors could be stored and indexed here long before they could be searched:
        the only path into the sparse index was the hybrid endpoint, which refuses a body with
        no dense vector and no text, and needs a `text-bm25` build. This needs neither.

        Args:
            table_name: Name of the table.
            sparse_vector: `{"indices": [...], "values": [...], "dimension": n}`.
            limit: How many hits to return.
            filter: Applied to the candidates the sparse index returns.
            with_payload: Include payloads.
            with_vector: Include the dense vectors.
            payload_fields: Return only these payload keys.
            score_threshold: Floor on the dot product; higher is nearer, and it only removes.

        Returns:
            The hits, best first.

        Raises:
            DXMTError: If the request fails.
        """
        body: dict[str, Any] = {
            "sparse_vector": sparse_vector,
            "limit": limit,
            "with_payload": with_payload,
            "with_vector": with_vector,
        }
        converted = convert_simple_filter_to_dxmt(filter)
        if converted is not None:
            body["filter"] = converted
        if payload_fields is not None:
            body["payload_fields"] = payload_fields
        if score_threshold is not None:
            body["score_threshold"] = score_threshold
        response = self._client.post(
            f"/tables/{quote(table_name, safe='')}/search/sparse", json=body
        )
        response.raise_for_status()
        return response.json().get("data", {}).get("hits", [])

    @staticmethod
    def _sparse_body(
        sparse_vector: dict[str, Any],
        limit: int,
        filter: Optional[Filter],
        with_payload: bool,
        with_vector: bool,
        payload_fields: Optional[list[str]],
        score_threshold: Optional[float],
    ) -> dict[str, Any]:
        """The request body for `/search/sparse`, shared by the sync and async pair.

        One function, because the two used to be the place where a field gets added to one
        call and not the other, and nothing notices until a caller on the async path finds
        an option that silently does nothing.
        """
        body: dict[str, Any] = {
            "sparse_vector": sparse_vector,
            "limit": limit,
            "with_payload": with_payload,
            "with_vector": with_vector,
        }
        converted = convert_simple_filter_to_dxmt(filter)
        if converted is not None:
            body["filter"] = converted
        if payload_fields is not None:
            body["payload_fields"] = payload_fields
        if score_threshold is not None:
            body["score_threshold"] = score_threshold
        return body

    async def asparse_search(
        self,
        table_name: str,
        sparse_vector: dict[str, Any],
        *,
        limit: int = 10,
        filter: Optional[Filter] = None,
        with_payload: bool = True,
        with_vector: bool = False,
        payload_fields: Optional[list[str]] = None,
        score_threshold: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """`sparse_search` on the event loop. See it for what the arguments mean.

        It was the one search endpoint with no async twin, which made
        `VectorStoreQueryMode.SPARSE` the one mode the LlamaIndex integration could not
        answer without blocking.
        """
        body = self._sparse_body(
            sparse_vector,
            limit,
            filter,
            with_payload,
            with_vector,
            payload_fields,
            score_threshold,
        )
        try:
            response = await self.async_client.post(
                f"/tables/{_seg(table_name)}/search/sparse", json=body
            )
            response.raise_for_status()
            return response.json().get("data", {}).get("hits", [])
        except httpx.HTTPStatusError as e:
            msg = f"Sparse search failed: {e.response.text}"
            raise DXMTSearchError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error during sparse search: {e}"
            raise DXMTConnectionError(msg) from e

    def search_groups(
        self,
        table_name: str,
        vector: list[float],
        group_by: str,
        *,
        groups: int = 10,
        group_size: int = 3,
        filter: Optional[Filter] = None,
        with_payload: bool = True,
        payload_fields: Optional[list[str]] = None,
        score_threshold: Optional[float] = None,
        search_mode: Optional[str] = None,
    ) -> dict[str, Any]:
        """Nearest neighbours grouped by a payload field.

        A document chunked for retrieval takes a whole top-10 by itself: the ten nearest
        vectors are ten pieces of the same text. Grouping by the document's id with
        `group_size=1` returns ten documents instead.

        Groups are filled from a window of nearest neighbours that widens until enough groups
        appear or the server's cap is reached, so `candidates_fetched` in the answer is worth
        reading: a short result means either the data ran out or the cap did.

        Args:
            table_name: Name of the table.
            vector: Query vector.
            group_by: Payload field whose value defines a group. A record with no value there
                is in no group and is counted in `ungrouped`.
            groups: How many groups to return.
            group_size: How many hits to keep per group, best first.
            filter: Applied before grouping, as in a plain search.
            with_payload: Include payloads in the hits.
            payload_fields: Return only these payload keys.
            score_threshold: Similarity floor; higher is nearer.
            search_mode: fast | normal | accurate | auto | exact.

        Returns:
            `{"groups": [...], "candidates_fetched": n, "ungrouped": n, "tookms": n}`.

        Raises:
            DXMTError: If the request fails.
        """
        body: dict[str, Any] = {
            "vector": vector,
            "group_by": group_by,
            "groups": groups,
            "group_size": group_size,
            "with_payload": with_payload,
        }
        converted = convert_simple_filter_to_dxmt(filter)
        if converted is not None:
            body["filter"] = converted
        if payload_fields is not None:
            body["payload_fields"] = payload_fields
        if score_threshold is not None:
            body["score_threshold"] = score_threshold
        if search_mode is not None:
            body["search_mode"] = search_mode
        response = self._client.post(
            f"/tables/{quote(table_name, safe='')}/search/groups", json=body
        )
        response.raise_for_status()
        return response.json().get("data", {})

    def aliases(self) -> dict[str, str]:
        """Every alias and the table it points at.

        Returns:
            `{alias: table}`.

        Raises:
            DXMTError: If the request fails.
        """
        response = self._client.get("/aliases")
        response.raise_for_status()
        return response.json().get("data", {}).get("aliases", {})

    def set_alias(self, alias: str, table_name: str) -> dict[str, Any]:
        """Point an alias at a table, creating it or moving it.

        This is how a table is rebuilt without a client deploy: build the new one under its own
        name, load it, then move the alias. Every reader that arrives during the move sees one
        table or the other, never neither. An alias may be used anywhere a table name is.

        Args:
            alias: The alias name. It may not be the name of an existing table.
            table_name: The table it resolves to.

        Returns:
            `{"alias", "table"}`.

        Raises:
            DXMTError: If the table does not exist, or a table already has that name.
        """
        response = self._client.put(
            f"/aliases/{quote(alias, safe='')}", json={"table": table_name}
        )
        response.raise_for_status()
        return response.json().get("data", {})

    def delete_alias(self, alias: str) -> dict[str, Any]:
        """Remove an alias. The table it pointed at is untouched.

        Args:
            alias: The alias name.

        Returns:
            The server's message.

        Raises:
            DXMTError: If there is no such alias.
        """
        response = self._client.delete(f"/aliases/{quote(alias, safe='')}")
        response.raise_for_status()
        return response.json().get("data", {})

    def export_table(self, table_name: str, path: str) -> dict[str, Any]:
        """Write one table -- configuration and every record, vectors included -- to a file.

        This is the unit `POST /backup` is not: a backup is the whole instance and restoring it
        means stopping the server and replacing its directory. The file is written **by the
        server**, on the server's filesystem, inside its `DXMT_BACKUP_ROOT`; `path` is not a
        path on this machine unless the two are the same machine.

        The graph is not in the file. `import_table` rebuilds it from the records, which is the
        same cost a restore from the WAL pays.

        Args:
            table_name: Table to export.
            path: Destination file on the server, inside `DXMT_BACKUP_ROOT`.

        Returns:
            `{"table", "path", "records", "bytes", "records_without_vectors"}`.
            `records_without_vectors` is zero on a healthy table; anything else is a record
            whose vector was neither in memory nor readable from the row file.

        Raises:
            DXMTError: If the path is outside the root, or the table does not exist.
        """
        response = self._client.post(
            f"/tables/{quote(table_name, safe='')}/export", json={"path": path}
        )
        response.raise_for_status()
        return response.json().get("data", {})

    def import_table(
        self,
        path: str,
        *,
        name: Optional[str] = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Read a table back out of a file written by `export_table`.

        Records keep their ids, payloads, versions and `created_at`: they are applied the way
        WAL replay applies them rather than re-inserted through the API, so an import does not
        restamp every row and reorder listings.

        Args:
            path: The file on the server, inside `DXMT_BACKUP_ROOT`.
            name: Import under this name instead of the one in the file -- which is how a table
                is copied beside itself, to try a different index on the same data.
            overwrite: Replace a table of that name if one exists. Without it an import onto a
                live name is refused, because a merge and a replacement are different
                operations and neither is the obvious reading of "import".

        Returns:
            `{"table", "path", "records", "records_in_file", "replaced"}`.

        Raises:
            DXMTError: If the file is unreadable, outside the root, or the name is in use.
        """
        body: dict[str, Any] = {"path": path, "overwrite": overwrite}
        if name is not None:
            body["name"] = name
        response = self._client.post("/tables/import", json=body)
        response.raise_for_status()
        return response.json().get("data", {})

    def payload_index(self, table_name: str) -> dict[str, Any]:
        """What the payload (filter) index holds for each field, and what it costs.

        Every payload field is indexed by default, so this is the answer to "what is that
        costing me": distinct values, postings and an estimate of the bytes, worst field
        first. It does not build the index in order to describe it -- on a table that has not
        filtered since it was restored, `built` is false and `fields` is empty.

        Args:
            table_name: Name of the table.

        Returns:
            The report, as the server sends it.

        Raises:
            DXMTError: If the request fails.
        """
        response = self._client.get(f"/tables/{quote(table_name, safe='')}/payload_index")
        response.raise_for_status()
        return response.json().get("data", {})

    def index_payload_field(self, table_name: str, field: str) -> dict[str, Any]:
        """Index a payload field for filtering.

        Idempotent, and a no-op on a table that indexes every field -- which is the default,
        so code written against a database where indexes must be declared works unchanged.
        The index is rebuilt from the records, so the field is filterable over rows written
        before it was indexed.

        Args:
            table_name: Name of the table.
            field: Payload field name.

        Returns:
            The server's message and the table's resulting policy.

        Raises:
            DXMTError: If the request fails.
        """
        response = self._client.put(
            f"/tables/{quote(table_name, safe='')}/payload_index/{quote(field, safe='')}"
        )
        response.raise_for_status()
        return response.json().get("data", {})

    def drop_payload_index_field(self, table_name: str, field: str) -> dict[str, Any]:
        """Stop indexing a payload field, freeing what its postings cost.

        The price is filtering on it: a filter naming an unindexed field is refused with a
        400, because this engine answers a filter only from the index and the only other
        honest answer would be an empty page -- which looks exactly like "nothing matched".

        Args:
            table_name: Name of the table.
            field: Payload field name.

        Returns:
            The server's message and the table's resulting policy.

        Raises:
            DXMTError: If the request fails.
        """
        response = self._client.delete(
            f"/tables/{quote(table_name, safe='')}/payload_index/{quote(field, safe='')}"
        )
        response.raise_for_status()
        return response.json().get("data", {})

    def count(
        self,
        table_name: str,
        *,
        filter: Optional[Filter] = None,
    ) -> int:
        """How many records a filter matches, or how many the table holds.

        The count is computed server-side from the same index that answers the filter: before
        this existed the only way to get it was to page every match and count them here, which
        put every matching record on the wire to produce one integer.

        Args:
            table_name: Name of the table.
            filter: Count only what this matches. Omitted, it counts the table.

        Returns:
            The number of matching records.

        Raises:
            DXMTError: If the request fails.
        """
        body: dict[str, Any] = {}
        converted = convert_simple_filter_to_dxmt(filter)
        if converted is not None:
            body["filter"] = converted
        try:
            response = self._client.post(
                f"/tables/{_seg(table_name)}/records/count", json=body
            )
            response.raise_for_status()
            return int(self._response_data(response.json())["count"])
        except httpx.HTTPStatusError as e:
            msg = f"Failed to count records: {e.response.text}"
            raise DXMTError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error while counting: {e}"
            raise DXMTConnectionError(msg) from e

    def facets(
        self,
        table_name: str,
        key: str,
        *,
        filter: Optional[Filter] = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """How many records hold each value of a payload field.

        Computed from the payload index, which is already a value-to-records map, so this
        does not page the table to count it.

        **Read `covered` against `matched` before drawing anything.** A facet built from an
        index describes what the index holds, and what it does not hold — a record with no
        value for the field, or one whose value is an object — is missing from every bucket
        rather than having a bucket of its own. Their difference is how many records this
        answer cannot describe, and `hits` alone will happily show a corpus that is not the
        corpus.

        Args:
            table_name: Name of the table.
            key: The payload field to count values of. A dotted path works, as in a filter.
            filter: Count only within what this matches. Omitted, it counts the table.
            limit: How many of the most common values to return; `distinct_values` in the
                result says how many there were.

        Returns:
            `{"hits": [{"value": ..., "count": n}, ...], "distinct_values": n,
              "covered": n, "matched": n}`.

        Raises:
            DXMTError: If the field is not indexed, or the request fails.
        """
        body: dict[str, Any] = {"key": key, "limit": limit}
        converted = convert_simple_filter_to_dxmt(filter)
        if converted is not None:
            body["filter"] = converted
        try:
            response = self._client.post(f"/tables/{_seg(table_name)}/facets", json=body)
            response.raise_for_status()
            return self._response_data(response.json())
        except httpx.HTTPStatusError as e:
            msg = f"Failed to facet '{key}': {e.response.text}"
            raise DXMTError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error while faceting: {e}"
            raise DXMTConnectionError(msg) from e

    def delete_records(
        self,
        table_name: str,
        ids: Optional[list[str]] = None,
        *,
        filter: Optional[Filter] = None,
    ) -> dict[str, Any]:
        """Delete records by id, or everything a filter matches.

        Exactly one of `ids` and `filter`. Deleting by filter is one request whatever it
        matches: before it existed, "remove everything belonging to tenant X" meant paging the
        table, collecting ids and sending them back, a round trip per page and at the mercy of
        anything written meanwhile.

        Args:
            table_name: Name of the table.
            ids: Record IDs to delete.
            filter: Delete every record this matches instead (a `Field` expression or a dict).

        Returns:
            Response data from server, including `deleted_count`.

        Raises:
            ValueError: If both or neither selector is given.
            DXMTError: If deletion fails.
        """
        converted = convert_simple_filter_to_dxmt(filter)
        if (ids is None) == (converted is None):
            raise ValueError("give either ids or filter, not both and not neither")
        payload = {"ids": ids} if ids is not None else {"filter": converted}
        try:
            response = self._client.request(
                "DELETE",
                f"/tables/{_seg(table_name)}/records",
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
        ids: Optional[list[str]] = None,
        *,
        filter: Optional[Filter] = None,
    ) -> dict[str, Any]:
        """Async version of :meth:`delete_records`; same one-selector rule."""
        converted = convert_simple_filter_to_dxmt(filter)
        if (ids is None) == (converted is None):
            raise ValueError("give either ids or filter, not both and not neither")
        payload = {"ids": ids} if ids is not None else {"filter": converted}
        try:
            response = await self.async_client.request(
                "DELETE",
                f"/tables/{_seg(table_name)}/records",
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

    def list_records(
        self,
        table_name: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List a table's records, without a query vector.

        The server has had `POST /tables/{t}/records/list` all along; this client did not
        expose it, and three call sites worked around the gap in ways that are worse than the
        round trip: `mem0.py` and `metagpt_vector_store.py` search with a zero vector and a
        large `k` to mean "everything", and `langgraph_store.py` logged a warning and returned
        nothing at all for a metadata-only search.

        `limit` is capped at 1,000 by the server. Page with `offset`.

        Args:
            table_name: Name of the table.
            limit: Records per page, at most 1,000.
            offset: Records to skip.

        Returns:
            The page, normalised the way search hits and single-record reads are.

        Raises:
            DXMTError: If the listing fails.
        """
        try:
            response = self._client.post(
                f"/tables/{_seg(table_name)}/records/list",
                json={"limit": limit, "offset": offset},
            )
            response.raise_for_status()
            data = self._response_data(response.json())
            records = data.get("records") or []
            return [self._normalize_record(r) for r in records]
        except httpx.HTTPStatusError as e:
            msg = f"Failed to list records in '{table_name}': {e.response.text}"
            raise DXMTRecordError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error while listing records: {e}"
            raise DXMTConnectionError(msg) from e

    async def alist_records(
        self,
        table_name: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Async twin of :meth:`list_records`."""
        try:
            response = await self.async_client.post(
                f"/tables/{_seg(table_name)}/records/list",
                json={"limit": limit, "offset": offset},
            )
            response.raise_for_status()
            data = self._response_data(response.json())
            records = data.get("records") or []
            return [self._normalize_record(r) for r in records]
        except httpx.HTTPStatusError as e:
            msg = f"Failed to list records in '{table_name}': {e.response.text}"
            raise DXMTRecordError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error while listing records: {e}"
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
        # Note: No batch get endpoint, so we fetch individually
        records = []
        for record_id in ids:
            try:
                response = self._client.get(
                    f"/tables/{_seg(table_name)}/records/{_seg(record_id)}"
                )
                response.raise_for_status()
                data = self._response_data(response.json())
                # Extract record from response envelope
                record = self._normalize_record(data)
                if record:
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
        # Note: No batch get endpoint, so we fetch individually
        records = []
        for record_id in ids:
            try:
                response = await self.async_client.get(
                    f"/tables/{_seg(table_name)}/records/{_seg(record_id)}"
                )
                response.raise_for_status()
                data = self._response_data(response.json())
                # Extract record from response envelope
                record = self._normalize_record(data)
                if record:
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

    # Documents: a long text stored as overlapping chunks, one record each

    @staticmethod
    def _document_body(text, metadata, chunk_size, chunk_overlap) -> dict[str, Any]:
        body: dict[str, Any] = {"text": text}
        if metadata is not None:
            body["metadata"] = metadata
        if chunk_size is not None:
            body["chunk_size"] = chunk_size
        if chunk_overlap is not None:
            body["chunk_overlap"] = chunk_overlap
        return body

    def put_document(
        self,
        table_name: str,
        doc_id: str,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> dict[str, Any]:
        """Store ``text`` as the document ``doc_id``, replacing any previous version.

        The server splits it into overlapping chunks (500 characters, 50 shared, by default)
        on paragraph and sentence boundaries and embeds each with the table's model, so the
        table needs one -- the ``kb`` table has the server's default. Each chunk is a record
        ``{doc_id}#{n}`` whose payload carries ``doc_id``, ``chunk_index``, ``chunk_count``
        and ``metadata``; search returns chunks.

        Returns:
            ``{"doc_id", "chunks", "removed"}``: chunks stored, and chunks of the previous
            version the new one no longer has.
        """
        try:
            response = self._client.put(
                f"/tables/{_seg(table_name)}/documents/{_seg(doc_id)}",
                json=self._document_body(text, metadata, chunk_size, chunk_overlap),
            )
            response.raise_for_status()
            return self._response_data(response.json())
        except httpx.HTTPStatusError as e:
            msg = f"Failed to store document {doc_id}: {e.response.text}"
            raise DXMTRecordError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error storing document {doc_id}: {e}"
            raise DXMTConnectionError(msg) from e

    async def aput_document(
        self,
        table_name: str,
        doc_id: str,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> dict[str, Any]:
        """Async version of :meth:`put_document`."""
        try:
            response = await self.async_client.put(
                f"/tables/{_seg(table_name)}/documents/{_seg(doc_id)}",
                json=self._document_body(text, metadata, chunk_size, chunk_overlap),
            )
            response.raise_for_status()
            return self._response_data(response.json())
        except httpx.HTTPStatusError as e:
            msg = f"Failed to store document {doc_id}: {e.response.text}"
            raise DXMTRecordError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error storing document {doc_id}: {e}"
            raise DXMTConnectionError(msg) from e

    def get_document(self, table_name: str, doc_id: str) -> Optional[dict[str, Any]]:
        """A document's chunks in order, with its metadata; ``None`` if there is none.

        Returns:
            ``{"doc_id", "metadata", "chunks": [{"id", "chunk_index", "text"}, ...]}``.
        """
        try:
            response = self._client.get(
                f"/tables/{_seg(table_name)}/documents/{_seg(doc_id)}"
            )
            # Both a missing table and a missing document are 404 NOT_FOUND; only the second
            # is an answer. The server says "no document '<id>'" for it.
            if response.status_code == 404 and "no document" in response.text:
                return None
            response.raise_for_status()
            return self._response_data(response.json())
        except httpx.HTTPStatusError as e:
            msg = f"Failed to get document {doc_id}: {e.response.text}"
            raise DXMTRecordError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error getting document {doc_id}: {e}"
            raise DXMTConnectionError(msg) from e

    async def aget_document(self, table_name: str, doc_id: str) -> Optional[dict[str, Any]]:
        """Async version of :meth:`get_document`."""
        try:
            response = await self.async_client.get(
                f"/tables/{_seg(table_name)}/documents/{_seg(doc_id)}"
            )
            # Both a missing table and a missing document are 404 NOT_FOUND; only the second
            # is an answer. The server says "no document '<id>'" for it.
            if response.status_code == 404 and "no document" in response.text:
                return None
            response.raise_for_status()
            return self._response_data(response.json())
        except httpx.HTTPStatusError as e:
            msg = f"Failed to get document {doc_id}: {e.response.text}"
            raise DXMTRecordError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error getting document {doc_id}: {e}"
            raise DXMTConnectionError(msg) from e

    def delete_document(self, table_name: str, doc_id: str) -> int:
        """Delete every chunk of ``doc_id``; returns how many there were (0 if none)."""
        try:
            response = self._client.delete(
                f"/tables/{_seg(table_name)}/documents/{_seg(doc_id)}"
            )
            response.raise_for_status()
            return int(self._response_data(response.json())["deleted_chunks"])
        except httpx.HTTPStatusError as e:
            msg = f"Failed to delete document {doc_id}: {e.response.text}"
            raise DXMTRecordError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error deleting document {doc_id}: {e}"
            raise DXMTConnectionError(msg) from e

    async def adelete_document(self, table_name: str, doc_id: str) -> int:
        """Async version of :meth:`delete_document`."""
        try:
            response = await self.async_client.delete(
                f"/tables/{_seg(table_name)}/documents/{_seg(doc_id)}"
            )
            response.raise_for_status()
            return int(self._response_data(response.json())["deleted_chunks"])
        except httpx.HTTPStatusError as e:
            msg = f"Failed to delete document {doc_id}: {e.response.text}"
            raise DXMTRecordError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error deleting document {doc_id}: {e}"
            raise DXMTConnectionError(msg) from e

    # Search Operations

    def search(
        self,
        table_name: str,
        vector: Optional[list[float]] = None,
        k: Optional[int] = None,
        limit: Optional[int] = None,
        filter: Optional[Filter] = None,
        search_mode: Optional[str] = None,
        score_threshold: Optional[float] = None,
        payload_fields: Optional[list[str]] = None,
        offset: int = 0,
        mmr: Optional[dict[str, Any]] = None,
        text: Optional[str] = None,
        rerank: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Vector similarity search.

        Args:
            table_name: Name of the table.
            vector: Query vector. Omit it when passing `text`.
            text: Query text, embedded by the table's own model -- which is the only model it
                may be embedded with, since a query vector from a different model than the
                corpus is not a worse answer but an unrelated one. Requires the table to have
                been created with `embedding=...`.
            k: Number of results to return (preferred parameter name).
            limit: Number of results to return (alias for k, for compatibility).
            filter: Optional metadata filter.
            search_mode: How hard to search -- "fast", "normal" (default), or "accurate",
                which scale the graph's ef_search by roughly 1x, 1x and 4x. The server has
                accepted this on every search since SearchMode existed; the SDK simply never
                passed it, so no caller could reach the recall/latency trade it offers.
                Omitted from the request when None, leaving the server's default.

            offset: Skip this many hits before the page starts -- page two of a search,
                without fetching page one again. Not free: the server asks the index for
                `offset + k` and drops the prefix, because an approximate index cannot start in
                the middle of a ranking it has not computed. To walk a whole table, `scroll`
                pages by cursor and stays flat.

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

        # `text` is embedded by the table's own model, server-side. The two are exclusive
        # and the server refuses a body carrying both, rather than quietly preferring one --
        # which is how a caller ends up measuring the wrong query for a week.
        if (vector is None) == (text is None):
            msg = "search takes either a vector or text, not both and not neither"
            raise ValueError(msg)
        payload: dict[str, Any] = {
            "top_k": k,
            "limit": k,
            "filter": convert_simple_filter_to_dxmt(filter),
        }
        if text is not None:
            payload["text"] = text
        else:
            payload["vector"] = vector
        if search_mode is not None:
            payload["search_mode"] = search_mode
        # A floor on the similarity the server returns (higher is nearer), not on a distance.
        if score_threshold is not None:
            payload["score_threshold"] = score_threshold
        # Only these payload keys come back. A payload is often mostly the text the embedding
        # came from, so a page of hits can be a megabyte of JSON to deliver two fields.
        if payload_fields is not None:
            payload["payload_fields"] = payload_fields
        if offset:
            # Page two of a search. Not free: the server asks the index for `offset + k` and
            # drops the prefix, because an approximate index cannot start in the middle of a
            # ranking it has not computed. To walk a whole table, `scroll` stays flat.
            payload["offset"] = offset
        if mmr is not None:
            # Diversity reranking, done by the server, which already holds the vectors:
            # `{"lambda_mult": 0.5, "fetch_k": 20}`. 1.0 is the plain ranking, 0.0 ignores the
            # query. Only candidates that were fetched can contribute diversity, so `fetch_k`
            # is what it costs.
            payload["mmr"] = mmr
        if rerank is not None:
            # Re-score the page with the table's rerank provider: `{}` for the defaults,
            # `{"top_n": 50, "field": "body"}` to size the pool or name the text field. Needs
            # `text`, and a table created with `rerank=...`; each hit gains `rerank_score`.
            payload["rerank"] = rerank
        try:
            response = self._client.post(
                f"/tables/{_seg(table_name)}/search",
                content=_encode_json(payload),
                headers=_JSON_HEADERS,
            )
            response.raise_for_status()
            return self._normalize_hits(response.json())
        except httpx.HTTPStatusError as e:
            msg = f"Search failed: {e.response.text}"
            raise DXMTSearchError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error during search: {e}"
            raise DXMTConnectionError(msg) from e

    async def asearch(
        self,
        table_name: str,
        vector: Optional[list[float]] = None,
        k: Optional[int] = None,
        limit: Optional[int] = None,
        filter: Optional[Filter] = None,
        search_mode: Optional[str] = None,
        score_threshold: Optional[float] = None,
        payload_fields: Optional[list[str]] = None,
        offset: int = 0,
        mmr: Optional[dict[str, Any]] = None,
        text: Optional[str] = None,
        rerank: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Async vector similarity search.

        Args:
            table_name: Name of the table.
            vector: Query vector. Omit it when passing `text`.
            text: Query text, embedded by the table's own model -- which is the only model it
                may be embedded with, since a query vector from a different model than the
                corpus is not a worse answer but an unrelated one. Requires the table to have
                been created with `embedding=...`.
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

        # `text` is embedded by the table's own model, server-side. The two are exclusive
        # and the server refuses a body carrying both, rather than quietly preferring one --
        # which is how a caller ends up measuring the wrong query for a week.
        if (vector is None) == (text is None):
            msg = "search takes either a vector or text, not both and not neither"
            raise ValueError(msg)
        payload: dict[str, Any] = {
            "top_k": k,
            "limit": k,
            "filter": convert_simple_filter_to_dxmt(filter),
        }
        if text is not None:
            payload["text"] = text
        else:
            payload["vector"] = vector
        if search_mode is not None:
            payload["search_mode"] = search_mode
        # A floor on the similarity the server returns (higher is nearer), not on a distance.
        if score_threshold is not None:
            payload["score_threshold"] = score_threshold
        # Only these payload keys come back. A payload is often mostly the text the embedding
        # came from, so a page of hits can be a megabyte of JSON to deliver two fields.
        if payload_fields is not None:
            payload["payload_fields"] = payload_fields
        if offset:
            payload["offset"] = offset
        if mmr is not None:
            payload["mmr"] = mmr
        if rerank is not None:
            # Re-score the page with the table's rerank provider: `{}` for the defaults,
            # `{"top_n": 50, "field": "body"}` to size the pool or name the text field. Needs
            # `text`, and a table created with `rerank=...`; each hit gains `rerank_score`.
            payload["rerank"] = rerank
        try:
            response = await self.async_client.post(
                f"/tables/{_seg(table_name)}/search",
                json=payload,
            )
            response.raise_for_status()
            return self._normalize_hits(response.json())
        except httpx.HTTPStatusError as e:
            msg = f"Search failed: {e.response.text}"
            raise DXMTSearchError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error during search: {e}"
            raise DXMTConnectionError(msg) from e

    @staticmethod
    def _fusion_body(vector_weight, alpha, fusion_method) -> dict[str, Any]:
        """The fusion half of a hybrid-search body.

        `dbsf` normalises each side by its own distribution and **ignores `vector_weight`**,
        and it was the server's default for every request until 2026-10-13 (now: `weighted_sum`
        at 0.6, or `dbsf` when there is a sparse vector). This SDK sent `vector_weight` and never
        a `fusion_method`, so every weight a
        caller passed, and the 0.7 default it invented for them, went to a method that does not
        read it: `vector_weight=0.0` and `vector_weight=1.0` returned the same rows with the
        same scores. Measured, not assumed.

        So a weight now implies a method that reads one (`weighted_sum`, min-max normalisation
        then a weighted sum, which is what "70% vector" means to a reader), and asking for no
        weight sends neither field and leaves the choice to the server's default.
        """
        if vector_weight is None:
            vector_weight = alpha
        body: dict[str, Any] = {}
        if fusion_method is not None:
            body["fusion_method"] = fusion_method
            if vector_weight is not None:
                body["vector_weight"] = vector_weight
        elif vector_weight is not None:
            body["vector_weight"] = vector_weight
            body["fusion_method"] = "weighted_sum"
        return body

    def hybrid_search(
        self,
        table_name: str,
        vector: Optional[list[float]] = None,
        text: Optional[str] = None,
        text_query: Optional[str] = None,  # Alias for text
        vector_weight: Optional[float] = None,
        alpha: Optional[float] = None,  # Alias for vector_weight
        fusion_method: Optional[str] = None,
        k: Optional[int] = None,
        limit: Optional[int] = None,  # Alias for k
        filter: Optional[Filter] = None,
        rerank: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search (vector + text).

        Args:
            table_name: Name of the table.
            vector: Query vector (optional if text provided).
            text: Query text (optional if vector provided).
            text_query: Alias for text parameter (for compatibility).
            vector_weight: Weight for vector similarity (0-1). Sending one selects
                `weighted_sum` fusion, which is the method that reads it; leaving it unset
                leaves the server's default (`weighted_sum` at 0.6).
            alpha: Alias for vector_weight (for compatibility).
            fusion_method: "weighted_sum" (the server's default), "dbsf" (its default with a
                sparse vector) or "reciprocal_rank_fusion".
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

        fusion = self._fusion_body(vector_weight, alpha, fusion_method)

        # Support limit as alias for k
        if k is None and limit is None:
            k = 4
        elif k is None:
            k = limit

        payload = {
            "vector": vector,
            "text": text,
            "limit": k,
            "filter": convert_simple_filter_to_dxmt(filter),
            **fusion,
        }
        if rerank is not None:
            # The fused page, re-scored by the table's rerank provider against `text`.
            payload["rerank"] = rerank
        try:
            response = self._client.post(
                f"/tables/{_seg(table_name)}/search/hybrid",
                json=payload,
            )
            response.raise_for_status()
            return self._normalize_hits(response.json())
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
        fusion_method: Optional[str] = None,
        k: Optional[int] = None,
        limit: Optional[int] = None,  # Alias for k
        filter: Optional[Filter] = None,
        rerank: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Async hybrid search.

        Args:
            table_name: Name of the table.
            vector: Query vector (optional if text provided).
            text: Query text (optional if vector provided).
            text_query: Alias for text parameter (for compatibility).
            vector_weight: Weight for vector similarity (0-1). Sending one selects
                `weighted_sum` fusion, which is the method that reads it; leaving it unset
                leaves the server's default (`weighted_sum` at 0.6).
            alpha: Alias for vector_weight (for compatibility).
            fusion_method: "weighted_sum" (the server's default), "dbsf" (its default with a
                sparse vector) or "reciprocal_rank_fusion".
            k: Number of results to return (preferred parameter name).
            limit: Number of results to return (alias for k).
            filter: Optional metadata filter.

        Returns:
            List of search results.
        """
        # Support text_query as alias for text
        if text is None and text_query is not None:
            text = text_query

        fusion = self._fusion_body(vector_weight, alpha, fusion_method)

        # Support limit as alias for k
        if k is None and limit is None:
            k = 4
        elif k is None:
            k = limit

        payload = {
            "vector": vector,
            "text": text,
            "limit": k,
            "filter": convert_simple_filter_to_dxmt(filter),
            **fusion,
        }
        if rerank is not None:
            # The fused page, re-scored by the table's rerank provider against `text`.
            payload["rerank"] = rerank
        try:
            response = await self.async_client.post(
                f"/tables/{_seg(table_name)}/search/hybrid",
                json=payload,
            )
            response.raise_for_status()
            return self._normalize_hits(response.json())
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
        filter: Optional[Filter] = None,
        rerank: Optional[dict[str, Any]] = None,
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
            # Converted, like every other search method. This was passed through raw, so a
            # filter written the documented way -- {"year": {"$gte": 2022}} -- worked on
            # `search` and `hybrid_search` and reached the server unconverted here.
            "filter": convert_simple_filter_to_dxmt(filter),
        }
        if rerank is not None:
            # The BM25 page, re-scored by the table's rerank provider against `query`.
            payload["rerank"] = rerank
        try:
            response = self._client.post(
                f"/tables/{_seg(table_name)}/search/text",
                json=payload,
            )
            response.raise_for_status()
            return self._normalize_hits(response.json())
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
            data = self._response_data(response.json())
            return data if isinstance(data, dict) else {}
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
            data = self._response_data(response.json())
            return data if isinstance(data, dict) else {}
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
        filter: Optional[Filter] = None,
        rerank: Optional[dict[str, Any]] = None,
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
            # Converted, like every other search method. This was passed through raw, so a
            # filter written the documented way -- {"year": {"$gte": 2022}} -- worked on
            # `search` and `hybrid_search` and reached the server unconverted here.
            "filter": convert_simple_filter_to_dxmt(filter),
        }
        if rerank is not None:
            # The BM25 page, re-scored by the table's rerank provider against `query`.
            payload["rerank"] = rerank
        try:
            response = await self.async_client.post(
                f"/tables/{_seg(table_name)}/search/text",
                json=payload,
            )
            response.raise_for_status()
            return self._normalize_hits(response.json())
        except httpx.HTTPStatusError as e:
            msg = f"Full-text search failed: {e.response.text}"
            raise DXMTSearchError(msg) from e
        except httpx.RequestError as e:
            msg = f"Connection error during full-text search: {e}"
            raise DXMTConnectionError(msg) from e
