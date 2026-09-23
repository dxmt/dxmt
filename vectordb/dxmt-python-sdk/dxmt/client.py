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
        memory_optimization: Optional[bool] = None,
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
            "memory_optimization": memory_optimization,
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
        ids: list[str],
    ) -> dict[str, Any]:
        """Async delete records by IDs."""
        payload = {"ids": ids}
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

    # Search Operations

    def search(
        self,
        table_name: str,
        vector: list[float],
        k: Optional[int] = None,
        limit: Optional[int] = None,
        filter: Optional[Filter] = None,
        search_mode: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Vector similarity search.

        Args:
            table_name: Name of the table.
            vector: Query vector.
            k: Number of results to return (preferred parameter name).
            limit: Number of results to return (alias for k, for compatibility).
            filter: Optional metadata filter.
            search_mode: How hard to search -- "fast", "normal" (default), or "accurate",
                which scale the graph's ef_search by roughly 1x, 1x and 4x. The server has
                accepted this on every search since SearchMode existed; the SDK simply never
                passed it, so no caller could reach the recall/latency trade it offers.
                Omitted from the request when None, leaving the server's default.

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
            "top_k": k,
            "limit": k,
            "filter": convert_simple_filter_to_dxmt(filter),
        }
        if search_mode is not None:
            payload["search_mode"] = search_mode
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
        vector: list[float],
        k: Optional[int] = None,
        limit: Optional[int] = None,
        filter: Optional[Filter] = None,
        search_mode: Optional[str] = None,
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
            "top_k": k,
            "limit": k,
            "filter": convert_simple_filter_to_dxmt(filter),
        }
        if search_mode is not None:
            payload["search_mode"] = search_mode
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

        The server's default fusion is `dbsf`, which normalises each side by its own
        distribution and **ignores `vector_weight`** -- that is the point of it, no weight to
        tune. This SDK sent `vector_weight` and never a `fusion_method`, so every weight a
        caller passed, and the 0.7 default it invented for them, went to a method that does not
        read it: `vector_weight=0.0` and `vector_weight=1.0` returned the same rows with the
        same scores. Measured, not assumed.

        So a weight now implies a method that reads one (`weighted_sum`, min-max normalisation
        then a weighted sum, which is what "70% vector" means to a reader), and asking for no
        weight sends neither field and leaves the server on `dbsf`.
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
    ) -> list[dict[str, Any]]:
        """Hybrid search (vector + text).

        Args:
            table_name: Name of the table.
            vector: Query vector (optional if text provided).
            text: Query text (optional if vector provided).
            text_query: Alias for text parameter (for compatibility).
            vector_weight: Weight for vector similarity (0-1). Sending one selects
                `weighted_sum` fusion, which is the method that reads it; leaving it unset
                leaves the server on `dbsf`, which needs no weight.
            alpha: Alias for vector_weight (for compatibility).
            fusion_method: "dbsf" (default), "weighted_sum" or "reciprocal_rank_fusion".
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
    ) -> list[dict[str, Any]]:
        """Async hybrid search.

        Args:
            table_name: Name of the table.
            vector: Query vector (optional if text provided).
            text: Query text (optional if vector provided).
            text_query: Alias for text parameter (for compatibility).
            vector_weight: Weight for vector similarity (0-1). Sending one selects
                `weighted_sum` fusion, which is the method that reads it; leaving it unset
                leaves the server on `dbsf`, which needs no weight.
            alpha: Alias for vector_weight (for compatibility).
            fusion_method: "dbsf" (default), "weighted_sum" or "reciprocal_rank_fusion".
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
