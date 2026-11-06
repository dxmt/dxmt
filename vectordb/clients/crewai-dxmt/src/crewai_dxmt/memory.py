"""DXMT as a ``StorageBackend`` for CrewAI's unified memory.

CrewAI's ``Memory`` does the thinking -- an LLM infers scope, categories and importance on
``remember``, and ``recall`` blends semantic similarity with recency and importance -- and
hands the storage a ``MemoryRecord`` with its embedding already computed. This backend keeps
one record as one DXMT row: the embedding is the vector; ``content``, ``scope``, ``categories``,
``importance``, ``created_at``, ``last_accessed``, ``source`` and ``private`` are payload
fields; the record's free-form ``metadata`` sits under ``meta`` so a metadata filter on ``k``
is the payload path ``meta.k``.

Every selection the protocol asks for is a server filter, so nothing is scanned here:

* a scope prefix ``/a`` is ``scope == "/a" OR scope starts with "/a/"`` -- ``/ab`` is not
  under ``/a``, which a plain prefix match would say it was;
* ``categories`` is ``In`` on the array field, which the server reads as "any element";
* ``metadata_filter`` is one ``Equal`` per key under ``meta.``;
* ``older_than`` is ``created_at < t`` on the epoch-seconds field.

The scores ``search`` returns are the server's similarity (1.0 is identical under cosine),
which is what ``compute_composite_score`` expects as the semantic term.

Because the server is a separate process, several ``Memory`` instances -- agent memory and
crew memory, or several crews on one machine -- share the store with no file lock, which is
the case the LanceDB default serialises with a lock and retries.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from crewai.memory.storage.backend import EmbeddingDimensionMismatchError
from crewai.memory.types import MemoryRecord, ScopeInfo
from dxmt import DXMTClient
from dxmt.exceptions import DXMTError

logger = logging.getLogger(__name__)

META = "meta"


def _epoch(t: datetime) -> float:
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.timestamp()


def _from_epoch(x: Any, default: Optional[datetime] = None) -> datetime:
    if isinstance(x, (int, float)):
        return datetime.fromtimestamp(float(x), tz=timezone.utc).replace(tzinfo=None)
    return default or datetime.utcnow()


def _norm_scope(scope: Optional[str]) -> Optional[str]:
    """``/a/b/`` -> ``/a/b``; ``a`` -> ``/a``; ``/`` and ``""`` -> ``None`` (everything)."""
    if scope is None:
        return None
    s = scope.strip()
    s = s.rstrip("/")
    if not s:
        return None
    return s if s.startswith("/") else "/" + s


def _deleted(out: Any) -> int:
    """The count from a delete response, which the SDK hands back in the server's envelope."""
    if isinstance(out, dict):
        data = out.get("data") if isinstance(out.get("data"), dict) else out
        return int(data.get("deleted_count", data.get("deleted", 0)) or 0)
    return 0


def _cond(key: str, type_: str, value: Any = ...) -> dict[str, Any]:
    op = {"type": type_} if value is ... else {"type": type_, "value": value}
    return {"key": key, "operation": op}


def _filter(
    scope_prefix: Optional[str] = None,
    categories: Optional[list[str]] = None,
    metadata_filter: Optional[dict[str, Any]] = None,
    older_than: Optional[datetime] = None,
    record_ids: Optional[list[str]] = None,
) -> Optional[dict[str, Any]]:
    must: list[dict[str, Any]] = []
    should: list[dict[str, Any]] = []
    prefix = _norm_scope(scope_prefix)
    if prefix is not None:
        should = [_cond("scope", "Equal", prefix), _cond("scope", "StartsWith", prefix + "/")]
    if categories:
        must.append(_cond("categories", "In", list(categories)))
    if metadata_filter:
        for k, v in metadata_filter.items():
            must.append(_cond(f"{META}.{k}", "Equal", v))
    if older_than is not None:
        must.append(_cond("created_at", "LessThan", _epoch(older_than)))
    if record_ids:
        must.append(_cond("id", "In", list(record_ids)))
    if not must and not should:
        return None
    out: dict[str, Any] = {}
    if must:
        out["must"] = must
    if should:
        out["should"] = should
    return out


class DXMTStorageBackend:
    """A CrewAI ``StorageBackend`` over one DXMT table.

    Args:
        table_name: The table; created on the first ``save`` with the embedding's width.
        host, port, ssl, api_key: Where the server is; ignored when ``client`` is given.
        client: A ``DXMTClient`` to use instead of building one.
        dimension: Vector width, if the table should be created before the first save.
        distance_metric: ``cosine`` (default), ``euclidean`` or ``dot``.
    """

    def __init__(
        self,
        table_name: str = "crewai_memory",
        *,
        host: str = "localhost",
        port: int = 9999,
        ssl: bool = False,
        api_key: Optional[str] = None,
        client: Optional[DXMTClient] = None,
        dimension: Optional[int] = None,
        distance_metric: str = "cosine",
    ) -> None:
        self.table_name = table_name
        self.distance_metric = distance_metric
        self._client = client or DXMTClient(host=host, port=port, ssl=ssl, api_key=api_key)
        self._dimension: Optional[int] = dimension
        self._exists: Optional[bool] = None

    # ---- table ---------------------------------------------------------------------------

    @property
    def client(self) -> DXMTClient:
        return self._client

    def _table_exists(self) -> bool:
        if self._exists:
            return True
        try:
            info = self._client.get_table(self.table_name)
            self._dimension = self._dimension or info.get("dimension")
            self._exists = True
            return True
        except DXMTError as e:
            if "TABLE_NOT_FOUND" in str(e):
                self._exists = False
                return False
            raise

    def _ensure_table(self, dim: int) -> None:
        if self._table_exists():
            if self._dimension and self._dimension != dim:
                raise EmbeddingDimensionMismatchError(self._dimension, dim)
            return
        self._client.create_table(name=self.table_name, dimension=dim, distance_metric=self.distance_metric)
        self._dimension = dim
        self._exists = True

    # ---- conversion ----------------------------------------------------------------------

    @staticmethod
    def _payload(record: MemoryRecord) -> dict[str, Any]:
        return {
            "content": record.content,
            "scope": _norm_scope(record.scope) or "/",
            "categories": list(record.categories),
            META: dict(record.metadata),
            "importance": float(record.importance),
            "created_at": _epoch(record.created_at),
            "last_accessed": _epoch(record.last_accessed),
            "source": record.source,
            "private": bool(record.private),
        }

    @staticmethod
    def _record(row: dict[str, Any]) -> MemoryRecord:
        p = row.get("payload") or row.get("metadata") or {}
        return MemoryRecord(
            id=row["id"],
            content=p.get("content", ""),
            scope=p.get("scope", "/"),
            categories=list(p.get("categories") or []),
            metadata=dict(p.get(META) or {}),
            importance=float(p.get("importance", 0.5)),
            created_at=_from_epoch(p.get("created_at")),
            last_accessed=_from_epoch(p.get("last_accessed")),
            embedding=row.get("vector"),
            source=p.get("source"),
            private=bool(p.get("private", False)),
        )

    def _rows(self, filter: Optional[dict[str, Any]], with_vector: bool = False) -> list[dict[str, Any]]:
        if not self._table_exists():
            return []
        return list(self._client.scroll(self.table_name, filter=filter, with_vector=with_vector, batch_size=1000))

    # ---- protocol: writes ----------------------------------------------------------------

    def save(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        rows = []
        for r in records:
            if r.embedding is None:
                raise ValueError(f"memory record {r.id} has no embedding; Memory embeds before save")
            self._ensure_table(len(r.embedding))
            rows.append({"id": r.id, "vector": list(r.embedding), "payload": self._payload(r)})
        self._client.insert_records(self.table_name, rows)

    def update(self, record: MemoryRecord) -> None:
        if not self._table_exists():
            return
        vector = record.embedding
        if vector is None:
            existing = self._client.get_records(self.table_name, [record.id])
            if not existing:
                return
            vector = existing[0].get("vector")
        self._client.update_record(
            self.table_name, record.id, vector=list(vector) if vector else None,
            payload=self._payload(record), merge_mode="replace",
        )

    def touch_records(self, record_ids: list[str]) -> None:
        """Bump ``last_accessed`` on recalled records; a payload merge, no vector sent."""
        if not record_ids or not self._table_exists():
            return
        now = _epoch(datetime.utcnow())
        for rid in record_ids:
            try:
                self._client.update_record(self.table_name, rid, payload={"last_accessed": now}, merge_mode="merge")
            except DXMTError as e:
                if "NOT_FOUND" not in str(e):
                    raise

    def delete(
        self,
        scope_prefix: Optional[str] = None,
        categories: Optional[list[str]] = None,
        record_ids: Optional[list[str]] = None,
        older_than: Optional[datetime] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> int:
        if not self._table_exists():
            return 0
        f = _filter(scope_prefix, categories, metadata_filter, older_than, record_ids)
        if f is None:
            # Delete everything: the whole table's rows, the table stays.
            n = self._client.count(self.table_name)
            if n:
                self._client.delete_records(self.table_name, filter={"must": [_cond("id", "NotEqual", "")]})
            return n
        return _deleted(self._client.delete_records(self.table_name, filter=f))

    def reset(self, scope_prefix: Optional[str] = None) -> None:
        if not self._table_exists():
            return
        if _norm_scope(scope_prefix) is None:
            self._client.delete_table(self.table_name)
            self._exists = False
            return
        self.delete(scope_prefix=scope_prefix)

    # ---- protocol: reads -----------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        scope_prefix: Optional[str] = None,
        categories: Optional[list[str]] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        if not self._table_exists() or limit <= 0:
            return []
        if self._dimension and len(query_embedding) != self._dimension:
            raise EmbeddingDimensionMismatchError(self._dimension, len(query_embedding))
        hits = self._client.search(
            self.table_name, vector=list(query_embedding), k=limit,
            filter=_filter(scope_prefix, categories, metadata_filter),
        )
        out = []
        for h in hits:
            score = float(h.get("score", 0.0))
            if score >= min_score:
                out.append((self._record(h), score))
        return out

    def get_record(self, record_id: str) -> Optional[MemoryRecord]:
        if not self._table_exists():
            return None
        try:
            rows = self._client.get_records(self.table_name, [record_id])
        except DXMTError as e:
            if "NOT_FOUND" in str(e):
                return None
            raise
        return self._record(rows[0]) if rows else None

    def list_records(self, scope_prefix: Optional[str] = None, limit: int = 200, offset: int = 0) -> list[MemoryRecord]:
        if limit <= 0 or offset < 0:
            return []
        rows = self._rows(_filter(scope_prefix))
        rows.sort(key=lambda r: float((r.get("payload") or {}).get("created_at", 0.0)), reverse=True)
        return [self._record(r) for r in rows[offset : offset + limit]]

    def get_scope_info(self, scope: str) -> ScopeInfo:
        path = _norm_scope(scope) or "/"
        rows = self._rows(_filter(path if path != "/" else None))
        cats: set[str] = set()
        oldest = newest = None
        children: set[str] = set()
        prefix = "" if path == "/" else path
        for r in rows:
            p = r.get("payload") or {}
            cats.update(p.get("categories") or [])
            t = p.get("created_at")
            if isinstance(t, (int, float)):
                oldest = t if oldest is None else min(oldest, t)
                newest = t if newest is None else max(newest, t)
            sc = p.get("scope", "/")
            if sc != path and sc.startswith(prefix + "/"):
                first = sc[len(prefix) + 1 :].split("/", 1)[0]
                if first:
                    children.add(prefix + "/" + first)
        return ScopeInfo(
            path=path,
            record_count=len(rows),
            categories=sorted(cats),
            oldest_record=_from_epoch(oldest) if oldest is not None else None,
            newest_record=_from_epoch(newest) if newest is not None else None,
            child_scopes=sorted(children),
        )

    def list_scopes(self, parent: str = "/") -> list[str]:
        return self.get_scope_info(parent).child_scopes

    def list_categories(self, scope_prefix: Optional[str] = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self._rows(_filter(scope_prefix)):
            for c in (r.get("payload") or {}).get("categories") or []:
                counts[c] = counts.get(c, 0) + 1
        return counts

    def count(self, scope_prefix: Optional[str] = None) -> int:
        if not self._table_exists():
            return 0
        return int(self._client.count(self.table_name, filter=_filter(scope_prefix)))

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    # ---- async: the client is synchronous HTTP, so these run it off the event loop -------

    async def asave(self, records: list[MemoryRecord]) -> None:
        await asyncio.to_thread(self.save, records)

    async def asearch(
        self,
        query_embedding: list[float],
        scope_prefix: Optional[str] = None,
        categories: Optional[list[str]] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        return await asyncio.to_thread(
            self.search, query_embedding, scope_prefix, categories, metadata_filter, limit, min_score
        )

    async def adelete(
        self,
        scope_prefix: Optional[str] = None,
        categories: Optional[list[str]] = None,
        record_ids: Optional[list[str]] = None,
        older_than: Optional[datetime] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> int:
        return await asyncio.to_thread(self.delete, scope_prefix, categories, record_ids, older_than, metadata_filter)
