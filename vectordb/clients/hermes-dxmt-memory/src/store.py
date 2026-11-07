"""The DXMT side: one client, scope tags, a background writer, and the row shapes.

Everything the plugin writes is a DXMT *memory* (``POST /api/v1/memories``): facts under
their category as ``kind``, conversation turns under kind ``turn``. Scope is tags --
``hermes:ws:<workspace>``, ``hermes:user:<id>``, ``hermes:agent:<identity>`` -- and recall
filters on the workspace tag when ``retrieval.isolate_workspace`` is on. Provenance, the
abstract and the session live in ``metadata``, which the server returns flattened on every
hit, so nothing here needs a second lookup.

The server embeds the text (its table or server-level model); this module never sees a
vector, and the plugin needs no embedding key.
"""
from __future__ import annotations

import hashlib
import logging
import queue
import threading
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

TURN_KIND = "turn"
FACT_CATEGORIES = ("preference", "entity", "event", "case", "pattern", "general")
WRITER_TAG = "hermes"

# A server older than `GET /memories/{id}` (1.7) is read by paging the list endpoint; this
# bounds that walk.
MAX_LIST_PAGES = 40
LIST_PAGE = 50


def closeness(hit: dict[str, Any]) -> float:
    """How close a recalled memory is to the query, for thresholds: its cosine ``similarity``
    (servers from 1.7), else the fused ``score`` -- a rank, the best hit near 1.0 however poor,
    which is all an older server reports."""
    value = hit.get("similarity", hit.get("score"))
    return float(value or 0.0)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def stable_turn_id(session_id: str, index: int, role: str, content: str) -> str:
    digest = hashlib.sha256(f"{session_id}|{index}|{role}|{content}".encode()).hexdigest()[:24]
    return f"turn-{digest}"


def scope_tags(*, workspace: str = "", user_id: str = "", agent_identity: str = "") -> list[str]:
    tags = [WRITER_TAG]
    if workspace:
        tags.append(f"hermes:ws:{workspace}")
    if user_id:
        tags.append(f"hermes:user:{user_id}")
    if agent_identity:
        tags.append(f"hermes:agent:{agent_identity}")
    return tags


def workspace_tag(workspace: str) -> str:
    return f"hermes:ws:{workspace}"


def client_from_url(url: str, *, api_key: str | None, timeout: float):
    """A ``dxmt.DXMTClient`` for ``http[s]://host[:port]``."""
    from dxmt import DXMTClient

    parsed = urlparse(url if "://" in url else f"http://{url}")
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"DXMT server URL must look like http://host:port, got {url!r}")
    ssl = parsed.scheme == "https"
    port = parsed.port or (443 if ssl else 9999)
    return DXMTClient(host=parsed.hostname, port=port, ssl=ssl, api_key=api_key, timeout=timeout)


class Writer:
    """Runs remembers on one background thread so ``sync_turn`` never blocks a turn."""

    def __init__(self) -> None:
        self._queue: queue.Queue[Callable[[], None] | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._run, daemon=True, name="dxmt-memory-writer")
            self._thread.start()

    def enqueue(self, job: Callable[[], None]) -> None:
        self.start()
        self._queue.put(job)

    def flush(self, timeout: float = 30.0) -> None:
        """Wait until every queued write has run (tests, shutdown)."""
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
        if not running:
            # Nothing was ever queued, or the writer has stopped: no thread would run the marker,
            # and shutdown used to wait out the whole timeout for it.
            return
        done = threading.Event()
        self._queue.put(done.set)
        done.wait(timeout)

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread and thread.is_alive():
            self._queue.put(None)
            thread.join(timeout)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                job()
            except Exception as exc:  # a failed write must not kill the writer
                logger.warning("dxmt memory write failed: %s", exc)


class MemoryStore:
    """What the provider and the tools call. Thin over the SDK; every method is synchronous."""

    def __init__(self, client, *, min_score: float = 0.0) -> None:
        self.client = client
        self.min_score = float(min_score or 0.0)
        self.writer = Writer()

    # -- writes --------------------------------------------------------------

    def remember(
        self,
        text: str,
        *,
        kind: str,
        tags: Iterable[str],
        source: str = "",
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
        supersedes: list[str] | None = None,
    ) -> dict[str, Any]:
        """``{"id", "superseded", "duplicate"?}`` from the server."""
        return self.client.remember(
            text,
            kind=kind,
            tags=list(dict.fromkeys(tags)),
            source=source or None,
            supersedes=supersedes or None,
            memory_id=memory_id,
            metadata=metadata or None,
        )

    def remember_later(self, **kwargs: Any) -> None:
        self.writer.enqueue(lambda: self.remember(**kwargs))

    def forget(self, memory_id: str) -> int:
        return int(self.client.forget(memory_id))

    # -- reads ---------------------------------------------------------------

    def recall(
        self,
        query: str,
        *,
        limit: int,
        kind: str | None = None,
        tags: list[str] | None = None,
        exclude_kinds: Iterable[str] = (),
        category: str = "",
    ) -> list[dict[str, Any]]:
        """Memories matching ``query``, best first; ``exclude_kinds`` and ``category`` are
        applied here, over a wider fetch, because the server filters on one kind at most."""
        excluded = set(exclude_kinds)
        fetch = limit if not excluded and not category else min(max(limit * 3, 20), 100)
        hits = self.client.recall(query, limit=fetch, kind=kind, tags=tags or None)
        out = []
        for hit in hits:
            if hit.get("kind") in excluded:
                continue
            if category and hit.get("kind") != category:
                continue
            if closeness(hit) < self.min_score:
                continue
            out.append(hit)
            if len(out) >= limit:
                break
        if len(out) < limit and excluded and not category and kind is None:
            # Turns outnumber facts many times over, so the widened fetch can be all turns and
            # leave no fact at all. Ask for each fact category by name, which the server filters.
            out = self._fill_by_category(query, out, limit=limit, tags=tags, excluded=excluded)
        return out

    def _fill_by_category(self, query, found, *, limit, tags, excluded):
        seen = {hit.get("id") for hit in found}
        extra = []
        for category in FACT_CATEGORIES:
            if category in excluded:
                continue
            for hit in self.client.recall(query, limit=limit, kind=category, tags=tags or None):
                if hit.get("id") not in seen and closeness(hit) >= self.min_score:
                    seen.add(hit.get("id"))
                    extra.append(hit)
        # Across separate queries only a similarity compares; without one (a server before
        # 1.7), each category's own order is kept and the categories are interleaved.
        if all("similarity" in hit for hit in extra):
            extra.sort(key=lambda hit: -float(hit["similarity"]))
        return (found + extra)[:limit]

    def get(self, memory_id: str, *, kind: str | None = None) -> dict[str, Any] | None:
        """One memory by id, or ``None``."""
        direct = self._get_by_id(memory_id)
        if direct != "unsupported":
            if direct is not None and kind is not None and direct.get("kind") != kind:
                return None
            return direct
        offset = 0
        for _ in range(MAX_LIST_PAGES):
            page = self.client.list_memories(kind=kind, limit=LIST_PAGE, offset=offset)
            memories = page.get("memories") or []
            for memory in memories:
                if memory.get("id") == memory_id:
                    return memory
            offset += len(memories)
            if not memories or offset >= int(page.get("total") or 0):
                return None
        return None

    def _get_by_id(self, memory_id: str):
        """``GET /memories/{id}``: the memory, ``None`` when there is none, or "unsupported" from
        a server older than the route (it answers 405), where the list has to be paged."""
        http = getattr(self.client, "_client", None)
        if http is None:
            return "unsupported"
        from urllib.parse import quote

        response = http.get(f"/memories/{quote(memory_id, safe='')}")
        if response.status_code == 404:
            return None
        if response.status_code == 405:
            return "unsupported"
        response.raise_for_status()
        return (response.json() or {}).get("data")

    def get_many(self, ids: Iterable[str], *, kind: str | None = None) -> list[dict[str, Any]]:
        wanted = [i for i in ids if i]
        if not wanted:
            return []
        if len(wanted) <= 20 and self._get_by_id(wanted[0]) != "unsupported":
            rows = [self.get(i, kind=kind) for i in wanted]
            return [row for row in rows if row is not None]
        found: dict[str, dict[str, Any]] = {}
        offset = 0
        for _ in range(MAX_LIST_PAGES):
            page = self.client.list_memories(kind=kind, limit=LIST_PAGE, offset=offset)
            memories = page.get("memories") or []
            for memory in memories:
                if memory.get("id") in wanted:
                    found[memory["id"]] = memory
            if len(found) == len(wanted):
                break
            offset += len(memories)
            if not memories or offset >= int(page.get("total") or 0):
                break
        return [found[i] for i in wanted if i in found]

    def health(self) -> dict[str, Any]:
        return self.client.health_check()

    def close(self) -> None:
        self.writer.flush(timeout=10.0)
        self.writer.stop()
        try:
            self.client.close()
        except Exception:
            pass
