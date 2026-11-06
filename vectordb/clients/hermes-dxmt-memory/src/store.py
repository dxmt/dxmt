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

# Reads page the list endpoint when a memory is wanted by id (the server has no GET by id);
# this bounds that walk.
MAX_LIST_PAGES = 40
LIST_PAGE = 50


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
            if float(hit.get("score") or 0.0) < self.min_score:
                continue
            out.append(hit)
            if len(out) >= limit:
                break
        return out

    def get(self, memory_id: str, *, kind: str | None = None) -> dict[str, Any] | None:
        """One memory by id, or ``None``. Pages the list endpoint, newest first."""
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

    def get_many(self, ids: Iterable[str], *, kind: str | None = None) -> list[dict[str, Any]]:
        wanted = [i for i in ids if i]
        if not wanted:
            return []
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
