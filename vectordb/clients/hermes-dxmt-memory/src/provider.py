"""The DXMT-backed Hermes ``MemoryProvider``.

Lifecycle, in the order Hermes calls it: ``initialize`` opens one client to the DXMT server
for this profile; ``queue_prefetch`` recalls in the background after each turn and
``prefetch`` hands the result to the next; ``sync_turn`` keeps the turn (kind ``turn``) off
the critical path through a writer thread; ``on_pre_compress`` and ``on_session_end`` run
LLM extraction and store the facts; ``on_memory_write`` mirrors the built-in memory tool;
``shutdown`` flushes the writer.

Scope: facts and turns carry ``hermes:ws:<workspace>`` / ``hermes:user:<id>`` /
``hermes:agent:<identity>`` tags, and recall filters on the workspace tag when
``retrieval.isolate_workspace`` is on. The server deduplicates ("saying what is already
remembered refreshes it") and keeps superseded memories as history, so none of that is
reimplemented here.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from agent.memory_provider import MemoryProvider

from .config import DEFAULTS, api_key, load_config, save_plugin_config, server_url
from .extraction import extract
from .retrieval import format_prefetch
from .store import (
    FACT_CATEGORIES,
    TURN_KIND,
    MemoryStore,
    client_from_url,
    scope_tags,
    stable_turn_id,
    workspace_tag,
)
from .tools import TOOL_SCHEMAS, DXMTToolDispatcher

logger = logging.getLogger(__name__)

PLUGIN_VERSION = "0.1.0"


class DXMTMemoryProvider(MemoryProvider):
    """Recall by meaning and keyword from a DXMT server; facts extracted at session boundaries."""

    def __init__(self) -> None:
        self._config: dict[str, Any] = load_config()
        self._session_id = ""
        self._hermes_home = ""
        self._platform = ""
        self._agent_context = "primary"
        self._agent_identity = ""
        self._agent_workspace = ""
        self._user_id = ""
        self._message_index = 0
        self._initialized = False
        self._store: MemoryStore | None = None
        self._tools = DXMTToolDispatcher(self)
        self._prefetch_result = ""
        self._prefetch_generation = 0
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: threading.Thread | None = None
        self.llm: Any = None  # the host-owned ctx.llm facade, when register() was given one

    # -- identity --------------------------------------------------------------

    @property
    def name(self) -> str:
        return "dxmt"

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    @property
    def store(self) -> MemoryStore:
        if self._store is None:
            server = self._config.get("server") or {}
            client = client_from_url(
                server_url(self._config),
                api_key=api_key(self._config),
                timeout=float(server.get("timeout") or 15),
            )
            self._store = MemoryStore(client, min_score=self._config.get("retrieval", {}).get("min_score", 0.0))
        return self._store

    def is_available(self) -> bool:
        """The SDK is importable. No network here, per the contract; a server that is down
        shows up as a tool error and an empty prefetch, never as a crash."""
        try:
            import dxmt  # noqa: F401
            import yaml  # noqa: F401
        except ImportError as exc:
            logger.debug("dxmt provider not available: %s", exc)
            return False
        return True

    def unavailable_reason(self) -> str:
        return "Install the client into Hermes's interpreter: uv pip install dxmt-python-sdk pyyaml"

    # -- lifecycle -------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._hermes_home = str(kwargs.get("hermes_home") or "")
        self._config = load_config(self._hermes_home or None)
        self._session_id = session_id
        self._platform = str(kwargs.get("platform") or "")
        self._agent_context = str(kwargs.get("agent_context") or "primary")
        self._agent_identity = str(kwargs.get("agent_identity") or "")
        self._agent_workspace = str(kwargs.get("agent_workspace") or "")
        self._user_id = str(kwargs.get("user_id") or "")
        self._message_index = 0
        self._store = None  # rebind to this profile's server
        self.store.writer.start()
        self._initialized = True
        logger.info(
            "dxmt provider initialized (session=%s, server=%s, workspace=%s)",
            session_id,
            server_url(self._config),
            self._agent_workspace or "-",
        )

    def system_prompt_block(self) -> str:
        return (
            "# DXMT Memory\n"
            "Active. Durable memory lives in a DXMT server shared by every agent on this machine. "
            "Recall it with dxmt_recall before answering anything that may depend on an earlier "
            "session. Use dxmt_remember when the user asks you to remember something or states a "
            "durable fact or preference; pass supersedes when it replaces an older memory. Use "
            "dxmt_read for the full text and provenance. For forgetting, preview with dxmt_forget "
            "first, then delete one exact id."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
            # A recall still running after the wait belongs to this turn, which has now passed;
            # its answer must not arrive as the next turn's context.
            self._prefetch_generation += 1
        return result

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not query or not self._initialized:
            return
        limit = int(self._config.get("retrieval", {}).get("prefetch_k", 5))
        with self._prefetch_lock:
            self._prefetch_generation += 1
            generation = self._prefetch_generation

        def _run() -> None:
            try:
                rows = self.recall(query, kind="fact", limit=limit)
                if not rows and self._config.get("turns", {}).get("enabled", True):
                    rows = self.recall(query, kind="turn", limit=min(limit, 3))
                formatted = format_prefetch(rows, max_items=limit)
                with self._prefetch_lock:
                    if generation == self._prefetch_generation:
                        self._prefetch_result = formatted
            except Exception as exc:
                logger.debug("dxmt prefetch failed: %s", exc)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="dxmt-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "", **kwargs: Any) -> None:
        if not self._should_write() or not self._config.get("turns", {}).get("enabled", True):
            return
        sid = session_id or self._session_id
        for role, content in (("user", user_content), ("assistant", assistant_content)):
            content = str(content or "").strip()
            index = self._message_index
            self._message_index += 1
            if not content:
                continue
            self.store.remember_later(
                text=content,
                kind=TURN_KIND,
                tags=self._tags(),
                source=sid,
                memory_id=stable_turn_id(sid, index, role, content),
                metadata={**self._context(sid), "role": role, "turn_index": index, "origin": "sync_turn"},
            )

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        return self._tools.handle(tool_name, args or {})

    def on_session_switch(
        self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False, **kwargs: Any
    ) -> None:
        self._session_id = new_session_id
        if reset:
            self._message_index = 0
        for key in ("agent_workspace", "agent_identity", "user_id"):
            attr = f"_{key}"
            if kwargs.get(key) is not None:
                setattr(self, attr, str(kwargs.get(key) or ""))

    def on_memory_write(self, action: str, target: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        if action != "add" or not content or not self._should_write():
            return
        category = "preference" if target == "user" else "general"
        try:
            self.remember_fact(content, category=category, origin="memory_write_mirror")
        except Exception as exc:
            logger.debug("dxmt memory_write mirror failed: %s", exc)

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        inserted = self._extract_and_store(messages, origin="pre_compress")
        if not inserted:
            return ""
        lines = ["DXMT kept these durable facts before compression:"]
        lines.extend(f"- {row['content']}" for row in inserted[:8])
        return "\n".join(lines)

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        self._extract_and_store(messages, origin="session_end")

    def shutdown(self) -> None:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2.0)
        if self._store is not None:
            self._store.close()
            self._store = None
        self._initialized = False
        logger.info("dxmt provider shutdown")

    # -- setup -----------------------------------------------------------------

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "url",
                "description": "DXMT server URL (the server embeds; no embedding key is needed here)",
                "default": server_url(self._config),
                "url": "https://github.com/dxmt/dxmt#readme",
            },
            {
                "key": "api_key",
                "description": "DXMT API key, if the server has one",
                "secret": True,
                "required": False,
                "env_var": "DXMT_API_KEY",
            },
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        cfg = dict(self._config)
        if values.get("url"):
            cfg.setdefault("server", {})["url"] = str(values["url"]).strip()
        save_plugin_config(cfg, hermes_home)
        self._config = load_config(hermes_home)

    def post_setup(self, hermes_home: str, config: dict[str, Any]) -> None:
        import sys

        if not self.is_available():
            print("\n  ⚠ The DXMT client is not importable in Hermes's interpreter.")
            print(f"  Run: uv pip install --python {sys.executable} dxmt-python-sdk pyyaml")
            print("  Then re-run: hermes memory setup\n")
            return
        try:
            from hermes_cli.config import save_config

            if not isinstance(config.get("memory"), dict):
                config["memory"] = {}
            config["memory"]["provider"] = "dxmt"
            save_config(config)
        except Exception as exc:  # pragma: no cover - outside Hermes
            logger.debug("could not write memory.provider: %s", exc)
        self._config = load_config(hermes_home)
        if "dxmt" not in (self._read_plugins_block(hermes_home) or {}):
            save_plugin_config(DEFAULTS, hermes_home)
        url = server_url(self._config)
        try:
            self._store = None
            health = self.store.health()
            status = health.get("status") or "ok"
            print(f"\n  ✓ DXMT memory configured ({url}: {status})")
        except Exception as exc:
            print(f"\n  ✓ DXMT memory configured ({url})")
            print(f"  ⚠ The server did not answer: {exc}")
            print("  Start it (`vectordb serve`, or docker run -p 127.0.0.1:9999:9999 dxmtai/dxmt) and chat again.")
        print("  Start a new session to activate.\n")

    @staticmethod
    def _read_plugins_block(hermes_home: str) -> dict[str, Any] | None:
        try:
            from pathlib import Path

            import yaml

            raw = yaml.safe_load((Path(hermes_home) / "config.yaml").read_text(encoding="utf-8-sig")) or {}
            return raw.get("plugins") if isinstance(raw, dict) else None
        except Exception:
            return None

    # -- operations the tools call -----------------------------------------------

    def recall(self, query: str, *, kind: str = "fact", category: str = "", limit: int = 10) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        kind = kind if kind in {"fact", "turn", "any"} else "fact"
        tags = None
        if self._agent_workspace and self._config.get("retrieval", {}).get("isolate_workspace", True):
            tags = [workspace_tag(self._agent_workspace)]
        if kind == "turn":
            return self.store.recall(query, limit=limit, kind=TURN_KIND, tags=tags)
        if kind == "any":
            return self.store.recall(query, limit=limit, tags=tags)
        if category in FACT_CATEGORIES:
            return self.store.recall(query, limit=limit, kind=category, tags=tags)
        return self.store.recall(query, limit=limit, tags=tags, exclude_kinds=(TURN_KIND,))

    def remember_fact(
        self,
        content: str,
        *,
        abstract: str = "",
        category: str = "general",
        tags: list[str] | None = None,
        provenance: list[str] | None = None,
        supersedes: list[str] | None = None,
        origin: str = "remember",
    ) -> dict[str, Any]:
        """Store one fact now (not through the writer) and return the server's answer."""
        if category not in FACT_CATEGORIES:
            category = "general"
        metadata = {**self._context(self._session_id), "origin": origin, "category": category}
        if abstract:
            metadata["abstract"] = abstract
        if provenance:
            metadata["provenance"] = provenance
        return self.store.remember(
            content,
            kind=category,
            tags=[*self._tags(), *(tags or [])],
            source=self._session_id,
            metadata=metadata,
            supersedes=supersedes,
        )

    # -- internals ---------------------------------------------------------------

    def _extract_and_store(self, messages: list[dict[str, Any]], *, origin: str) -> list[dict[str, Any]]:
        extraction = self._config.get("extraction", {}) or {}
        if not self._should_write() or not extraction.get("enabled", True):
            return []
        user_turns = sum(1 for m in messages if m.get("role") == "user")
        if user_turns < int(extraction.get("min_turns", 3)):
            return []
        inserted = []
        for fact in extract(messages, self.llm):
            try:
                result = self.remember_fact(
                    fact["content"],
                    abstract=fact.get("abstract", ""),
                    category=fact.get("category", "general"),
                    tags=fact.get("tags", []),
                    provenance=self._evidence_to_turn_ids(messages, fact.get("evidence") or []),
                    origin=origin,
                )
            except Exception as exc:
                logger.warning("dxmt could not store an extracted fact: %s", exc)
                continue
            if not result.get("duplicate"):
                inserted.append({"id": result.get("id"), "content": fact["content"]})
        return inserted

    def _evidence_to_turn_ids(self, messages: list[dict[str, Any]], evidence: list[int]) -> list[str]:
        """The ids ``sync_turn`` gave the turns a fact rests on.

        ``sync_turn`` keeps two messages a turn -- the user's, numbered ``2k``, and the
        assistant's final reply, ``2k + 1`` -- while ``evidence`` indexes this list, which also
        holds system and tool messages and the assistant's intermediate tool-calling steps.
        Using the list index named turns that were never kept. The k-th user message is
        ``2k``; the last assistant message before the next user message is ``2k + 1``; any
        other message was never kept and has no id. (A list that starts after an earlier
        compression restarts the count, and its ids can still miss.)
        """
        turn_number: dict[int, int] = {}
        k = -1
        last_assistant = None
        for i, message in enumerate(messages):
            role = str(message.get("role") or "")
            if role == "user":
                if last_assistant is not None:
                    turn_number[last_assistant] = 2 * k + 1
                k += 1
                turn_number[i] = 2 * k
                last_assistant = None
            elif role == "assistant" and k >= 0:
                last_assistant = i
        if last_assistant is not None:
            turn_number[last_assistant] = 2 * k + 1
        ids = []
        for idx in evidence:
            if idx not in turn_number:
                continue
            role = str(messages[idx].get("role") or "")
            content = str(messages[idx].get("content") or "").strip()
            if content:
                ids.append(stable_turn_id(self._session_id, turn_number[idx], role, content))
        return ids

    def _tags(self) -> list[str]:
        return scope_tags(workspace=self._agent_workspace, user_id=self._user_id, agent_identity=self._agent_identity)

    def _context(self, session_id: str) -> dict[str, Any]:
        out: dict[str, Any] = {"session_id": session_id}
        for key, value in (
            ("platform", self._platform),
            ("agent_identity", self._agent_identity),
            ("agent_workspace", self._agent_workspace),
            ("user_id", self._user_id),
        ):
            if value:
                out[key] = value
        return out

    def _should_write(self) -> bool:
        return self._agent_context not in {"cron", "subagent", "flush"}
