"""Tool schemas and dispatch for DXMT memory."""
from __future__ import annotations

import json
from typing import Any

try:
    from tools.registry import tool_error
except Exception:  # pragma: no cover - outside Hermes
    def tool_error(message: str, **extra: Any) -> str:
        return json.dumps({"error": message, **extra})

from .store import FACT_CATEGORIES, TURN_KIND

DXMT_RECALL = {
    "name": "dxmt_recall",
    "description": (
        "Recall durable memory from DXMT by meaning and keyword. Returns memory ids, text, "
        "category, score (1.0 = identical) and provenance turn ids. Use it before answering "
        "anything that may depend on what the user told you in an earlier session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search memory for."},
            "kind": {
                "type": "string",
                "enum": ["fact", "turn", "any"],
                "description": "fact (default): extracted facts; turn: raw conversation turns; any: both.",
            },
            "category": {"type": "string", "enum": list(FACT_CATEGORIES)},
            "limit": {"type": "integer"},
        },
        "required": ["query"],
    },
}

DXMT_REMEMBER = {
    "name": "dxmt_remember",
    "description": (
        "Store a durable fact the user would expect Hermes to remember. Saying what is already "
        "remembered refreshes it instead of adding a copy. Pass supersedes when the fact "
        "replaces older ones (a move, a changed preference): they are kept as history and left "
        "out of recall."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "abstract": {"type": "string", "description": "Optional one-sentence summary."},
            "category": {"type": "string", "enum": list(FACT_CATEGORIES)},
            "tags": {"type": "array", "items": {"type": "string"}},
            "supersedes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ids of memories this one replaces.",
            },
        },
        "required": ["content"],
    },
}

DXMT_READ = {
    "name": "dxmt_read",
    "description": "Read one memory in full by id, optionally with the conversation turns it came from.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "include_provenance": {"type": "boolean"},
        },
        "required": ["id"],
    },
}

DXMT_FORGET = {
    "name": "dxmt_forget",
    "description": (
        "Preview or delete a memory. When the user asks to forget something by description, "
        "preview candidates first and confirm; delete takes one exact id and is permanent."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["preview", "delete"]},
            "query": {"type": "string", "description": "Required for preview."},
            "id": {"type": "string", "description": "Required for delete."},
            "limit": {"type": "integer"},
        },
        "required": ["action"],
    },
}

TOOL_SCHEMAS = [DXMT_RECALL, DXMT_REMEMBER, DXMT_READ, DXMT_FORGET]


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _clean_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


class DXMTToolDispatcher:
    def __init__(self, provider) -> None:
        self.provider = provider

    def handle(self, tool_name: str, args: dict[str, Any]) -> str:
        try:
            if tool_name == "dxmt_recall":
                return self._recall(args)
            if tool_name == "dxmt_remember":
                return self._remember(args)
            if tool_name == "dxmt_read":
                return self._read(args)
            if tool_name == "dxmt_forget":
                return self._forget(args)
        except Exception as exc:  # the server being down is an answer, not a crash
            return tool_error(f"DXMT memory call failed: {exc}")
        return tool_error(f"Unknown DXMT memory tool: {tool_name}")

    def _recall(self, args: dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return tool_error("query is required")
        rows = self.provider.recall(
            query,
            kind=str(args.get("kind") or "fact"),
            category=str(args.get("category") or ""),
            limit=_limit(args.get("limit"), self.provider.config["retrieval"]["top_k"]),
        )
        return _json({"results": [format_result(row) for row in rows], "total": len(rows)})

    def _remember(self, args: dict[str, Any]) -> str:
        content = str(args.get("content") or "").strip()
        if not content:
            return tool_error("content is required")
        category = str(args.get("category") or "general").strip()
        if category not in FACT_CATEGORIES:
            category = "general"
        result = self.provider.remember_fact(
            content,
            abstract=str(args.get("abstract") or "").strip(),
            category=category,
            tags=_clean_list(args.get("tags")),
            supersedes=_clean_list(args.get("supersedes")),
            origin="remember",
        )
        payload = {
            "status": "exists" if result.get("duplicate") else "stored",
            "id": result.get("id"),
            "content": content,
        }
        if result.get("superseded"):
            payload["superseded"] = result["superseded"]
        return _json(payload)

    def _read(self, args: dict[str, Any]) -> str:
        memory_id = str(args.get("id") or "").strip()
        if not memory_id:
            return tool_error("id is required")
        row = self.provider.store.get(memory_id)
        if not row:
            return tool_error(f"memory not found: {memory_id}")
        payload: dict[str, Any] = {"memory": format_result(row, include_full=True)}
        if args.get("include_provenance"):
            ids = row.get("provenance") or []
            payload["provenance"] = [
                format_result(item, include_full=True)
                for item in self.provider.store.get_many(ids, kind=TURN_KIND)
            ]
        return _json(payload)

    def _forget(self, args: dict[str, Any]) -> str:
        action = str(args.get("action") or "").strip()
        if action == "preview":
            query = str(args.get("query") or "").strip()
            if not query:
                return tool_error("query is required for preview")
            rows = self.provider.recall(query, kind="fact", limit=_limit(args.get("limit"), 5))
            return _json({
                "action": "preview",
                "candidates": [format_result(row) for row in rows],
                "instruction": "Confirm the exact id with the user before deleting if there is any ambiguity.",
            })
        if action == "delete":
            memory_id = str(args.get("id") or "").strip()
            if not memory_id:
                return tool_error("id is required for delete")
            row = self.provider.store.get(memory_id)
            if not row:
                return tool_error(f"memory not found: {memory_id}")
            self.provider.store.forget(memory_id)
            return _json({"action": "delete", "deleted": format_result(row, include_full=True)})
        return tool_error("action must be preview or delete")


def _limit(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return max(1, min(parsed, 50))


def format_result(row: dict[str, Any], *, include_full: bool = False) -> dict[str, Any]:
    text = str(row.get("text") or "")
    snippet = " ".join(str(row.get("abstract") or text).split())
    if len(snippet) > 700 and not include_full:
        snippet = snippet[:697] + "..."
    kind = row.get("kind") or ""
    payload: dict[str, Any] = {
        "id": row.get("id"),
        "kind": "turn" if kind == TURN_KIND else "fact",
        "category": "" if kind == TURN_KIND else kind,
        "snippet": snippet,
        "tags": [t for t in (row.get("tags") or []) if not str(t).startswith("hermes")],
        "provenance_turn_ids": row.get("provenance") or [],
        "at": row.get("at"),
    }
    if "score" in row:
        payload["score"] = row["score"]
    if include_full:
        payload["content"] = text
        payload["session_id"] = row.get("session_id") or ""
        payload["role"] = row.get("role") or ""
        payload["origin"] = row.get("origin") or ""
    return payload
