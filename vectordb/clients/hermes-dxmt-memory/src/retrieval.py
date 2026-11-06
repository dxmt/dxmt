"""Formatting recalled memories for the prompt."""
from __future__ import annotations

from typing import Any

from .store import TURN_KIND


def format_prefetch(rows: list[dict[str, Any]], *, max_items: int = 5) -> str:
    if not rows:
        return ""
    lines = ["## DXMT Memory"]
    for row in rows[:max_items]:
        content = row.get("abstract") or row.get("text") or ""
        content = " ".join(str(content).split())
        if len(content) > 500:
            content = content[:497] + "..."
        kind = row.get("kind") or "memory"
        label = "turn" if kind == TURN_KIND else kind
        lines.append(f"- ({label}, id={row.get('id')}) {content}")
    return "\n".join(lines)
