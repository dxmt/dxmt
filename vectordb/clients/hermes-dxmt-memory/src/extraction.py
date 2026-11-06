"""LLM fact extraction.

Through the host-owned ``ctx.llm`` facade (``complete_structured``, task ``dxmt_extraction``)
when the plugin was given one, else through Hermes's auxiliary client directly -- the memory
activation path does not expose the facade to providers.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from .store import FACT_CATEGORIES

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM = """Extract durable memories about the user and their work from this conversation.
Return a JSON object with a "facts" array. Each fact has:
  content    - one concise durable fact, in the third person
  abstract   - optional one-sentence summary
  category   - preference, entity, event, case, pattern, or general
  tags       - short search tags
  evidence   - indexes of the messages that support the fact

Skip trivial, temporary, repeated, or purely session-local details.
Return {"facts": []} when there is nothing durable."""


def format_messages_with_indexes(messages: list[dict[str, Any]]) -> str:
    lines = []
    for idx, message in enumerate(messages):
        role = message.get("role", "")
        if role not in {"user", "assistant"}:
            continue
        content = message.get("content")
        if isinstance(content, list):  # multimodal parts
            content = " ".join(str(p.get("text") or "") for p in content if isinstance(p, dict))
        content = str(content or "").strip()
        if not content:
            continue
        lines.append(f"[{idx}] {role}: {content}")
    return "\n\n".join(lines)


TASK = "dxmt_extraction"


def extract(messages: list[dict[str, Any]], llm: Any = None) -> list[dict[str, Any]]:
    """Durable facts from ``messages``; ``[]`` when no LLM route is available."""
    if not messages:
        return []
    transcript = format_messages_with_indexes(messages)
    if llm is not None and hasattr(llm, "complete_structured"):
        try:
            result = llm.complete_structured(
                instructions=EXTRACTION_SYSTEM, input=[{"type": "text", "text": transcript}],
                json_mode=True, timeout=30, task=TASK, purpose="dxmt memory extraction",
            )
        except Exception as exc:
            logger.debug("dxmt extraction through ctx.llm failed: %s", exc)
            return []
        parsed = getattr(result, "parsed", None)
        return parse_facts(json.dumps(parsed) if parsed is not None else _response_text(result))
    try:
        from agent.auxiliary_client import call_llm
    except Exception as exc:
        logger.debug("auxiliary client unavailable for dxmt extraction: %s", exc)
        return []
    try:
        response = call_llm(
            task=TASK,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": transcript},
            ],
            timeout=30,
            extra_body={"response_format": {"type": "json_object"}},
        )
    except Exception as exc:
        logger.debug("dxmt extraction call failed: %s", exc)
        return []
    return parse_facts(_response_text(response))


def _response_text(response: Any) -> str:
    text = getattr(response, "content", None)
    if text is None:
        text = getattr(response, "text", None)
    if text is None:
        choices = getattr(response, "choices", None)
        if choices:
            text = getattr(getattr(choices[0], "message", None), "content", None)
    if text is None:
        text = response
    return text if isinstance(text, str) else str(text)


def parse_facts(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        payload = json.loads(text)
    except Exception as exc:
        logger.debug("dxmt extraction returned non-json: %s", exc)
        return []
    facts = payload.get("facts", []) if isinstance(payload, dict) else payload
    if not isinstance(facts, list):
        return []
    cleaned = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        content = str(fact.get("content") or "").strip()
        if not content:
            continue
        category = str(fact.get("category") or "general").strip()
        if category not in FACT_CATEGORIES:
            category = "general"
        evidence = fact.get("evidence") or []
        tags = fact.get("tags") or []
        cleaned.append(
            {
                "content": content,
                "abstract": str(fact.get("abstract") or "").strip(),
                "category": category,
                "tags": [str(t).strip() for t in tags if str(t).strip()] if isinstance(tags, list) else [],
                "evidence": [int(i) for i in evidence if isinstance(i, int) or str(i).isdigit()]
                if isinstance(evidence, list) else [],
            }
        )
    return cleaned
