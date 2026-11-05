"""Haystack filters to the server's filter shape, or a verdict that it has none.

A Haystack filter is a tree: comparison leaves ``{"field", "operator", "value"}`` under
``AND`` / ``OR`` / ``NOT`` nodes with ``conditions``. The server takes one shape, three flat
lists -- every ``must`` holds, at least one ``should`` holds, no ``must_not`` holds -- so not
every tree has a server form, and the ones that do not must not be approximated: a filter that
runs and returns the wrong documents is worse than one that is refused.

``translate`` therefore answers in one of two ways. A ``Filter`` (the SDK's builder, which
produces the flat shape exactly or raises) when the tree has a server form; ``None`` when it
does not. ``filter_documents`` then evaluates such a tree itself, with Haystack's own
``document_matches_filter`` over a scroll of the table, which is always correct and only slower;
the retrievers, where a ranked page cannot be post-filtered without changing its meaning,
refuse it by name.

What has a server form, with ``haystack.utils.filters`` as the reference:

* ``==`` and ``!=`` against a string, number or boolean. A missing field is ``!=`` anything and
  ``==`` nothing, on both sides.
* ``>`` ``>=`` ``<`` ``<=`` against a number. Against a string or an ISO date the reference
  parses dates and refuses other strings; the server compares numbers only, so those go to the
  evaluator.
* ``in`` against a list of scalars. ``not in`` too, as ``NotIn OR IsEmpty``: the reference
  counts a missing field as "not in", the server's ``NotIn`` does not, and the disjunction is
  what makes them agree -- when it is the only disjunction in the filter.
* ``AND``, ``OR`` of leaves, and ``NOT`` of a single leaf (Haystack's ``NOT`` is "not all of
  these", which for one condition is its negation).
* ``== None`` / ``!= None`` go to the evaluator: the reference's ``None`` is "missing or null"
  and ``[]`` is not ``None``, which no single server test says.

Field names: ``meta.x`` is the payload path ``meta.x`` (the store keeps a document's metadata
under ``meta``), ``content`` is ``content``, ``id`` is the record id.
"""

from __future__ import annotations

import datetime
from typing import Any, Optional

from dxmt.filters import Field, Filter
from haystack.errors import FilterError

_COMPARISONS = {"==", "!=", ">", ">=", "<", "<=", "in", "not in"}
_LOGIC = {"AND", "OR", "NOT"}


def validate(filters: dict[str, Any]) -> None:
    """Raise ``FilterError`` for a malformed tree, as the reference evaluator would.

    Done up front and separately from translation so that a malformed filter is refused the
    same way whether or not the well-formed version of it would have had a server form.
    """
    if "field" in filters:
        if "operator" not in filters:
            raise FilterError(f"'operator' key missing in {filters}")
        if "value" not in filters:
            raise FilterError(f"'value' key missing in {filters}")
        if filters["operator"] not in _COMPARISONS:
            raise FilterError(
                f"Unknown comparison operator '{filters['operator']}'. "
                f"Valid operators are: {sorted(_COMPARISONS)}"
            )
        if filters["operator"] in ("in", "not in") and not isinstance(filters["value"], list):
            raise FilterError(
                "Filter value must be a `list` when using operator 'in' or 'not in', "
                f"received type '{type(filters['value'])}'"
            )
        return
    if "operator" not in filters:
        raise FilterError(f"'operator' key missing in {filters}")
    if "conditions" not in filters:
        raise FilterError(f"'conditions' key missing in {filters}")
    if filters["operator"] not in _LOGIC:
        raise FilterError(
            f"Unknown logical operator '{filters['operator']}'. Valid operators are: {sorted(_LOGIC)}"
        )
    for condition in filters["conditions"]:
        validate(condition)


def translate(filters: Optional[dict[str, Any]]) -> Optional[Filter]:
    """The server filter meaning the same as ``filters``, or ``None`` when there is none.

    ``{}`` and ``None`` translate to ``None`` too: callers treat "no filter" and "no server
    form" the same way for an unfiltered request, and distinguish them with ``bool(filters)``.
    """
    if not filters:
        return None
    validate(filters)
    try:
        return _node(filters)
    except _NoServerForm:
        return None
    except ValueError:
        # The SDK builder refusing a composition it cannot express flat.
        return None


class _NoServerForm(Exception):
    pass


def _field(name: str) -> Field:
    if name == "id":
        return Field("id")
    if name == "content" or name.startswith("meta.") or name == "meta":
        return Field(name)
    # Haystack's reference treats a bare key as `meta.<key>` for converted legacy filters.
    return Field(f"meta.{name}")


def _scalar(value: Any) -> bool:
    return isinstance(value, (str, bool)) or (
        isinstance(value, (int, float)) and not isinstance(value, bool)
    )


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _looks_like_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        datetime.datetime.fromisoformat(value)
        return True
    except ValueError:
        return False


def _leaf(condition: dict[str, Any]) -> Filter:
    field, op, value = _field(condition["field"]), condition["operator"], condition["value"]
    if op == "==":
        if not _scalar(value) or _looks_like_date(value):
            raise _NoServerForm
        return field == value
    if op == "!=":
        if not _scalar(value) or _looks_like_date(value):
            raise _NoServerForm
        return field != value
    if op in (">", ">=", "<", "<="):
        if not _number(value):
            raise _NoServerForm
        return {">": field.__gt__, ">=": field.__ge__, "<": field.__lt__, "<=": field.__le__}[op](value)
    if op == "in":
        if not all(_scalar(v) and not _looks_like_date(v) for v in value):
            raise _NoServerForm
        return field.is_in(list(value))
    if op == "not in":
        if not all(_scalar(v) and not _looks_like_date(v) for v in value):
            raise _NoServerForm
        return field.not_in(list(value)) | field.is_empty()
    raise _NoServerForm


def _node(node: dict[str, Any]) -> Filter:
    if "field" in node:
        return _leaf(node)
    parts = [_node(c) for c in node["conditions"]]
    if not parts:
        raise _NoServerForm
    op = node["operator"]
    if op == "AND":
        out = parts[0]
        for p in parts[1:]:
            out = out & p
        return out
    if op == "OR":
        out = parts[0]
        for p in parts[1:]:
            out = out | p
        return out
    # NOT: "not all of these hold". One condition negates cleanly; several do not flatten.
    if len(parts) != 1:
        raise _NoServerForm
    return ~parts[0]
