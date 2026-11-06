"""Agent Framework portable filters to the server's filter, or a refusal.

The server takes one shape, three flat lists of conditions -- every ``must`` holds, at least
one ``should`` holds, no ``must_not`` holds -- so not every filter tree has a form there. This
translation is built on the SDK's ``Field`` builder, which produces that shape exactly or raises
``ValueError`` naming the formula it cannot flatten, and that is re-raised here as
``NotImplementedError``, which is what the framework's own connectors raise for a portable
filter they cannot honour. Nothing is approximated.

Operators (the framework's ``FilterOperator``), with the server's semantics -- a field holding an
array matches when any element does, which is what makes the ``contains*`` family natural:

==================  ==========================================================
``eq`` ``ne``        ``Equal`` / ``NotEqual``
``gt`` ``gte``       numeric comparisons; ``between`` is ``gte`` and ``lte``
``lt`` ``lte``
``in`` ``not_in``    ``In`` / ``NotIn``
``is_null``          ``IsNull``; ``is_not_null`` and ``exists`` are ``not IsEmpty``
``contains``         ``Equal`` on the array field; ``contains_any`` is ``In``;
                     ``contains_all`` is one ``Equal`` per value, all required
``starts_with``      ``StartsWith``; ``ends_with`` is ``EndsWith``
``contains_text``    ``Contains`` (a substring test, not tokenised search)
==================  ==========================================================

Field names are the collection definition's logical names and are mapped to storage names;
the key field is the record id (server key ``id``).
"""

from __future__ import annotations

import functools
import operator as op
from typing import Any

from agent_framework import Filter as AFFilter
from agent_framework import FilterGroup, VectorStoreCollectionDefinition
from dxmt.filters import Field, Filter


def translate(expression: Any, definition: VectorStoreCollectionDefinition) -> dict[str, Any] | None:
    """The server filter for ``expression``, or ``None`` for no filter."""
    if expression is None:
        return None
    try:
        return _node(expression, definition).to_dict()
    except ValueError as exc:
        raise NotImplementedError(f"DXMT has no flat form for this filter: {exc}") from exc


def _storage_name(name: str, definition: VectorStoreCollectionDefinition) -> str:
    field = definition.try_get_field(name)
    if field is None or "." in name:
        raise NotImplementedError("DXMT filters support declared top-level logical fields only.")
    if field.field_type == "vector":
        raise NotImplementedError("Portable DXMT filters do not operate on vector fields.")
    if field.field_type == "key":
        return "id"
    return field.storage_name or field.name


def _leaf(f: AFFilter, definition: VectorStoreCollectionDefinition) -> Filter:
    field = Field(_storage_name(f.field_name, definition))
    o, v = f.operator, f.value
    if o == "eq":
        return field == v
    if o == "ne":
        return field != v
    if o == "gt":
        return field > v
    if o == "gte":
        return field >= v
    if o == "lt":
        return field < v
    if o == "lte":
        return field <= v
    if o == "between":
        return (field >= v[0]) & (field <= v[1])
    if o == "in":
        return field.is_in(list(v))
    if o == "not_in":
        return field.not_in(list(v))
    if o == "is_null":
        return field.is_null()
    if o in ("is_not_null", "exists"):
        return ~field.is_empty()
    if o == "contains":
        return field == v
    if o == "contains_any":
        return field.is_in(list(v))
    if o == "contains_all":
        values = list(v)
        if not values:
            raise NotImplementedError("contains_all with no values has no meaning on the server.")
        return functools.reduce(op.and_, (field == item for item in values))
    if o == "starts_with":
        return field.starts_with(str(v))
    if o == "ends_with":
        return field.ends_with(str(v))
    if o == "contains_text":
        return field.contains(str(v))
    raise NotImplementedError(f"DXMT does not support portable filter operator '{o}'.")


def _node(expression: Any, definition: VectorStoreCollectionDefinition) -> Filter:
    if isinstance(expression, FilterGroup):
        parts = [_node(child, definition) for child in expression.filters]
        if not parts:
            raise NotImplementedError("An empty filter group has no server form.")
        if expression.operator == "and":
            return functools.reduce(op.and_, parts)
        if expression.operator == "or":
            return functools.reduce(op.or_, parts)
        # "not" takes exactly one child.
        return ~parts[0]
    if isinstance(expression, AFFilter):
        return _leaf(expression, definition)
    raise TypeError(f"Unexpected filter node {type(expression).__name__}.")
