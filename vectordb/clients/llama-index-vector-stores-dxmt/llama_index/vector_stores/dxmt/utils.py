"""Utility functions for DXMT vector store integration."""

import functools
import operator
from typing import Any, Dict, List, Optional

from dxmt.filters import Field, Filter
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.utils import (
    node_to_metadata_dict,
    metadata_dict_to_node,
)


def convert_llamaindex_filters_to_dxmt(
    filters: Optional[MetadataFilters],
) -> Dict[str, Any]:
    """Translate LlamaIndex ``MetadataFilters`` into the server's filter shape.

    The server takes one shape: flat ``must`` / ``should`` / ``must_not`` lists of conditions,
    meaning every ``must`` holds, at least one ``should`` holds, and no ``must_not`` holds. That
    is not every boolean formula, so this translation is built on the SDK's ``Field`` builder,
    which either produces that shape exactly or raises ``ValueError`` naming what it cannot
    express. It never approximates: the translation this replaced turned ``ALL`` into ``IN``
    (any value instead of every value), ``NOT`` into ``AND``, dropped ``IS_EMPTY`` on the floor
    and nested a ``{"must": [...]}`` group inside a ``must`` list, each a query that ran and
    returned the wrong records.

    Operators, with the semantics of ``llama_index.core.vector_stores.utils
    .build_metadata_filter_fn`` as the reference:

    * ``EQ`` ``NE`` ``GT`` ``GTE`` ``LT`` ``LTE`` ``IN`` ``NIN`` map one to one.
    * ``CONTAINS`` ("metadata array contains value") is ``Equal``: on the server a field holding
      an array matches when any element does.
    * ``ANY`` is ``In`` for the same reason; ``ALL`` is one ``Equal`` per value, all of which
      must hold, so it is only expressible under ``AND`` (or on its own).
    * ``TEXT_MATCH`` (substring of a string field) is ``Contains``. ``TEXT_MATCH_INSENSITIVE``
      has no server form and raises.
    * ``IS_EMPTY`` is ``IsEmpty``: the field is absent, ``null`` or an empty array.

    Conditions: ``AND`` is ``&``, ``OR`` is ``|``, and ``NOT`` -- which in LlamaIndex means
    "none of these filters match" -- is ``~(a | b | ...)``. Nested ``MetadataFilters`` are
    composed the same way, so ``AND`` of ``AND``, ``OR`` of ``OR`` and ``NOT`` inside ``AND``
    all work; ``(a AND b) OR c`` and ``(a OR b) AND (c OR d)`` raise.

    One deviation from the reference: ``NIN`` on a record that lacks the field. LlamaIndex's
    in-memory evaluator counts it a match; the server's ``NotIn`` does not (its ``NotEqual``
    does). A record without the field is the rare case; the common one, a present value that
    is not in the list, agrees.

    Returns ``{}`` for ``None`` or an empty filter set.
    """
    if filters is None or not filters.filters:
        return {}
    return _filter_of(filters).to_dict()


def _filter_of(filters: MetadataFilters) -> Filter:
    parts = []
    for f in filters.filters:
        if isinstance(f, MetadataFilters):
            if f.filters:
                parts.append(_filter_of(f))
        else:
            parts.append(_condition_of(f))
    if not parts:
        raise ValueError("a MetadataFilters group with nothing in it")
    condition = filters.condition or FilterCondition.AND
    if condition == FilterCondition.AND:
        return functools.reduce(operator.and_, parts)
    if condition == FilterCondition.OR:
        return functools.reduce(operator.or_, parts)
    if condition == FilterCondition.NOT:
        return ~functools.reduce(operator.or_, parts)
    raise ValueError(f"unknown FilterCondition {condition!r}")


def _condition_of(f: MetadataFilter) -> Filter:
    field = Field(f.key)
    op, value = f.operator, f.value
    if op == FilterOperator.EQ:
        return field == value
    if op == FilterOperator.NE:
        return field != value
    if op == FilterOperator.GT:
        return field > value
    if op == FilterOperator.GTE:
        return field >= value
    if op == FilterOperator.LT:
        return field < value
    if op == FilterOperator.LTE:
        return field <= value
    if op == FilterOperator.IN:
        return field.is_in(_as_list(value))
    if op == FilterOperator.NIN:
        return field.not_in(_as_list(value))
    if op == FilterOperator.CONTAINS:
        return field == value
    if op == FilterOperator.ANY:
        return field.is_in(_as_list(value))
    if op == FilterOperator.ALL:
        values = _as_list(value)
        if not values:
            raise ValueError(f"ALL on {f.key!r} with no values matches nothing meaningful")
        return functools.reduce(operator.and_, (field == v for v in values))
    if op == FilterOperator.TEXT_MATCH:
        return field.contains(str(value))
    if op == FilterOperator.IS_EMPTY:
        return field.is_empty()
    if op == FilterOperator.TEXT_MATCH_INSENSITIVE:
        raise ValueError(
            f"TEXT_MATCH_INSENSITIVE on {f.key!r}: the server matches substrings case-sensitively "
            "and has no case-insensitive form; use TEXT_MATCH, or a full-text query "
            "(VectorStoreQueryMode.TEXT_SEARCH) on an indexed text field"
        )
    raise ValueError(f"unsupported filter operator {op!r} on {f.key!r}")


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def node_to_dxmt_record(
    node: BaseNode,
    include_text: bool = True,
    flat_metadata: bool = False,
) -> Dict[str, Any]:
    """Convert LlamaIndex node to DXMT record format.

    Args:
        node: LlamaIndex BaseNode or subclass.
        include_text: Whether to include text content in payload.
        flat_metadata: Whether to flatten nested metadata.

    Returns:
        DXMT record dictionary with id, vector, and payload.

    Examples:
        >>> from llama_index.core.schema import TextNode
        >>> node = TextNode(
        ...     text="Example document",
        ...     id_="doc1",
        ...     embedding=[0.1, 0.2, 0.3],
        ...     metadata={"category": "example"}
        ... )
        >>> record = node_to_dxmt_record(node)
        >>> record['id']
        'doc1'
        >>> record['vector']
        [0.1, 0.2, 0.3]
        >>> record['payload']['category']
        'example'
    """
    # Get metadata dict from node
    metadata = node_to_metadata_dict(
        node,
        remove_text=not include_text,
        flat_metadata=flat_metadata
    )

    # Create payload with metadata
    payload = metadata.copy()

    # Store node content
    if hasattr(node, "get_content"):
        payload["_node_content"] = node.get_content()
    elif hasattr(node, "text"):
        payload["_node_content"] = node.text

    return {
        "id": node.node_id,
        "vector": node.get_embedding(),
        "payload": payload,
    }


def dxmt_hit_to_node(hit: Dict[str, Any]) -> BaseNode:
    """Convert DXMT search hit to LlamaIndex node.

    Args:
        hit: DXMT search result hit containing id, score, payload, etc.

    Returns:
        LlamaIndex BaseNode (typically TextNode).

    Examples:
        >>> hit = {
        ...     "id": "doc1",
        ...     "score": 0.95,
        ...     "payload": {
        ...         "_node_content": "Example text",
        ...         "category": "example"
        ...     },
        ...     "vector": [0.1, 0.2, 0.3]
        ... }
        >>> node = dxmt_hit_to_node(hit)
        >>> node.node_id
        'doc1'
        >>> node.text
        'Example text'
    """
    # Extract payload - handle both dict and JSON string formats
    payload = hit.get("payload", {})
    if isinstance(payload, str):
        import json
        payload = json.loads(payload)

    # Extract node content and type
    text = payload.get("_node_content", "")
    node_type = payload.get("_node_type", "TextNode")

    # Extract metadata (remove special fields)
    metadata = {}
    for key, value in payload.items():
        if not key.startswith("_"):
            # Convert string "None" back to None
            if value == "None":
                metadata[key] = None
            else:
                metadata[key] = value

    # Construct node based on type
    # Note: We construct the node manually instead of using metadata_dict_to_node
    # because we store the metadata as a flat dict, not a serialized node JSON
    if node_type == "TextNode":
        from llama_index.core.schema import TextNode
        node = TextNode(
            text=text,
            metadata=metadata,
        )
    else:
        # Fallback to base node type
        from llama_index.core.schema import BaseNode
        node = BaseNode(
            text=text,
            metadata=metadata,
        )

    # Set node ID from hit
    node.node_id = hit["id"]

    # Add embedding if present
    if "vector" in hit:
        node.embedding = hit["vector"]

    return node


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """Chunk a list into smaller lists of specified size.

    Args:
        items: List to chunk.
        chunk_size: Size of each chunk.

    Returns:
        List of chunks.

    Examples:
        >>> chunk_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def determine_optimal_batch_size(num_records: int, default_batch_size: int = 100) -> int:
    """Determine optimal batch size based on number of records.

    Args:
        num_records: Total number of records to process.
        default_batch_size: Default batch size.

    Returns:
        Optimal batch size.

    Examples:
        >>> determine_optimal_batch_size(50)
        50
        >>> determine_optimal_batch_size(500)
        100
        >>> determine_optimal_batch_size(5000)
        500
        >>> determine_optimal_batch_size(50000)
        1000
    """
    if num_records < default_batch_size:
        return num_records
    elif num_records < 1000:
        return default_batch_size
    elif num_records < 10000:
        return 500
    else:
        return 1000
