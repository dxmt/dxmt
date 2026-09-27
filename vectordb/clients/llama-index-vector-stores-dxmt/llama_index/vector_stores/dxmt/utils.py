"""Utility functions for DXMT vector store integration."""

from typing import Dict, List, Any, Optional
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator, FilterCondition
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.utils import (
    node_to_metadata_dict,
    metadata_dict_to_node,
)


def convert_llamaindex_filters_to_dxmt(
    filters: MetadataFilters
) -> Dict[str, Any]:
    """Convert LlamaIndex MetadataFilters to DXMT filter format.

    LlamaIndex uses a structured filter format with operators like EQ, NE, GT, etc.
    DXMT uses a nested dictionary format with operators like "==", "!=", ">", etc.

    This function bridges the two formats, similar to how the LangChain integration
    converts MongoDB-style filters.

    Args:
        filters: LlamaIndex MetadataFilters object.

    Returns:
        DXMT filter dictionary.

    Examples:
        Single filter:
        >>> from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator
        >>> filters = MetadataFilters(filters=[
        ...     MetadataFilter(key="category", value="AI", operator=FilterOperator.EQ)
        ... ])
        >>> convert_llamaindex_filters_to_dxmt(filters)
        {'must': [{'key': 'category', 'operation': {'type': 'Equal', 'value': 'AI'}}]}

        Multiple filters with AND:
        >>> filters = MetadataFilters(filters=[
        ...     MetadataFilter(key="year", value=2022, operator=FilterOperator.GTE),
        ...     MetadataFilter(key="rating", value=8.0, operator=FilterOperator.GT),
        ... ], condition=FilterCondition.AND)
        >>> result = convert_llamaindex_filters_to_dxmt(filters)
        >>> result['must']  # Returns list of filter conditions
        [...]
    """
    if not filters.filters:
        return {}

    # Convert individual filters
    dxmt_filters = []

    for f in filters.filters:
        if isinstance(f, MetadataFilters):
            # Nested filters - recursively convert
            nested = convert_llamaindex_filters_to_dxmt(f)
            if nested:
                dxmt_filters.append(nested)
        elif isinstance(f, MetadataFilter):
            # Single filter - convert to DXMT format
            dxmt_filter = _convert_single_filter(f)
            if dxmt_filter:
                dxmt_filters.append(dxmt_filter)

    # If no filters, return empty dict
    if not dxmt_filters:
        return {}

    # Always wrap filters in must/should, even for single filter
    # This ensures compatibility with DXMT SDK's filter processing
    # LlamaIndex uses FilterCondition.AND or FilterCondition.OR
    # DXMT uses "must" (AND) or "should" (OR) at the root level
    condition = filters.condition if hasattr(filters, 'condition') else FilterCondition.AND

    if condition == FilterCondition.AND:
        return {"must": dxmt_filters}
    elif condition == FilterCondition.OR:
        return {"should": dxmt_filters}
    else:
        # Default to AND
        return {"must": dxmt_filters}


def _convert_single_filter(filter: MetadataFilter) -> Optional[Dict[str, Any]]:
    """Convert a single MetadataFilter to DXMT format.

    Args:
        filter: Single MetadataFilter.

    Returns:
        DXMT filter dictionary or None if unsupported.
    """
    key = filter.key
    value = filter.value
    operator = filter.operator

    # Map LlamaIndex operators to DXMT operation types
    if operator == FilterOperator.EQ:
        return {
            "key": key,
            "operation": {
                "type": "Equal",
                "value": value
            }
        }
    elif operator == FilterOperator.NE:
        return {
            "key": key,
            "operation": {
                "type": "NotEqual",
                "value": value
            }
        }
    elif operator == FilterOperator.GT:
        return {
            "key": key,
            "operation": {
                "type": "GreaterThan",
                "value": value
            }
        }
    elif operator == FilterOperator.GTE:
        return {
            "key": key,
            "operation": {
                "type": "GreaterThanOrEqual",
                "value": value
            }
        }
    elif operator == FilterOperator.LT:
        return {
            "key": key,
            "operation": {
                "type": "LessThan",
                "value": value
            }
        }
    elif operator == FilterOperator.LTE:
        return {
            "key": key,
            "operation": {
                "type": "LessThanOrEqual",
                "value": value
            }
        }
    elif operator == FilterOperator.IN:
        return {
            "key": key,
            "operation": {
                "type": "In",
                "value": value if isinstance(value, list) else [value]
            }
        }
    elif operator == FilterOperator.NIN:
        return {
            "key": key,
            "operation": {
                "type": "NotIn",
                "value": value if isinstance(value, list) else [value]
            }
        }
    elif operator == FilterOperator.CONTAINS:
        return {
            "key": key,
            "operation": {
                "type": "Contains",
                "value": value
            }
        }
    elif operator == FilterOperator.TEXT_MATCH:
        # Text match - use Contains as approximation
        return {
            "key": key,
            "operation": {
                "type": "Contains",
                "value": value
            }
        }
    elif operator == FilterOperator.ANY:
        # ANY - at least one value matches
        # For DXMT, use In operator
        return {
            "key": key,
            "operation": {
                "type": "In",
                "value": value if isinstance(value, list) else [value]
            }
        }
    elif operator == FilterOperator.ALL:
        # ALL - all values must match
        # This is more complex - for now, use In as approximation
        return {
            "key": key,
            "operation": {
                "type": "In",
                "value": value if isinstance(value, list) else [value]
            }
        }
    else:
        # Unsupported operator - return None
        return None


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
