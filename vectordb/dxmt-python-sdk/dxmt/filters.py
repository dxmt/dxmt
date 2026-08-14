"""Filter utilities for DXMT Python SDK.

Converts simple dict filters to DXMT's Filter format.
Supports MongoDB-style operators ($and, $or, $gte, $lt, etc.).
"""

from typing import Any, Dict, List, Optional, Union


# MongoDB operator to DXMT operation type mapping
OPERATOR_MAP = {
    "$eq": "Equal",
    "$ne": "NotEqual",
    "$gt": "GreaterThan",
    "$gte": "GreaterThanOrEqual",
    "$lt": "LessThan",
    "$lte": "LessThanOrEqual",
    "$in": "In",
    "$nin": "NotIn",
    "$contains": "Contains",
    "$startswith": "StartsWith",
    "$endswith": "EndsWith",
}


def _map_operator(mongo_op: str) -> str:
    """Map MongoDB operator to DXMT operation type.

    Args:
        mongo_op: MongoDB operator like $gte, $lt, etc.

    Returns:
        DXMT operation type like GreaterThanOrEqual, LessThan, etc.

    Raises:
        ValueError: If operator is not supported.
    """
    if mongo_op not in OPERATOR_MAP:
        raise ValueError(
            f"Unsupported operator: {mongo_op}. "
            f"Supported operators: {list(OPERATOR_MAP.keys())}"
        )
    return OPERATOR_MAP[mongo_op]


def _parse_condition(cond_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse a condition dict to DXMT condition format.

    Supports:
        - Simple equality: {"year": 2022}
        - Operator form: {"year": {"$gte": 2022}}
        - Multiple operators: {"year": {"$gte": 2020, "$lte": 2023}}

    Args:
        cond_dict: Condition dictionary.

    Returns:
        List of DXMT condition dicts.
    """
    conditions = []

    for key, value in cond_dict.items():
        if isinstance(value, dict):
            # Handle operators like {"year": {"$gte": 2022}}
            for op, val in value.items():
                if op.startswith("$"):
                    op_type = _map_operator(op)
                    conditions.append({
                        "key": key,
                        "operation": {"type": op_type, "value": val}
                    })
                else:
                    # Nested dict without operator, treat as equality
                    conditions.append({
                        "key": key,
                        "operation": {"type": "Equal", "value": value}
                    })
                    break
        else:
            # Simple equality: {"year": 2022}
            conditions.append({
                "key": key,
                "operation": {"type": "Equal", "value": value}
            })

    return conditions


def convert_simple_filter_to_dxmt(
    filter_dict: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Convert simple filter dict to DXMT Filter format.

    Supported formats:

    1. Simple equality:
        {"category": "AI", "score": 0.8}

    2. MongoDB-style operators:
        {"year": {"$gte": 2022}, "rating": {"$lt": 9.0}}

    3. MongoDB logical operators:
        {
            "$and": [
                {"year": {"$gte": 2022}},
                {"rating": {"$gte": 8.0}}
            ]
        }

        {
            "$or": [
                {"category": "AI"},
                {"category": "ML"}
            ]
        }

    4. DXMT native format (passed through):
        {
            "must": [
                {"key": "category", "operation": {"type": "Equal", "value": "AI"}},
                {"key": "score", "operation": {"type": "GreaterThan", "value": 0.8}}
            ]
        }

    Args:
        filter_dict: Filter dictionary in any supported format.

    Returns:
        DXMT Filter format dictionary, or None if input is None.

    Raises:
        ValueError: If filter format is invalid or uses unsupported operators.
    """
    if filter_dict is None or not filter_dict:
        return None

    # Check if already in DXMT format
    if any(key in filter_dict for key in ["must", "must_not", "should"]):
        return filter_dict

    # Handle MongoDB logical operators
    if "$and" in filter_dict:
        must_conditions = []
        for cond in filter_dict["$and"]:
            must_conditions.extend(_parse_condition(cond))
        return {"must": must_conditions}

    if "$or" in filter_dict:
        should_conditions = []
        for cond in filter_dict["$or"]:
            should_conditions.extend(_parse_condition(cond))
        return {"should": should_conditions}

    if "$not" in filter_dict:
        must_not_conditions = []
        for cond in filter_dict["$not"]:
            must_not_conditions.extend(_parse_condition(cond))
        return {"must_not": must_not_conditions}

    # Convert simple dict or operator-based dict to DXMT format
    must_conditions = _parse_condition(filter_dict)
    return {"must": must_conditions} if must_conditions else None
