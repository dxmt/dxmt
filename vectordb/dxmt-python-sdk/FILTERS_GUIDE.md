# DXMT Filter Guide

This guide explains how to use filters in DXMT Python SDK for metadata-based search filtering.

## Overview

DXMT supports flexible filtering on document metadata using both simple Python dictionaries and MongoDB-style query operators. The `convert_simple_filter_to_dxmt()` function automatically converts your filters to the internal DXMT format.

## Filter Formats

### 1. Simple Equality Filter

The simplest format for exact matches:

```python
# Match documents where category = "AI" AND status = "active"
filter = {
    "category": "AI",
    "status": "active"
}
```

### 2. MongoDB-Style Operators

Use MongoDB-style operators for advanced queries:

#### Comparison Operators

| Operator | DXMT Type | Description | Example |
|----------|-----------|-------------|---------|
| `$eq` | Equal | Exact match | `{"year": {"$eq": 2023}}` |
| `$ne` | NotEqual | Not equal | `{"status": {"$ne": "deleted"}}` |
| `$gt` | GreaterThan | Greater than | `{"price": {"$gt": 100}}` |
| `$gte` | GreaterThanOrEqual | Greater than or equal | `{"rating": {"$gte": 8.0}}` |
| `$lt` | LessThan | Less than | `{"age": {"$lt": 18}}` |
| `$lte` | LessThanOrEqual | Less than or equal | `{"year": {"$lte": 2023}}` |

#### Array Operators

| Operator | DXMT Type | Description | Example |
|----------|-----------|-------------|---------|
| `$in` | In | Value in list | `{"category": {"$in": ["AI", "ML", "DL"]}}` |
| `$nin` | NotIn | Value not in list | `{"status": {"$nin": ["deleted", "archived"]}}` |

#### String Operators

| Operator | DXMT Type | Description | Example |
|----------|-----------|-------------|---------|
| `$contains` | Contains | String contains substring | `{"title": {"$contains": "Space"}}` |
| `$startswith` | StartsWith | String starts with | `{"category": {"$startswith": "tech"}}` |
| `$endswith` | EndsWith | String ends with | `{"filename": {"$endswith": ".pdf"}}` |

#### Examples

```python
# Greater than or equal
filter = {"year": {"$gte": 2022}}

# Less than
filter = {"price": {"$lt": 100}}

# In list
filter = {"category": {"$in": ["AI", "ML", "DL"]}}

# String contains
filter = {"title": {"$contains": "machine learning"}}
```

### 3. Multiple Operators on Same Field

You can apply multiple operators to the same field (e.g., range queries):

```python
# year between 2020 and 2023 (inclusive)
filter = {
    "year": {
        "$gte": 2020,
        "$lte": 2023
    }
}

# price between $100 and $500
filter = {
    "price": {
        "$gte": 100,
        "$lte": 500
    }
}
```

### 4. Logical Operators

Combine multiple conditions using logical operators:

#### `$and` - All conditions must match

```python
filter = {
    "$and": [
        {"year": {"$gte": 2022}},
        {"rating": {"$gte": 8.0}},
        {"category": "sci-fi"}
    ]
}
```

#### `$or` - At least one condition must match

```python
filter = {
    "$or": [
        {"category": "AI"},
        {"category": "ML"},
        {"category": "DL"}
    ]
}
```

#### `$nor` - None of the conditions may match

```python
filter = {
    "$nor": [
        {"status": "deleted"},
        {"status": "archived"}
    ]
}
```

`$not` with a list means the same, and is still accepted.

#### What can be combined

The server takes a filter as three flat lists: every `must` holds, at least one `should`
holds, no `must_not` holds. So:

- `$and` of anything, `$or` of single conditions, `$nor` of single conditions, and one `$or`
  beside any number of other conditions all translate exactly;
- `(a and b) or c`, `(a or b) and (c or d)` and `not (a and b)` have no such form and raise
  `ValueError` rather than run as a different query;
- keys written beside `$and`/`$or` are ANDed with them, as in MongoDB.

#### Ranges compare numbers

`$gt`/`$gte`/`$lt`/`$lte` order numbers only; a string bound would match nothing, so it raises.
Store dates as a number, such as a Unix timestamp.

#### Arrays and missing fields

- A field holding an array matches when any element does: `{"tags": "ai"}` matches
  `["ai", "ml"]`. Comparing against a whole array (`{"tags": ["ai", "ml"]}`) is refused.
- `$ne` is true of a record without the field; `$nin` is not.
- Numbers compare as numbers: `5` matches `5.0`.

### 5. DXMT Native Format

Advanced users can use the DXMT native format directly (automatically detected and passed through):

```python
filter = {
    "must": [
        {"key": "category", "operation": {"type": "Equal", "value": "AI"}},
        {"key": "rating", "operation": {"type": "GreaterThan", "value": 8.0}}
    ],
    "must_not": [
        {"key": "status", "operation": {"type": "Equal", "value": "deleted"}}
    ],
    "should": [
        {"key": "tags", "operation": {"type": "Contains", "value": "featured"}}
    ]
}
```

## Real-World Examples

### E-Commerce Product Search

```python
# Find high-rated products in specific price range
filter = {
    "$and": [
        {"price": {"$gte": 100, "$lte": 500}},
        {"rating": {"$gte": 4.5}},
        {"in_stock": True},
        {"category": {"$in": ["electronics", "computers"]}}
    ]
}

results = client.search(
    table_name="products",
    vector=query_embedding,
    k=10,
    filter=filter
)
```

### Movie Recommendation

```python
# Find recent high-rated sci-fi movies
filter = {
    "$and": [
        {"year": {"$gte": 2020}},
        {"rating": {"$gte": 8.0}},
        {"genres": {"$contains": "sci-fi"}}
    ]
}

results = client.search(
    table_name="movies",
    vector=user_preference_embedding,
    k=5,
    filter=filter
)
```

### Document Search

```python
# Find recent PDF documents by specific authors
filter = {
    "$and": [
        {"file_type": "pdf"},
        {"created_ts": {"$gte": 1672531200}},  # 2023-01-01 as a Unix timestamp
        {"author": {"$in": ["John Doe", "Jane Smith"]}},
        {"status": {"$ne": "archived"}}
    ]
}

results = client.search(
    table_name="documents",
    vector=search_query_embedding,
    k=20,
    filter=filter
)
```

### User Profile Matching

```python
# Find users in specific age range and location
filter = {
    "$and": [
        {"age": {"$gte": 25, "$lte": 35}},
        {"city": {"$in": ["New York", "San Francisco", "Austin"]}},
        {"verified": True}
    ]
}

results = client.search(
    table_name="users",
    vector=profile_embedding,
    k=10,
    filter=filter
)
```

## Best Practices

1. **Use Specific Filters**: Narrow down results with filters before vector search to improve performance
2. **Index Filtered Fields**: Ensure frequently filtered fields are indexed for better performance
3. **Combine with Vector Search**: Use filters to constrain the search space, then let vector similarity rank results
4. **Test Filter Logic**: Validate your filters return expected results before deploying to production
5. **Profile Performance**: Monitor query performance and adjust filter complexity as needed

## Building Filters in Python

`Field` builds the same filters with operators, and is accepted wherever a filter is:

```python
from dxmt import Field

f = (
    (Field("category") == "sci-fi")
    & Field("year").between(2020, 2024)
    & ~Field("status").is_in(["deleted", "archived"])
)
results = client.search(table_name="movies", vector=v, k=5, filter=f)
```

`==` `!=` `>` `>=` `<` `<=`, `.is_in()`, `.not_in()`, `.between()`, `.contains()`,
`.starts_with()`, `.ends_with()`; combine with `&`, `|` and `~`. Python's `and`, `or`, `not`
and chained comparisons (`2020 <= Field("year") <= 2024`) raise `TypeError`, because each would
silently keep one side. `Field("id")` is the record id.

## Filter Conversion

The `convert_simple_filter_to_dxmt()` function handles all conversions automatically:

```python
from dxmt.filters import convert_simple_filter_to_dxmt

# Any of these formats work automatically
simple_filter = {"category": "AI", "year": 2023}
operator_filter = {"rating": {"$gte": 8.0}}
complex_filter = {
    "$and": [
        {"year": {"$gte": 2020}},
        {"category": {"$in": ["AI", "ML"]}}
    ]
}

# All are automatically converted when you call search()
results = client.search(
    table_name="documents",
    vector=query_vector,
    k=10,
    filter=complex_filter  # Automatic conversion
)
```

## Error Handling

Unsupported operators will raise a `ValueError`:

```python
try:
    filter = {"field": {"$invalid_operator": "value"}}
    results = client.search(table_name="test", vector=[1,2,3], filter=filter)
except ValueError as e:
    print(f"Filter error: {e}")
    # Output: Unsupported operator: $invalid_operator. Supported operators: ['$eq', '$ne', ...]
```

## API Reference

### Supported Operators

- **Comparison**: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`
- **Array**: `$in`, `$nin`
- **String**: `$contains`, `$startswith`, `$endswith`
- **Logical**: `$and`, `$or`, `$nor` (`$not` with a list is the same)

### Field Types

Filters work on any JSON-serializable field in your document's `payload`:
- **Numbers**: integers, floats
- **Strings**: text fields
- **Booleans**: true/false
- **Arrays**: a condition matches when any element does

### Notes

- Filters are applied BEFORE vector similarity ranking
- Filter operations are case-sensitive by default
- Nested field access is not currently supported
- All filter conditions within `$and` must match (AND logic)
- At least one condition within `$or` must match (OR logic)
- Use `must`, `must_not`, `should` keys for DXMT native format

## Testing

See `tests/test_filters.py` for comprehensive unit tests demonstrating all filter formats.

## See Also

- [Recommendation System Example](examples/recommendation_system.py) - Demonstrates filtered search in practice
- [Filtered Search Example](examples/filtered_search.py) - Advanced filtering techniques
- [API Documentation](https://dxmt.online/api/search) - Complete search API reference
