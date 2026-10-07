# PRE-FILTERING Usage Example

## Overview

DXMT VectorDB now supports intelligent PRE/POST-FILTERING with enhanced cardinality estimation.

## Filter Format

Filters support three logical operators:

- `must`: AND condition (all must match)
- `must_not`: NOT condition (none can match)
- `should`: OR condition (at least one must match)

## Supported Operations

1. **Equal**: Exact match
2. **NotEqual**: Negation
3. **GreaterThan / GreaterThanOrEqual**: Numeric comparison
4. **LessThan / LessThanOrEqual**: Numeric comparison
5. **In**: Multi-value match
6. **NotIn**: Multi-value negation
7. **Contains**: Substring match
8. **StartsWith**: Prefix match
9. **EndsWith**: Suffix match

## API Example

```bash
# Search with range filter
curl -X POST http://localhost:9999/api/v1/tables/products/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, ...],
    "limit": 20,
    "filter": {
      "must": [
        {
          "key": "price",
          "operation": {
            "type": "LessThan",
            "value": 100
          }
        },
        {
          "key": "in_stock",
          "operation": {
            "type": "Equal",
            "value": true
          }
        }
      ]
    }
  }'
```

## MCP Tool Example

```json
{
  "name": "vector_search",
  "arguments": {
    "table": "documents",
    "vector": [0.1, 0.2, 0.3, ...],
    "limit": 10,
    "filter": {
      "must": [
        {
          "key": "language",
          "operation": {
            "type": "In",
            "value": ["en", "es", "fr"]
          }
        }
      ],
      "should": [
        {
          "key": "tags",
          "operation": {
            "type": "Contains",
            "value": "important"
          }
        },
        {
          "key": "priority",
          "operation": {
            "type": "GreaterThan",
            "value": 5
          }
        }
      ]
    }
  }
}
```

## Performance Characteristics

### PRE-FILTERING (Highly Selective)
- **Triggered when**: selectivity < 10% AND estimated matches < 10,000
- **Strategy**: Get matching IDs first, then search with increased fetch limit
- **Best for**: Filters that match <10% of records (e.g., `status=rare_value`)
- **Performance**: ~2-3x faster for selective filters

### POST-FILTERING (Less Selective)
- **Triggered when**: selectivity >= 10% OR estimated matches >= 10,000
- **Strategy**: Search first with 3x limit, filter results after
- **Best for**: Filters that match >10% of records
- **Performance**: More predictable, avoids over-fetching

## Enhanced Cardinality Estimation

### Range Queries (Numeric)
Previously: Fixed 30% estimate
Now: Uses min/max statistics for accurate estimation

Example:
```
Field: age (min=18, max=80)
Filter: age > 60
Old estimate: 30% (conservative)
New estimate: (80-60)/(80-18) = 32% (accurate)
```

### String Matching
Previously: Fixed 20% estimate
Now: Uses prefix index for StartsWith estimation

Example:
```
Field: name with prefixes ["Jo", "Ma", "Pe", "Ro"]
Filter: StartsWith "Jo"
Old estimate: 20% (conservative)
New estimate: 25% (1 matching prefix / 4 total)
```

## Implementation Details

### Statistics Collection
- **Numeric fields**: Tracks min, max values
- **String fields**: Maintains prefix set (up to 1000 unique prefixes, 2-3 chars)
- **Updates**: Automatic on insert/update operations

### Decision Logic
```rust
// In intelligent_indexed_storage.rs
let selectivity = cardinality.exp / total_records;
let use_pre_filtering = selectivity < 0.1 && cardinality.exp < 10000;

if use_pre_filtering {
    // Get candidate IDs first
    let candidate_ids = payload_index.get_matching_ids(filter);
    let fetch_limit = result_limit / selectivity * 2.0;
} else {
    // Search first, filter later
    let fetch_limit = result_limit * 3;
}
```

## Testing Filter Operations

```bash
# Create table
curl -X POST http://localhost:9999/api/v1/tables \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test_filters",
    "dimension": 128,
    "distance_metric": "cosine"
  }'

# Insert test data
curl -X POST http://localhost:9999/api/v1/tables/test_filters/records \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "id": "1",
        "vector": [0.1, ...],
        "payload": {
          "category": "A",
          "score": 95,
          "tags": ["important", "urgent"]
        }
      }
    ]
  }'

# Test Equal filter
curl -X POST http://localhost:9999/api/v1/tables/test_filters/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, ...],
    "limit": 10,
    "filter": {
      "must": [
        {
          "key": "category",
          "operation": {"type": "Equal", "value": "A"}
        }
      ]
    }
  }'

# Test Range filter
curl -X POST http://localhost:9999/api/v1/tables/test_filters/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, ...],
    "limit": 10,
    "filter": {
      "must": [
        {
          "key": "score",
          "operation": {"type": "GreaterThan", "value": 90}
        }
      ]
    }
  }'
```

## Benefits of Enhanced PRE-FILTERING

1. **Intelligent Decision Making**: Automatic choice between PRE and POST filtering
2. **Accurate Estimation**: Statistics-based cardinality estimation
3. **Cost Efficiency**: Reduced over-fetching and computation
4. **Better Recall**: Dynamic fetch limit adjustment
5. **API Consistency**: Works across the REST API, gRPC and MCP

## Limitations and Future Work

### Current Limitations
- In-index filtering not yet implemented (HNSW searches all vectors)
- String matching uses conservative estimates for Contains/EndsWith
- Histogram-based estimation not yet implemented

### Future Enhancements
- HNSW in-index filtering during graph traversal
- Bloom filters for tombstone checking
- Learned cardinality models
- Field-level B-tree indexes for range queries
- Trie-based indexes for string prefix matching
