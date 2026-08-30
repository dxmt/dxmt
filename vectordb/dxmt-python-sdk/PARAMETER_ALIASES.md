# Parameter Aliases Guide

## Overview

To improve compatibility and ease of use, the DXMT Python SDK supports **parameter aliases** for common search methods. This allows you to use familiar parameter names from different ecosystems without breaking your code.

**Updated**: 2026-01-21
**SDK Version**: 0.1.0+

---

## Supported Aliases

### 1. Result Count: `k` ⟺ `limit`

**Preferred**: `k` (following scikit-learn and vector DB conventions)
**Alias**: `limit` (for SQL/database familiarity)

**Methods**:
- `search()` / `asearch()`
- `hybrid_search()` / `ahybrid_search()`
- `full_text_search()` / `afull_text_search()`

**Examples**:
```python
# Both work identically
results = client.search(table_name="demo", vector=[...], k=5)
results = client.search(table_name="demo", vector=[...], limit=5)

# If both provided, k takes precedence
results = client.search(table_name="demo", vector=[...], k=5, limit=10)  # Uses k=5
```

---

### 2. Text Query: `text` ⟺ `text_query`

**Preferred**: `text` (shorter, cleaner)
**Alias**: `text_query` (more explicit)

**Methods**:
- `hybrid_search()` / `ahybrid_search()`

**Examples**:
```python
# Both work identically
results = client.hybrid_search(
    table_name="demo",
    vector=[...],
    text="search query",
    k=5
)

results = client.hybrid_search(
    table_name="demo",
    vector=[...],
    text_query="search query",  # Alias
    k=5
)
```

---

### 3. Search Query: `query` ⟺ `text_query`

**Preferred**: `query` (standard for text search)
**Alias**: `text_query` (consistency with hybrid_search)

**Methods**:
- `full_text_search()` / `afull_text_search()`

**Examples**:
```python
# Both work identically
results = client.full_text_search(
    table_name="demo",
    query="neural networks",
    k=10
)

results = client.full_text_search(
    table_name="demo",
    text_query="neural networks",  # Alias
    k=10
)
```

---

### 4. Vector Weight: `vector_weight` ⟺ `alpha`

**Preferred**: `vector_weight` (more descriptive)
**Alias**: `alpha` (mathematical/academic convention)

**Methods**:
- `hybrid_search()` / `ahybrid_search()`

**Default**: 0.7 (70% vector, 30% text)

**Examples**:
```python
# Both work identically
results = client.hybrid_search(
    table_name="demo",
    vector=[...],
    text="query",
    vector_weight=0.8,  # 80% vector, 20% text
    k=5
)

results = client.hybrid_search(
    table_name="demo",
    vector=[...],
    text="query",
    alpha=0.8,  # Alias - same meaning
    k=5
)

# Understanding the weight/alpha parameter:
# - 0.0 = 100% text search (pure text matching)
# - 0.5 = 50/50 balanced
# - 1.0 = 100% vector search (pure semantic similarity)
```

---

## Complete Alias Reference Table

| Method | Preferred Parameter | Alias | Notes |
|--------|-------------------|-------|-------|
| `search()` | `k` | `limit` | Number of results |
| `asearch()` | `k` | `limit` | Number of results |
| `hybrid_search()` | `k` | `limit` | Number of results |
| `hybrid_search()` | `text` | `text_query` | Query text |
| `hybrid_search()` | `vector_weight` | `alpha` | Vector similarity weight (0-1) |
| `ahybrid_search()` | `k` | `limit` | Number of results |
| `ahybrid_search()` | `text` | `text_query` | Query text |
| `ahybrid_search()` | `vector_weight` | `alpha` | Vector similarity weight (0-1) |
| `full_text_search()` | `k` | `limit` | Number of results |
| `full_text_search()` | `query` | `text_query` | Search query |
| `afull_text_search()` | `k` | `limit` | Number of results |
| `afull_text_search()` | `query` | `text_query` | Search query |

---

## Migration Guide

### From Other Vector Databases

#### From Pinecone/Weaviate (uses `top_k`)
```python
# Before (Pinecone)
results = pinecone_index.query(vector=[...], top_k=5)

# After (DXMT)
results = client.search(table_name="demo", vector=[...], k=5)  # ✅ Use k
# Or
results = client.search(table_name="demo", vector=[...], limit=5)  # ✅ Also works
```

#### From Milvus (uses `limit`)
```python
# Before (Milvus)
results = collection.search(data=[...], limit=10)

# After (DXMT)
results = client.search(table_name="demo", vector=[...], limit=10)  # ✅ Works!
```

#### From Chroma (uses `n_results`)
```python
# Before (Chroma)
results = collection.query(query_embeddings=[...], n_results=5)

# After (DXMT)
results = client.search(table_name="demo", vector=[...], k=5)  # ✅ Use k
```

---

## Best Practices

### 1. Consistency Within Your Codebase

**Recommended**: Pick one style and use it consistently

```python
# ✅ Good - consistent use of preferred parameters
results1 = client.search(table_name="demo", vector=[...], k=5)
results2 = client.hybrid_search(table_name="demo", vector=[...], text="query", vector_weight=0.7, k=5)
results3 = client.full_text_search(table_name="demo", query="text", k=10)

# ⚠️ OK but inconsistent - mixing styles
results1 = client.search(table_name="demo", vector=[...], limit=5)  # Using limit
results2 = client.hybrid_search(table_name="demo", vector=[...], text_query="query", alpha=0.7, k=5)  # Using k
```

### 2. Use Preferred Parameters for New Code

For new code, we recommend using the **preferred parameter names**:
- `k` instead of `limit`
- `text` instead of `text_query`
- `query` instead of `text_query` (for full_text_search)
- `vector_weight` instead of `alpha`

### 3. Aliases for Migration

Use aliases when:
- Migrating from other vector databases
- Maintaining backward compatibility
- Working with existing codebases

---

## Precedence Rules

When both a parameter and its alias are provided, the **preferred parameter takes precedence**:

```python
# k takes precedence over limit
results = client.search(table_name="demo", vector=[...], k=5, limit=10)
# Result: Returns 5 results (uses k=5)

# text takes precedence over text_query
results = client.hybrid_search(
    table_name="demo",
    vector=[...],
    text="query A",
    text_query="query B",
    k=5
)
# Result: Searches for "query A" (uses text)

# vector_weight takes precedence over alpha
results = client.hybrid_search(
    table_name="demo",
    vector=[...],
    text="query",
    vector_weight=0.8,
    alpha=0.6,
    k=5
)
# Result: Uses vector_weight=0.8 (ignores alpha)
```

---

## Default Values

If neither the preferred parameter nor its alias is provided, these defaults apply:

| Method | Parameter | Default Value |
|--------|-----------|---------------|
| `search()` | `k` / `limit` | 4 |
| `hybrid_search()` | `k` / `limit` | 4 |
| `hybrid_search()` | `vector_weight` / `alpha` | 0.7 (70% vector) |
| `full_text_search()` | `k` / `limit` | 4 |

**Note**: `text`/`text_query` and `query`/`text_query` have no defaults - at least one must be provided.

---

## Examples

### Example 1: Flexible Hybrid Search

```python
from dxmt import Client

client = Client(host="localhost", port=9999)

# All of these work and are equivalent:

# Style 1: Preferred parameters
results = client.hybrid_search(
    table_name="articles",
    vector=[0.1] * 128,
    text="machine learning",
    vector_weight=0.7,
    k=10
)

# Style 2: Using aliases (backward compatibility)
results = client.hybrid_search(
    table_name="articles",
    vector=[0.1] * 128,
    text_query="machine learning",  # Alias for text
    alpha=0.7,  # Alias for vector_weight
    limit=10  # Alias for k
)

# Style 3: Mixed (not recommended but works)
results = client.hybrid_search(
    table_name="articles",
    vector=[0.1] * 128,
    text="machine learning",
    alpha=0.7,  # Using alias
    limit=10  # Using alias
)
```

### Example 2: Migrating from SQL Background

```python
# If you come from SQL/database background, use limit
results = client.search(
    table_name="products",
    vector=product_embedding,
    limit=20  # Familiar SQL-like syntax
)

# Gradually migrate to k if preferred
results = client.search(
    table_name="products",
    vector=product_embedding,
    k=20  # Vector DB convention
)
```

### Example 3: Full-Text Search Compatibility

```python
# Both work for full-text search
results = client.full_text_search(
    table_name="documents",
    query="artificial intelligence",  # Preferred
    k=15
)

results = client.full_text_search(
    table_name="documents",
    text_query="artificial intelligence",  # Alias
    limit=15  # Alias
)
```

---

## Why Parameter Aliases?

### Benefits

1. **Ease of Migration**: Smooth transition from other vector databases
2. **Flexibility**: Use familiar terminology from your background (SQL, ML, etc.)
3. **Backward Compatibility**: Old code continues to work
4. **Developer Choice**: Use what makes sense for your team

### Design Philosophy

We follow these principles:
- ✅ **Prefer standard conventions** (k for kNN, query for search)
- ✅ **Support common aliases** (limit, text_query, alpha)
- ✅ **Clear precedence rules** (preferred parameter wins)
- ✅ **No breaking changes** (aliases stay forever)

---

## FAQ

### Q: Which parameter name should I use?

**A**: Use the **preferred parameter name** for new code. Use aliases for:
- Migration from other systems
- Maintaining existing code
- Team preference/familiarity

### Q: Will aliases be deprecated?

**A**: No. Parameter aliases are a permanent feature for compatibility.

### Q: What happens if I use both?

**A**: The preferred parameter takes precedence. Example:
```python
client.search(..., k=5, limit=10)  # Uses k=5
```

### Q: Can I mix parameter styles?

**A**: Yes, but we recommend consistency within your codebase for readability.

### Q: Are there performance differences?

**A**: No. Aliases are resolved immediately and have zero performance impact.

---

## Support

If you have questions or suggestions about parameter names:
- GitHub Issues: https://github.com/dxmt/dxmt/issues
- Documentation: https://github.com/dxmt/dxmt

---

**Last Updated**: 2026-01-21
**SDK Version**: 0.1.0+
