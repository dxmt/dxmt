# What's New - Parameter Aliases Support

## 🎉 Now Supporting Multiple Parameter Names!

**Updated**: 2026-01-21
**SDK Version**: 0.1.0+

---

## Quick Summary

You can now use **either parameter name** - both work!

```python
from dxmt import Client

client = Client(host="localhost", port=9999)

# ✅ Both of these work identically:
results = client.search(table_name="demo", vector=[...], k=5)
results = client.search(table_name="demo", vector=[...], limit=5)

# ✅ All of these work identically:
results = client.hybrid_search(
    table_name="demo",
    vector=[...],
    text="query",           # or text_query="query"
    vector_weight=0.7,      # or alpha=0.7
    k=5                     # or limit=5
)
```

---

## Supported Aliases

### For Result Count
- `k` (preferred) ⟺ `limit` (alias)
- Works in: `search()`, `hybrid_search()`, `full_text_search()` and async versions

### For Text Query
- `text` (preferred) ⟺ `text_query` (alias) - in `hybrid_search()`
- `query` (preferred) ⟺ `text_query` (alias) - in `full_text_search()`

### For Vector Weight
- `vector_weight` (preferred) ⟺ `alpha` (alias)
- Works in: `hybrid_search()` and `ahybrid_search()`

---

## Why This Matters

### 1. Your Old Code Still Works ✅

If you had errors like:
```python
TypeError: search() got an unexpected keyword argument 'limit'
TypeError: hybrid_search() got an unexpected keyword argument 'text_query'
```

**These are now fixed!** All your existing code works without changes.

### 2. Use What You're Comfortable With

**From SQL background?** Use `limit`:
```python
results = client.search(table_name="products", vector=[...], limit=10)
```

**From ML background?** Use `k`:
```python
results = client.search(table_name="products", vector=[...], k=10)
```

**Academic preference?** Use `alpha`:
```python
results = client.hybrid_search(table_name="papers", vector=[...], text="query", alpha=0.8)
```

**Prefer descriptive names?** Use `vector_weight`:
```python
results = client.hybrid_search(table_name="papers", vector=[...], text="query", vector_weight=0.8)
```

---

## Examples

### Example 1: Basic Search

```python
# All work identically ✅
results = client.search(table_name="demo", vector=[0.1] * 128, k=5)
results = client.search(table_name="demo", vector=[0.1] * 128, limit=5)
```

### Example 2: Hybrid Search

```python
# All work identically ✅
results = client.hybrid_search(
    table_name="articles",
    vector=[0.1] * 128,
    text="machine learning",
    vector_weight=0.7,
    k=10
)

results = client.hybrid_search(
    table_name="articles",
    vector=[0.1] * 128,
    text_query="machine learning",  # Alias
    alpha=0.7,                       # Alias
    limit=10                         # Alias
)
```

### Example 3: Full-Text Search

```python
# All work identically ✅
results = client.full_text_search(table_name="docs", query="AI", k=5)
results = client.full_text_search(table_name="docs", text_query="AI", limit=5)
```

---

## Important: Use `payload` Not `metadata`

**Remember**: When inserting records, use `payload` field:

```python
# ✅ Correct
record = {
    "id": "doc1",
    "vector": [0.1] * 128,
    "payload": {"title": "Document 1"}
}

# ❌ Wrong - metadata not stored
record = {
    "id": "doc1",
    "vector": [0.1] * 128,
    "metadata": {"title": "Document 1"}  # Won't work!
}
```

---

## Migration Guide

### From Milvus
```python
# Before (Milvus)
results = collection.search(data=[...], limit=10)

# After (DXMT) - works immediately!
results = client.search(table_name="demo", vector=[...], limit=10)  # ✅
```

### From Pinecone
```python
# Before (Pinecone)
results = index.query(vector=[...], top_k=5)

# After (DXMT)
results = client.search(table_name="demo", vector=[...], k=5)  # ✅
```

---

## Testing Your Code

Run the updated examples:

```bash
# Test basic usage
python3 examples/basic_usage.py

# Test hybrid search with aliases
python3 examples/hybrid_search.py

# Test async with aliases
python3 examples/async_usage.py
```

All should work without errors! ✅

---

## Documentation

- **Full Guide**: See `PARAMETER_ALIASES.md`
- **Implementation Details**: See `ALIAS_IMPLEMENTATION_SUMMARY.md`
- **Bug Fixes**: See `BUG_FIXES_SUMMARY.md`
- **Quick Start**: See `QUICK_START.md`

---

## Questions?

**Q: Which parameter name should I use?**
A: Use whichever you're comfortable with! For new code, we recommend the preferred names (`k`, `text`, `vector_weight`), but aliases work forever.

**Q: Will my old code break?**
A: No! All changes are backward compatible. If your code worked before (or should have worked), it works now.

**Q: Can I mix parameter styles?**
A: Yes, but pick one style for consistency within your codebase.

**Q: What if I provide both (e.g., `k=5` and `limit=10`)?**
A: The preferred parameter (`k`) takes precedence. No error is raised.

---

## Summary

✅ **12 parameter aliases** added across all search methods
✅ **100% backward compatible** - no breaking changes
✅ **Zero performance impact** - aliases are free
✅ **All examples updated** and working
✅ **Comprehensive documentation** provided

**Your code just got more flexible!** 🚀

---

**Need Help?**
- GitHub Issues: https://github.com/dxmt/dxmt/issues
- Documentation: https://github.com/dxmt/dxmt
