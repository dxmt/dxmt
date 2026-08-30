# Changelog

All notable changes to the DXMT Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### ⚠️ Breaking Changes - Field Naming (2026-01-22)

**Critical Update: Full-Text Search Field Names**

Full-text search and hybrid search now require specific field names in the payload:

#### What Changed

- **REQUIRED:** Use `"content"` field for main text content (not `"text"`)
- **RECOMMENDED:** Use `"title"` for document titles
- **OPTIONAL:** Use `"description"` and `"tags"` for additional searchable fields

#### Migration Required

If you're using full-text search or hybrid search, you must update your code:

```python
# ❌ Old code (will not work)
records = [{
    "id": "doc1",
    "vector": embedding,
    "payload": {
        "text": "Document content..."  # ❌ Wrong field name
    }
}]

# ✅ New code (required)
records = [{
    "id": "doc1",
    "vector": embedding,
    "payload": {
        "content": "Document content..."  # ✅ Correct field name
    }
}]
```

#### Why This Change

The full-text search engine (Tantivy) expects specific field names to map to its internal schema. Using incorrect field names causes:
- Full-text search returns 0 results (silent failure)
- Hybrid search text component doesn't work
- No error messages (making debugging difficult)

#### Migration Steps

1. **Update your code** to use `"content"` instead of `"text"`
2. **Re-create tables** with `enable_full_text_search=True`
3. **Re-insert all records** with corrected field names
4. **Test full-text search** to verify it returns results

#### Impact

- ✅ **No impact** if you only use vector search (no full-text/hybrid search)
- ⚠️ **Breaking change** if you use full-text or hybrid search with old field names

For complete details, see:
- [Field Naming Conventions](README.md#field-naming-conventions)

### Fixed

- **Full-text search now works correctly** with proper field names
- **Hybrid search text component** now contributes non-zero scores
- **Min-Max normalization** handles edge case when all scores are similar

### Documentation

- Added comprehensive [Field Naming Conventions](README.md#field-naming-conventions) section
- Updated all examples to use `"content"` field
- Added warning about field naming in README
- Created detailed field naming standard documentation

## [0.1.0] - 2024-XX-XX

### Added

- Initial release of DXMT Python SDK
- Support for vector operations (insert, search, delete)
- Multiple distance metrics (Cosine, Euclidean, Manhattan, Dot Product)
- Multiple index types (HNSW, IVF, IVF-PQ, LSH, Brute Force)
- Full-text search with BM25
- Hybrid search (vector + text)
- Chinese text support with Jieba tokenizer
- Async/await support for all operations
- Context manager support
- Comprehensive filtering with MongoDB-style operators
- Type hints with mypy support
- Zero external dependencies (only httpx)

### Documentation

- Complete README with examples
- API reference documentation
- Filter guide (FILTERS_GUIDE.md)
- Example scripts for common use cases

## Notes

### Versioning Strategy

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for new functionality (backward-compatible)
- **PATCH** version for backward-compatible bug fixes

### Field Naming History

**2026-01-22:** Standardized field naming for full-text search
- Required field: `"content"` (was: various names like `"text"`)
- Reason: Internal Tantivy schema mapping
- Impact: Breaking change for full-text/hybrid search users

[Unreleased]: https://github.com/dxmt/vectordb/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dxmt/vectordb/releases/tag/v0.1.0
