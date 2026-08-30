# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial LangChain integration for DXMT VectorDB
- Full VectorStore implementation with all required and recommended methods
- Native async support for all operations
- Hybrid search capability (vector + BM25 full-text)
- Support for multiple index types (HNSW, IVF, IVF-PQ, LSH, Brute Force)
- Support for multiple distance metrics (Cosine, Euclidean, Manhattan, Dot Product)
- Advanced insert modes (Standard, LockFree, Streaming, MemoryPool, Ultimate)
- Chinese text search support with Jieba tokenizer
- Comprehensive test suite (65+ tests)
- Integration tests for real-world scenarios
- Performance benchmarks
- Complete API reference documentation
- Multiple usage examples (basic, hybrid search, async, Chinese text, advanced config)

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- N/A (initial release)

### Security
- N/A (initial release)

---

## [0.1.0] - TBD

### Added
- Initial release of langchain-dxmt
- Core functionality:
  - `DXMTVectorDB` class implementing LangChain VectorStore interface
  - `DXMTClient` for HTTP communication with DXMT server
  - Custom exception types (DXMTError, DXMTConnectionError, DXMTTableError, DXMTSearchError)

- LangChain VectorStore Methods:
  - `add_texts()` / `aadd_texts()` - Add texts with embeddings
  - `add_documents()` / `aadd_documents()` - Add Document objects
  - `similarity_search()` / `asimilarity_search()` - Vector similarity search
  - `similarity_search_with_score()` / `asimilarity_search_with_score()` - Search with scores
  - `similarity_search_by_vector()` / `asimilarity_search_by_vector()` - Direct vector search
  - `similarity_search_with_score_by_vector()` / `asimilarity_search_with_score_by_vector()` - Vector search with scores
  - `get_by_ids()` / `aget_by_ids()` - Retrieve documents by IDs
  - `delete()` / `adelete()` - Delete documents by IDs
  - `from_texts()` - Factory method to create from texts
  - `from_documents()` - Factory method to create from documents
  - `from_existing_table()` - Connect to existing DXMT table
  - `as_retriever()` - Use as LangChain retriever

- DXMT-Specific Features:
  - `hybrid_search()` / `ahybrid_search()` - Hybrid vector + text search
  - Configurable vector weight for hybrid search
  - Full-text search with BM25
  - Chinese text support with Jieba tokenization
  - Multiple index type support
  - Multiple distance metric support
  - Advanced insert mode optimization

- Testing:
  - 45+ unit tests with mocked HTTP requests
  - 20+ integration tests with real DXMT server
  - 7 performance benchmarks
  - pytest configuration
  - Test coverage reporting

- Documentation:
  - README with quick start guide
  - Complete API reference (400+ lines)
  - Migration guides from 5 major vector databases
  - Testing guide
  - Contributing guide
  - 5 comprehensive examples
  - Inline docstrings (Google-style)
  - Type hints throughout

- Development:
  - CI/CD with GitHub Actions
  - Ruff for linting and formatting
  - mypy for type checking
  - Pre-commit hooks support
  - PyPI-ready package configuration

---

## Version History

- **[Unreleased]** - Development in progress
- **[0.1.0]** - Initial release (TBD)

---

## Links

- [Repository](https://github.com/dxmt/dxmt)
- [PyPI Package](https://pypi.org/project/langchain-dxmt/) (coming soon)
- [Documentation](https://github.com/dxmt/dxmt/tree/main/vectordb/langchain-dxmt)
- [LangChain Integration Guide](https://python.langchain.com/docs/integrations/vectorstores/dxmt)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

---

**Last Updated:** 2026-01-20
