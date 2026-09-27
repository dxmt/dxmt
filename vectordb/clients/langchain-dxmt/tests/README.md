# Testing Guide - LangChain DXMT

Comprehensive testing documentation for the `langchain-dxmt` package.

## Table of Contents

- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Unit Tests](#unit-tests)
- [Integration Tests](#integration-tests)
- [Performance Benchmarks](#performance-benchmarks)
- [Test Coverage](#test-coverage)
- [Continuous Integration](#continuous-integration)

---

## Test Structure

```
tests/
├── unit_tests/              # Unit tests (no external dependencies)
│   └── test_dxmt_vectorstore.py  # VectorStore tests (mocked)
├── integration_tests/       # Integration tests (require DXMT server)
│   └── test_dxmt_integration.py  # End-to-end tests
├── langchain_standard/      # LangChain compliance tests ⭐ NEW
│   ├── test_standard_vectorstore.py          # Standard interface tests
│   ├── test_vectorstore_integration_tests.py # Official LangChain tests
│   └── test_advanced_features.py             # Advanced features (MMR, etc.)
├── performance/             # Performance benchmarks
│   └── bench_dxmt.py             # Benchmark suite
└── README.md                # This file
```

---

## Running Tests

### Prerequisites

Install development dependencies:

```bash
cd langchain-dxmt
pip install -e ".[dev]"
```

This installs:
- `pytest` - Test runner
- `pytest-asyncio` - Async test support
- `pytest-cov` - Coverage reporting
- `pytest-mock` - Mocking utilities

### Quick Start

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit_tests/test_dxmt_client.py

# Run specific test
pytest tests/unit_tests/test_dxmt_client.py::test_create_table_success
```

---

## Unit Tests

Unit tests use mocked HTTP requests and don't require a running DXMT server.

### Running Unit Tests

```bash
# All unit tests
pytest tests/unit_tests/

# With coverage
pytest tests/unit_tests/ --cov=langchain_dxmt --cov-report=html

# Specific test file
pytest tests/unit_tests/test_dxmt_client.py -v

# Async tests only
pytest tests/unit_tests/test_dxmt_vectorstore.py -k async
```

### Unit Test Coverage

**Client Tests** (`test_dxmt_client.py`):
- ✅ Table management (create, get, delete, list)
- ✅ Record operations (insert, get, delete)
- ✅ Search operations (vector, hybrid, full-text)
- ✅ Error handling (connection errors, HTTP errors)
- ✅ Async equivalents
- ✅ Context manager support

**VectorStore Tests** (`test_dxmt_vectorstore.py`):
- ✅ Initialization (various configurations)
- ✅ Adding texts and documents
- ✅ Similarity search (all variants)
- ✅ Hybrid search
- ✅ Retrieval by IDs
- ✅ Deletion
- ✅ Factory methods (from_texts, from_documents)
- ✅ Async methods
- ✅ Error conditions
- ✅ Edge cases

### Example: Running Unit Tests

```bash
$ pytest tests/unit_tests/ -v

tests/unit_tests/test_dxmt_client.py::test_client_init PASSED
tests/unit_tests/test_dxmt_client.py::test_create_table_success PASSED
tests/unit_tests/test_dxmt_client.py::test_search_success PASSED
...
tests/unit_tests/test_dxmt_vectorstore.py::test_add_texts_success PASSED
tests/unit_tests/test_dxmt_vectorstore.py::test_similarity_search PASSED
...

======================== 45 passed in 2.5s ========================
```

---

## Integration Tests

Integration tests require a running DXMT server and OpenAI API key.

### Prerequisites

1. **Start DXMT server:**

```bash
# In vectordb root directory
cargo run --release -- serve

# Or if using Docker
docker run -p 9999:9999 dxmt/vectordb
```

2. **Set environment variables:**

```bash
export OPENAI_API_KEY="your-openai-api-key"

# Optional: Custom DXMT server location
export DXMT_HOST="localhost"
export DXMT_PORT="9999"
```

### Running Integration Tests

```bash
# All integration tests
pytest tests/integration_tests/ -v

# Skip if OPENAI_API_KEY not set
pytest tests/integration_tests/ -v --skip-missing-env

# Run specific test class
pytest tests/integration_tests/test_dxmt_integration.py::TestBasicOperations -v

# Run with output
pytest tests/integration_tests/ -v -s
```

### Integration Test Coverage

**TestBasicOperations:**
- ✅ Adding texts
- ✅ Adding documents with metadata
- ✅ Similarity search
- ✅ Similarity search with scores
- ✅ Search by embedding vector
- ✅ Retrieval by IDs
- ✅ Deletion

**TestAdvancedFeatures:**
- ✅ Metadata filtering
- ✅ Hybrid search
- ✅ Different index types (HNSW, IVF, Brute Force)
- ✅ Different distance metrics (Cosine, Euclidean, Dot Product, Manhattan)

**TestAsyncOperations:**
- ✅ Async add texts
- ✅ Async similarity search
- ✅ Async deletion

**TestLangChainIntegration:**
- ✅ Using as retriever
- ✅ Factory methods (from_texts, from_documents)

**TestErrorHandling:**
- ✅ Invalid table names
- ✅ Missing dimensions
- ✅ Connection errors

**TestPerformance:**
- ✅ Batch insert performance
- ✅ Search performance

### Example: Running Integration Tests

```bash
$ export OPENAI_API_KEY="sk-..."
$ cargo run --release -- serve &  # Start DXMT server
$ pytest tests/integration_tests/ -v

tests/integration_tests/test_dxmt_integration.py::TestBasicOperations::test_add_texts PASSED
tests/integration_tests/test_dxmt_integration.py::TestBasicOperations::test_similarity_search PASSED
tests/integration_tests/test_dxmt_integration.py::TestAdvancedFeatures::test_hybrid_search PASSED
...

======================== 20 passed in 45.2s ========================
```

---

## LangChain Standard Tests ⭐ NEW

**Purpose**: Verify full compliance with LangChain VectorStore standards using official test suites and advanced feature tests.

### Running LangChain Standard Tests

```bash
# All standard compliance tests
pytest tests/langchain_standard/ -v

# Official LangChain integration tests (requires langchain-standard-tests)
pytest tests/langchain_standard/test_vectorstore_integration_tests.py -v

# Advanced features (MMR, score threshold, etc.)
pytest tests/langchain_standard/test_advanced_features.py -v
```

### Prerequisites

```bash
# Install LangChain testing dependencies
pip install langchain-openai langchain-standard-tests

# Or use dev dependencies
pip install -e ".[dev]"
```

### Test Coverage

**test_standard_vectorstore.py** - Core LangChain interface:
- ✅ Required methods (add_texts, similarity_search, from_texts)
- ✅ Recommended methods (delete, get_by_ids, as_retriever, etc.)
- ✅ Async operations (aadd_texts, asimilarity_search, etc.)
- ✅ Metadata filtering (single and multiple fields)
- ✅ Retriever interface (similarity, with filters)
- ✅ Edge cases (empty inputs, non-existent IDs)
- ✅ DXMT-specific features (hybrid_search, from_existing_table)

**test_vectorstore_integration_tests.py** - Official LangChain tests:
- Uses `VectorStoreIntegrationTests` base class from `langchain-standard-tests`
- Automatically tests all standard VectorStore interface requirements
- Ensures full LangChain compatibility and best practices
- Note: Requires `langchain-standard-tests` package

**test_advanced_features.py** - Advanced LangChain features:
- ✅ **MMR Search** (Maximal Marginal Relevance for diversity)
  - Test different lambda_mult values (relevance vs diversity)
  - Verify diverse results in search output
- ✅ **Score Threshold** (Filter by minimum similarity score)
  - Test similarity_score_threshold search type
  - Verify threshold filtering works correctly
- ✅ **Complex Metadata Filtering**
  - Numeric range filters (if supported)
  - Nested metadata fields
  - List-valued metadata
- ✅ **Document Integrity**
  - Original documents not mutated during add_documents()
  - Metadata survives round-trip storage
  - Custom IDs preserved exactly
  - Search results ordered by score
- ✅ **Validation Errors**
  - Metadata count mismatch detection
  - Dimension mismatch handling
  - Empty text handling
  - None ID handling
- ✅ **Batch Handling**
  - Single document insertion
  - Small batches (< 10 docs)
  - Medium batches (10-100 docs)
  - Large batches (100+ docs)
- ✅ **Async Operations**
  - Concurrent async operations
  - Async search after sync add

### Example: Running Standard Tests

```bash
$ pytest tests/langchain_standard/ -v

tests/langchain_standard/test_standard_vectorstore.py::TestStandardVectorStoreInterface::test_add_texts PASSED
tests/langchain_standard/test_standard_vectorstore.py::TestAsyncInterface::test_aadd_texts PASSED
tests/langchain_standard/test_advanced_features.py::TestMMRSearch::test_mmr_search_type PASSED
tests/langchain_standard/test_advanced_features.py::TestScoreThreshold::test_score_threshold_search PASSED
tests/langchain_standard/test_advanced_features.py::TestDocumentIntegrity::test_original_documents_not_mutated PASSED
...

======================== 35 passed in 28.5s ========================
```

### Compliance Report

Run this to see full compliance report:

```bash
pytest tests/langchain_standard/test_standard_vectorstore.py::test_langchain_compliance_report -v -s
```

---

## Performance Benchmarks

Performance benchmarks measure throughput, latency, and resource usage.

### Prerequisites

Same as integration tests (DXMT server + OpenAI API key).

### Running Benchmarks

```bash
# Run all benchmarks
python tests/performance/bench_dxmt.py

# Or via pytest
pytest tests/performance/ -v -s
```

### Benchmark Coverage

1. **Insert Throughput** (1000 docs)
   - Measures document insertion speed
   - Reports: ops/sec, total time

2. **Search Latency** (100 queries, k=10)
   - Measures query response time
   - Reports: p50, p95, p99 latencies

3. **Batch Size Comparison** (10, 50, 100, 500, 1000)
   - Optimal batch size for insertions
   - Reports: throughput per batch size

4. **Index Type Comparison** (HNSW, IVF, Brute Force)
   - Insert and search performance
   - Reports: insert throughput, search latency

5. **Distance Metric Comparison** (Cosine, Euclidean, Dot Product)
   - Search performance by metric
   - Reports: search latency

6. **Hybrid Search** (Vector vs Hybrid)
   - Compare pure vector vs hybrid search
   - Reports: latency for each type

7. **Concurrent Operations** (20 concurrent searches)
   - Async concurrency performance
   - Reports: total time, throughput

### Example: Benchmark Output

```bash
$ python tests/performance/bench_dxmt.py

================================================================================
DXMT VectorStore Performance Benchmarks
================================================================================

Configuration:
  Host: localhost
  Port: 9999
  Embedding: OpenAI (1536-dim)

[1/7] Benchmarking insert throughput (1000 docs)...
[2/7] Benchmarking search latency (100 queries, k=10)...
[3/7] Benchmarking different batch sizes...
...

================================================================================
BENCHMARK RESULTS
================================================================================

Insert Throughput
--------------------------------------------------------------------------------
  Duration:    2.345s
  Operations:  1000
  Throughput:  426.4 ops/sec
  Latency:     2.35ms per operation
  Mode:        ultimate
  Dimension:   1536

Search Latency
--------------------------------------------------------------------------------
  Duration:    1.234s
  Operations:  100
  Throughput:  81.0 ops/sec
  Latency:     12.34ms per operation
  K:           10
  P50 Ms:      11.2
  P95 Ms:      18.5
  P99 Ms:      24.3

...

✅ All benchmarks completed successfully!
```

### Performance Targets

Based on 1M vectors (1536-dim) on single server:

| Metric | Target | Notes |
|--------|--------|-------|
| Insert throughput | >1000 docs/sec | With ultimate mode |
| Search latency (p50) | <20ms | k=10 |
| Search latency (p95) | <50ms | k=10 |
| Memory usage | <6GB | HNSW index |
| Memory usage (IVF-PQ) | <2GB | Compressed |

---

## Test Coverage

### Measuring Coverage

```bash
# Run tests with coverage
pytest --cov=langchain_dxmt --cov-report=term-missing

# Generate HTML report
pytest --cov=langchain_dxmt --cov-report=html

# Open HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Coverage Targets

- **Target**: 90%+ overall coverage
- **Critical paths**: 100% coverage for:
  - Error handling
  - API endpoints
  - Data transformations

### Example: Coverage Report

```bash
$ pytest --cov=langchain_dxmt --cov-report=term-missing

---------- coverage: platform linux, python 3.11.0 -----------
Name                                Stmts   Miss  Cover   Missing
-----------------------------------------------------------------
langchain_dxmt/__init__.py            10      0   100%
langchain_dxmt/client.py             320     15    95%   245-250, 380-382
langchain_dxmt/vectorstores.py       380     20    95%   412-415, 550-555
-----------------------------------------------------------------
TOTAL                                 710     35    95%
```

---

## Continuous Integration

### GitHub Actions (Recommended)

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      dxmt:
        image: dxmt/vectordb:latest
        ports:
          - 9999:9999

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run unit tests
        run: |
          pytest tests/unit_tests/ -v --cov=langchain_dxmt

      - name: Run integration tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          pytest tests/integration_tests/ -v

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: pytest-unit
        name: pytest-unit
        entry: pytest tests/unit_tests/ -v
        language: system
        pass_filenames: false
        always_run: true
```

Install hooks:

```bash
pip install pre-commit
pre-commit install
```

---

## Troubleshooting

### Common Issues

**1. ImportError: No module named 'langchain_dxmt'**

```bash
# Install in editable mode
pip install -e .
```

**2. Connection refused (integration tests)**

```bash
# Make sure DXMT server is running
cargo run --release -- serve

# Or check if port is correct
export DXMT_PORT=9999
```

**3. OpenAI API rate limits**

```bash
# Reduce number of test documents
pytest tests/integration_tests/ -v -k "not performance"
```

**4. Async tests hanging**

```bash
# Check pytest-asyncio is installed
pip install pytest-asyncio

# Run with timeout
pytest --timeout=60 tests/integration_tests/
```

### Debug Mode

```bash
# Run tests with debug output
pytest -v -s --log-cli-level=DEBUG

# Run single test with full output
pytest tests/unit_tests/test_dxmt_client.py::test_create_table_success -vvv -s
```

---

## Writing New Tests

### Unit Test Template

```python
def test_new_feature(mock_client: Mock) -> None:
    """Test description."""
    # Arrange
    mock_client.some_method.return_value = {"success": True}

    # Act
    result = function_under_test()

    # Assert
    assert result is not None
    mock_client.some_method.assert_called_once()
```

### Integration Test Template

```python
def test_new_integration(vector_store: DXMTVectorDB) -> None:
    """Test description."""
    # Arrange
    documents = [Document(page_content="Test")]

    # Act
    ids = vector_store.add_documents(documents)

    # Assert
    assert len(ids) == 1

    # Cleanup
    vector_store.delete(ids=ids)
```

### Async Test Template

```python
@pytest.mark.asyncio
async def test_async_feature(vector_store: DXMTVectorDB) -> None:
    """Test description."""
    # Arrange
    texts = ["Test 1", "Test 2"]

    # Act
    ids = await vector_store.aadd_texts(texts)

    # Assert
    assert len(ids) == 2

    # Cleanup
    await vector_store.adelete(ids=ids)
```

---

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Cleanup**: Always cleanup test data in integration tests
3. **Meaningful Names**: Use descriptive test function names
4. **Arrange-Act-Assert**: Follow AAA pattern
5. **Mock External Calls**: Mock HTTP requests in unit tests
6. **Test Edge Cases**: Test empty inputs, None values, etc.
7. **Async Testing**: Use proper async fixtures and markers
8. **Performance Tests**: Run separately from unit/integration tests

---

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-asyncio documentation](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [LangChain testing guide](https://python.langchain.com/docs/contributing/testing)

---

**Last Updated:** 2026-01-20
**Version:** 0.1.0
