# Tests for DXMT Vector Store Integration

This directory contains tests for the LlamaIndex DXMT Vector Store integration.

## Test Structure

### Unit Tests

- **`test_dxmt_vector_store.py`** - Core vector store functionality tests
  - Initialization and configuration
  - Adding nodes
  - Querying (vector, hybrid, text search)
  - Deleting nodes
  - Getting nodes
  - Properties and utilities

- **`test_filters.py`** - Filter conversion tests
  - Single filter operators (EQ, NE, GT, LT, etc.)
  - Complex filters (AND, OR conditions)
  - Nested filters
  - Edge cases

### Integration Tests

- **`test_integration.py`** - End-to-end integration tests
  - Basic CRUD operations
  - Metadata filtering
  - Retrieval functionality
  - RAG pipeline
  - *Requires running DXMT server*

## Running Tests

### Install Dependencies

```bash
# Install package with dev dependencies
poetry install --with dev

# Or with pip
pip install -e ".[dev]"
```

### Run All Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=llama_index --cov-report=html

# Run with verbose output
pytest -v
```

### Run Specific Test Files

```bash
# Unit tests only
pytest tests/test_dxmt_vector_store.py
pytest tests/test_filters.py

# Integration tests only
pytest tests/test_integration.py -m integration
```

### Run Specific Test Classes or Functions

```bash
# Run specific test class
pytest tests/test_dxmt_vector_store.py::TestDXMTVectorStoreAdd

# Run specific test function
pytest tests/test_dxmt_vector_store.py::TestDXMTVectorStoreAdd::test_add_single_node
```

### Skip Integration Tests

Integration tests are marked with `@pytest.mark.integration` and require a running DXMT server.

```bash
# Skip integration tests
pytest -m "not integration"
```

### Run Only Integration Tests

```bash
# Run only integration tests
pytest -m integration

# With custom DXMT server
DXMT_HOST=example.com DXMT_PORT=9999 pytest -m integration
```

## Environment Variables

Integration tests use the following environment variables:

- `DXMT_HOST` - DXMT server host (default: `localhost`)
- `DXMT_PORT` - DXMT server port (default: `9999`)
- `OPENAI_API_KEY` - OpenAI API key (required for embedding and LLM tests)

Example:

```bash
export DXMT_HOST=localhost
export DXMT_PORT=9999
export OPENAI_API_KEY=sk-...

pytest -m integration
```

## Test Coverage

Current test coverage includes:

- ✅ Vector store initialization and configuration
- ✅ Adding single and multiple nodes
- ✅ Batch operations with custom batch size
- ✅ Vector similarity search
- ✅ Hybrid search (vector + text)
- ✅ Full-text search
- ✅ Metadata filtering (all operators)
- ✅ Complex filter conditions (AND, OR, nested)
- ✅ Deleting nodes (by ID, by ref_doc_id, by filter)
- ✅ Getting nodes (by ID, by filter)
- ✅ Table creation and management
- ✅ From existing table
- ✅ Retriever functionality
- ✅ End-to-end RAG pipeline

## Writing New Tests

### Unit Test Template

```python
import pytest
from llama_index.vector_stores.dxmt import DXMTVectorStore

class TestNewFeature:
    """Tests for new feature."""

    def test_feature_basic(self, vector_store):
        """Test basic functionality."""
        # Arrange
        ...

        # Act
        result = vector_store.some_method()

        # Assert
        assert result is not None
```

### Integration Test Template

```python
import pytest

pytestmark = pytest.mark.integration

class TestNewIntegration:
    """Integration tests for new feature."""

    def test_feature_integration(self, vector_store, embed_model):
        """Test feature integration."""
        # Requires running DXMT server
        ...
```

## Continuous Integration

Tests are run automatically on:

- Pull requests
- Commits to main branch
- Daily schedule (integration tests)

## Debugging Tests

### Run with PDB

```bash
# Drop into debugger on failure
pytest --pdb

# Drop into debugger on first failure
pytest -x --pdb
```

### Show Print Statements

```bash
# Show stdout/stderr
pytest -s

# Or
pytest --capture=no
```

### Run Last Failed Tests

```bash
# Re-run only failed tests from last run
pytest --lf

# Re-run failed tests first, then others
pytest --ff
```

## Performance Testing

For performance benchmarks, see the separate benchmark suite:

```bash
# Run benchmarks (if implemented)
pytest tests/test_performance.py --benchmark-only
```

## Test Fixtures

Common fixtures available:

- `mock_dxmt_client` - Mocked DXMT client for unit tests
- `vector_store` - Fresh vector store instance
- `sample_nodes` - Sample TextNode objects
- `sample_documents` - Sample Document objects
- `embed_model` - OpenAI embedding model (integration)
- `dxmt_host` - DXMT server host (integration)
- `dxmt_port` - DXMT server port (integration)

## Troubleshooting

### Integration Tests Failing

1. Ensure DXMT server is running:
   ```bash
   cd /path/to/vectordb
   cargo run -- serve --host 0.0.0.0 --port 9999
   ```

2. Check server is accessible:
   ```bash
   curl http://localhost:9999/health
   ```

3. Set correct environment variables:
   ```bash
   export DXMT_HOST=localhost
   export DXMT_PORT=9999
   ```

### OpenAI API Key Required

Some integration tests require OpenAI API key:

```bash
export OPENAI_API_KEY=sk-...
```

### Import Errors

Ensure package is installed in development mode:

```bash
pip install -e .
```

## Contributing

When adding new features:

1. Write unit tests first (TDD approach)
2. Add integration tests for end-to-end workflows
3. Update this README if adding new test categories
4. Ensure all tests pass before submitting PR
5. Aim for >80% code coverage

## Resources

- [LlamaIndex Testing Guide](https://docs.llamaindex.ai/en/stable/contributing/contributing.html)
- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
