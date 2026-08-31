# Contributing to LangChain-DXMT

Thank you for your interest in contributing to LangChain-DXMT! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code.

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- DXMT VectorDB server (for integration tests)
- OpenAI API key (for integration tests)

### Finding Issues

- Check the [issue tracker](https://github.com/dxmt/vectordb/issues)
- Look for issues labeled `good first issue` or `help wanted`
- Feel free to create new issues for bugs or feature requests

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/vectordb.git
cd vectordb/langchain-dxmt

# Add upstream remote
git remote add upstream https://github.com/dxmt/vectordb.git
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install package in editable mode with dev dependencies
pip install -e ".[dev]"
```

### 4. Install Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

### 5. Start DXMT Server (for integration tests)

```bash
# In the vectordb root directory
cd ../..
cargo run --release -- serve
```

---

## Making Changes

### 1. Create a Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

Follow these guidelines:

- **Keep changes focused**: One feature/fix per branch
- **Write clear commit messages**: Use conventional commits format
- **Add tests**: All new features must have tests
- **Update documentation**: Update docstrings and README if needed
- **Follow code style**: Use ruff and mypy (see below)

### 3. Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**

```bash
git commit -m "feat(vectorstore): add MMR search support"
git commit -m "fix(client): handle connection timeout errors"
git commit -m "docs(api): update similarity_search examples"
git commit -m "test(integration): add hybrid search tests"
```

---

## Testing

### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=langchain_dxmt --cov-report=html
```

### Run Specific Test Types

```bash
# Unit tests only
pytest tests/unit_tests/ -v

# Integration tests only (requires DXMT server)
pytest tests/integration_tests/ -v

# Performance benchmarks
python tests/performance/bench_dxmt.py
```

### Writing Tests

**Unit Test Example:**

```python
def test_new_feature(mock_client: Mock) -> None:
    """Test description of what this tests."""
    # Arrange
    mock_client.method.return_value = {"success": True}

    # Act
    result = function_under_test()

    # Assert
    assert result is not None
    mock_client.method.assert_called_once()
```

**Integration Test Example:**

```python
def test_new_integration(vector_store: DXMTVectorDB) -> None:
    """Test description of what this tests."""
    # Arrange
    documents = [Document(page_content="Test")]

    # Act
    ids = vector_store.add_documents(documents)

    # Assert
    assert len(ids) == 1

    # Cleanup
    vector_store.delete(ids=ids)
```

### Test Coverage Requirements

- **Minimum coverage**: 90%
- **Critical paths**: 100% coverage
- Run coverage report:

```bash
pytest --cov=langchain_dxmt --cov-report=term-missing
```

---

## Code Style

### Automatic Formatting with Ruff

```bash
# Check code
ruff check langchain_dxmt

# Fix auto-fixable issues
ruff check --fix langchain_dxmt

# Format code
ruff format langchain_dxmt
```

### Type Checking with mypy

```bash
# Run type checker
mypy langchain_dxmt

# Check specific file
mypy langchain_dxmt/client.py
```

### Code Style Guidelines

1. **Line Length**: Max 88 characters (Black standard)
2. **Quotes**: Use double quotes `"`
3. **Type Hints**: Required for all public functions
4. **Docstrings**: Google-style docstrings for all public APIs
5. **Imports**: Sorted with isort (handled by ruff)

**Docstring Example:**

```python
def add_texts(
    self,
    texts: Iterable[str],
    metadatas: Optional[list[dict]] = None,
    **kwargs: Any,
) -> list[str]:
    """Add texts to the vector store.

    Args:
        texts: Texts to add to the vector store.
        metadatas: Optional metadata for each text.
        **kwargs: Additional keyword arguments.

    Returns:
        List of IDs for the added documents.

    Raises:
        ValueError: If embedding function not provided.
        DXMTError: If insertion fails.

    Example:
        >>> store = DXMTVectorDB(...)
        >>> ids = store.add_texts(["text1", "text2"])
        >>> print(ids)
        ['id1', 'id2']
    """
```

### Pre-commit Checks

Before committing, ensure:

```bash
# Format code
ruff format langchain_dxmt

# Check for issues
ruff check langchain_dxmt

# Type check
mypy langchain_dxmt

# Run tests
pytest tests/unit_tests/ -v
```

---

## Submitting Changes

### 1. Push Changes

```bash
# Push to your fork
git push origin feature/your-feature-name
```

### 2. Create Pull Request

1. Go to the [repository on GitHub](https://github.com/dxmt/vectordb)
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill in the PR template:
   - Clear description of changes
   - Related issue numbers (if any)
   - Testing performed
   - Screenshots (if applicable)

### 3. PR Checklist

Before submitting, ensure:

- [ ] All tests pass (`pytest`)
- [ ] Code is formatted (`ruff format`)
- [ ] No linting errors (`ruff check`)
- [ ] Type checking passes (`mypy`)
- [ ] Documentation updated (if needed)
- [ ] CHANGELOG.md updated (for significant changes)
- [ ] Commit messages follow convention
- [ ] Branch is up to date with main

### 4. Code Review Process

- Maintainers will review your PR
- Address feedback by pushing new commits
- Once approved, maintainers will merge
- Squash and merge is preferred

---

## Release Process

(For maintainers)

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Steps

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md** with release notes
3. **Create release commit**:
   ```bash
   git commit -m "chore(release): bump version to X.Y.Z"
   ```
4. **Tag release**:
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```
5. **Create GitHub Release**: GitHub Actions will automatically publish to PyPI

---

## Development Workflow Examples

### Adding a New Feature

```bash
# 1. Create branch
git checkout -b feature/add-mmr-search

# 2. Implement feature
# Edit files...

# 3. Add tests
# Edit test files...

# 4. Run tests
pytest tests/unit_tests/ -v

# 5. Update documentation
# Edit README.md, docstrings, etc.

# 6. Format and check
ruff format langchain_dxmt
ruff check langchain_dxmt
mypy langchain_dxmt

# 7. Commit
git add .
git commit -m "feat(vectorstore): add MMR search support

Implements maximal marginal relevance search for diversity-aware retrieval.

Closes #123"

# 8. Push and create PR
git push origin feature/add-mmr-search
```

### Fixing a Bug

```bash
# 1. Create branch
git checkout -b fix/connection-timeout

# 2. Write failing test
# Add test that reproduces bug

# 3. Fix bug
# Edit source files

# 4. Verify fix
pytest tests/unit_tests/test_client.py::test_connection_timeout -v

# 5. Run all tests
pytest

# 6. Commit
git commit -m "fix(client): handle connection timeout gracefully

Add proper timeout handling and error messages for connection failures.

Fixes #456"

# 7. Push and create PR
git push origin fix/connection-timeout
```

---

## Getting Help

- **Documentation**: [README.md](../README.md), [API Reference](API_REFERENCE.md)
- **Issues**: [GitHub Issues](https://github.com/dxmt/vectordb/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dxmt/vectordb/discussions)
- **Discord**: [Join our community](#)

---

## Recognition

Contributors will be:

- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

Thank you for contributing! 🎉

---

**Last Updated:** 2026-01-20
