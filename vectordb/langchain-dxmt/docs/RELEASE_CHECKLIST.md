# Release Checklist - LangChain-DXMT

This checklist ensures all steps are completed before releasing a new version.

## Pre-Release Checklist

### Code Quality

- [ ] All unit tests pass locally
  ```bash
  pytest tests/unit_tests/ -v
  ```

- [ ] All integration tests pass with real DXMT server
  ```bash
  pytest tests/integration_tests/ -v
  ```

- [ ] Performance benchmarks run successfully
  ```bash
  python tests/performance/bench_dxmt.py
  ```

- [ ] Code coverage is ≥90%
  ```bash
  pytest --cov=langchain_dxmt --cov-report=term-missing
  ```

- [ ] No linting errors
  ```bash
  ruff check langchain_dxmt
  ```

- [ ] Code is properly formatted
  ```bash
  ruff format langchain_dxmt
  ```

- [ ] Type checking passes
  ```bash
  mypy langchain_dxmt
  ```

- [ ] No TODO or FIXME comments in production code

### Documentation

- [ ] README.md is up to date
  - [ ] Installation instructions correct
  - [ ] Quick start example works
  - [ ] Feature list complete
  - [ ] Links are valid

- [ ] API_REFERENCE.md is complete
  - [ ] All public methods documented
  - [ ] Examples are correct
  - [ ] Parameter descriptions accurate

- [ ] CHANGELOG.md updated
  - [ ] New version section added
  - [ ] All changes categorized (Added, Changed, Fixed, etc.)
  - [ ] Breaking changes clearly marked
  - [ ] Release date added

- [ ] Migration guides reviewed
  - [ ] Code examples tested
  - [ ] Version compatibility noted

- [ ] Examples work correctly
  - [ ] basic_usage.py runs
  - [ ] hybrid_search.py runs
  - [ ] async_usage.py runs
  - [ ] chinese_text_search.py runs
  - [ ] advanced_configuration.py runs

### Version Management

- [ ] Version number updated in `pyproject.toml`
  - [ ] Follows semantic versioning
  - [ ] Consistent across all files

- [ ] Version matches CHANGELOG.md

- [ ] Git tag created
  ```bash
  git tag -a vX.Y.Z -m "Release vX.Y.Z"
  ```

### Package Building

- [ ] Package builds successfully
  ```bash
  python -m build
  ```

- [ ] Built package checked
  ```bash
  twine check dist/*
  ```

- [ ] Package installs in clean environment
  ```bash
  python -m venv test-env
  source test-env/bin/activate
  pip install dist/langchain_dxmt-X.Y.Z-py3-none-any.whl
  python -c "from langchain_dxmt import DXMTVectorDB; print('Success')"
  deactivate
  rm -rf test-env
  ```

- [ ] Test installation from wheel works
  ```bash
  pip install langchain-dxmt==X.Y.Z
  ```

### Dependencies

- [ ] All dependencies have version constraints
- [ ] Dependencies are up to date
- [ ] No security vulnerabilities in dependencies
  ```bash
  pip-audit
  ```
- [ ] Optional dependencies work correctly

### Testing in Real Environment

- [ ] Test with minimum Python version (3.9)
  ```bash
  pyenv local 3.9
  pytest
  ```

- [ ] Test with latest Python version (3.12)
  ```bash
  pyenv local 3.12
  pytest
  ```

- [ ] Test with different OS
  - [ ] Linux
  - [ ] macOS
  - [ ] Windows

- [ ] Integration with actual DXMT server works
  - [ ] Create table
  - [ ] Insert documents
  - [ ] Search works
  - [ ] Delete works

- [ ] Integration with LangChain ecosystem
  - [ ] Works with different embedding providers
  - [ ] Works as retriever in chains
  - [ ] Compatible with LangChain version constraints

### CI/CD

- [ ] GitHub Actions workflows pass
  - [ ] test.yml passes
  - [ ] publish.yml configured

- [ ] All checks on PR pass
  - [ ] Linting
  - [ ] Type checking
  - [ ] Unit tests
  - [ ] Integration tests (if OpenAI key available)

### Security

- [ ] No hardcoded secrets or API keys
- [ ] No sensitive data in examples
- [ ] Dependencies have no known vulnerabilities
- [ ] API endpoints use proper authentication (if required)

### Legal

- [ ] LICENSE file present (Apache-2.0)
- [ ] Copyright notices correct
- [ ] Attribution for third-party code (if any)
- [ ] No license conflicts with dependencies

---

## Release Process

### 1. Prepare Release Branch

```bash
git checkout main
git pull origin main
git checkout -b release/vX.Y.Z
```

### 2. Update Version and Changelog

```bash
# Edit pyproject.toml - update version
# Edit CHANGELOG.md - add release notes with date
git add pyproject.toml CHANGELOG.md
git commit -m "chore(release): bump version to X.Y.Z"
```

### 3. Run Final Checks

```bash
# Run all tests
pytest

# Check code quality
ruff check langchain_dxmt
mypy langchain_dxmt

# Build and check package
python -m build
twine check dist/*
```

### 4. Create Pull Request

```bash
git push origin release/vX.Y.Z
# Create PR on GitHub
# Wait for CI to pass
# Get approval from maintainer
```

### 5. Merge and Tag

```bash
# After PR approval, merge to main
git checkout main
git pull origin main

# Create and push tag
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z

# Create GitHub Release
# Go to GitHub → Releases → Draft a new release
# Select tag vX.Y.Z
# Copy CHANGELOG.md content for this version
# Publish release
```

### 6. PyPI Publication

GitHub Actions will automatically publish to PyPI when a release is created.

Alternatively, manual publication:

```bash
# Build package
python -m build

# Upload to Test PyPI (optional)
twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ langchain-dxmt==X.Y.Z

# Upload to PyPI
twine upload dist/*
```

### 7. Verify Publication

```bash
# Check PyPI page
# https://pypi.org/project/langchain-dxmt/

# Test installation
pip install langchain-dxmt==X.Y.Z

# Quick test
python -c "from langchain_dxmt import DXMTVectorDB; print('✅ Installation successful')"
```

### 8. Post-Release

- [ ] Verify package on PyPI
- [ ] Update documentation links
- [ ] Announce release
  - [ ] GitHub Discussions
  - [ ] Discord/Slack
  - [ ] Reddit (r/LangChain)
  - [ ] Twitter/X
- [ ] Monitor for issues
- [ ] Respond to feedback

---

## Hotfix Release Process

For critical bug fixes:

```bash
# Create hotfix branch from latest release tag
git checkout vX.Y.Z
git checkout -b hotfix/vX.Y.Z+1

# Make fix
# Update version to X.Y.Z+1
# Update CHANGELOG.md

# Test thoroughly
pytest

# Commit, push, and create PR
git commit -m "fix(critical): description of fix"
git push origin hotfix/vX.Y.Z+1

# After merge, tag and release
git tag -a vX.Y.Z+1 -m "Hotfix release vX.Y.Z+1"
git push origin vX.Y.Z+1
```

---

## Rollback Procedure

If issues are discovered after release:

1. **Yank the release from PyPI**
   ```bash
   # Contact PyPI support or use web interface
   # Mark version as "yanked" (not deleted)
   ```

2. **Publish hotfix or revert**
   - If fixable quickly: hotfix release
   - If complex: revert and release previous version

3. **Communicate clearly**
   - GitHub issue explaining problem
   - Update documentation
   - Notify users via Discord/Slack

---

## Version Numbering Guide

Following [Semantic Versioning](https://semver.org/):

- **MAJOR (X.0.0)**: Breaking changes
  - Incompatible API changes
  - Removal of deprecated features
  - Major refactoring

- **MINOR (0.X.0)**: New features
  - New functionality (backward compatible)
  - Deprecations (with warnings)
  - Performance improvements

- **PATCH (0.0.X)**: Bug fixes
  - Bug fixes (backward compatible)
  - Documentation updates
  - Security patches

---

## Contact

For release questions:
- GitHub Issues: https://github.com/dxmt/vectordb/issues
- Maintainer: [Contact info]

---

**Last Updated:** 2026-01-20
