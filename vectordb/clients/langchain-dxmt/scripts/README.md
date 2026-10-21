# Development Scripts

Collection of utility scripts for developing and releasing langchain-dxmt.

## Overview

| Script | Purpose | Usage |
|--------|---------|-------|
| `setup_dev.sh` | Set up development environment | `./scripts/setup_dev.sh` |
| `check_quality.sh` | Run code quality checks | `./scripts/check_quality.sh [all\|lint\|format\|type]` |
| `run_tests.sh` | Run test suites | `./scripts/run_tests.sh [all\|unit\|integration\|standard\|performance]` |
| `release.sh` | Build and release package | `./scripts/release.sh [check\|build\|test-pypi\|pypi\|full]` |

---

## setup_dev.sh

Sets up a complete development environment.

### What it does:

1. ✅ Checks Python version (3.9+)
2. ✅ Creates virtual environment
3. ✅ Installs all dependencies
4. ✅ Installs development tools
5. ✅ Makes scripts executable
6. ✅ Optionally installs pre-commit hooks
7. ✅ Checks for DXMT server
8. ✅ Runs initial tests

### Usage:

```bash
./scripts/setup_dev.sh
```

### First time setup:

```bash
cd langchain-dxmt

# Run setup script
./scripts/setup_dev.sh

# Activate virtual environment (when prompted)
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

---

## check_quality.sh

Runs code quality checks (linting, formatting, type checking).

### Commands:

```bash
# Run all checks
./scripts/check_quality.sh all

# Run linting only
./scripts/check_quality.sh lint

# Run formatting check only
./scripts/check_quality.sh format

# Run type checking only
./scripts/check_quality.sh type
```

### What it checks:

1. **Linting** (ruff check)
   - Code style
   - Potential bugs
   - Import sorting
   - Complexity

2. **Formatting** (ruff format)
   - Code formatting consistency
   - Line length
   - Quote style

3. **Type Checking** (mypy)
   - Type hint correctness
   - Type safety
   - Missing type annotations

### Fixing issues:

```bash
# Auto-fix linting issues
ruff check --fix langchain_dxmt

# Auto-format code
ruff format langchain_dxmt

# Type errors usually need manual fixes
```

---

## run_tests.sh

Runs various test suites.

### Commands:

```bash
# Run all tests
./scripts/run_tests.sh all

# Run unit tests only (fast, no dependencies)
./scripts/run_tests.sh unit

# Run integration tests (requires DXMT server)
./scripts/run_tests.sh integration

# Run LangChain standard compliance tests
./scripts/run_tests.sh standard

# Run performance benchmarks
./scripts/run_tests.sh performance
```

### Prerequisites:

**Unit tests:**
- ✅ No prerequisites (mocked)

**Integration tests:**
- ✅ DXMT server running on localhost:9999
- ✅ OpenAI API key (optional, for embeddings)

**Standard tests:**
- ✅ DXMT server running
- ✅ OpenAI API key required

**Performance tests:**
- ✅ DXMT server running
- ✅ OpenAI API key required

### Example workflow:

```bash
# Start DXMT server (in separate terminal)
cd ../../
cargo run --release -- serve

# Set OpenAI API key
export OPENAI_API_KEY="sk-..."

# Run all tests
./scripts/run_tests.sh all
```

---

## release.sh

Handles package building and publication to PyPI.

### Commands:

```bash
# Run pre-release checks
./scripts/release.sh check

# Build package (create dist/)
./scripts/release.sh build

# Upload to Test PyPI (for testing)
./scripts/release.sh test-pypi

# Upload to production PyPI (permanent!)
./scripts/release.sh pypi

# Full release workflow (all steps)
./scripts/release.sh full
```

### Release workflow:

#### 1. Prepare release

```bash
# Update version in pyproject.toml
# Update CHANGELOG.md with release notes

# Commit changes
git add pyproject.toml CHANGELOG.md
git commit -m "chore(release): prepare v0.1.0"
```

#### 2. Run pre-release checks

```bash
./scripts/release.sh check
```

This checks:
- ✅ On main branch (or prompts to confirm)
- ✅ No uncommitted changes
- ✅ Code quality passes
- ✅ Unit tests pass
- ✅ Version tag doesn't exist

#### 3. Build package

```bash
./scripts/release.sh build
```

Creates:
- `dist/langchain_dxmt-X.Y.Z-py3-none-any.whl`
- `dist/langchain_dxmt-X.Y.Z.tar.gz`

#### 4. Test on Test PyPI

```bash
./scripts/release.sh test-pypi
```

Then verify:
```bash
# Install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ langchain-dxmt==X.Y.Z

# Test it works
python -c "from langchain_dxmt import DXMTVectorDB; print('OK')"
```

#### 5. Publish to production PyPI

```bash
./scripts/release.sh pypi
```

⚠️ **WARNING**: This is permanent and cannot be undone!

#### 6. Create Git tag and GitHub release

```bash
# Create tag
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z

# Create GitHub release (manual)
# Go to: https://github.com/dxmt/dxmt/releases/new
# - Select tag vX.Y.Z
# - Copy CHANGELOG.md content
# - Publish
```

### Full automated workflow:

```bash
./scripts/release.sh full
```

This runs all steps with prompts for confirmation.

---

## Environment Variables

Scripts respect these environment variables:

| Variable | Purpose | Default |
|----------|---------|---------|
| `DXMT_HOST` | DXMT server hostname | `localhost` |
| `DXMT_PORT` | DXMT server port | `9999` |
| `OPENAI_API_KEY` | OpenAI API key | (none) |

### Setting variables:

```bash
# Temporary (current session)
export OPENAI_API_KEY="sk-..."
export DXMT_HOST="dxmt.example.com"
export DXMT_PORT="8080"

# Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
```

---

## Troubleshooting

### Script permission denied

```bash
# Make scripts executable
chmod +x scripts/*.sh
```

### Python not found

```bash
# Install Python 3.9+
# Ubuntu/Debian
sudo apt install python3.9 python3.9-venv

# macOS
brew install python@3.9

# Windows
# Download from python.org
```

### DXMT server not found

```bash
# Build and run DXMT server
cd ../../
cargo build --release
cargo run --release -- serve
```

### Dependencies not installed

```bash
# Install development dependencies
pip install -e ".[dev]"
```

### Pre-commit hooks fail

```bash
# Update pre-commit
pre-commit autoupdate

# Run manually
pre-commit run --all-files
```

---

## Best Practices

### Before committing:

```bash
# 1. Format code
ruff format langchain_dxmt

# 2. Fix linting issues
ruff check --fix langchain_dxmt

# 3. Run type checker
mypy langchain_dxmt

# 4. Run tests
./scripts/run_tests.sh unit
```

### Before pushing:

```bash
# Run full quality check
./scripts/check_quality.sh all

# Run all tests
./scripts/run_tests.sh all
```

### Before releasing:

```bash
# Full release checklist
./scripts/release.sh check

# Review RELEASE_CHECKLIST.md
cat RELEASE_CHECKLIST.md
```

---

## CI/CD Integration

CI does not call these scripts. This package is tested by the `sdk-live` job of the
repository's `.github/workflows/ci.yml` (at the repository root), which installs it with
`pip install -e . langchain-tests` and runs its suite against a live server. There is no
publishing workflow yet.

---

## Contributing

When adding new scripts:

1. ✅ Make them executable (`chmod +x`)
2. ✅ Add usage documentation here
3. ✅ Follow existing patterns (colors, error handling)
4. ✅ Test on Linux/macOS/Windows (if possible)

---

## Support

For issues with scripts:
- [GitHub Issues](https://github.com/dxmt/dxmt/issues)
- [Contributing Guide](../docs/CONTRIBUTING.md)

---

**Last Updated:** 2026-01-20
