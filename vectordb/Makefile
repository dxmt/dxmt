# DXMT VectorDB Makefile
# Provides common development and deployment commands

.PHONY: help build test clean run docker setup lint bench docs release

# Default target - show help
help:
	@echo "DXMT VectorDB - Available targets:"
	@echo ""
	@echo "Development:"
	@echo "  setup      - Install development dependencies"
	@echo "  build      - Build the project in debug mode"
	@echo "  release    - Build the project in release mode"
	@echo "  run        - Run the server in debug mode"
	@echo "  run-release- Run the server in release mode"
	@echo ""
	@echo "Testing:"
	@echo "  test       - Run all tests"
	@echo "  test-unit  - Run unit tests only"
	@echo "  test-integ - Run integration tests"
	@echo "  bench      - Run benchmarks"
	@echo "  coverage   - Generate test coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  fmt        - Format code with rustfmt"
	@echo "  lint       - Run clippy linter"
	@echo "  check      - Run cargo check"
	@echo "  audit      - Check for security vulnerabilities"
	@echo ""
	@echo "Documentation:"
	@echo "  docs       - Build documentation"
	@echo "  docs-open  - Build and open documentation"
	@echo ""
	@echo "Cluster Operations:"
	@echo "  cluster-start  - Start a local cluster"
	@echo "  cluster-stop   - Stop the local cluster"
	@echo "  cluster-test   - Test cluster functionality"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Run Docker container"
	@echo "  docker-compose - Run with docker-compose"
	@echo ""
	@echo "Utilities:"
	@echo "  clean      - Clean build artifacts"
	@echo "  verify     - Run all checks (fmt, lint, test)"

# Setup development environment
setup:
	@echo "Setting up development environment..."
	@rustup component add rustfmt clippy
	@cargo install cargo-audit cargo-tarpaulin
	@pip3 install -r scripts/requirements.txt || true
	@echo "Setup complete!"

# Build targets
build:
	@echo "Building DXMT VectorDB (debug)..."
	@cargo build

release:
	@echo "Building DXMT VectorDB (release)..."
	@cargo build --release

# Run targets
run: build
	@echo "Starting DXMT VectorDB server (debug)..."
	@cargo run

run-release: release
	@echo "Starting DXMT VectorDB server (release)..."
	@cargo run --release

# Testing targets
test:
	@echo "Running all tests..."
	@cargo test

test-unit:
	@echo "Running unit tests..."
	@cargo test --lib

test-integ:
	@echo "Running integration tests..."
	@cargo test --test '*'
	@python3 scripts/test/run-integration-tests.py || true

bench:
	@echo "Running benchmarks..."
	@cargo bench
	@python3 scripts/benchmark/performance-test.py || true

coverage:
	@echo "Generating test coverage..."
	@cargo tarpaulin --out Html --output-dir target/coverage

# Code quality targets
fmt:
	@echo "Formatting code..."
	@cargo fmt

# Checks formatting without rewriting. `verify` uses this, not `fmt`: a gate that edits
# your working tree is not a gate, and `cargo fmt` on an unformatted repo buries the change
# under review in hundreds of reformatted files.
fmt-check:
	@echo "Checking formatting..."
	@cargo fmt -- --check

lint:
	@echo "Running clippy..."
	@cargo clippy --all-targets -- -D warnings

check:
	@echo "Running cargo check..."
	@cargo check

audit:
	@echo "Checking for security vulnerabilities..."
	@cargo audit

# Documentation
docs:
	@echo "Building documentation..."
	@cargo doc --no-deps

docs-open:
	@echo "Building and opening documentation..."
	@cargo doc --no-deps --open

# Cluster operations
cluster-start:
	@echo "Starting local cluster..."
	@scripts/cluster/start_cluster.sh || true

cluster-stop:
	@echo "Stopping local cluster..."
	@scripts/cluster/stop_cluster.sh || true

cluster-test:
	@echo "Testing cluster functionality..."
	@python3 scripts/test/test_cluster_resilience.py || true

# Docker operations
docker-build:
	@echo "Building Docker image..."
	@docker build -t dxmt-vectordb:latest .

docker-run:
	@echo "Running Docker container..."
	@docker run -p 8000:8000 dxmt-vectordb:latest

docker-compose:
	@echo "Starting with docker-compose..."
	@docker-compose up -d

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@cargo clean
	@rm -rf target/
	@rm -rf /tmp/dxmt_*
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -delete

# Verify everything is working
verify: fmt-check lint test
	@echo "All checks passed!"

# Release process
release-patch:
	@cargo bump patch
	@git add Cargo.toml Cargo.lock
	@git commit -m "chore: bump patch version"
	@cargo build --release

release-minor:
	@cargo bump minor
	@git add Cargo.toml Cargo.lock
	@git commit -m "chore: bump minor version"
	@cargo build --release

release-major:
	@cargo bump major
	@git add Cargo.toml Cargo.lock
	@git commit -m "chore: bump major version"
	@cargo build --release

# Quick commands for common workflows
quick-test: fmt lint test-unit
	@echo "Quick test suite passed!"

dev: fmt
	@cargo watch -x run

# Performance testing
perf-test:
	@echo "Running performance tests..."
	@python3 scripts/benchmark/performance-test.py

perf-compare:
	@echo "Comparing performance with baseline..."
	@python3 scripts/benchmark/compare-versions.sh || true

# API testing shortcuts
api-test:
	@echo "Testing API endpoints..."
	@./scripts/test/simple_test.sh || true

api-test-detailed:
	@echo "Running detailed API tests..."
	@./scripts/test/detailed_test.sh || true

# Default target
.DEFAULT_GOAL := help