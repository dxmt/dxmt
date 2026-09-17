# DXMT VectorDB Makefile
# Provides common development and deployment commands

# Keep local gates reproducible on small developer machines and CI runners. Environment values
# supplied by the caller still win because these use `?=`.
CARGO_BUILD_JOBS ?= 1
CARGO_INCREMENTAL ?= 0
CARGO_PROFILE_DEV_DEBUG ?= 0
export CARGO_BUILD_JOBS CARGO_INCREMENTAL CARGO_PROFILE_DEV_DEBUG

.PHONY: help build test test-unit test-integ test-recovery test-contract clean run docker setup lint lint-strict bench docs release verify verify-core ci bench-recall test-api test-api-quick verify-live dashboard-install dashboard-build dashboard-dev run-dashboard verify-dashboard dashboard-clean

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
	@echo "  test-api   - Run the HTTP functional suite in both storage modes (needs python3+pytest+httpx)"
	@echo "  test-api-quick - Same, skipping the property suite"
	@echo "  bench      - Run the criterion benchmarks"
	@echo "  bench-recall - Measure ANN recall against exact ground truth"
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
	@echo "Docker:"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Run Docker container"
	@echo "  docker-compose - Run with docker-compose"
	@echo ""
	@echo "Utilities:"
	@echo "  clean      - Clean build artifacts"
	@echo "  verify     - Run all checks (fmt, lint, test)"
	@echo "  verify-live - verify plus the live HTTP suite"

# Setup development environment
setup:
	@echo "Setting up development environment..."
	@rustup component add rustfmt clippy
	@cargo install cargo-audit cargo-tarpaulin
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
	@cargo test --tests

test-recovery:
	@echo "Running persistence and recovery tests..."
	@cargo test --test wal_recovery_test --test index_persistence_test --test partitioned_ivf_test

test-contract:
	@echo "Running server/API contract tests..."
	@cargo test --test api_contract_test

# The HTTP surface, against a server this starts itself -- once with records and vectors in
# memory and once with them in row files. Needs python3 with pytest and httpx; deliberately
# NOT part of `verify`, which stays pure cargo so it runs on a checkout with no Python, and
# because CI runs this in `sdk-live` where the server and the interpreter already live.
test-api:
	@echo "Running functional API tests in both storage modes..."
	@scripts/api_tests.sh

test-api-quick:
	@echo "Running functional API tests (skipping the property suite)..."
	@scripts/api_tests.sh -m "not slow"

# Dashboard (the read-only web console). Node 20+; nothing else in this Makefile needs it.
dashboard-install:
	@command -v npm >/dev/null || { echo "the dashboard needs Node 20+; the rest of the build does not"; exit 1; }
	@cd dashboard && npm ci

dashboard-build:
	@command -v npm >/dev/null || { echo "the dashboard needs Node 20+; the rest of the build does not"; exit 1; }
	@[ -d dashboard/node_modules ] || $(MAKE) dashboard-install
	@cd dashboard && npm run build

dashboard-dev:
	@echo "Vite on :3000, proxying the API to :9999 — start the server separately."
	@cd dashboard && npm run dev

run-dashboard: dashboard-build
	@cargo run --features api-dashboard -- serve --dashboard

# The opt-in gate, parallel to verify-live. `verify` itself stays pure cargo so it runs on a
# checkout with neither Python nor Node.
verify-dashboard: dashboard-build
	@cargo clippy --all-targets --features api-dashboard -- -D warnings
	@cargo test --features api-dashboard --test dashboard_test
	@cd dashboard && npm run lint && npm test
	@DXMT_FEATURES=api-dashboard DXMT_MODES=memory scripts/api_tests.sh -q tests/api/test_dashboard.py

dashboard-clean:
	@rm -rf dashboard/dist dashboard/node_modules

verify-live: verify test-api
	@echo "All checks passed, including the live API suite."

# `filter_performance_bench` declares `required-features = ["text-bm25"]`, and text-bm25
# left the default set. Without naming it here, cargo silently skips the only bench target
# and this recipe exits 0 having measured nothing.
bench:
	@echo "Running benchmarks..."
	@cargo bench --features text-bm25

# Recall against an exact brute-force reference, over the production storage path. The test
# is feature-gated AND #[ignore]d, so nothing reaches it by accident; release-unwind because
# it asserts, and the numbers mean nothing at opt-level 0.
bench-recall:
	@echo "Measuring ANN recall..."
	@cargo test --profile release-unwind --features benchmarks --test storage_ann_benchmark_test -- --ignored --nocapture

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

# Kept as an alias: `lint` is already the strict gate, and it passes.
lint-strict: lint

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
# The fast gate: everything except the full integration suite. Use it while iterating.
verify-core: fmt-check check lint test-unit test-recovery test-contract

# The real gate, and what CI runs. `test-integ` is `cargo test --tests`, which includes the
# recovery and contract targets above; running them first just fails faster.
verify: verify-core test-integ
	@echo "All checks passed!"

ci: verify

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

# Default target
.DEFAULT_GOAL := help
