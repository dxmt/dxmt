# DXMT VectorDB Makefile
# Provides common development and deployment commands

# Keep local gates reproducible on small developer machines and CI runners. Environment values
# supplied by the caller still win because these use `?=`.
CARGO_BUILD_JOBS ?= 1
CARGO_INCREMENTAL ?= 0
CARGO_PROFILE_DEV_DEBUG ?= 0
export CARGO_BUILD_JOBS CARGO_INCREMENTAL CARGO_PROFILE_DEV_DEBUG

.PHONY: examples examples-list help build test test-unit test-integ test-recovery test-contract clean run docker setup lint lint-strict bench docs release verify verify-core ci bench-recall test-api test-api-quick verify-live dashboard-install dashboard-build dashboard-dev run-dashboard verify-dashboard dashboard-clean

# Default target - show help
# Every target below is listed here, and `make check-help` fails if one is not: this listed
# about half of them, so the way to find `verify-core` or `dashboard-build` was to read the file.
help:
	@echo "DXMT VectorDB - Available targets:"
	@echo ""
	@echo "Development:"
	@echo "  setup        - Install development dependencies (rustfmt, clippy, audit, tarpaulin)"
	@echo "  build        - Build the project in debug mode"
	@echo "  release      - Build the project in release mode"
	@echo "  run          - Run the server in debug mode"
	@echo "  run-release  - Run the server in release mode"
	@echo "  dev          - Format, then run under cargo-watch"
	@echo ""
	@echo "Testing:"
	@echo "  test         - Run all tests"
	@echo "  test-unit    - Run unit tests only"
	@echo "  test-integ   - Run integration tests"
	@echo "  test-recovery  - WAL recovery tests"
	@echo "  test-contract  - The SDK/router contract test"
	@echo "  test-api     - The HTTP functional suite in both storage modes (needs python3+pytest+httpx)"
	@echo "  test-api-quick - Same, skipping the property suite"
	@echo "  examples     - Run examples/, naming every one it could not run and why"
	@echo "  examples-list - What would run and what would not, without running it"
	@echo "  quick-test   - fmt + lint + unit tests"
	@echo "  bench        - Run the criterion benchmarks"
	@echo "  bench-recall - Measure ANN recall against exact ground truth"
	@echo "  coverage     - Generate test coverage report"
	@echo ""
	@echo "Web console (needs Node 20+):"
	@echo "  dashboard-install - npm ci in dashboard/"
	@echo "  dashboard-build   - Build the bundle the binary embeds"
	@echo "  dashboard-dev     - Vite dev server against a running API"
	@echo "  dashboard-clean   - Remove the built bundle"
	@echo "  run-dashboard     - Build the bundle and serve it at /dashboard"
	@echo "  verify-dashboard  - Console tests plus the dashboard API suite"
	@echo ""
	@echo "Code Quality:"
	@echo "  fmt          - Format code with rustfmt"
	@echo "  fmt-check    - Check formatting without writing"
	@echo "  lint         - Run clippy linter"
	@echo "  lint-strict  - clippy with -D warnings"
	@echo "  check        - Run cargo check"
	@echo "  audit        - Check for security vulnerabilities"
	@echo ""
	@echo "Documentation:"
	@echo "  docs         - Build documentation"
	@echo "  docs-open    - Build and open documentation"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build   - Build the image (dxmt/vectordb:alpha)"
	@echo "  docker-run     - Run it on http://localhost:9999 with a data volume"
	@echo "  docker-compose - Run with docker-compose"
	@echo ""
	@echo "Utilities:"
	@echo "  clean        - Clean build artifacts"
	@echo "  verify-core  - fmt-check, check, clippy, unit, recovery and contract tests"
	@echo "  verify       - verify-core plus the full integration suite"
	@echo "  verify-live  - verify plus the live HTTP suite"
	@echo "  ci           - Alias for verify"
	@echo "  check-help   - Fail if a target is missing from this list"

# Every target in this file must appear in `help`, or the help text drifts into a subset of
# what exists -- which is what it had already become.
check-help:
	@listed=$$(mktemp); declared=$$(mktemp); \
	grep -oE '^	@echo "  [a-z][a-z0-9-]*' $(firstword $(MAKEFILE_LIST)) \
		| grep -oE '[a-z][a-z0-9-]*$$' | sort -u > $$listed; \
	grep -oE '^[a-z][a-z0-9-]*:' $(firstword $(MAKEFILE_LIST)) | tr -d ':' | grep -v '^help$$' | sort -u > $$declared; \
	missing=$$(comm -13 $$listed $$declared); \
	rm -f $$listed $$declared; \
	if [ -n "$$missing" ]; then \
		echo "targets missing from 'make help':"; echo "$$missing"; exit 1; \
	fi; \
	echo "help lists every target"

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

# The examples, against a server this starts. Not part of `verify`: it needs frameworks and
# keys that a checkout does not have, and reports what it could not run rather than pretending.
examples:
	@scripts/run_examples.py

examples-list:
	@scripts/run_examples.py --list

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

# Docker operations. The image name is the one docker-compose.yml uses, so building here and
# bringing it up there are the same image rather than two.
docker-build:
	@echo "Building Docker image..."
	@docker build -f docker/Dockerfile -t dxmt/vectordb:alpha .

# The server listens on 9999 (Dockerfile: EXPOSE 9999, CMD --port 9999). This published 8000,
# so the container came up healthy and nothing could reach it. The volume is the other half:
# DXMT_ROOT_DIR is /data in the image, so without it every restart is an empty database.
docker-run:
	@echo "Running Docker container on http://localhost:9999 ..."
	@docker run --rm -p 9999:9999 -v dxmt-vectordb-data:/data dxmt/vectordb:alpha

docker-compose:
	@echo "Starting with docker-compose..."
	@docker-compose -f docker/docker-compose.yml up -d

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
verify-core: check-help fmt-check check lint test-unit test-recovery test-contract

# The real gate, and what CI runs. `test-integ` is `cargo test --tests`, which includes the
# recovery and contract targets above; running them first just fails faster.
verify: verify-core test-integ
	@echo "All checks passed!"

ci: verify

# `release-patch`, `release-minor` and `release-major` were here and could not run: they began
# with `cargo bump`, which is not a cargo subcommand and which `setup` does not install, so
# each one stopped on its first line. They also ran `git commit` after `git add Cargo.toml
# Cargo.lock`, which commits whatever else was already staged. Nothing has been released yet;
# when that changes, the version is one line in Cargo.toml and the commit is yours to write.

# Quick commands for common workflows
quick-test: fmt lint test-unit
	@echo "Quick test suite passed!"

dev: fmt
	@cargo watch -x run

# Default target
.DEFAULT_GOAL := help
