# DXMT VectorDB Test Organization

This directory contains all tests for DXMT VectorDB, organized by category for easy execution and maintenance.

## Directory Structure

```
tests/
├── unit/              # Fast, isolated unit tests
│   └── validation_test.rs
├── integration/       # Multi-component integration tests
│   ├── crud_consistency_test.rs
│   ├── full_text_search_test.rs
│   ├── phase3_integration_tests.rs
│   ├── wal_recovery_test.rs
│   └── wal_concurrency_test.rs
├── benchmarks/        # Performance validation tests
│   └── wal_performance_test.rs
├── property/          # Property-based tests (future)
├── fixtures/          # Shared test data and utilities
│   └── mod.rs
├── common/            # Common test utilities
│   └── mod.rs
├── unit.rs            # Unit tests harness
├── integration.rs     # Integration tests harness
├── benchmarks.rs      # Benchmark tests harness
└── property.rs        # Property-based tests harness
```

## Running Tests

### Run All Tests
```bash
make test              # All tests (Rust + Python)
cargo test             # All Rust tests
```

### Run by Category
```bash
# Unit tests (fast, < 1s)
cargo test --test unit

# Integration tests (slower, multi-component)
cargo test --test integration

# Benchmarks (performance validation)
cargo test --test benchmarks

# Property-based tests
cargo test --test property
```

### Run Specific Tests
```bash
# Run a specific test file
cargo test --test unit validation_test

# Run a specific test function
cargo test test_vector_validation_config
```

### Run with Features
```bash
# Run full-text search tests
cargo test --test integration --features text-bm25

# Run all tests with all features
cargo test --all-features
```

## Test Categories

### Unit Tests (`tests/unit/`)

**Purpose**: Fast, isolated tests of individual components.

**Characteristics**:
- Execute in < 100ms
- No external dependencies (filesystem, network)
- Test single functions or small modules
- Mock dependencies when needed

**When to add**: When testing individual functions, validation logic, data structures, or algorithms.

**Example**: `validation_test.rs` - Tests vector validation logic

### Integration Tests (`tests/integration/`)

**Purpose**: Verify multi-component interactions and end-to-end workflows.

**Characteristics**:
- May take > 1s to execute
- Can use filesystem, storage backends
- Test complete workflows (CRUD operations, WAL recovery)
- May spawn async tasks

**When to add**: When testing storage operations, API handlers, index building, or cross-component interactions.

**Examples**:
- `crud_consistency_test.rs` - CRUD + full-text search consistency
- `wal_recovery_test.rs` - WAL crash recovery scenarios

### Benchmarks (`tests/benchmarks/`)

**Purpose**: Measure performance and validate against thresholds.

**Characteristics**:
- May take several seconds
- Report latency (P50, P95, P99)
- Report throughput (QPS)
- Assert minimum performance requirements

**When to add**: When validating performance of critical paths (insert, search, index build).

**Example**: `wal_performance_test.rs` - WAL write/recovery performance

### Property-Based Tests (`tests/property/`)

**Purpose**: Verify properties hold across random inputs using `proptest`.

**Characteristics**:
- Generate random test cases
- Run 100-10000 iterations
- Test mathematical properties (commutativity, associativity)
- Compare against reference implementations

**When to add**: When testing core algorithms, distance metrics, or data structures.

**Status**: To be implemented in Phase 2.2

## Shared Utilities

### Fixtures (`tests/fixtures/mod.rs`)

Provides common test data and setup functions:

```rust
use fixtures::*;

// Create test storage
let storage = create_test_storage().await;

// Generate test vectors
let vectors = generate_test_vectors(100, 128);

// Create test records
let records = create_test_records(vectors);

// Use sample datasets
let sample = SiftSample::new_small();
```

### Common (`tests/common/mod.rs`)

Provides test utilities and helpers:

```rust
use common::*;

// Floating point assertions
assertions::assert_float_eq(1.0, 1.0001, 0.001);
assertions::assert_vec_eq(&vec1, &vec2, 0.001);

// Performance measurement
let (result, duration) = perf::measure(|| expensive_operation());
println!("Took {}", perf::format_duration(duration));

// Statistics
let mean = stats::mean(&values);
let p95 = stats::percentile(&sorted_values, 95.0);
```

## Writing New Tests

### 1. Choose the Right Category

- **Unit**: Testing individual functions, validation, data structures
- **Integration**: Testing workflows, storage operations, multi-component interactions
- **Benchmark**: Performance validation, latency/throughput measurement
- **Property**: Mathematical properties, algorithm correctness

### 2. Create Test File

```bash
# Unit test
touch tests/unit/my_feature_test.rs

# Integration test
touch tests/integration/my_workflow_test.rs

# Benchmark
touch tests/benchmarks/my_performance_test.rs

# Property test
touch tests/property/my_property_test.rs
```

### 3. Add to Harness

Edit the appropriate harness file (`unit.rs`, `integration.rs`, etc.):

```rust
mod my_feature_test {
    include!("unit/my_feature_test.rs");
}
```

### 4. Write Test

```rust
/// Test description
use vectordb::types::vector::VectorValidator;
use crate::fixtures::*;
use crate::common::assertions::*;

#[test]
fn test_my_feature() {
    // Arrange
    let vectors = generate_test_vectors(10, 128);

    // Act
    let result = my_function(&vectors);

    // Assert
    assert_eq!(result.len(), 10);
    assert_float_eq(result[0], 1.0, 0.001);
}
```

### 5. Run and Verify

```bash
cargo test --test unit my_feature
```

## Coverage Requirements

- **Overall**: 70% line coverage, 60% branch coverage
- **Critical paths** (`src/storage/`, `src/search/`, `src/types/`): 80% line coverage

Check coverage:
```bash
cargo tarpaulin --out Html --output-dir coverage/
open coverage/index.html
```

## Best Practices

1. **Use shared fixtures**: Reduce code duplication, use `fixtures::create_test_storage()`, `generate_test_vectors()`, etc.

2. **Clear test names**: Use descriptive names like `test_vector_validation_rejects_nan` instead of `test1`

3. **Arrange-Act-Assert**: Structure tests clearly with setup, execution, and verification sections

4. **Async tests**: Use `#[tokio::test]` for async tests

5. **Feature gates**: Use `#[cfg(feature = "...")]` for feature-specific tests

6. **Error messages**: Use assertions with clear messages: `assert!(condition, "Expected X but got {}", actual)`

7. **Test isolation**: Each test should be independent and not rely on execution order

8. **Cleanup**: Tests should clean up resources (though most use in-memory storage)

## CI/CD Integration

Tests run automatically in CI with different presets:

- **Fast**: `cargo test --test unit` (< 5 minutes)
- **Normal**: `cargo test --test unit --test integration` (< 15 minutes)
- **Comprehensive**: `cargo test --all-features --workspace` (< 45 minutes)

## Troubleshooting

### Tests not discovered

Make sure the test file is included in the harness (`unit.rs`, `integration.rs`, etc.)

### Feature-gated tests failing

Run with required features:
```bash
cargo test --test integration --features text-bm25
```

### Slow tests timing out

Increase timeout or move to benchmark category:
```rust
#[tokio::test]
#[timeout(Duration::from_secs(60))]
async fn test_slow_operation() {
    // ...
}
```

## Next Steps

See `docs/TESTING_GUIDE.md` for comprehensive testing methodology and patterns.

Phase roadmap:
- ✅ Phase 1.1: Enhanced Input Validation
- ✅ Phase 1.2: Test Organization Restructure
- 🔄 Phase 1.3: Basic Recall Metrics (in progress)
- 📋 Phase 2: Dataset Integration & Advanced Metrics
- 📋 Phase 3: Performance Thresholds & Documentation
