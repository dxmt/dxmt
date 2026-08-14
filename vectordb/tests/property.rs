// Property-based tests harness
//
// This file allows running all property-based tests with: cargo test --test property
//
// Property-based tests use proptest to verify properties hold across random inputs.
//
// Control test cases with PROPTEST_CASES environment variable:
//   PROPTEST_CASES=50 cargo test --test property     # Fast (default for vector tests)
//   PROPTEST_CASES=100 cargo test --test property    # Normal
//   PROPTEST_CASES=1000 cargo test --test property   # Comprehensive

mod common;
mod fixtures;

// Vector and distance metric properties
mod vector_properties {
    include!("property/vector_properties.rs");
}

// Search operation properties
mod search_properties {
    include!("property/search_properties.rs");
}
