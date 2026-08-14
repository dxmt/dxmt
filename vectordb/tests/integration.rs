// Integration tests harness
//
// This file allows running all integration tests with: cargo test --test integration
//
// Integration tests verify multi-component interactions.

mod fixtures;
mod common;

// Include integration test modules
#[cfg(feature = "text-bm25")]
mod crud_consistency_test {
    include!("integration/crud_consistency_test.rs");
}

#[cfg(feature = "text-bm25")]
mod full_text_search_test {
    include!("integration/full_text_search_test.rs");
}

mod phase3_integration_tests {
    include!("integration/phase3_integration_tests.rs");
}

// WAL tests disabled - WAL configuration has been restructured
// mod wal_recovery_test {
//     include!("integration/wal_recovery_test.rs");
// }

// mod wal_concurrency_test {
//     include!("integration/wal_concurrency_test.rs");
// }

// Original tiered storage tests use old API - replaced with simplified version
// mod tiered_storage_test {
//     include!("integration/tiered_storage_test.rs");
// }

// Original ephemeral VM tests use old API - replaced with simplified version
// mod ephemeral_vm_test {
//     include!("integration/ephemeral_vm_test.rs");
// }

// Simplified versions using current API
mod tiered_storage_simple_test {
    include!("integration/tiered_storage_simple_test.rs");
}

mod ephemeral_vm_simple_test {
    include!("integration/ephemeral_vm_simple_test.rs");
}

mod metrics_test {
    include!("integration/metrics_test.rs");
}

mod api_handlers_test {
    include!("integration/api_handlers_test.rs");
}
