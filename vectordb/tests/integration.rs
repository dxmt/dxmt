// Integration tests harness
//
// This file allows running all integration tests with: cargo test --test integration
//
// Integration tests verify multi-component interactions.

mod common;
mod fixtures;

// Include integration test modules
#[cfg(feature = "text-bm25")]
mod crud_consistency_test {
    include!("integration/crud_consistency_test.rs");
}

#[cfg(feature = "text-bm25")]
mod full_text_search_test {
    include!("integration/full_text_search_test.rs");
}

#[cfg(feature = "index-management")]
mod phase3_integration_tests {
    include!("integration/phase3_integration_tests.rs");
}

// WAL recovery now lives in tests/wal_recovery_test.rs, rewritten against the current API.
// The files that used to be included here targeted a StorageConfig with enable_wal / wal_dir /
// wal_batch_size / wal_sync_interval_ms / wal_consistency_mode and a WalConsistencyMode type,
// none of which exist any more, so they were unrevivable rather than merely disabled.

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

#[cfg(feature = "benchmarks")]
mod metrics_test {
    include!("integration/metrics_test.rs");
}

mod api_handlers_test {
    include!("integration/api_handlers_test.rs");
}
