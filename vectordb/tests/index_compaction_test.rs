//! An upsert-heavy table gets its graph rebuilt, and answers the same afterwards.
//!
//! An HNSW node cannot move: where it sits is decided by the vector it had when it was linked
//! in. So an update deletes the old slot and appends a new one, and nothing gives the slot
//! back. A table of constant size under steady upserts therefore grows a slot per write, with
//! no bound at all -- measured in a 17-hour soak at 200 rows/s over a 400,000-row key space:
//!
//! * 11.1 million slots holding 400,000 records
//! * `index_size_bytes` 5.3 GB where the live rows need about 200 MB
//! * resident memory climbing 8.4 GB a day, with the row count flat for sixteen of those hours
//! * snapshots from 1.2 s to 18.9 s, and 196 s to come back from `kill -9`
//!
//! The re-evaluation sweep now compares live entries against slots and rebuilds when the gap is
//! worth an O(n) pass. This asserts both halves: that the reclamation happens, and that the
//! table still holds every record and returns the same neighbours afterwards -- a compaction
//! that loses a row is worse than the leak it fixes.

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{CreateRecordRequest, SearchQuery},
        table::TableConfig,
        vector::DistanceMetric,
    },
};
use tempfile::TempDir;

const ROWS: usize = 500;
const DIM: usize = 64;
/// Ten passes over 500 rows leaves 5,000 slots for 500 live entries: ten times the garbage,
/// well past the one-in-two the sweep acts on.
const PASSES: usize = 10;
/// The sweep's floor, lowered from its 100,000-slot default. What is under test is the ratio
/// and what the rebuild preserves, neither of which cares about the size of the constant.
const MIN_SLOTS: &str = "1000";

fn config() -> StorageConfig {
    StorageConfig {
        r#type: "memory".to_string(),
        memory: Some(MemoryConfig {
            max_size_mb: 4096,
            eviction_policy: "lru".to_string(),
        }),
        ..Default::default()
    }
}

/// The vector for row `i`, fixed for the life of the table: a compaction that lost or
/// mismatched a row shows up as a changed answer, not as a nearby one.
fn vector(i: usize) -> Vec<f32> {
    let mut state = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..DIM)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 16_777_216.0 - 0.5
        })
        .collect()
}

async fn write_pass(storage: &IntelligentIndexedStorage, pass: usize) {
    let batch: Vec<CreateRecordRequest> = (0..ROWS)
        .map(|i| CreateRecordRequest {
            id: i.to_string(),
            // The same vector every pass: only the payload changes, which is the shape of a
            // real upsert workload and keeps the expected answers fixed.
            vector: vector(i),
            sparse_vector: None,
            named_vectors: None,
            payload: Some(serde_json::json!({ "pass": pass })),
            text: None,
        })
        .collect();
    storage.insert_records("t", batch).await.expect("insert");
}

async fn index_bytes(storage: &IntelligentIndexedStorage) -> u64 {
    let stats = storage.get_table_stats("t").await.expect("stats");
    stats
        .get("index_size_bytes")
        .and_then(|v| v.as_u64())
        .expect("index_size_bytes")
}

async fn nearest(storage: &IntelligentIndexedStorage, i: usize) -> String {
    storage
        .search(
            "t",
            SearchQuery {
                vector: vector(i),
                top_k: 1,
                limit: 1,
                ..Default::default()
            },
        )
        .await
        .expect("search")
        .hits
        .first()
        .expect("a hit")
        .id
        .clone()
}

#[tokio::test]
async fn upserts_leave_tombstones_and_the_sweep_rebuilds_them_away() {
    std::env::set_var("DXMT_INDEX_COMPACTION_MIN_SLOTS", MIN_SLOTS);
    let tmp = TempDir::new().expect("tempdir");
    let storage = IntelligentIndexedStorage::new_with_wal(&config(), Some(tmp.path().join("wal")))
        .await
        .expect("storage");
    storage
        .create_table(TableConfig {
            name: "t".to_string(),
            dimension: DIM,
            distance_metric: DistanceMetric::Cosine,
            description: None,
            index_type: None,
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
            hnsw: None,
            quantization: None,
            payload_index: None,
            embedding: None,
        })
        .await
        .expect("create table");

    write_pass(&storage, 0).await;
    let after_first_pass = index_bytes(&storage).await;
    for pass in 1..PASSES {
        write_pass(&storage, pass).await;
    }
    let bloated = index_bytes(&storage).await;

    // Without this the test could pass on a table that never accumulated anything.
    assert!(
        bloated > after_first_pass * 3,
        "index grew from {after_first_pass} to {bloated} bytes over {PASSES} passes; the \
         tombstones this test is about are not there"
    );

    let before: Vec<String> = [0usize, 1, ROWS / 2, ROWS - 1]
        .iter()
        .map(|&i| i.to_string())
        .collect();
    for id in &before {
        assert_eq!(
            &nearest(&storage, id.parse().expect("id")).await,
            id,
            "own vector should rank first before the rebuild"
        );
    }

    let rebuilt = storage
        .reevaluate_indexes()
        .await
        .expect("reevaluate_indexes");
    assert!(
        rebuilt.contains(&"t".to_string()),
        "the sweep did not rebuild a table with ten times more slots than entries: {rebuilt:?}"
    );

    let compacted = index_bytes(&storage).await;
    assert!(
        compacted * 2 < bloated,
        "index was {bloated} bytes before the rebuild and {compacted} after; the tombstones \
         were not reclaimed"
    );

    // Everything the table held, it still holds, and it answers the same.
    let table = storage.get_table("t").await.expect("get_table");
    assert_eq!(table.record_count, ROWS as u64, "every record survived");
    for i in 0..ROWS {
        let record = storage
            .get_record("t", &i.to_string())
            .await
            .unwrap_or_else(|e| panic!("record {i} is gone after the rebuild: {e}"));
        assert_eq!(record.vector, vector(i), "record {i} came back changed");
    }
    for id in &before {
        assert_eq!(
            &nearest(&storage, id.parse().expect("id")).await,
            id,
            "own vector should still rank first after the rebuild"
        );
    }

    // And it does not rebuild again on the next sweep: the garbage is gone, so the trigger
    // must not fire. A sweep that rebuilds every time costs O(n) every time.
    let again = storage
        .reevaluate_indexes()
        .await
        .expect("second reevaluate_indexes");
    assert!(
        !again.contains(&"t".to_string()),
        "the sweep rebuilt a freshly compacted table: {again:?}"
    );
}
