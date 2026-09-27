//! An upsert-heavy table gives its disk back, and every record still reads correctly.
//!
//! Row files are append-only: an update appends a new row and repoints the id, and the row it
//! left behind keeps its blocks. Nothing reclaimed them, and the only thing that claimed to --
//! a comment in `vector_file.rs` about a rebuild rewriting the file -- described code that did
//! not exist. A soak at 200 rows/s over a fixed 400,000-row key space grew the vector file to
//! 5.2 GB holding 1.2 GB of live rows in 2.4 hours; the growth tracked the number of writes,
//! not the size of the table, so it had no bound.
//!
//! The two things worth asserting are in tension, so both are here: that the space actually
//! comes back, and that reclaiming it did not disturb a single live row. The second is the
//! one that would make this feature worse than the bug.

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{CreateRecordRequest, SearchQuery},
        table::TableConfig,
        vector::DistanceMetric,
    },
};
use std::path::{Path, PathBuf};
use tempfile::TempDir;

const ROWS: usize = 1_000;
const DIM: usize = 128;
const PASSES: usize = 8;
/// The sweep's floor, lowered from its 64 MB default: reaching 64 MB of dead rows honestly
/// costs tens of thousands of graph insertions in a debug build, which is minutes of test time
/// spent on the size of a constant rather than on the behaviour. Eight passes of 1,000 rows at
/// 128 dimensions leave 3.6 MB dead, comfortably over this.
const RECLAIM_FLOOR: &str = "1048576";

fn config() -> StorageConfig {
    StorageConfig {
        memory: Some(MemoryConfig {
            max_size_mb: 4096,
            eviction_policy: "lru".to_string(),
        }),
        ..Default::default()
    }
}

/// Vector for row `i` on pass `p`. Every pass moves every vector, so no row's bytes can be
/// mistaken for another's and a stale row is visible as a wrong answer rather than a near one.
fn vector(i: usize, pass: usize) -> Vec<f32> {
    let mut state = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (pass as u64).wrapping_mul(31);
    (0..DIM)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 16_777_216.0 - 0.5
        })
        .collect()
}

fn vector_file(wal: &Path) -> PathBuf {
    std::fs::read_dir(wal.join("vectors"))
        .expect("a vectors directory")
        .map(|e| e.expect("entry").path())
        .find(|p| p.extension().is_some_and(|e| e == "f32"))
        .expect("a .f32 row file")
}

/// Blocks the filesystem is actually holding for `path`. `metadata().len()` is the apparent
/// length, which a punched file keeps.
fn allocated(path: &Path) -> u64 {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata(path)
        .map(|m| m.blocks() * 512)
        .unwrap_or(0)
}

async fn write_pass(storage: &IntelligentIndexedStorage, pass: usize) {
    for chunk in (0..ROWS).collect::<Vec<_>>().chunks(1_000) {
        let batch: Vec<CreateRecordRequest> = chunk
            .iter()
            .map(|&i| CreateRecordRequest {
                id: i.to_string(),
                vector: vector(i, pass),
                sparse_vector: None,
                named_vectors: None,
                payload: Some(serde_json::json!({ "pass": pass })),
                text: None,
            })
            .collect();
        storage.insert_records("t", batch).await.expect("insert");
    }
}

#[tokio::test]
async fn upserts_give_their_disk_back_and_every_live_row_survives() {
    std::env::set_var("DXMT_VECTOR_STORE", "disk");
    std::env::set_var("DXMT_RECORD_STORE", "disk");
    std::env::set_var("DXMT_ROW_RECLAIM_MIN_BYTES", RECLAIM_FLOOR);
    let tmp = TempDir::new().expect("tempdir");
    let wal = tmp.path().join("wal");

    let storage = IntelligentIndexedStorage::new_with_wal(&config(), Some(wal.clone()))
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

    for pass in 0..PASSES {
        write_pass(&storage, pass).await;
    }

    let path = vector_file(&wal);
    let live_bytes = (ROWS * DIM * 4) as u64;
    let apparent = std::fs::metadata(&path).expect("metadata").len();
    assert_eq!(
        apparent,
        live_bytes * PASSES as u64,
        "the file holds a row per write, not a row per record -- that is the bug"
    );
    let before = allocated(&path);
    assert!(
        before >= live_bytes * PASSES as u64,
        "{before} bytes allocated for {PASSES} passes over {live_bytes} bytes of live rows"
    );

    // What every record should read back as, taken before the reclamation so the comparison
    // is against the answers the table gave while its dead rows were still there.
    let last = PASSES - 1;
    let expected: Vec<(String, Vec<f32>)> = (0..ROWS)
        .map(|i| (i.to_string(), vector(i, last)))
        .collect();

    let reclaimed = storage.reclaim_dead_rows().await.expect("reclaim");
    let after = allocated(&path);

    assert!(
        !reclaimed.is_empty(),
        "nothing was reclaimed from a file holding {} passes of dead rows",
        PASSES - 1
    );
    assert_eq!(
        std::fs::metadata(&path).expect("metadata").len(),
        apparent,
        "punching a hole must not change the file's length: every row keeps its number"
    );
    assert!(
        after < before / 2,
        "allocated bytes {before} -> {after}, expected most of it back"
    );

    // The live rows are untouched: same vector, byte for byte, for every id.
    for (id, want) in &expected {
        let record = storage
            .get_record("t", id)
            .await
            .unwrap_or_else(|e| panic!("record {id} is gone after reclamation: {e}"));
        assert_eq!(
            &record.vector, want,
            "record {id} came back with a different vector after reclamation"
        );
    }

    // And search still finds each record by its own vector, which is the path that reads the
    // row file rather than the record: the rerank.
    for &i in &[0usize, 7, ROWS / 2, ROWS - 1] {
        let hits = storage
            .search(
                "t",
                SearchQuery {
                    vector: vector(i, last),
                    top_k: 1,
                    limit: 1,
                    ..Default::default()
                },
            )
            .await
            .expect("search")
            .hits;
        assert_eq!(hits[0].id, i.to_string(), "own vector should rank first");
    }

    // Running it again releases nothing and breaks nothing: the second sweep finds the same
    // rows dead and already punched, and must not report their bytes a second time.
    let again = storage.reclaim_dead_rows().await.expect("second reclaim");
    assert!(
        again.is_empty(),
        "a second sweep reported {again:?} from a file it already punched"
    );
}

#[tokio::test]
async fn a_reclaimed_table_still_recovers_from_its_row_files() {
    std::env::set_var("DXMT_VECTOR_STORE", "disk");
    std::env::set_var("DXMT_RECORD_STORE", "disk");
    std::env::set_var("DXMT_ROW_RECLAIM_MIN_BYTES", RECLAIM_FLOOR);
    let tmp = TempDir::new().expect("tempdir");
    let wal = tmp.path().join("wal");

    {
        let storage = IntelligentIndexedStorage::new_with_wal(&config(), Some(wal.clone()))
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
        for pass in 0..PASSES {
            write_pass(&storage, pass).await;
        }
        storage.reclaim_dead_rows().await.expect("reclaim");
        storage.snapshot().await.expect("snapshot");
    }

    // A restart rebuilds the id-to-row map from the `.rows` sidecar, which still names every
    // row including the punched ones. The ids must resolve to the rows that are still there.
    let storage = IntelligentIndexedStorage::new_with_wal(&config(), Some(wal.clone()))
        .await
        .expect("reopen");
    let last = PASSES - 1;
    for i in [0usize, 1, ROWS / 2, ROWS - 1] {
        let id = i.to_string();
        let record = storage
            .get_record("t", &id)
            .await
            .unwrap_or_else(|e| panic!("record {id} did not survive the restart: {e}"));
        assert_eq!(
            record.vector,
            vector(i, last),
            "record {id} came back from disk with the wrong vector"
        );
    }
}
