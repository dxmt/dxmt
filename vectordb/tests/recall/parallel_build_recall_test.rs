//! Recall of a table built through the parallel batch path, end to end.
//!
//! Every storage-level recall assertion in this repository runs on at most 6,000 rows, and
//! `HnswIndex::prepare_batch` falls back to inserting one record at a time below
//! `PARALLEL_PLAN_MIN_NODES` — 10,000. So the build every production table above that size
//! uses, where a whole batch plans its edges against the graph as it stood *before* any of them
//! were added, had no end-to-end guard at all. `hnsw::batch_insert_tests` compares a bulk build
//! with a serial one at 12,000 nodes, which is the same question asked of the index alone; this
//! asks it of the thing the server runs: records on disk, vectors in a row file, the search
//! path that reads them back.
//!
//! Ground truth is this server's own `SearchMode::Exact`, which scores every live row and skips
//! the index entirely. That is what makes the test cheap enough to keep in the default gate:
//! no external dataset, no published ground truth, and the comparison is against the same
//! distance function the ranking uses.
//!
//! What it fails on: a change to the batch planner that links the wrong neighbours, an entry
//! point that stops being shared across a batch, a reverse-edge pass that prunes too hard. What
//! it deliberately does not do is measure absolute recall on a real corpus -- that is the
//! benchmark harness's job, and the numbers in README.md come from there.

use std::path::Path;

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{CreateRecordRequest, SearchMode, SearchQuery},
        table::TableConfig,
        vector::DistanceMetric,
    },
};
use tempfile::TempDir;

const DIM: usize = 48;
/// Past `PARALLEL_PLAN_MIN_NODES` by enough that most of the corpus arrives through the
/// parallel path rather than the serial warm-up.
const ROWS: usize = 15_000;
const BATCH: usize = 1_000;
const QUERIES: usize = 25;
const K: usize = 10;

/// Measured on this corpus: normal 0.984, accurate 0.992, fast 0.780. The floor leaves room
/// for the tie-breaking a different rayon thread count can change, and nothing else.
const FLOOR: f32 = 0.95;

fn config() -> StorageConfig {
    StorageConfig {
        memory: Some(MemoryConfig {
            max_size_mb: 512,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    }
}

/// Deterministic and clustered. A uniformly random corpus is the hardest case for any graph
/// index and the least like an embedding, so recall on one says little about a regression.
fn vector(i: usize) -> Vec<f32> {
    let cluster = (i % 60) as f32;
    (0..DIM)
        .map(|d| {
            let mut h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (d as u64 + 1);
            h ^= h >> 29;
            h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h ^= h >> 32;
            let jitter = (h & 0xFFFF) as f32 / 65_536.0 - 0.5;
            (cluster * 0.29 + d as f32 * 0.07).sin() + jitter * 0.2
        })
        .collect()
}

async fn open(wal: &Path) -> IntelligentIndexedStorage {
    IntelligentIndexedStorage::new_with_wal(&config(), Some(wal.to_path_buf()))
        .await
        .unwrap()
}

async fn ids_for(
    storage: &IntelligentIndexedStorage,
    query: &[f32],
    mode: SearchMode,
) -> Vec<String> {
    storage
        .search(
            "t",
            SearchQuery {
                vector: query.to_vec(),
                top_k: K,
                limit: K,
                search_mode: mode,
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .hits
        .into_iter()
        .map(|hit| hit.id)
        .collect()
}

#[tokio::test]
async fn a_table_built_in_parallel_batches_finds_what_an_exact_scan_finds() {
    // Row files, which is what a server with a WAL directory gives every table: the search has
    // to read each candidate's vector back to score it, and a recall test that kept everything
    // in memory would not exercise that.
    std::env::set_var("DXMT_VECTOR_STORE", "disk");
    std::env::set_var("DXMT_RECORD_STORE", "disk");
    let tmp = TempDir::new().unwrap();
    let storage = open(&tmp.path().join("wal")).await;
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
            // A deliberately lean graph. The shipped defaults answer this corpus at recall
            // 1.000, and a test whose answer is 1.000 cannot show a build getting worse -- it
            // can only show one getting catastrophic. At m=4 the recall sits just below 1 and
            // has somewhere to fall, and the build takes 4 s instead of 12.
            hnsw: Some(dxmt_vectordb::types::table::HnswParams {
                m: Some(4),
                ef_construction: Some(16),
                ef_search: None,
            }),
            quantization: None,
            payload_index: None,
            embedding: None,
            rerank: None,
        })
        .await
        .unwrap();

    for start in (0..ROWS).step_by(BATCH) {
        let records: Vec<CreateRecordRequest> = (start..(start + BATCH).min(ROWS))
            .map(|i| CreateRecordRequest {
                id: i.to_string(),
                vector: vector(i),
                sparse_vector: None,
                named_vectors: None,
                payload: None,
                text: None,
            })
            .collect();
        storage.insert_records("t", records).await.unwrap();
    }

    // Queries near a record rather than the record itself: an exact hit on a stored vector is a
    // weaker question, since the graph only has to return the node it starts from.
    let mut normal_hits = 0usize;
    let mut accurate_hits = 0usize;
    let mut fast_hits = 0usize;
    let mut expected = 0usize;
    for q in 0..QUERIES {
        let base = (q * 601) % ROWS;
        let mut query = vector(base);
        for (d, value) in query.iter_mut().enumerate() {
            *value += ((d + q) as f32 * 0.013).sin() * 0.05;
        }

        let truth: std::collections::HashSet<String> = ids_for(&storage, &query, SearchMode::Exact)
            .await
            .into_iter()
            .collect();
        assert_eq!(
            truth.len(),
            K,
            "the exact scan returned {} ids",
            truth.len()
        );
        expected += K;
        normal_hits += ids_for(&storage, &query, SearchMode::Normal)
            .await
            .into_iter()
            .filter(|id| truth.contains(id))
            .count();
        fast_hits += ids_for(&storage, &query, SearchMode::Fast)
            .await
            .into_iter()
            .filter(|id| truth.contains(id))
            .count();
        accurate_hits += ids_for(&storage, &query, SearchMode::Accurate)
            .await
            .into_iter()
            .filter(|id| truth.contains(id))
            .count();
    }

    let normal = normal_hits as f32 / expected as f32;
    let accurate = accurate_hits as f32 / expected as f32;
    let fast = fast_hits as f32 / expected as f32;
    println!(
        "recall@{K} over {ROWS} rows: normal {normal:.3}, accurate {accurate:.3}, fast {fast:.3}"
    );
    assert!(
        normal >= FLOOR,
        "recall@{K} is {normal:.3} on a table built through the parallel batch path, below the \
         floor of {FLOOR}. The graph that build produces has got worse: a batch plans its edges \
         against the graph as it stood before the batch arrived, and this is the only test that \
         asks what that costs end to end."
    );
    // Asking harder must not answer worse. It has been the other way round before: a mode that
    // scaled `ef_search` the wrong way looked fine on averages and lost recall where it counted.
    assert!(
        accurate >= normal - 0.01,
        "the accurate mode ({accurate:.3}) is below the normal one ({normal:.3})"
    );
    // And the harness has to be able to see a worse search at all. If `fast` -- which is the
    // same graph with a smaller traversal budget -- scores as well as `normal`, then this
    // corpus has stopped discriminating and the floor above would pass whatever the build did.
    // A test that cannot fail is worse than no test, so it fails here instead.
    assert!(
        fast < normal,
        "fast ({fast:.3}) matched normal ({normal:.3}): this corpus no longer tells a weaker \
         search from a stronger one, so it cannot tell a worse graph from a better one either. \
         Make it harder -- more dimensions, or a leaner graph -- rather than deleting this line."
    );
}
