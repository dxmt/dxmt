//! Round-trip tests for on-disk indexes.
//!
//! The property that matters is not "the file parses" but "the reloaded index answers the same
//! queries". An index can deserialize into a structurally valid object and still return
//! different neighbours — a dropped entry point, a codebook that did not travel with its codes.
//! So every test here compares search results before and after, not field values.

use dxmt_vectordb::search::{
    hnsw::{HnswConfig, HnswIndex},
    hnsw_pq::{HnswPqConfig, HnswPqIndex},
    index_traits::{SerializableIndex, VectorIndex},
};
use dxmt_vectordb::types::vector::DistanceMetric;
use tempfile::TempDir;

fn vectors(count: usize, dim: usize) -> Vec<(String, Vec<f32>)> {
    (0..count)
        .map(|i| {
            let v = (0..dim)
                .map(|d| ((i * 31 + d * 17) % 97) as f32 / 97.0)
                .collect();
            (format!("v{}", i), v)
        })
        .collect()
}

#[tokio::test]
async fn hnsw_returns_the_same_neighbours_after_a_reload() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("hnsw.idx");
    let data = vectors(200, 16);

    let mut index = HnswIndex::new(DistanceMetric::Euclidean, Some(HnswConfig::default()));
    for (id, v) in &data {
        index.add(id.clone(), v.clone()).await.unwrap();
    }

    let query = data[42].1.clone();
    let before = VectorIndex::search(&index, &query, 10).await.unwrap();
    index.save(&path).unwrap();

    let reloaded = HnswIndex::load(&path).unwrap();
    let after = VectorIndex::search(&reloaded, &query, 10).await.unwrap();

    assert_eq!(before.len(), after.len());
    for (b, a) in before.iter().zip(after.iter()) {
        assert_eq!(b.0, a.0, "neighbour order changed across save/load");
        assert!(
            (b.1 - a.1).abs() < f32::EPSILON,
            "distance changed across save/load: {} vs {}",
            b.1,
            a.1
        );
    }
}

#[tokio::test]
async fn hnsw_keeps_deletions_across_a_reload() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("hnsw.idx");

    let mut index = HnswIndex::new(DistanceMetric::Cosine, None);
    for (id, v) in vectors(50, 8) {
        index.add(id, v).await.unwrap();
    }
    index.delete("v7").await.unwrap();
    index.save(&path).unwrap();

    // Deletion is a tombstone on the node, not a removal from the vector. If the flag did not
    // survive serialization the record would come back from the dead on the next restart.
    let reloaded = HnswIndex::load(&path).unwrap();
    let hits = VectorIndex::search(&reloaded, &vectors(50, 8)[7].1, 50)
        .await
        .unwrap();
    assert!(
        !hits.iter().any(|(id, _)| id == "v7"),
        "a deleted record reappeared after reload"
    );
}

#[tokio::test]
async fn hnsw_pq_carries_its_codebooks() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("hnswpq.idx");
    let dim = 16;
    let data = vectors(300, dim);

    let config = HnswPqConfig {
        num_subvectors: 4,
        num_centroids: 16,
        enable_reranking: false,
        ..Default::default()
    };
    let mut index = HnswPqIndex::new(dim, DistanceMetric::Euclidean, Some(config)).unwrap();
    for (id, v) in &data {
        index.add(id.clone(), v.clone()).await.unwrap();
    }

    let query = data[100].1.clone();
    let before = VectorIndex::search(&index, &query, 5).await.unwrap();
    index.save(&path).unwrap();

    // PQ codes are indices into the codebooks. If the codebooks are not in the file, every
    // reloaded distance is computed against a different table and is silently wrong.
    let reloaded = HnswPqIndex::load(&path).unwrap();
    let after = VectorIndex::search(&reloaded, &query, 5).await.unwrap();

    assert_eq!(
        before.len(),
        after.len(),
        "reloaded index returned a different result count"
    );
    let before_ids: Vec<_> = before.iter().map(|(id, _)| id.as_str()).collect();
    let after_ids: Vec<_> = after.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(
        before_ids, after_ids,
        "PQ neighbours changed across save/load"
    );
}

#[tokio::test]
async fn an_hnsw_file_is_refused_as_hnsw_pq() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("mixed.idx");

    let mut index = HnswIndex::new(DistanceMetric::Cosine, None);
    index.add("a".into(), vec![1.0, 0.0]).await.unwrap();
    index.save(&path).unwrap();

    // Loading the wrong index type has to fail loudly. The alternative — bincode happening to
    // parse the bytes into a valid-looking HnswPqIndex — is a corrupted index nobody notices.
    let err = match HnswPqIndex::load(&path) {
        Ok(_) => panic!("an hnsw file loaded as hnsw-pq"),
        Err(e) => e.to_string(),
    };
    // "hnsw-v2": the kind was renamed when cosine vectors started being stored unit length, so
    // that a v1 file is refused rather than scored against normalised queries.
    assert!(
        err.contains("holds a 'hnsw-v2' index"),
        "expected a kind mismatch, got: {}",
        err
    );
}

/// A snapshot written before the record map became `Arc`-shared must still be readable.
///
/// The map holds `Arc<Record>` so a snapshot can copy handles instead of records -- 10M of
/// them were 90 GB of copy on top of 90 GB resident, and the process was OOM-killed for it
/// (design doc section 34.3). That is only safe if the bytes are unchanged. `Record` has a
/// hand-written `Serialize` that puts the payload through a JSON string, and serde's `rc`
/// feature serialises an `Arc<T>` by delegating to that same impl -- so they should agree.
/// Asserted rather than assumed, because the alternative is a snapshot nobody can restore.
#[test]
fn an_arc_serialises_to_the_same_bytes_as_the_record_inside_it() {
    use dxmt_vectordb::types::record::{Record, RecordStatus};
    use std::sync::Arc;

    let record = Record {
        id: "r1".to_string(),
        vector: vec![0.5, -0.25, 0.125, 0.0],
        sparse_vector: None,
        named_vectors: None,
        payload: serde_json::json!({"bucket": 3, "name": "x"}),
        created_at: chrono::DateTime::parse_from_rfc3339("2026-01-01T00:00:00Z")
            .unwrap()
            .into(),
        updated_at: chrono::DateTime::parse_from_rfc3339("2026-01-02T03:04:05Z")
            .unwrap()
            .into(),
        version: 7,
        status: RecordStatus::Active,
    };

    let plain = bincode::serialize(&vec![record.clone()]).expect("Vec<Record>");
    let shared = bincode::serialize(&vec![Arc::new(record)]).expect("Vec<Arc<Record>>");
    assert_eq!(
        plain, shared,
        "Arc<Record> must serialise to the same bytes as Record, or every snapshot written \
         before this change becomes unreadable"
    );
}
