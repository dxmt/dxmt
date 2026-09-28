//! Tests for the partition-on-demand IVF index.
//!
//! The point of this index is that most of it is *not* in memory, so the tests assert on
//! residency as much as on results. An index that quietly kept everything resident would pass a
//! correctness-only suite while delivering none of the benefit.

use dxmt_vectordb::search::index_traits::{SerializableIndex, VectorIndex};
use dxmt_vectordb::search::partitioned_ivf::{PartitionedIvfConfig, PartitionedIvfIndex};
use dxmt_vectordb::storage::object::filesystem::FilesystemStorage;
use dxmt_vectordb::storage::object::traits::ObjectStorage;
use dxmt_vectordb::types::vector::DistanceMetric;
use std::sync::Arc;
use tempfile::TempDir;

fn vectors(count: usize, dim: usize) -> Vec<(String, Vec<f32>)> {
    (0..count)
        .map(|i| {
            let v = (0..dim)
                .map(|d| (((i * 7919 + d * 104729) % 1000) as f32) / 1000.0)
                .collect();
            (format!("v{}", i), v)
        })
        .collect()
}

fn exact_nearest(data: &[(String, Vec<f32>)], query: &[f32], k: usize) -> Vec<String> {
    let mut scored: Vec<_> = data
        .iter()
        .map(|(id, v)| {
            let d: f32 = v.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum();
            (id.clone(), d)
        })
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

fn store(dir: &TempDir) -> Arc<dyn ObjectStorage + Send + Sync> {
    Arc::new(FilesystemStorage::new(dir.path(), "bucket").unwrap())
}

#[tokio::test]
async fn partitions_are_not_resident_until_a_query_probes_them() {
    let dir = TempDir::new().unwrap();
    let data = vectors(400, 16);

    let mut index = PartitionedIvfIndex::new(
        DistanceMetric::Euclidean,
        "t",
        Some(PartitionedIvfConfig {
            nlist: 16,
            nprobe: 3,
            max_resident_partitions: 4,
            max_iterations: 10,
            ..Default::default()
        }),
    )
    .with_store(store(&dir));

    index.rebuild(data.clone()).await.unwrap();
    assert_eq!(index.partition_count(), 16);
    assert_eq!(
        index.resident_partition_count().await,
        0,
        "a freshly built index must hold nothing but centroids"
    );

    index.search(&data[0].1, 5).await.unwrap();
    let resident = index.resident_partition_count().await;
    assert!(resident > 0, "a query loaded nothing");
    assert!(
        resident <= 3,
        "a query loaded {} partitions but nprobe is 3",
        resident
    );
}

#[tokio::test]
async fn resident_partitions_stay_under_the_cap() {
    let dir = TempDir::new().unwrap();
    let data = vectors(600, 8);

    let mut index = PartitionedIvfIndex::new(
        DistanceMetric::Euclidean,
        "t",
        Some(PartitionedIvfConfig {
            nlist: 24,
            nprobe: 4,
            max_resident_partitions: 6,
            max_iterations: 10,
            ..Default::default()
        }),
    )
    .with_store(store(&dir));
    index.rebuild(data.clone()).await.unwrap();

    // Query all over the space so the probe sets differ and the cache is forced to turn over.
    for (_, v) in data.iter().step_by(37) {
        index.search(v, 5).await.unwrap();
    }

    assert!(
        index.resident_partition_count().await <= 6,
        "resident set grew to {}, cap is 6 — memory is not bounded",
        index.resident_partition_count().await
    );
}

#[tokio::test]
async fn probing_every_partition_matches_an_exact_scan() {
    let dir = TempDir::new().unwrap();
    let data = vectors(300, 12);

    // nprobe == nlist removes the approximation, so any difference from a brute-force scan is a
    // bug in partitioning or distance, not the expected recall loss of IVF.
    let mut index = PartitionedIvfIndex::new(
        DistanceMetric::Euclidean,
        "t",
        Some(PartitionedIvfConfig {
            nlist: 8,
            nprobe: 8,
            max_resident_partitions: 8,
            max_iterations: 15,
            ..Default::default()
        }),
    )
    .with_store(store(&dir));
    index.rebuild(data.clone()).await.unwrap();

    for (_, query) in data.iter().step_by(53) {
        let got: Vec<String> = index
            .search(query, 5)
            .await
            .unwrap()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(
            got,
            exact_nearest(&data, query, 5),
            "exhaustive probe missed a neighbour"
        );
    }
}

#[tokio::test]
async fn writes_after_a_rebuild_are_searchable_immediately() {
    let dir = TempDir::new().unwrap();
    let data = vectors(200, 8);

    let mut index =
        PartitionedIvfIndex::new(DistanceMetric::Euclidean, "t", None).with_store(store(&dir));
    index.rebuild(data).await.unwrap();

    // A partitioned table must stay writable. The new vector lives only in the overflow buffer
    // until the next rebuild, so search has to scan that too.
    index.add("fresh".into(), vec![9.0; 8]).await.unwrap();
    let hits = index.search(&[9.0; 8], 1).await.unwrap();
    assert_eq!(hits[0].0, "fresh", "a just-written vector was invisible");
}

#[tokio::test]
async fn deletes_apply_to_partitions_that_are_not_even_loaded() {
    let dir = TempDir::new().unwrap();
    let data = vectors(200, 8);

    let mut index = PartitionedIvfIndex::new(
        DistanceMetric::Euclidean,
        "t",
        Some(PartitionedIvfConfig {
            nlist: 8,
            nprobe: 8,
            max_resident_partitions: 8,
            max_iterations: 10,
            ..Default::default()
        }),
    )
    .with_store(store(&dir));
    index.rebuild(data.clone()).await.unwrap();

    let target = &data[42];
    index.delete(&target.0).await.unwrap();

    let hits = index.search(&target.1, 5).await.unwrap();
    assert!(
        !hits.iter().any(|(id, _)| id == &target.0),
        "a deleted vector came back from a segment"
    );
}

#[tokio::test]
async fn saving_the_index_does_not_write_the_partitions_again() {
    let dir = TempDir::new().unwrap();
    let out = TempDir::new().unwrap();
    let data = vectors(500, 32);

    let mut index =
        PartitionedIvfIndex::new(DistanceMetric::Euclidean, "t", None).with_store(store(&dir));
    index.rebuild(data).await.unwrap();

    let path = out.path().join("index.bin");
    index.save(&path).unwrap();
    let size = std::fs::metadata(&path).unwrap().len();

    // 500 vectors × 32 dims × 4 bytes is 64 KB. The saved index holds centroids and metadata,
    // not vectors — if it were duplicating the segments this would be at least that large.
    assert!(
        size < 32 * 1024,
        "index file is {} bytes; it looks like the partitions were written into it",
        size
    );

    let reloaded = PartitionedIvfIndex::load(&path).unwrap();
    assert_eq!(reloaded.partition_count(), index.partition_count());
}

fn maintained(nlist: usize, split: usize, merge: usize) -> PartitionedIvfConfig {
    PartitionedIvfConfig {
        nlist,
        nprobe: nlist,
        max_resident_partitions: nlist,
        max_iterations: 10,
        split_threshold: split,
        merge_threshold: merge,
    }
}

#[tokio::test]
async fn maintenance_rewrites_only_the_partitions_it_touched() {
    let dir = TempDir::new().unwrap();
    let data = vectors(600, 8);

    let mut index = PartitionedIvfIndex::new(
        DistanceMetric::Euclidean,
        "t",
        Some(maintained(24, 1000, 0)),
    )
    .with_store(store(&dir));
    index.rebuild(data.clone()).await.unwrap();
    let partitions = index.partition_count();
    assert!(
        partitions >= 20,
        "expected a spread of partitions, got {}",
        partitions
    );

    // Three vectors, clustered together so they land in the same one or two partitions.
    for i in 0..3 {
        index
            .add(format!("late{}", i), vec![0.5 + i as f32 * 0.001; 8])
            .await
            .unwrap();
    }

    let report = index.maintain().await.unwrap();

    assert_eq!(report.absorbed, 3);
    // The point of incremental maintenance: cost tracks the new data, not the total. A full
    // rebuild would rewrite all 24 segments and re-run k-means over 600 vectors.
    assert!(
        report.rewritten <= 3,
        "maintenance rewrote {} of {} partitions for 3 new vectors",
        report.rewritten,
        partitions
    );
    assert_eq!(report.split, 0);
    assert_eq!(report.merged, 0);
}

#[tokio::test]
async fn an_oversized_partition_splits() {
    let dir = TempDir::new().unwrap();

    // One partition to start with, then push it past the split threshold.
    let mut index =
        PartitionedIvfIndex::new(DistanceMetric::Euclidean, "t", Some(maintained(1, 40, 0)))
            .with_store(store(&dir));
    index.rebuild(vectors(30, 8)).await.unwrap();
    assert_eq!(index.partition_count(), 1);

    for (id, v) in vectors(60, 8).into_iter().skip(30) {
        index.add(format!("x{}", id), v).await.unwrap();
    }
    let report = index.maintain().await.unwrap();

    assert_eq!(
        report.split, 1,
        "a 60-vector partition did not split at a threshold of 40"
    );
    assert_eq!(index.partition_count(), 2);

    // Splitting must not lose anything.
    assert_eq!(index.len(), 60, "vectors went missing across the split");
}

#[tokio::test]
async fn an_undersized_partition_merges_away() {
    let dir = TempDir::new().unwrap();
    let data = vectors(200, 8);

    let mut index = PartitionedIvfIndex::new(
        DistanceMetric::Euclidean,
        "t",
        Some(maintained(8, 10_000, 10_000)),
    )
    .with_store(store(&dir));
    index.rebuild(data.clone()).await.unwrap();
    let before = index.partition_count();
    assert!(before > 1);

    // Every partition is under a 10,000 merge threshold, so touching them all collapses the
    // index down. Merging must stop at one rather than emptying itself.
    index
        .add("trigger".into(), data[0].1.clone())
        .await
        .unwrap();
    let report = index.maintain().await.unwrap();

    assert!(report.merged > 0 || before == 1, "nothing merged");
    assert!(
        index.partition_count() >= 1,
        "merging removed the last partition"
    );
    assert!(
        index.partition_count() < before,
        "partition count did not fall: {} -> {}",
        before,
        index.partition_count()
    );
}

#[tokio::test]
async fn maintenance_keeps_search_results_correct() {
    let dir = TempDir::new().unwrap();
    let data = vectors(400, 12);

    let mut index =
        PartitionedIvfIndex::new(DistanceMetric::Euclidean, "t", Some(maintained(8, 60, 20)))
            .with_store(store(&dir));
    index.rebuild(data.clone()).await.unwrap();

    // Add, delete, then maintain — the combination that exercises absorb, split, merge and
    // tombstone purging in one pass.
    let extra = vectors(500, 12);
    for (id, v) in extra.iter().skip(400) {
        index.add(id.clone(), v.clone()).await.unwrap();
    }
    index.delete(&data[7].0).await.unwrap();
    index.maintain().await.unwrap();

    // nprobe == nlist, so this is exhaustive and any divergence is a bug in maintenance rather
    // than IVF's usual recall loss.
    let mut expected: Vec<_> = extra.clone();
    expected.retain(|(id, _)| id != &data[7].0);

    for (_, query) in expected.iter().step_by(97) {
        let got: Vec<String> = index
            .search(query, 3)
            .await
            .unwrap()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(
            got,
            exact_nearest(&expected, query, 3),
            "maintenance changed the answer"
        );
    }

    assert!(
        !index
            .search(&data[7].1, 5)
            .await
            .unwrap()
            .iter()
            .any(|(id, _)| id == &data[7].0),
        "a deleted vector came back after maintenance"
    );
}

#[tokio::test]
async fn many_deletes_stay_linear() {
    use std::time::Instant;

    let dir = TempDir::new().unwrap();
    let data = vectors(20_000, 8);

    let mut index = PartitionedIvfIndex::new(
        DistanceMetric::Euclidean,
        "t",
        Some(maintained(32, 10_000, 0)),
    )
    .with_store(store(&dir));
    index.rebuild(data.clone()).await.unwrap();

    // Tombstones are consulted for every candidate a query scans, and checked again on every
    // delete. Holding them in a Vec makes deleting n records O(n^2) and every query
    // O(candidates x tombstones). A hash set makes both O(1); caching a snapshot of the set
    // for the query path would take it further still.
    let started = Instant::now();
    for (id, _) in data.iter().take(10_000) {
        index.delete(id).await.unwrap();
    }
    let deleting = started.elapsed();

    let started = Instant::now();
    for (_, query) in data.iter().step_by(2000).take(10) {
        index.search(query, 5).await.unwrap();
    }
    let searching = started.elapsed();

    assert!(
        deleting < std::time::Duration::from_secs(10),
        "10,000 deletes took {:?}",
        deleting
    );
    assert!(
        searching < std::time::Duration::from_secs(10),
        "10 searches over 10,000 tombstones took {:?}",
        searching
    );
}

#[tokio::test]
async fn a_wide_probe_over_a_warm_cache_is_not_dominated_by_bookkeeping() {
    use std::time::Instant;

    let dir = TempDir::new().unwrap();
    // Realistic dimensionality: at 8 dimensions a partition is a kilobyte and copying it costs
    // nothing, which hides the thing this measures.
    let data = vectors(4_000, 1536);

    // Everything resident, so no fetches happen and what is left is per-probe bookkeeping: the
    // cache hit itself, and the LRU update.
    let mut index = PartitionedIvfIndex::new(
        DistanceMetric::Euclidean,
        "t",
        Some(PartitionedIvfConfig {
            nlist: 40,
            nprobe: 8,
            max_resident_partitions: 40,
            max_iterations: 5,
            split_threshold: 100_000,
            merge_threshold: 0,
        }),
    )
    .with_store(store(&dir));
    index.rebuild(data.clone()).await.unwrap();

    // Warm every partition first.
    index.search(&data[0].1, 5).await.unwrap();

    let started = Instant::now();
    for (_, query) in data.iter().step_by(80).take(50) {
        index.search(query, 5).await.unwrap();
    }
    let elapsed = started.elapsed();
    println!("50 warm probes at 1536 dimensions: {:?}", elapsed);

    assert!(
        elapsed < std::time::Duration::from_secs(30),
        "50 warm probes took {:?}",
        elapsed
    );
}
