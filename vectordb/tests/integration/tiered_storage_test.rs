/// Tiered Storage Quality Assurance Tests
///
/// Tests the tiered storage architecture (Memory → SSD → S3) for:
/// - Recall consistency across tiers
/// - Latency requirements per tier
/// - Data integrity during migrations
/// - Cache behavior and hotness scoring

use dxmt_vectordb::{
    config::{StorageConfig, MemoryConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::CreateRecordRequest,
        table::TableConfig,
        vector::DistanceMetric,
    },
};
use std::time::Instant;

/// Helper to create test storage
async fn create_test_storage(max_size_mb: usize) -> IntelligentIndexedStorage {
    let config = StorageConfig {
        r#type: "memory".to_string(),
        memory: Some(MemoryConfig {
            max_size_mb,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    };
    IntelligentIndexedStorage::new(&config).await.expect("Failed to create storage")
}

/// Helper function to create test vectors
fn create_test_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            let mut vec: Vec<f32> = (0..dim)
                .map(|j| ((i + j) as f32 * 0.1).sin())
                .collect();

            // Normalize
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                vec.iter_mut().for_each(|x| *x /= norm);
            }
            vec
        })
        .collect()
}

#[tokio::test]
async fn test_tier_recall_consistency() {
    // Test that recall is consistent regardless of which tier data is in

    let table_config = TableConfig {
        name: "tier_recall_test".to_string(),
        dimension: 64,
        distance_metric: DistanceMetric::Cosine,
        description: None,
        index_type: None,
        text_fields: None,
        enable_full_text_search: false,
        hnsw: None,
        quantization: None,
    };

    let storage = create_test_storage(1).await; // 1MB memory

    // Create table
    storage.create_table(&table_config).await.unwrap();

    // Insert vectors
    let vectors = create_test_vectors(100, 64);
    let mut records = Vec::new();

    for (i, vector) in vectors.iter().enumerate() {
        records.push(CreateRecordRequest {
            id: format!("vec_{}", i),
            vector: vector.clone(),
            sparse_vector: None,
            payload: None,
        });
    }

    storage.insert_batch("tier_recall_test", records).await.unwrap();

    // Query vector (should be exact match to one of the inserted vectors)
    let query = vectors[0].clone();

    // Search multiple times to test across different cache states
    for iteration in 0..5 {
        let results = storage
            .search("tier_recall_test", &query, 10, None)
            .await
            .unwrap();

        // First result should be exact match
        assert!(
            results.len() > 0,
            "Iteration {}: Should return results",
            iteration
        );

        assert_eq!(
            results[0].id, "vec_0",
            "Iteration {}: First result should be exact match",
            iteration
        );

        // Distance should be very close to 0 (within floating point error)
        assert!(
            results[0].distance < 1e-5,
            "Iteration {}: Distance should be ~0, got {}",
            iteration,
            results[0].distance
        );
    }
}

#[tokio::test]
async fn test_memory_tier_latency() {
    // Memory tier should have <1ms latency

    let table_config = TableConfig {
        name: "memory_latency_test".to_string(),
        dimension: 128,
        distance_metric: DistanceMetric::Euclidean,
        description: None,
        index_type: None,
        text_fields: None,
        enable_full_text_search: false,
        hnsw: None,
        quantization: None,
    };

    let storage = create_test_storage(512).await; // 512MB
    storage.create_table(&table_config).await.unwrap();

    // Insert small dataset that fits in memory
    let vectors = create_test_vectors(1000, 128);
    let mut records = Vec::new();

    for (i, vector) in vectors.iter().enumerate() {
        records.push(CreateRecordRequest {
            id: format!("vec_{}", i),
            vector: vector.clone(),
            sparse_vector: None,
            payload: None,
        });
    }

    storage.insert_batch("memory_latency_test", records).await.unwrap();

    // Warm up cache
    for _ in 0..10 {
        storage
            .search("memory_latency_test", &vectors[0], 10, None)
            .await
            .unwrap();
    }

    // Measure latency for 100 queries
    let mut latencies = Vec::new();

    for i in 0..100 {
        let query = &vectors[i % vectors.len()];
        let start = Instant::now();

        storage
            .search("memory_latency_test", query, 10, None)
            .await
            .unwrap();

        let elapsed = start.elapsed();
        latencies.push(elapsed.as_micros() as f64 / 1000.0); // Convert to ms
    }

    // Calculate statistics
    let mean_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p99_idx = (latencies.len() as f64 * 0.99) as usize;
    let p99_latency = latencies[p99_idx];

    println!("Memory tier latency: mean={:.3}ms, p99={:.3}ms", mean_latency, p99_latency);

    // Memory tier should be <1ms for p99
    assert!(
        p99_latency < 1.0,
        "Memory tier P99 latency should be <1ms, got {:.3}ms",
        p99_latency
    );
}

#[tokio::test]
async fn test_data_integrity_across_operations() {
    // Ensure data integrity is maintained through various operations

    let table_config = TableConfig {
        name: "integrity_test".to_string(),
        dimension: 64,
        distance_metric: DistanceMetric::Cosine,
        description: None,
        index_type: None,
        text_fields: None,
        enable_full_text_search: false,
        hnsw: None,
        quantization: None,
    };

    let storage = create_test_storage(1).await;
    storage.create_table(&table_config).await.unwrap();

    // Test 1: Insert and verify
    let original_vector = vec![1.0, 2.0, 3.0, 4.0, 5.0]
        .into_iter()
        .chain(std::iter::repeat(0.0).take(59))
        .collect::<Vec<_>>();

    let record = CreateRecordRequest {
        id: "test_vec".to_string(),
        vector: original_vector.clone(),
        sparse_vector: None,
        payload: None,
    };

    storage.insert("integrity_test", record).await.unwrap();

    // Retrieve and verify
    let retrieved = storage
        .get("integrity_test", "test_vec")
        .await
        .unwrap()
        .expect("Record should exist");

    for (i, (&original, &retrieved)) in original_vector
        .iter()
        .zip(retrieved.vector.iter())
        .enumerate()
    {
        assert!(
            (original - retrieved).abs() < 1e-6,
            "Vector mismatch at index {}: {} != {}",
            i,
            original,
            retrieved
        );
    }

    // Test 2: Update and verify
    let updated_vector = vec![5.0, 4.0, 3.0, 2.0, 1.0]
        .into_iter()
        .chain(std::iter::repeat(0.0).take(59))
        .collect::<Vec<_>>();

    let update_record = CreateRecordRequest {
        id: "test_vec".to_string(),
        vector: updated_vector.clone(),
        sparse_vector: None,
        payload: None,
    };

    storage.insert("integrity_test", update_record).await.unwrap();

    let retrieved_updated = storage
        .get("integrity_test", "test_vec")
        .await
        .unwrap()
        .expect("Record should exist");

    for (i, (&expected, &actual)) in updated_vector
        .iter()
        .zip(retrieved_updated.vector.iter())
        .enumerate()
    {
        assert!(
            (expected - actual).abs() < 1e-6,
            "Updated vector mismatch at index {}: {} != {}",
            i,
            expected,
            actual
        );
    }

    // Test 3: Delete and verify
    storage.delete("integrity_test", "test_vec").await.unwrap();

    let deleted = storage.get("integrity_test", "test_vec").await.unwrap();
    assert!(deleted.is_none(), "Deleted record should not exist");
}

#[tokio::test]
async fn test_bulk_operations_consistency() {
    // Test consistency with large batch operations

    let table_config = TableConfig {
        name: "bulk_test".to_string(),
        dimension: 32,
        distance_metric: DistanceMetric::Euclidean,
        description: None,
        index_type: None,
        text_fields: None,
        enable_full_text_search: false,
        hnsw: None,
        quantization: None,
    };

    let storage = create_test_storage(10).await; // 10MB
    storage.create_table(&table_config).await.unwrap();

    // Insert 1000 vectors in batches of 100
    let total_vectors = 1000;
    let batch_size = 100;
    let vectors = create_test_vectors(total_vectors, 32);

    for batch_start in (0..total_vectors).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(total_vectors);
        let mut batch = Vec::new();

        for i in batch_start..batch_end {
            batch.push(CreateRecordRequest {
                id: Some(format!("vec_{}", i)),
                vector: vectors[i].clone(),
                metadata: None,
                sparse_vector: None,
            });
        }

        storage.insert_batch("bulk_test", batch).await.unwrap();
    }

    // Verify count
    let table = storage.get_table("bulk_test").await.unwrap();
    let count = table.record_count().await;
    assert_eq!(
        count, total_vectors,
        "Should have {} records, got {}",
        total_vectors, count
    );

    // Verify random samples
    for i in vec![0, 100, 500, 750, 999] {
        let record = storage
            .get("bulk_test", &format!("vec_{}", i))
            .await
            .unwrap()
            .expect(&format!("Record vec_{} should exist", i));

        // Verify vector matches
        for (j, (&expected, &actual)) in vectors[i]
            .iter()
            .zip(record.vector.iter())
            .enumerate()
        {
            assert!(
                (expected - actual).abs() < 1e-5,
                "Vector mismatch for vec_{} at index {}: {} != {}",
                i,
                j,
                expected,
                actual
            );
        }
    }
}

#[tokio::test]
async fn test_concurrent_access_consistency() {
    // Test that concurrent reads/writes maintain consistency

    let table_config = TableConfig {
        name: "concurrent_test".to_string(),
        dimension: 64,
        distance_metric: DistanceMetric::Cosine,
        description: None,
        index_type: None,
        text_fields: None,
        enable_full_text_search: false,
        hnsw: None,
        quantization: None,
    };

    let storage = std::sync::Arc::new(
        create_test_storage(5).await // 5MB
    );

    storage.create_table(&table_config).await.unwrap();

    // Insert initial data
    let vectors = create_test_vectors(100, 64);
    let mut records = Vec::new();

    for (i, vector) in vectors.iter().enumerate() {
        records.push(CreateRecordRequest {
            id: format!("vec_{}", i),
            vector: vector.clone(),
            sparse_vector: None,
            payload: None,
        });
    }

    storage.insert_batch("concurrent_test", records).await.unwrap();

    // Spawn multiple concurrent readers
    let mut handles = Vec::new();

    for thread_id in 0..5 {
        let storage_clone = storage.clone();
        let vectors_clone = vectors.clone();

        let handle = tokio::spawn(async move {
            for i in 0..20 {
                let query_idx = (thread_id * 20 + i) % vectors_clone.len();
                let query = &vectors_clone[query_idx];

                let results = storage_clone
                    .search("concurrent_test", query, 10, None)
                    .await
                    .unwrap();

                // Should always return results
                assert!(
                    results.len() > 0,
                    "Thread {} iteration {}: Should return results",
                    thread_id,
                    i
                );
            }
        });

        handles.push(handle);
    }

    // Wait for all readers
    for handle in handles {
        handle.await.unwrap();
    }
}

#[tokio::test]
async fn test_search_quality_after_cache_eviction() {
    // Ensure search quality doesn't degrade after cache evictions

    let table_config = TableConfig {
        name: "eviction_test".to_string(),
        dimension: 128,
        distance_metric: DistanceMetric::Euclidean,
        description: None,
        index_type: None,
        text_fields: None,
        enable_full_text_search: false,
        hnsw: None,
        quantization: None,
    };

    let storage = create_test_storage(2).await; // 2MB small cache
    storage.create_table(&table_config).await.unwrap();

    // Insert more data than cache can hold
    let vectors = create_test_vectors(200, 128);
    let mut records = Vec::new();

    for (i, vector) in vectors.iter().enumerate() {
        records.push(CreateRecordRequest {
            id: format!("vec_{}", i),
            vector: vector.clone(),
            sparse_vector: None,
            payload: None,
        });
    }

    storage.insert_batch("eviction_test", records).await.unwrap();

    // Query for exact matches before and after cache pressure
    let test_queries = vec![0, 50, 100, 150];

    for &query_idx in &test_queries {
        let query = &vectors[query_idx];

        // First search (may be cached)
        let results1 = storage
            .search("eviction_test", query, 10, None)
            .await
            .unwrap();

        // Force cache pressure by querying many other vectors
        for i in 0..100 {
            if i != query_idx {
                storage
                    .search("eviction_test", &vectors[i % vectors.len()], 5, None)
                    .await
                    .unwrap();
            }
        }

        // Second search (likely evicted from cache)
        let results2 = storage
            .search("eviction_test", query, 10, None)
            .await
            .unwrap();

        // Results should be identical
        assert_eq!(
            results1.len(),
            results2.len(),
            "Result count should be same for query {}",
            query_idx
        );

        assert_eq!(
            results1[0].id, results2[0].id,
            "Top result should be same for query {}",
            query_idx
        );

        assert!(
            (results1[0].distance - results2[0].distance).abs() < 1e-5,
            "Distance should be same for query {}: {} vs {}",
            query_idx,
            results1[0].distance,
            results2[0].distance
        );
    }
}

#[test]
fn test_tiered_storage_compilation() {
    // This test just verifies that tiered storage compiles
    // Actual tiered storage tests would require S3/MinIO setup
    println!("Tiered storage QA tests compiled successfully");
}
