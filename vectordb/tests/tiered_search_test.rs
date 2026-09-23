// Tiered Search Integration Tests
//
// Tests for the tiered storage search implementation, including:
// - Cross-tier search (Memory → SSD → S3)
// - Result merging and reranking
// - Vector promotion based on access patterns
// - Performance benchmarks

use dxmt_vectordb::{
    storage::{
        memory::MemoryStorage, tier_promotion::TierPromotionConfig, tiered::TieredStorage, Storage,
        StorageConfig,
    },
    types::{
        record::{CreateRecordRequest, SearchMode, SearchQuery},
        table::TableConfig,
        DistanceMetric, IndexType,
    },
};
use std::sync::Arc;

/// Helper function to create a test table config
fn create_test_table_config(name: &str) -> TableConfig {
    TableConfig {
        name: name.to_string(),
        dimension: 128,
        distance_metric: DistanceMetric::Cosine,
        index_type: Some(IndexType::BruteForce),
        enable_full_text_search: false,
        text_fields: None,
        description: None,
        memory_optimization: None,
        payload_schema: None,
        schema_version: 1,
        named_vectors: None,
    }
}

/// Helper function to create test vectors
fn create_test_vectors(count: usize, dimension: usize) -> Vec<CreateRecordRequest> {
    (0..count)
        .map(|i| CreateRecordRequest {
            id: format!("vec_{}", i),
            vector: vec![(i + 1) as f32 / count as f32; dimension],
            sparse_vector: None,
            payload: Some(serde_json::json!({
                "index": i,
                "category": if i % 2 == 0 { "even" } else { "odd" }
            })),
            named_vectors: None,
        })
        .collect()
}

#[tokio::test]
async fn test_tiered_search_memory_only() {
    // Test: All data fits in memory, should only search hot tier
    let hot_tier = Arc::new(MemoryStorage::new());
    let warm_tier = Arc::new(MemoryStorage::new());
    let cold_tier = Arc::new(MemoryStorage::new());

    let tiered_storage = TieredStorage::new(
        hot_tier.clone(),
        warm_tier.clone(),
        cold_tier.clone(),
        StorageConfig::default(),
    );

    // Create table
    let table_config = create_test_table_config("test_memory_only");
    tiered_storage
        .create_table(table_config.clone())
        .await
        .unwrap();

    // Insert vectors into hot tier only
    let vectors = create_test_vectors(100, 128);
    tiered_storage
        .insert_records("test_memory_only", vectors)
        .await
        .unwrap();

    // Search
    let query = SearchQuery {
        vector: vec![0.5; 128],
        top_k: 10,
        filter: None,
        with_payload: true,
        with_vector: false,
        limit: 10,
        search_mode: SearchMode::Normal,
        vector_name: None,
        ..Default::default()
    };

    let result = tiered_storage
        .search("test_memory_only", query)
        .await
        .unwrap();

    // Assertions
    assert_eq!(result.hits.len(), 10);
    assert!(result.tookms < 100); // Should be very fast (<100ms)

    // Verify results are sorted by score
    for i in 1..result.hits.len() {
        assert!(result.hits[i - 1].score <= result.hits[i].score);
    }
}

#[tokio::test]
async fn test_tiered_search_with_warm_tier() {
    // Test: Data in both hot and warm tiers, should merge results
    let hot_tier = Arc::new(MemoryStorage::new());
    let warm_tier = Arc::new(MemoryStorage::new());
    let cold_tier = Arc::new(MemoryStorage::new());

    let tiered_storage = TieredStorage::new(
        hot_tier.clone(),
        warm_tier.clone(),
        cold_tier.clone(),
        StorageConfig::default(),
    );

    // Create table in all tiers
    let table_config = create_test_table_config("test_warm_tier");
    tiered_storage
        .create_table(table_config.clone())
        .await
        .unwrap();

    // Insert vectors into hot tier
    let hot_vectors = create_test_vectors(50, 128);
    hot_tier
        .insert_records("test_warm_tier", hot_vectors)
        .unwrap();

    // Insert different vectors into warm tier
    let warm_vectors: Vec<CreateRecordRequest> = (50..100)
        .map(|i| CreateRecordRequest {
            id: format!("vec_{}", i),
            vector: vec![i as f32 / 100.0; 128],
            sparse_vector: None,
            payload: Some(serde_json::json!({
                "index": i,
                "tier": "warm"
            })),
            named_vectors: None,
        })
        .collect();
    warm_tier
        .insert_records("test_warm_tier", warm_vectors)
        .unwrap();

    // Search
    let query = SearchQuery {
        vector: vec![0.5; 128],
        top_k: 20,
        filter: None,
        with_payload: true,
        with_vector: false,
        limit: 20,
        search_mode: SearchMode::Normal,
        vector_name: None,
        ..Default::default()
    };

    let result = tiered_storage
        .search("test_warm_tier", query)
        .await
        .unwrap();

    // Assertions
    assert_eq!(result.hits.len(), 20); // Should get results from both tiers
    assert!(result.tookms < 500); // Should be reasonably fast

    // Verify results are sorted
    for i in 1..result.hits.len() {
        assert!(result.hits[i - 1].score <= result.hits[i].score);
    }
}

#[tokio::test]
async fn test_tiered_search_all_tiers() {
    // Test: Data in all three tiers, should merge from all
    let hot_tier = Arc::new(MemoryStorage::new());
    let warm_tier = Arc::new(MemoryStorage::new());
    let cold_tier = Arc::new(MemoryStorage::new());

    let tiered_storage = TieredStorage::new(
        hot_tier.clone(),
        warm_tier.clone(),
        cold_tier.clone(),
        StorageConfig::default(),
    );

    // Create table
    let table_config = create_test_table_config("test_all_tiers");
    tiered_storage
        .create_table(table_config.clone())
        .await
        .unwrap();

    // Insert vectors into each tier
    let hot_vectors = create_test_vectors(10, 128);
    hot_tier
        .insert_records("test_all_tiers", hot_vectors)
        .unwrap();

    let warm_vectors: Vec<CreateRecordRequest> = (10..20)
        .map(|i| CreateRecordRequest {
            id: format!("vec_{}", i),
            vector: vec![i as f32 / 30.0; 128],
            sparse_vector: None,
            payload: None,
            named_vectors: None,
        })
        .collect();
    warm_tier
        .insert_records("test_all_tiers", warm_vectors)
        .unwrap();

    let cold_vectors: Vec<CreateRecordRequest> = (20..30)
        .map(|i| CreateRecordRequest {
            id: format!("vec_{}", i),
            vector: vec![i as f32 / 30.0; 128],
            sparse_vector: None,
            payload: None,
            named_vectors: None,
        })
        .collect();
    cold_tier
        .insert_records("test_all_tiers", cold_vectors)
        .unwrap();

    // Search
    let query = SearchQuery {
        vector: vec![0.5; 128],
        top_k: 30,
        filter: None,
        with_payload: false,
        with_vector: false,
        limit: 30,
        search_mode: SearchMode::Normal,
        vector_name: None,
        ..Default::default()
    };

    let result = tiered_storage
        .search("test_all_tiers", query)
        .await
        .unwrap();

    // Assertions
    assert_eq!(result.hits.len(), 30); // Should get all vectors from all tiers
    assert!(result.tookms < 1000); // Should complete within 1 second

    // Verify results are sorted
    for i in 1..result.hits.len() {
        assert!(result.hits[i - 1].score <= result.hits[i].score);
    }
}

#[tokio::test]
async fn test_hot_vector_promotion() {
    // Test: Frequently accessed vectors should be tracked for promotion
    let hot_tier = Arc::new(MemoryStorage::new());
    let warm_tier = Arc::new(MemoryStorage::new());
    let cold_tier = Arc::new(MemoryStorage::new());

    let promotion_config = TierPromotionConfig {
        hot_threshold: 0.3, // Lower threshold for testing
        warm_threshold: 0.1,
        demote_to_warm_threshold: 0.2,
        demote_to_cold_threshold: 0.05,
        time_window_secs: 60,
        auto_promotion_enabled: true,
        auto_demotion_enabled: true,
    };

    let tiered_storage = TieredStorage::new_with_promotion_config(
        hot_tier.clone(),
        warm_tier.clone(),
        cold_tier.clone(),
        StorageConfig::default(),
        promotion_config,
    );

    // Create table
    let table_config = create_test_table_config("test_promotion");
    tiered_storage
        .create_table(table_config.clone())
        .await
        .unwrap();

    // Insert vectors
    let vectors = create_test_vectors(100, 128);
    tiered_storage
        .insert_records("test_promotion", vectors)
        .await
        .unwrap();

    // Perform multiple searches to trigger access tracking
    let query = SearchQuery {
        vector: vec![0.1; 128],
        top_k: 10,
        filter: None,
        with_payload: true,
        with_vector: false,
        limit: 10,
        search_mode: SearchMode::Normal,
        vector_name: None,
        ..Default::default()
    };

    for _ in 0..10 {
        let _ = tiered_storage.search("test_promotion", query.clone()).await;
    }

    // Check promotion policy statistics
    let policy = tiered_storage.promotion_policy();
    let stats = policy.get_stats();

    assert!(stats.total_accesses > 0);
    println!("Total accesses tracked: {}", stats.total_accesses);

    // Verify that top results have high hotness scores
    let result = tiered_storage
        .search("test_promotion", query.clone())
        .await
        .unwrap();

    for hit in result.hits.iter().take(5) {
        let hotness = policy.get_hotness_score(&hit.id);
        println!("Vector {} hotness: {:.3}", hit.id, hotness);
        assert!(hotness > 0.0); // Should have some hotness score
    }
}

#[tokio::test]
async fn test_search_result_reranking() {
    // Test: Results from multiple tiers are correctly merged and reranked
    let hot_tier = Arc::new(MemoryStorage::new());
    let warm_tier = Arc::new(MemoryStorage::new());
    let cold_tier = Arc::new(MemoryStorage::new());

    let tiered_storage = TieredStorage::new(
        hot_tier.clone(),
        warm_tier.clone(),
        cold_tier.clone(),
        StorageConfig::default(),
    );

    // Create table
    let table_config = create_test_table_config("test_reranking");
    tiered_storage
        .create_table(table_config.clone())
        .await
        .unwrap();

    // Insert vectors with known scores
    // Hot tier: vectors close to [0.3; 128]
    let hot_vectors: Vec<CreateRecordRequest> = vec![CreateRecordRequest {
        id: "hot_1".to_string(),
        vector: vec![0.3; 128],
        sparse_vector: None,
        payload: None,
        named_vectors: None,
    }];
    hot_tier
        .insert_records("test_reranking", hot_vectors)
        .unwrap();

    // Warm tier: vectors close to [0.5; 128]
    let warm_vectors: Vec<CreateRecordRequest> = vec![CreateRecordRequest {
        id: "warm_1".to_string(),
        vector: vec![0.5; 128],
        sparse_vector: None,
        payload: None,
        named_vectors: None,
    }];
    warm_tier
        .insert_records("test_reranking", warm_vectors)
        .unwrap();

    // Cold tier: vectors close to [0.7; 128]
    let cold_vectors: Vec<CreateRecordRequest> = vec![CreateRecordRequest {
        id: "cold_1".to_string(),
        vector: vec![0.7; 128],
        sparse_vector: None,
        payload: None,
        named_vectors: None,
    }];
    cold_tier
        .insert_records("test_reranking", cold_vectors)
        .unwrap();

    // Query with [0.5; 128] - should rank warm_1 highest
    let query = SearchQuery {
        vector: vec![0.5; 128],
        top_k: 3,
        filter: None,
        with_payload: false,
        with_vector: false,
        limit: 3,
        search_mode: SearchMode::Normal,
        vector_name: None,
        ..Default::default()
    };

    let result = tiered_storage
        .search("test_reranking", query)
        .await
        .unwrap();

    // Verify reranking: warm_1 should be first (closest to query)
    assert_eq!(result.hits.len(), 3);
    assert_eq!(result.hits[0].id, "warm_1");

    // Verify scores are in ascending order (lower is better for distance)
    for i in 1..result.hits.len() {
        assert!(result.hits[i - 1].score <= result.hits[i].score);
    }
}

#[tokio::test]
async fn test_early_termination_optimization() {
    // Test: Search should terminate early if enough results found in hot tier
    let hot_tier = Arc::new(MemoryStorage::new());
    let warm_tier = Arc::new(MemoryStorage::new());
    let cold_tier = Arc::new(MemoryStorage::new());

    let tiered_storage = TieredStorage::new(
        hot_tier.clone(),
        warm_tier.clone(),
        cold_tier.clone(),
        StorageConfig::default(),
    );

    // Create table
    let table_config = create_test_table_config("test_early_termination");
    tiered_storage
        .create_table(table_config.clone())
        .await
        .unwrap();

    // Insert many vectors in hot tier
    let hot_vectors = create_test_vectors(100, 128);
    hot_tier
        .insert_records("test_early_termination", hot_vectors)
        .unwrap();

    // Insert vectors in warm tier (should not be accessed)
    let warm_vectors = create_test_vectors(100, 128);
    warm_tier
        .insert_records("test_early_termination", warm_vectors)
        .unwrap();

    // Search for only 10 results
    let query = SearchQuery {
        vector: vec![0.5; 128],
        top_k: 10,
        filter: None,
        with_payload: false,
        with_vector: false,
        limit: 10,
        search_mode: SearchMode::Normal,
        vector_name: None,
        ..Default::default()
    };

    let result = tiered_storage
        .search("test_early_termination", query)
        .await
        .unwrap();

    // Should return exactly 10 results
    assert_eq!(result.hits.len(), 10);

    // Should be very fast since it only searched hot tier
    assert!(result.tookms < 100);
}

#[cfg(test)]
mod performance_benchmarks {
    use super::*;
    use std::time::Instant;

    #[tokio::test]
    #[ignore] // Run with: cargo test --test tiered_search_test -- --ignored
    async fn benchmark_tiered_search_latency() {
        println!("\n=== Tiered Search Latency Benchmark ===\n");

        let hot_tier = Arc::new(MemoryStorage::new());
        let warm_tier = Arc::new(MemoryStorage::new());
        let cold_tier = Arc::new(MemoryStorage::new());

        let tiered_storage = TieredStorage::new(
            hot_tier.clone(),
            warm_tier.clone(),
            cold_tier.clone(),
            StorageConfig::default(),
        );

        // Create table
        let table_config = create_test_table_config("benchmark");
        tiered_storage.create_table(table_config).await.unwrap();

        // Insert 10K vectors in hot tier
        let hot_vectors = create_test_vectors(10000, 128);
        hot_tier.insert_records("benchmark", hot_vectors).unwrap();

        // Warmup
        let query = SearchQuery {
            vector: vec![0.5; 128],
            top_k: 10,
            filter: None,
            with_payload: true,
            with_vector: false,
            limit: 10,
            search_mode: SearchMode::Normal,
            vector_name: None,
            ..Default::default()
        };
        for _ in 0..10 {
            let _ = tiered_storage.search("benchmark", query.clone()).await;
        }

        // Benchmark
        let iterations = 1000;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = tiered_storage.search("benchmark", query.clone()).await;
        }
        let elapsed = start.elapsed();

        let avg_latency = elapsed.as_micros() / iterations;
        let qps = (iterations as f64 / elapsed.as_secs_f64()) as u64;

        println!("Iterations: {}", iterations);
        println!("Average latency: {} us", avg_latency);
        println!("Queries per second: {} QPS", qps);
        println!("\nTarget: <1ms P50 latency");
        assert!(avg_latency < 1000); // Should be <1ms average
    }
}
