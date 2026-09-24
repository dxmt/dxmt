// Basic Tiered Search Tests
//
// Simple tests to verify tiered storage search works correctly

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
        description: None,
        index_type: Some(IndexType::BruteForce),
        text_fields: None,
        enable_full_text_search: false,
        memory_optimization: None,
        payload_schema: None,
        schema_version: 1,
        named_vectors: None,
        hnsw: None,
        quantization: None,
        payload_index: None,
    }
}

/// Helper function to create test vectors
fn create_test_vectors(count: usize, dimension: usize) -> Vec<CreateRecordRequest> {
    (0..count)
        .map(|i| CreateRecordRequest {
            id: format!("vec_{}", i),
            // Add 1 to avoid zero vectors
            vector: vec![(i + 1) as f32 / (count + 1) as f32; dimension],
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
async fn test_basic_tiered_storage_creation() {
    // Test: Can create tiered storage with three tiers
    let hot_tier = Arc::new(MemoryStorage::new());
    let warm_tier = Arc::new(MemoryStorage::new());
    let cold_tier = Arc::new(MemoryStorage::new());

    let tiered_storage =
        TieredStorage::new(hot_tier, warm_tier, cold_tier, StorageConfig::default());

    // Verify promotion policy exists
    let policy = tiered_storage.promotion_policy();
    let stats = policy.get_stats();
    assert_eq!(stats.total_vectors, 0);
}

#[tokio::test]
async fn test_tiered_storage_create_table() {
    // Test: Can create table in tiered storage
    let hot_tier = Arc::new(MemoryStorage::new());
    let warm_tier = Arc::new(MemoryStorage::new());
    let cold_tier = Arc::new(MemoryStorage::new());

    let tiered_storage =
        TieredStorage::new(hot_tier, warm_tier, cold_tier, StorageConfig::default());

    // Create table
    let table_config = create_test_table_config("test_table");
    let result = tiered_storage.create_table(table_config).await;

    assert!(result.is_ok());
    let table = result.unwrap();
    assert_eq!(table.name, "test_table");
    assert_eq!(table.dimension, 128);
}

#[tokio::test]
async fn test_tiered_storage_insert_and_search() {
    // Test: Can insert and search vectors
    let hot_tier = Arc::new(MemoryStorage::new());
    let warm_tier = Arc::new(MemoryStorage::new());
    let cold_tier = Arc::new(MemoryStorage::new());

    let tiered_storage =
        TieredStorage::new(hot_tier, warm_tier, cold_tier, StorageConfig::default());

    // Create table
    let table_config = create_test_table_config("test_search");
    tiered_storage.create_table(table_config).await.unwrap();

    // Insert vectors
    let vectors = create_test_vectors(50, 128);
    let insert_result = tiered_storage.insert_records("test_search", vectors).await;

    assert!(insert_result.is_ok());
    let inserted = insert_result.unwrap();
    assert_eq!(inserted.len(), 50);

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

    let search_result = tiered_storage.search("test_search", query).await;

    assert!(search_result.is_ok());
    let result = search_result.unwrap();
    assert_eq!(result.hits.len(), 10);
    // tookms is u64, so always >= 0; just assert the search returned hits

    // Verify results are sorted by score
    for i in 1..result.hits.len() {
        assert!(result.hits[i - 1].score <= result.hits[i].score);
    }
}

#[tokio::test]
async fn test_tier_promotion_policy() {
    // Test: Tier promotion policy tracks access patterns
    let hot_tier = Arc::new(MemoryStorage::new());
    let warm_tier = Arc::new(MemoryStorage::new());
    let cold_tier = Arc::new(MemoryStorage::new());

    let promotion_config = TierPromotionConfig {
        hot_threshold: 0.5,
        warm_threshold: 0.3,
        demote_to_warm_threshold: 0.4,
        demote_to_cold_threshold: 0.2,
        time_window_secs: 60,
        auto_promotion_enabled: true,
        auto_demotion_enabled: true,
    };

    let tiered_storage = TieredStorage::new_with_promotion_config(
        hot_tier,
        warm_tier,
        cold_tier,
        StorageConfig::default(),
        promotion_config,
    );

    // Create table
    let table_config = create_test_table_config("test_promotion");
    tiered_storage.create_table(table_config).await.unwrap();

    // Insert vectors
    let vectors = create_test_vectors(20, 128);
    tiered_storage
        .insert_records("test_promotion", vectors)
        .await
        .unwrap();

    // Perform searches to trigger access tracking
    let query = SearchQuery {
        vector: vec![0.1; 128],
        top_k: 5,
        filter: None,
        with_payload: true,
        with_vector: false,
        limit: 5,
        search_mode: SearchMode::Normal,
        vector_name: None,
        ..Default::default()
    };

    // Search multiple times
    for _ in 0..10 {
        let _ = tiered_storage.search("test_promotion", query.clone()).await;
    }

    // Check policy statistics
    let policy = tiered_storage.promotion_policy();
    let stats = policy.get_stats();

    // Should have tracked accesses
    assert!(stats.total_accesses > 0);
    println!("Total accesses: {}", stats.total_accesses);

    // Check hotness scores
    let result = tiered_storage
        .search("test_promotion", query)
        .await
        .unwrap();

    for hit in result.hits.iter().take(3) {
        let hotness = policy.get_hotness_score(&hit.id);
        println!("Vector {} hotness: {:.3}", hit.id, hotness);
        assert!((0.0..=1.0).contains(&hotness));
    }
}

#[tokio::test]
async fn test_search_result_sorting() {
    // Test: Search results are correctly sorted by score
    let hot_tier = Arc::new(MemoryStorage::new());
    let warm_tier = Arc::new(MemoryStorage::new());
    let cold_tier = Arc::new(MemoryStorage::new());

    let tiered_storage =
        TieredStorage::new(hot_tier, warm_tier, cold_tier, StorageConfig::default());

    // Create table
    let table_config = create_test_table_config("test_sorting");
    tiered_storage.create_table(table_config).await.unwrap();

    // Insert vectors
    let vectors = create_test_vectors(100, 128);
    tiered_storage
        .insert_records("test_sorting", vectors)
        .await
        .unwrap();

    // Search
    let query = SearchQuery {
        vector: vec![0.5; 128],
        top_k: 20,
        filter: None,
        with_payload: false,
        with_vector: false,
        limit: 20,
        search_mode: SearchMode::Normal,
        vector_name: None,
        ..Default::default()
    };

    let result = tiered_storage.search("test_sorting", query).await.unwrap();

    // Verify results are sorted (lower score is better for distance)
    assert_eq!(result.hits.len(), 20);
    for i in 1..result.hits.len() {
        assert!(
            result.hits[i - 1].score <= result.hits[i].score,
            "Results not sorted: {} > {}",
            result.hits[i - 1].score,
            result.hits[i].score
        );
    }
}

#[tokio::test]
async fn test_search_with_filters() {
    // Test: Search with metadata filters
    use dxmt_vectordb::types::record::{Condition, Filter, FilterOperation};

    let hot_tier = Arc::new(MemoryStorage::new());
    let warm_tier = Arc::new(MemoryStorage::new());
    let cold_tier = Arc::new(MemoryStorage::new());

    let tiered_storage =
        TieredStorage::new(hot_tier, warm_tier, cold_tier, StorageConfig::default());

    // Create table
    let table_config = create_test_table_config("test_filters");
    tiered_storage.create_table(table_config).await.unwrap();

    // Insert vectors
    let vectors = create_test_vectors(50, 128);
    tiered_storage
        .insert_records("test_filters", vectors)
        .await
        .unwrap();

    // Search with filter (only "even" category)
    let filter = Filter {
        must: Some(vec![Condition {
            key: "category".to_string(),
            operation: FilterOperation::Equal(serde_json::json!("even")),
        }]),
        must_not: None,
        should: None,
    };

    let query = SearchQuery {
        vector: vec![0.5; 128],
        top_k: 10,
        filter: Some(filter),
        with_payload: true,
        with_vector: false,
        limit: 10,
        search_mode: SearchMode::Normal,
        vector_name: None,
        ..Default::default()
    };

    let result = tiered_storage.search("test_filters", query).await.unwrap();

    // Verify filtered results
    assert!(result.hits.len() <= 10);
    for hit in &result.hits {
        if let Some(payload) = &hit.payload {
            if let Some(category) = payload.get("category") {
                if let Some(cat_str) = category.as_str() {
                    assert_eq!(cat_str, "even");
                }
            }
        }
    }
}
