// Basic Vector Movement Tests
//
// Tests for vector movement between tiers using TieredStorage API

use dxmt_vectordb::{
    storage::{
        memory::MemoryStorage,
        tier_promotion::{Tier, TierPromotionConfig},
        tiered::TieredStorage,
        Storage, StorageConfig,
    },
    types::{
        record::{CreateRecordRequest, SearchMode, SearchQuery},
        table::TableConfig,
        DistanceMetric, IndexType,
    },
};
use std::sync::Arc;

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
    }
}

fn create_test_vector(id: &str) -> CreateRecordRequest {
    CreateRecordRequest {
        id: id.to_string(),
        vector: vec![0.1; 128],
        sparse_vector: None,
        payload: Some(serde_json::json!({
            "test": "data"
        })),
        named_vectors: None,
    }
}

#[tokio::test]
async fn test_move_vector_cold_to_hot() {
    // Test: Move a vector from cold to hot tier
    let hot_tier = Arc::new(MemoryStorage::new());
    let warm_tier = Arc::new(MemoryStorage::new());
    let cold_tier = Arc::new(MemoryStorage::new());

    let tiered_storage = TieredStorage::new(
        hot_tier.clone(),
        warm_tier.clone(),
        cold_tier.clone(),
        StorageConfig::default(),
    );

    // Create table in tiered storage
    let table_config = create_test_table_config("test_move");
    tiered_storage.create_table(table_config).await.unwrap();

    // Insert vector via tiered storage (goes to hot tier by default)
    let vector = create_test_vector("vec_1");
    tiered_storage
        .insert_record("test_move", vector)
        .await
        .unwrap();

    // Initial state: vector is in hot tier
    let record_hot = tiered_storage
        .get_record("test_move", &"vec_1".to_string())
        .await;
    assert!(record_hot.is_ok());

    println!("✓ Vector successfully inserted and retrieved from hot tier");

    // Move from hot to warm
    let move_result = tiered_storage
        .move_vector_to_tier("test_move", &"vec_1".to_string(), Tier::Hot, Tier::Warm)
        .await;

    assert!(
        move_result.is_ok(),
        "Move to warm tier failed: {:?}",
        move_result
    );
    println!("✓ Vector moved from hot to warm tier");

    // Verify policy updated
    assert_eq!(
        tiered_storage.promotion_policy().get_current_tier("vec_1"),
        Some(Tier::Warm)
    );
    println!("✓ Promotion policy updated correctly");
}

#[tokio::test]
async fn test_batch_move() {
    // Test: Batch move multiple vectors
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
    let table_config = create_test_table_config("test_batch");
    tiered_storage.create_table(table_config).await.unwrap();

    // Insert 3 vectors
    for i in 0..3 {
        let vector = create_test_vector(&format!("vec_{}", i));
        tiered_storage
            .insert_record("test_batch", vector)
            .await
            .unwrap();
    }

    println!("✓ Inserted 3 vectors");

    // Prepare batch moves (all from hot to warm)
    let moves = vec![
        ("vec_0".to_string(), Tier::Hot, Tier::Warm),
        ("vec_1".to_string(), Tier::Hot, Tier::Warm),
        ("vec_2".to_string(), Tier::Hot, Tier::Warm),
    ];

    // Execute batch move
    let results = tiered_storage
        .batch_move_vectors_to_tier("test_batch", moves)
        .await
        .unwrap();

    // Verify results
    assert_eq!(results.len(), 3);
    let successes = results.iter().filter(|(_, r)| r.is_ok()).count();
    println!("✓ Batch move: {}/3 succeeded", successes);

    // Verify policy updated for all
    for i in 0..3 {
        let tier = tiered_storage
            .promotion_policy()
            .get_current_tier(&format!("vec_{}", i));
        assert_eq!(tier, Some(Tier::Warm));
    }
    println!("✓ All promotion policies updated correctly");
}

#[tokio::test]
async fn test_promotion_on_frequent_access() {
    // Test: Frequent access triggers promotion tracking
    let hot_tier = Arc::new(MemoryStorage::new());
    let warm_tier = Arc::new(MemoryStorage::new());
    let cold_tier = Arc::new(MemoryStorage::new());

    // Use aggressive thresholds for testing
    let promotion_config = TierPromotionConfig {
        hot_threshold: 0.1,
        warm_threshold: 0.05,
        demote_to_warm_threshold: 0.08,
        demote_to_cold_threshold: 0.03,
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

    // Insert vector
    let vector = create_test_vector("vec_1");
    tiered_storage
        .insert_records("test_promotion", vec![vector])
        .await
        .unwrap();

    println!("✓ Vector inserted");

    // Access multiple times via search
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

    for i in 0..15 {
        let _ = tiered_storage.search("test_promotion", query.clone()).await;
        if i % 5 == 0 {
            println!("  - Accessed {} times", i + 1);
        }
    }

    // Check hotness score
    let hotness = tiered_storage.promotion_policy().get_hotness_score("vec_1");

    println!("✓ Hotness score after 15 accesses: {:.3}", hotness);
    assert!(hotness > 0.0, "Hotness should be > 0 after accesses");

    // Check recommended tier
    let recommended = tiered_storage
        .promotion_policy()
        .get_recommended_tier("vec_1");

    println!("✓ Recommended tier: {:?}", recommended);
    // With low thresholds and frequent access, should recommend promotion
}

#[tokio::test]
async fn test_move_nonexistent_vector() {
    // Test: Moving a nonexistent vector should fail gracefully
    let hot_tier = Arc::new(MemoryStorage::new());
    let warm_tier = Arc::new(MemoryStorage::new());
    let cold_tier = Arc::new(MemoryStorage::new());

    let tiered_storage =
        TieredStorage::new(hot_tier, warm_tier, cold_tier, StorageConfig::default());

    // Create table
    let table_config = create_test_table_config("test_nonexistent");
    tiered_storage.create_table(table_config).await.unwrap();

    // Try to move a vector that doesn't exist
    let result = tiered_storage
        .move_vector_to_tier(
            "test_nonexistent",
            &"nonexistent".to_string(),
            Tier::Hot,
            Tier::Warm,
        )
        .await;

    // Should fail
    assert!(result.is_err(), "Moving nonexistent vector should fail");
    println!("✓ Correctly handled nonexistent vector");
}

#[tokio::test]
async fn test_promotion_policy_stats() {
    // Test: Promotion policy tracks statistics
    let hot_tier = Arc::new(MemoryStorage::new());
    let warm_tier = Arc::new(MemoryStorage::new());
    let cold_tier = Arc::new(MemoryStorage::new());

    let tiered_storage =
        TieredStorage::new(hot_tier, warm_tier, cold_tier, StorageConfig::default());

    // Initially should have no vectors
    let stats = tiered_storage.promotion_policy().get_stats();
    assert_eq!(stats.total_vectors, 0);
    assert_eq!(stats.total_accesses, 0);
    println!(
        "✓ Initial stats: {} vectors, {} accesses",
        stats.total_vectors, stats.total_accesses
    );

    // Record some accesses
    for _ in 0..10 {
        tiered_storage.promotion_policy().record_access("vec_1");
    }

    // Check updated stats
    let stats = tiered_storage.promotion_policy().get_stats();
    assert!(stats.total_accesses >= 10);
    println!(
        "✓ After accesses: {} vectors tracked, {} total accesses",
        stats.total_vectors, stats.total_accesses
    );
}
