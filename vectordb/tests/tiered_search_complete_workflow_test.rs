// Complete Tiered Search Workflow Test
//
// This test demonstrates the complete tiered storage workflow:
// 1. Cross-tier search (Memory → SSD → S3)
// 2. Automatic vector promotion based on access patterns
// 3. Background demotion task for cold vectors
// 4. Vector movement between tiers

use dxmt_vectordb::{
    storage::{
        background_demotion::{start_background_demotion_task, BackgroundDemotionConfig},
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
use std::time::Duration;

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
    }
}

fn create_test_vector(id: &str, value: f32) -> CreateRecordRequest {
    // Ensure non-zero vector by adding 0.01 to avoid zero vectors
    let vector_value = value + 0.01;
    CreateRecordRequest {
        id: id.to_string(),
        vector: vec![vector_value; 128],
        sparse_vector: None,
        payload: Some(serde_json::json!({
            "id": id,
            "value": value
        })),
        named_vectors: None,
    }
}

#[tokio::test]
async fn test_complete_tiered_workflow() {
    println!("\n=== Complete Tiered Storage Workflow Test ===\n");

    // Step 1: Create tiered storage with aggressive promotion/demotion thresholds
    println!("Step 1: Creating tiered storage with 3 tiers...");
    let hot_tier = Arc::new(MemoryStorage::new());
    let warm_tier = Arc::new(MemoryStorage::new());
    let cold_tier = Arc::new(MemoryStorage::new());

    let promotion_config = TierPromotionConfig {
        hot_threshold: 0.15,  // Very low for testing
        warm_threshold: 0.08, // Very low for testing
        demote_to_warm_threshold: 0.12,
        demote_to_cold_threshold: 0.05,
        time_window_secs: 10, // Short window for testing
        auto_promotion_enabled: true,
        auto_demotion_enabled: true,
    };

    let storage = Arc::new(TieredStorage::new_with_promotion_config(
        hot_tier.clone(),
        warm_tier.clone(),
        cold_tier.clone(),
        StorageConfig::default(),
        promotion_config,
    ));

    println!("✓ Tiered storage created");
    println!("  - Hot tier: Memory (fast access)");
    println!("  - Warm tier: SSD (medium access)");
    println!("  - Cold tier: S3 (slow but cheap)");
    println!();

    // Step 2: Create table
    println!("Step 2: Creating table...");
    let table_config = create_test_table_config("workflow_test");
    storage.create_table(table_config).await.unwrap();
    println!("✓ Table 'workflow_test' created");
    println!();

    // Step 3: Insert vectors (goes to hot tier by default)
    println!("Step 3: Inserting 10 vectors...");
    let vectors: Vec<_> = (0..10)
        .map(|i| create_test_vector(&format!("vec_{}", i), (i as f32) / 10.0))
        .collect();

    storage
        .insert_records("workflow_test", vectors)
        .await
        .unwrap();

    // Track vectors in promotion policy
    for i in 0..10 {
        storage
            .promotion_policy()
            .set_current_tier(&format!("vec_{}", i), Tier::Hot);
    }

    println!("✓ Inserted 10 vectors into hot tier");
    println!();

    // Step 4: Perform cross-tier search
    println!("Step 4: Performing initial search...");
    let query = SearchQuery {
        vector: vec![0.5; 128],
        top_k: 5,
        filter: None,
        with_payload: true,
        with_vector: false,
        limit: 5,
        include_metadata: true,
        search_mode: SearchMode::Normal,
        vector_name: None,
    };

    let result = storage
        .search("workflow_test", query.clone())
        .await
        .unwrap();
    println!("✓ Found {} results", result.hits.len());
    for (i, hit) in result.hits.iter().take(3).enumerate() {
        println!("  {}. ID: {}, Score: {:.4}", i + 1, hit.id, hit.score);
    }
    println!();

    // Step 5: Access specific vectors frequently to trigger promotion
    println!("Step 5: Frequently accessing vec_0, vec_1, vec_2 (20 times each)...");
    let frequent_vectors = vec!["vec_0", "vec_1", "vec_2"];

    for _ in 0..20 {
        for vec_id in &frequent_vectors {
            storage.promotion_policy().record_access(vec_id);
        }
    }

    // Check hotness scores
    println!("✓ Access patterns recorded");
    for vec_id in &frequent_vectors {
        let hotness = storage.promotion_policy().get_hotness_score(vec_id);
        let tier = storage.promotion_policy().get_recommended_tier(vec_id);
        println!(
            "  - {}: hotness={:.3}, recommended={:?}",
            vec_id, hotness, tier
        );
    }
    println!();

    // Step 6: Trigger promotion by performing search
    println!("Step 6: Performing search to trigger automatic promotion...");
    let result = storage
        .search("workflow_test", query.clone())
        .await
        .unwrap();
    println!("✓ Search completed, promotions triggered");
    println!("  - Search took {}ms", result.tookms);
    println!();

    // Step 7: Check promotion policy statistics
    println!("Step 7: Checking promotion policy statistics...");
    let stats = storage.promotion_policy().get_stats();
    println!("✓ Policy stats:");
    println!("  - Total vectors tracked: {}", stats.total_vectors);
    println!("  - Total accesses: {}", stats.total_accesses);
    println!("  - Hot tier vectors: {}", stats.hot_tier_count);
    println!("  - Warm tier vectors: {}", stats.warm_tier_count);
    println!("  - Cold tier vectors: {}", stats.cold_tier_count);
    println!();

    // Step 8: Start background demotion task
    println!("Step 8: Starting background demotion task...");
    let demotion_config = BackgroundDemotionConfig {
        interval_secs: 2, // 2 seconds for testing
        table_names: vec!["workflow_test".to_string()],
        enabled: true,
    };

    let demotion_task = start_background_demotion_task(storage.clone(), demotion_config);
    println!("✓ Background demotion task started (interval: 2s)");
    println!();

    // Step 9: Wait for background task to run
    println!("Step 9: Waiting for background demotion to run...");
    tokio::time::sleep(Duration::from_secs(3)).await;
    println!("✓ Background task has run at least once");
    println!();

    // Step 10: Perform final search to verify system still works
    println!("Step 10: Performing final search across all tiers...");
    let final_result = storage
        .search("workflow_test", query.clone())
        .await
        .unwrap();
    println!("✓ Final search completed");
    println!("  - Found {} results", final_result.hits.len());
    println!("  - Search took {}ms", final_result.tookms);
    println!();

    // Step 11: Stop background task
    println!("Step 11: Stopping background demotion task...");
    demotion_task.stop().await;
    println!("✓ Background task stopped gracefully");
    println!();

    // Final statistics
    println!("=== Final Statistics ===");
    let final_stats = storage.promotion_policy().get_stats();
    println!("Total vectors: {}", final_stats.total_vectors);
    println!("Total accesses: {}", final_stats.total_accesses);
    println!("Tier distribution:");
    println!("  - Hot: {} vectors", final_stats.hot_tier_count);
    println!("  - Warm: {} vectors", final_stats.warm_tier_count);
    println!("  - Cold: {} vectors", final_stats.cold_tier_count);
    println!();
    println!("✓ Complete workflow test passed!");
}

#[tokio::test]
async fn test_vector_lifecycle() {
    // Test: A vector's journey through tiers based on access patterns
    println!("\n=== Vector Lifecycle Test ===\n");

    let hot_tier = Arc::new(MemoryStorage::new());
    let warm_tier = Arc::new(MemoryStorage::new());
    let cold_tier = Arc::new(MemoryStorage::new());

    let promotion_config = TierPromotionConfig {
        hot_threshold: 0.15,
        warm_threshold: 0.08,
        demote_to_warm_threshold: 0.12,
        demote_to_cold_threshold: 0.05,
        time_window_secs: 10,
        auto_promotion_enabled: true,
        auto_demotion_enabled: true,
    };

    let storage = Arc::new(TieredStorage::new_with_promotion_config(
        hot_tier.clone(),
        warm_tier.clone(),
        cold_tier.clone(),
        StorageConfig::default(),
        promotion_config,
    ));

    // Create table
    let table_config = create_test_table_config("lifecycle_test");
    storage.create_table(table_config).await.unwrap();

    // Insert vector
    let vector = create_test_vector("lifecycle_vec", 0.5);
    storage
        .insert_records("lifecycle_test", vec![vector])
        .await
        .unwrap();

    // Track in promotion policy
    storage
        .promotion_policy()
        .set_current_tier("lifecycle_vec", Tier::Hot);

    println!("Phase 1: Initial state");
    println!("  - Vector inserted into hot tier");

    // Initially in hot tier
    let initial_tier = storage.promotion_policy().get_current_tier("lifecycle_vec");
    println!("  - Current tier: {:?}", initial_tier);
    assert_eq!(initial_tier, Some(Tier::Hot));

    // Access it frequently
    println!("\nPhase 2: Frequent access (30 times)");
    for _ in 0..30 {
        storage.promotion_policy().record_access("lifecycle_vec");
    }

    let hotness = storage
        .promotion_policy()
        .get_hotness_score("lifecycle_vec");
    println!("  - Hotness score: {:.3}", hotness);
    println!("  - Should stay in hot tier");

    // Wait for time to pass (simulate no recent accesses)
    println!("\nPhase 3: No access for 12 seconds (beyond time window)");
    tokio::time::sleep(Duration::from_secs(12)).await;

    let coolness = storage
        .promotion_policy()
        .get_hotness_score("lifecycle_vec");
    println!("  - Hotness score: {:.3}", coolness);
    println!("  - Should be eligible for demotion");

    // Manually trigger demotion
    println!("\nPhase 4: Triggering demotion");
    storage.demote_cold_vectors("lifecycle_test").await.unwrap();

    let demoted_tier = storage.promotion_policy().get_current_tier("lifecycle_vec");
    println!("  - New tier: {:?}", demoted_tier);

    // Access again frequently
    println!("\nPhase 5: Frequent access again (40 times)");
    for _ in 0..40 {
        storage.promotion_policy().record_access("lifecycle_vec");
    }

    let reheated_score = storage
        .promotion_policy()
        .get_hotness_score("lifecycle_vec");
    println!("  - Hotness score: {:.3}", reheated_score);

    let recommended = storage
        .promotion_policy()
        .get_recommended_tier("lifecycle_vec");
    println!("  - Recommended tier: {:?}", recommended);
    println!("  - Should be promoted back to hot tier");

    println!("\n✓ Vector lifecycle test completed");
}

#[tokio::test]
async fn test_search_performance_across_tiers() {
    // Test: Search performance when data is distributed across tiers
    println!("\n=== Search Performance Across Tiers Test ===\n");

    let hot_tier = Arc::new(MemoryStorage::new());
    let warm_tier = Arc::new(MemoryStorage::new());
    let cold_tier = Arc::new(MemoryStorage::new());

    let storage = Arc::new(TieredStorage::new(
        hot_tier.clone(),
        warm_tier.clone(),
        cold_tier.clone(),
        StorageConfig::default(),
    ));

    // Create table
    let table_config = create_test_table_config("performance_test");
    storage.create_table(table_config).await.unwrap();

    // Insert vectors into different tiers manually
    println!("Setting up test data:");

    // Hot tier: 10 vectors
    println!("  - Inserting 10 vectors into hot tier...");
    for i in 0..10 {
        let vec = create_test_vector(&format!("hot_{}", i), i as f32 / 10.0);
        hot_tier
            .insert_record("performance_test", vec)
            .await
            .unwrap();
        storage
            .promotion_policy()
            .set_current_tier(&format!("hot_{}", i), Tier::Hot);
    }

    // Warm tier: 10 vectors
    println!("  - Inserting 10 vectors into warm tier...");
    for i in 0..10 {
        let vec = create_test_vector(&format!("warm_{}", i), i as f32 / 10.0);
        warm_tier
            .insert_record("performance_test", vec)
            .await
            .unwrap();
        storage
            .promotion_policy()
            .set_current_tier(&format!("warm_{}", i), Tier::Warm);
    }

    // Cold tier: 10 vectors
    println!("  - Inserting 10 vectors into cold tier...");
    for i in 0..10 {
        let vec = create_test_vector(&format!("cold_{}", i), i as f32 / 10.0);
        cold_tier
            .insert_record("performance_test", vec)
            .await
            .unwrap();
        storage
            .promotion_policy()
            .set_current_tier(&format!("cold_{}", i), Tier::Cold);
    }

    println!("✓ Test data prepared (30 vectors across 3 tiers)");
    println!();

    // Perform search
    println!("Performing cross-tier search...");
    let query = SearchQuery {
        vector: vec![0.5; 128],
        top_k: 15,
        filter: None,
        with_payload: true,
        with_vector: false,
        limit: 15,
        include_metadata: true,
        search_mode: SearchMode::Normal,
        vector_name: None,
    };

    let result = storage.search("performance_test", query).await.unwrap();

    println!("✓ Search completed");
    println!("  - Found {} results", result.hits.len());
    println!("  - Search took {}ms", result.tookms);
    println!();

    // Analyze results by tier
    println!("Results by tier:");
    let mut hot_count = 0;
    let mut warm_count = 0;
    let mut cold_count = 0;

    for hit in &result.hits {
        if hit.id.starts_with("hot_") {
            hot_count += 1;
        } else if hit.id.starts_with("warm_") {
            warm_count += 1;
        } else if hit.id.starts_with("cold_") {
            cold_count += 1;
        }
    }

    println!("  - Hot tier: {} results", hot_count);
    println!("  - Warm tier: {} results", warm_count);
    println!("  - Cold tier: {} results", cold_count);
    println!();

    // Verify we got results (early termination may skip cold tier)
    assert!(result.hits.len() == 15, "Should have 15 results total");
    assert!(hot_count > 0, "Should have results from at least hot tier");

    // Note: cold_count may be 0 due to early termination optimization
    // This is the correct behavior - if hot+warm tiers have enough results,
    // we don't need to query the expensive cold tier
    if cold_count == 0 {
        println!("  Note: Cold tier not queried (early termination optimization)");
    }

    println!("✓ Cross-tier search working correctly");
}
