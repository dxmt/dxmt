// Vector Movement Tests
//
// Tests for vector movement between tiers (Hot → Warm → Cold)

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
        payload_index: None,
        embedding: None,
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
        text: None,
    }
}

#[tokio::test]
async fn test_move_vector_between_tiers() {
    // Test: Move a single vector from one tier to another
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
    let table_config = create_test_table_config("test_move");
    tiered_storage.create_table(table_config).await.unwrap();

    // Insert vector into cold tier
    let vector = create_test_vector("vec_1");
    cold_tier
        .insert_record("test_move", vector.clone())
        .await
        .unwrap();

    // Set initial tier in policy
    tiered_storage
        .promotion_policy()
        .set_current_tier("vec_1", Tier::Cold);

    // Verify it's in cold tier
    assert!(cold_tier
        .get_record("test_move", &"vec_1".to_string())
        .is_ok());
    assert!(warm_tier
        .get_record("test_move", &"vec_1".to_string())
        .is_err());
    assert!(hot_tier
        .get_record("test_move", &"vec_1".to_string())
        .is_err());

    // Move from Cold to Warm
    tiered_storage
        .move_vector_to_tier("test_move", &"vec_1".to_string(), Tier::Cold, Tier::Warm)
        .await
        .unwrap();

    // Verify it's now in warm tier
    assert!(cold_tier
        .get_record("test_move", &"vec_1".to_string())
        .is_err());
    assert!(warm_tier
        .get_record("test_move", &"vec_1".to_string())
        .is_ok());
    assert!(hot_tier
        .get_record("test_move", &"vec_1".to_string())
        .is_err());

    // Verify policy updated
    assert_eq!(
        tiered_storage.promotion_policy().get_current_tier("vec_1"),
        Some(Tier::Warm)
    );

    // Move from Warm to Hot
    tiered_storage
        .move_vector_to_tier("test_move", &"vec_1".to_string(), Tier::Warm, Tier::Hot)
        .await
        .unwrap();

    // Verify it's now in hot tier
    assert!(cold_tier
        .get_record("test_move", &"vec_1".to_string())
        .is_err());
    assert!(warm_tier
        .get_record("test_move", &"vec_1".to_string())
        .is_err());
    assert!(hot_tier
        .get_record("test_move", &"vec_1".to_string())
        .is_ok());

    // Verify policy updated
    assert_eq!(
        tiered_storage.promotion_policy().get_current_tier("vec_1"),
        Some(Tier::Hot)
    );
}

#[tokio::test]
async fn test_batch_move_vectors() {
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
    let table_config = create_test_table_config("test_batch_move");
    tiered_storage.create_table(table_config).await.unwrap();

    // Insert vectors into cold tier
    for i in 0..5 {
        let vector = create_test_vector(&format!("vec_{}", i));
        cold_tier
            .insert_record("test_batch_move", vector)
            .await
            .unwrap();
        tiered_storage
            .promotion_policy()
            .set_current_tier(&format!("vec_{}", i), Tier::Cold);
    }

    // Prepare batch moves
    let moves = vec![
        ("vec_0".to_string(), Tier::Cold, Tier::Hot),
        ("vec_1".to_string(), Tier::Cold, Tier::Hot),
        ("vec_2".to_string(), Tier::Cold, Tier::Warm),
        ("vec_3".to_string(), Tier::Cold, Tier::Warm),
        ("vec_4".to_string(), Tier::Cold, Tier::Hot),
    ];

    // Execute batch move
    let results = tiered_storage
        .batch_move_vectors_to_tier("test_batch_move", moves)
        .await
        .unwrap();

    // Verify all moves succeeded
    assert_eq!(results.len(), 5);
    for (vec_id, result) in &results {
        assert!(result.is_ok(), "Move failed for {}: {:?}", vec_id, result);
    }

    // Verify final locations
    assert!(hot_tier
        .get_record("test_batch_move", &"vec_0".to_string())
        .is_ok());
    assert!(hot_tier
        .get_record("test_batch_move", &"vec_1".to_string())
        .is_ok());
    assert!(warm_tier
        .get_record("test_batch_move", &"vec_2".to_string())
        .is_ok());
    assert!(warm_tier
        .get_record("test_batch_move", &"vec_3".to_string())
        .is_ok());
    assert!(hot_tier
        .get_record("test_batch_move", &"vec_4".to_string())
        .is_ok());

    // Verify not in cold tier anymore
    for i in 0..5 {
        assert!(cold_tier
            .get_record("test_batch_move", &format!("vec_{}", i))
            .is_err());
    }
}

#[tokio::test]
async fn test_automatic_promotion_on_access() {
    // Test: Vectors are automatically promoted after frequent access
    let hot_tier = Arc::new(MemoryStorage::new());
    let warm_tier = Arc::new(MemoryStorage::new());
    let cold_tier = Arc::new(MemoryStorage::new());

    // Use aggressive thresholds for testing
    let promotion_config = TierPromotionConfig {
        hot_threshold: 0.1, // Very low threshold for testing
        warm_threshold: 0.05,
        demote_to_warm_threshold: 0.08,
        demote_to_cold_threshold: 0.03,
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
    let table_config = create_test_table_config("test_auto_promote");
    tiered_storage.create_table(table_config).await.unwrap();

    // Insert vector into hot tier (so search can find it)
    let vector = create_test_vector("vec_1");
    tiered_storage
        .insert_records("test_auto_promote", vec![vector])
        .await
        .unwrap();

    // Initially set as cold
    tiered_storage
        .promotion_policy()
        .set_current_tier("vec_1", Tier::Cold);

    // Access the vector multiple times via search
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

    for _ in 0..20 {
        let _ = tiered_storage
            .search("test_auto_promote", query.clone())
            .await;
    }

    // Check hotness score
    let hotness = tiered_storage.promotion_policy().get_hotness_score("vec_1");

    println!("Hotness after 20 accesses: {:.3}", hotness);
    assert!(hotness > 0.0);

    // With aggressive thresholds, it should recommend promotion
    let recommended = tiered_storage
        .promotion_policy()
        .get_recommended_tier("vec_1");

    println!("Recommended tier: {:?}", recommended);
    assert!(recommended == Tier::Hot || recommended == Tier::Warm);
}

#[tokio::test]
async fn test_demote_cold_vectors() {
    // Test: Demote cold vectors from hot/warm tiers
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
    let table_config = create_test_table_config("test_demote");
    tiered_storage.create_table(table_config).await.unwrap();

    // Insert vector into hot tier
    let vector = create_test_vector("vec_1");
    hot_tier.insert_record("test_demote", vector).await.unwrap();

    tiered_storage
        .promotion_policy()
        .set_current_tier("vec_1", Tier::Hot);

    // Verify it's in hot tier
    assert!(hot_tier
        .get_record("test_demote", &"vec_1".to_string())
        .is_ok());

    // Run demotion (it should find that vec_1 has no recent accesses)
    // and should be demoted
    tiered_storage
        .demote_cold_vectors("test_demote")
        .await
        .unwrap();

    // Note: With default config, demotion might not happen immediately
    // since the vector was just created. This test mainly verifies
    // the demotion mechanism doesn't crash.
}

#[tokio::test]
async fn test_find_vector_tier() {
    // Test: Can find which tier contains a vector
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
    let table_config = create_test_table_config("test_find");
    tiered_storage.create_table(table_config).await.unwrap();

    // Insert into hot tier
    let vector = create_test_vector("vec_hot");
    hot_tier
        .insert_record("test_find", vector.clone())
        .await
        .unwrap();

    // Insert into warm tier
    let vector2 = create_test_vector("vec_warm");
    warm_tier.insert_record("test_find", vector2).await.unwrap();

    // Insert into cold tier
    let vector3 = create_test_vector("vec_cold");
    cold_tier.insert_record("test_find", vector3).await.unwrap();

    // Test finding each vector (this tests the private method indirectly
    // by checking that moves work correctly)

    // For now, just verify they exist in their respective tiers
    assert!(hot_tier
        .get_record("test_find", &"vec_hot".to_string())
        .is_ok());
    assert!(warm_tier
        .get_record("test_find", &"vec_warm".to_string())
        .is_ok());
    assert!(cold_tier
        .get_record("test_find", &"vec_cold".to_string())
        .is_ok());
}

#[tokio::test]
async fn test_move_preserves_data() {
    // Test: Moving a vector preserves all its data
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
    let table_config = create_test_table_config("test_preserve");
    tiered_storage.create_table(table_config).await.unwrap();

    // Insert vector with specific data
    let original_vector = CreateRecordRequest {
        id: "vec_1".to_string(),
        vector: vec![0.5, 0.6, 0.7].into_iter().cycle().take(128).collect(),
        sparse_vector: None,
        payload: Some(serde_json::json!({
            "name": "test vector",
            "value": 42,
            "tags": ["important", "test"]
        })),
        named_vectors: None,
        text: None,
    };

    cold_tier
        .insert_record("test_preserve", original_vector.clone())
        .await
        .unwrap();

    tiered_storage
        .promotion_policy()
        .set_current_tier("vec_1", Tier::Cold);

    // Move to hot tier
    tiered_storage
        .move_vector_to_tier("test_preserve", &"vec_1".to_string(), Tier::Cold, Tier::Hot)
        .await
        .unwrap();

    // Get from hot tier and verify data
    let moved_record = hot_tier
        .get_record("test_preserve", &"vec_1".to_string())
        .unwrap();

    // Verify vector values
    assert_eq!(moved_record.id, "vec_1");
    assert_eq!(moved_record.vector.len(), 128);

    // Verify first few values
    assert!((moved_record.vector[0] - 0.5).abs() < 0.001);
    assert!((moved_record.vector[1] - 0.6).abs() < 0.001);
    assert!((moved_record.vector[2] - 0.7).abs() < 0.001);

    // Verify payload
    assert_eq!(moved_record.payload.get("name").unwrap(), "test vector");
    assert_eq!(moved_record.payload.get("value").unwrap(), 42);
}
