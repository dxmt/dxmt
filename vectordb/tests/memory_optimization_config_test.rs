// Memory Optimization Configuration Tests
// 测试配置层级决策逻辑

use dxmt_vectordb::config::AppConfig;
use dxmt_vectordb::types::table::TableConfig;
use dxmt_vectordb::types::vector::DistanceMetric;
use std::sync::Mutex;

// 全局锁确保环境变量测试串行运行
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn clear_env_vars() {
    std::env::remove_var("DXMT_MEMORY_OPTIMIZATION");
}

#[test]
fn test_global_config_default_memory_optimization() {
    let _guard = ENV_LOCK.lock().unwrap();
    clear_env_vars();

    // 测试默认值（应该为 true）
    let config = AppConfig::default();
    assert!(
        config.tables.default_memory_optimization,
        "Default memory_optimization should be true"
    );

    clear_env_vars();
}

#[test]
fn test_global_config_memory_optimization_true() {
    let _guard = ENV_LOCK.lock().unwrap();
    clear_env_vars();

    // 设置为 true
    std::env::set_var("DXMT_MEMORY_OPTIMIZATION", "true");

    let config = AppConfig::default();
    assert!(
        config.tables.default_memory_optimization,
        "memory_optimization should be true when DXMT_MEMORY_OPTIMIZATION=true"
    );

    clear_env_vars();
}

#[test]
fn test_global_config_memory_optimization_false() {
    let _guard = ENV_LOCK.lock().unwrap();
    clear_env_vars();

    // 设置为 false
    std::env::set_var("DXMT_MEMORY_OPTIMIZATION", "false");

    let config = AppConfig::default();
    assert!(
        !config.tables.default_memory_optimization,
        "memory_optimization should be false when DXMT_MEMORY_OPTIMIZATION=false"
    );

    clear_env_vars();
}

#[test]
fn test_table_config_memory_optimization_none() {
    // 测试 TableConfig 使用 None（应使用全局配置）
    let config = TableConfig {
        name: "test_table".to_string(),
        dimension: 768,
        distance_metric: DistanceMetric::Cosine,
        description: None,
        index_type: None,
        text_fields: None,
        enable_full_text_search: false,
        memory_optimization: None, // 使用全局配置
        named_vectors: None,
        payload_schema: None,
        schema_version: 1,
        hnsw: None,
        quantization: None,
        payload_index: None,
        embedding: None,
    };

    assert_eq!(
        config.memory_optimization, None,
        "memory_optimization should be None when not specified"
    );
}

#[test]
fn test_table_config_memory_optimization_true() {
    // 测试 TableConfig 显式设置为 true
    let config = TableConfig {
        name: "optimized_table".to_string(),
        dimension: 768,
        distance_metric: DistanceMetric::Cosine,
        description: None,
        index_type: None,
        text_fields: None,
        enable_full_text_search: false,
        memory_optimization: Some(true), // 显式启用
        named_vectors: None,
        payload_schema: None,
        schema_version: 1,
        hnsw: None,
        quantization: None,
        payload_index: None,
        embedding: None,
    };

    assert_eq!(
        config.memory_optimization,
        Some(true),
        "memory_optimization should be Some(true) when explicitly set"
    );
}

#[test]
fn test_table_config_memory_optimization_false() {
    // 测试 TableConfig 显式设置为 false
    let config = TableConfig {
        name: "precision_table".to_string(),
        dimension: 512,
        distance_metric: DistanceMetric::Euclidean,
        description: None,
        index_type: None,
        text_fields: None,
        enable_full_text_search: false,
        memory_optimization: Some(false), // 显式禁用
        named_vectors: None,
        payload_schema: None,
        schema_version: 1,
        hnsw: None,
        quantization: None,
        payload_index: None,
        embedding: None,
    };

    assert_eq!(
        config.memory_optimization,
        Some(false),
        "memory_optimization should be Some(false) when explicitly set to false"
    );
}

#[test]
fn test_config_hierarchy_table_overrides_global() {
    let _guard = ENV_LOCK.lock().unwrap();
    clear_env_vars();

    // 设置全局配置为 true
    std::env::set_var("DXMT_MEMORY_OPTIMIZATION", "true");
    let global_config = AppConfig::default();
    assert!(global_config.tables.default_memory_optimization);

    // 表级配置覆盖为 false
    let table_config = TableConfig {
        name: "override_table".to_string(),
        dimension: 384,
        distance_metric: DistanceMetric::Dot,
        description: None,
        index_type: None,
        text_fields: None,
        enable_full_text_search: false,
        memory_optimization: Some(false), // 覆盖全局配置
        named_vectors: None,
        payload_schema: None,
        schema_version: 1,
        hnsw: None,
        quantization: None,
        payload_index: None,
        embedding: None,
    };

    // 验证配置层级：表级配置应覆盖全局配置
    let effective_config = table_config
        .memory_optimization
        .unwrap_or(global_config.tables.default_memory_optimization);

    assert!(
        !effective_config,
        "Table-level config should override global config"
    );

    clear_env_vars();
}

#[test]
fn test_config_hierarchy_uses_global_when_none() {
    let _guard = ENV_LOCK.lock().unwrap();
    clear_env_vars();

    // 设置全局配置为 false
    std::env::set_var("DXMT_MEMORY_OPTIMIZATION", "false");
    let global_config = AppConfig::default();
    assert!(!global_config.tables.default_memory_optimization);

    // 表级配置为 None（使用全局配置）
    let table_config = TableConfig {
        name: "default_table".to_string(),
        dimension: 256,
        distance_metric: DistanceMetric::Cosine,
        description: None,
        index_type: None,
        text_fields: None,
        enable_full_text_search: false,
        memory_optimization: None, // 使用全局配置
        named_vectors: None,
        payload_schema: None,
        schema_version: 1,
        hnsw: None,
        quantization: None,
        payload_index: None,
        embedding: None,
    };

    // 验证配置层级：应使用全局配置
    let effective_config = table_config
        .memory_optimization
        .unwrap_or(global_config.tables.default_memory_optimization);

    assert!(
        !effective_config,
        "Should use global config when table-level config is None"
    );

    clear_env_vars();
}

#[test]
fn test_mixed_table_configurations() {
    let _guard = ENV_LOCK.lock().unwrap();
    clear_env_vars();

    // 设置全局配置为 true
    std::env::set_var("DXMT_MEMORY_OPTIMIZATION", "true");
    let global_config = AppConfig::default();

    // 表1：使用全局配置（None）
    let table1 = TableConfig {
        name: "table1".to_string(),
        dimension: 768,
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
    };

    // 表2：显式设置为 false（高精度）
    let table2 = TableConfig {
        name: "table2".to_string(),
        dimension: 512,
        distance_metric: DistanceMetric::Euclidean,
        description: None,
        index_type: None,
        text_fields: None,
        enable_full_text_search: false,
        memory_optimization: Some(false),
        named_vectors: None,
        payload_schema: None,
        schema_version: 1,
        hnsw: None,
        quantization: None,
        payload_index: None,
        embedding: None,
    };

    // 表3：显式设置为 true（内存优化）
    let table3 = TableConfig {
        name: "table3".to_string(),
        dimension: 384,
        distance_metric: DistanceMetric::Dot,
        description: None,
        index_type: None,
        text_fields: None,
        enable_full_text_search: false,
        memory_optimization: Some(true),
        named_vectors: None,
        payload_schema: None,
        schema_version: 1,
        hnsw: None,
        quantization: None,
        payload_index: None,
        embedding: None,
    };

    // 验证配置决策
    let config1 = table1
        .memory_optimization
        .unwrap_or(global_config.tables.default_memory_optimization);
    let config2 = table2
        .memory_optimization
        .unwrap_or(global_config.tables.default_memory_optimization);
    let config3 = table3
        .memory_optimization
        .unwrap_or(global_config.tables.default_memory_optimization);

    assert!(config1, "Table1 should use global config (true)");
    assert!(!config2, "Table2 should override to false");
    assert!(config3, "Table3 should explicitly use true");

    clear_env_vars();
}
