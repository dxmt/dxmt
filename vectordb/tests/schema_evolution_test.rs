use dxmt_vectordb::{
    config::StorageConfig,
    storage::IntelligentIndexedStorage,
    types::{
        table::{FieldDef, FieldType, TableAlterRequest, TableConfig},
        vector::DistanceMetric,
    },
};

fn make_storage_config() -> StorageConfig {
    StorageConfig {
        r#type: "memory".to_string(),
        memory: Some(dxmt_vectordb::config::MemoryConfig {
            max_size_mb: 64,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    }
}

fn make_table_config(name: &str) -> TableConfig {
    TableConfig {
        name: name.to_string(),
        dimension: 4,
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
    }
}

async fn make_storage() -> IntelligentIndexedStorage {
    IntelligentIndexedStorage::new(&make_storage_config())
        .await
        .unwrap()
}

#[tokio::test]
async fn test_alter_table_increments_schema_version() {
    let storage = make_storage().await;
    storage
        .create_table(make_table_config("ev_test"))
        .await
        .unwrap();

    let table = storage.get_table("ev_test").await.unwrap();
    assert_eq!(
        table.schema_version, 1,
        "Initial schema_version should be 1"
    );
    assert!(table.payload_schema.is_none());

    let req = TableAlterRequest {
        description: Some("updated description".to_string()),
        add_fields: vec![FieldDef {
            name: "category".to_string(),
            field_type: FieldType::String,
            nullable: true,
            default_value: None,
        }],
        ..Default::default()
    };

    let updated = storage.alter_table("ev_test", req).await.unwrap();
    assert_eq!(
        updated.schema_version, 2,
        "schema_version should be 2 after alter"
    );
    assert_eq!(updated.description.as_deref(), Some("updated description"));

    let schema = updated.payload_schema.as_ref().unwrap();
    assert_eq!(schema.len(), 1);
    assert_eq!(schema[0].name, "category");
}

#[tokio::test]
async fn test_alter_table_add_then_drop_field() {
    let storage = make_storage().await;
    storage
        .create_table(make_table_config("ev_drop_test"))
        .await
        .unwrap();

    // Add two fields
    let req_add = TableAlterRequest {
        add_fields: vec![
            FieldDef {
                name: "tag".to_string(),
                field_type: FieldType::String,
                nullable: true,
                default_value: None,
            },
            FieldDef {
                name: "score".to_string(),
                field_type: FieldType::Number,
                nullable: true,
                default_value: None,
            },
        ],
        ..Default::default()
    };
    let after_add = storage.alter_table("ev_drop_test", req_add).await.unwrap();
    assert_eq!(after_add.schema_version, 2);
    assert_eq!(after_add.payload_schema.as_ref().unwrap().len(), 2);

    // Drop one field
    let req_drop = TableAlterRequest {
        drop_fields: vec!["tag".to_string()],
        ..Default::default()
    };
    let after_drop = storage.alter_table("ev_drop_test", req_drop).await.unwrap();
    assert_eq!(after_drop.schema_version, 3);
    let schema = after_drop.payload_schema.as_ref().unwrap();
    assert_eq!(schema.len(), 1);
    assert_eq!(schema[0].name, "score");
}

#[tokio::test]
async fn test_alter_table_not_found_returns_error() {
    let storage = make_storage().await;
    let result = storage
        .alter_table("nonexistent", TableAlterRequest::default())
        .await;
    assert!(
        result.is_err(),
        "alter_table on missing table should return error"
    );
}

#[tokio::test]
async fn test_schema_persisted_across_get_table() {
    let storage = make_storage().await;
    storage
        .create_table(make_table_config("ev_persist"))
        .await
        .unwrap();

    let req = TableAlterRequest {
        add_fields: vec![FieldDef {
            name: "region".to_string(),
            field_type: FieldType::String,
            nullable: false,
            default_value: None,
        }],
        ..Default::default()
    };
    storage.alter_table("ev_persist", req).await.unwrap();

    // Re-fetch from the in-memory store — schema must be preserved
    let fetched = storage.get_table("ev_persist").await.unwrap();
    assert_eq!(fetched.schema_version, 2);
    let schema = fetched.payload_schema.as_ref().unwrap();
    assert_eq!(schema.len(), 1);
    assert_eq!(schema[0].name, "region");
    assert!(!schema[0].nullable);
}

#[tokio::test]
async fn test_alter_table_enable_full_text_search() {
    let storage = make_storage().await;
    storage
        .create_table(make_table_config("ev_fts"))
        .await
        .unwrap();

    let table = storage.get_table("ev_fts").await.unwrap();
    assert!(!table.enable_full_text_search);

    let req = TableAlterRequest {
        enable_full_text_search: Some(true),
        ..Default::default()
    };
    let updated = storage.alter_table("ev_fts", req).await.unwrap();
    assert!(updated.enable_full_text_search);
    assert_eq!(updated.schema_version, 2);
}
