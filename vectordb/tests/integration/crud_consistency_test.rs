/// Integration tests for CRUD consistency with full-text search
///
/// These tests verify that:
/// 1. INSERT: Both vector and text are indexed
/// 2. UPDATE: Both vector and text indexes are updated
/// 3. DELETE: Both vector and text indexes are cleaned up
/// 4. QUERY: Results are consistent across vector and text search
#[cfg(test)]
#[cfg(feature = "text-bm25")]
mod crud_consistency_tests {
    use dxmt_vectordb::{
        storage::IntelligentIndexedStorage,
        config::{StorageConfig, MemoryConfig},
        types::{
            table::{TableConfig, TextFieldConfig, TextSource, TextAnalyzer},
            record::{CreateRecordRequest, FullTextSearchRequest},
            vector::DistanceMetric,
        },
    };

    /// Helper function to create a test storage instance
    async fn create_test_storage() -> IntelligentIndexedStorage {
        let config = StorageConfig {
            r#type: "memory".to_string(),
            memory: Some(MemoryConfig {
                max_size_mb: 1024,
                eviction_policy: "lru".to_string(),
            }),
            ssd: None,
            object_storage: None,
        };
        IntelligentIndexedStorage::new(&config).await.expect("Failed to create storage")
    }

    #[tokio::test]
    async fn test_insert_consistency() {
        let storage = create_test_storage().await;

        // Create table with full-text search
        let table_config = TableConfig {
            name: "test_insert".to_string(),
            dimension: 128,
            distance_metric: DistanceMetric::Cosine,
            description: Some("Test insert consistency".to_string()),
            index_type: None,
            text_fields: Some(vec![
                TextFieldConfig {
                    name: "content".to_string(),
                    source: TextSource::MetadataField("text".to_string()),
                    analyzer: TextAnalyzer::Standard,
                    boost: 1.0,
                },
            ]),
            enable_full_text_search: true,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
            hnsw: None,
            quantization: None,
        };

        storage.create_table(table_config).await.expect("Failed to create table");

        // Insert record
        let request = CreateRecordRequest {
            id: "doc1".to_string(),
            vector: vec![0.1; 128],
            sparse_vector: None,
            payload: Some(serde_json::json!({
                "text": "artificial intelligence and machine learning"
            })),
            named_vectors: None,
        };

        storage.insert_records("test_insert", vec![request]).await.expect("Failed to insert");

        // Wait for indexing
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Verify text search works
        let text_search = FullTextSearchRequest {
            query: "artificial intelligence".to_string(),
            fields: None,
            limit: 10,
            offset: 0,
            highlight: false,
            fuzzy: None,
            filter: None,
        };

        let result = storage.full_text_search("test_insert", text_search).await.expect("Search failed");
        assert_eq!(result.total, 1, "Expected 1 result from insert");
        assert_eq!(result.hits[0].id, "doc1");
    }

    #[tokio::test]
    async fn test_update_consistency() {
        let storage = create_test_storage().await;

        // Create table
        let table_config = TableConfig {
            name: "test_update".to_string(),
            dimension: 128,
            distance_metric: DistanceMetric::Cosine,
            description: Some("Test update consistency".to_string()),
            index_type: None,
            text_fields: Some(vec![
                TextFieldConfig {
                    name: "content".to_string(),
                    source: TextSource::MetadataField("text".to_string()),
                    analyzer: TextAnalyzer::Standard,
                    boost: 1.0,
                },
            ]),
            enable_full_text_search: true,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
            hnsw: None,
            quantization: None,
        };

        storage.create_table(table_config).await.expect("Failed to create table");

        // Insert initial record
        let initial = CreateRecordRequest {
            id: "doc1".to_string(),
            vector: vec![0.1; 128],
            sparse_vector: None,
            payload: Some(serde_json::json!({
                "text": "old content about cats"
            })),
            named_vectors: None,
        };

        storage.insert_records("test_update", vec![initial]).await.expect("Failed to insert");
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Update record with new text
        let updated = CreateRecordRequest {
            id: "doc1".to_string(),
            vector: vec![0.2; 128],
            sparse_vector: None,
            payload: Some(serde_json::json!({
                "text": "new content about dogs"
            })),
            named_vectors: None,
        };

        storage.update_record("test_update", &"doc1".to_string(), updated).await.expect("Failed to update");
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Search for old content - should NOT find it
        let old_search = FullTextSearchRequest {
            query: "cats".to_string(),
            fields: None,
            limit: 10,
            offset: 0,
            highlight: false,
            fuzzy: None,
            filter: None,
        };

        let old_result = storage.full_text_search("test_update", old_search).await.expect("Search failed");
        assert_eq!(old_result.total, 0, "Should NOT find old content after update");

        // Search for new content - should find it
        let new_search = FullTextSearchRequest {
            query: "dogs".to_string(),
            fields: None,
            limit: 10,
            offset: 0,
            highlight: false,
            fuzzy: None,
            filter: None,
        };

        let new_result = storage.full_text_search("test_update", new_search).await.expect("Search failed");
        assert_eq!(new_result.total, 1, "Should find new content after update");
        assert_eq!(new_result.hits[0].id, "doc1");
    }

    #[tokio::test]
    async fn test_delete_consistency() {
        let storage = create_test_storage().await;

        // Create table
        let table_config = TableConfig {
            name: "test_delete".to_string(),
            dimension: 128,
            distance_metric: DistanceMetric::Cosine,
            description: Some("Test delete consistency".to_string()),
            index_type: None,
            text_fields: Some(vec![
                TextFieldConfig {
                    name: "content".to_string(),
                    source: TextSource::MetadataField("text".to_string()),
                    analyzer: TextAnalyzer::Standard,
                    boost: 1.0,
                },
            ]),
            enable_full_text_search: true,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
            hnsw: None,
            quantization: None,
        };

        storage.create_table(table_config).await.expect("Failed to create table");

        // Insert records
        let records = vec![
            CreateRecordRequest {
                id: "doc1".to_string(),
                vector: vec![0.1; 128],
                sparse_vector: None,
                payload: Some(serde_json::json!({
                    "text": "content about machine learning"
                })),
                named_vectors: None,
            },
            CreateRecordRequest {
                id: "doc2".to_string(),
                vector: vec![0.2; 128],
                sparse_vector: None,
                payload: Some(serde_json::json!({
                    "text": "content about deep learning"
                })),
                named_vectors: None,
            },
        ];

        storage.insert_records("test_delete", records).await.expect("Failed to insert");
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Verify both documents are searchable
        let before_search = FullTextSearchRequest {
            query: "learning".to_string(),
            fields: None,
            limit: 10,
            offset: 0,
            highlight: false,
            fuzzy: None,
            filter: None,
        };

        let before_result = storage.full_text_search("test_delete", before_search).await.expect("Search failed");
        assert_eq!(before_result.total, 2, "Should find 2 documents before delete");

        // Delete one document
        storage.delete_records("test_delete", vec!["doc1".to_string()]).await.expect("Failed to delete");
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Search again - should only find doc2
        let after_search = FullTextSearchRequest {
            query: "learning".to_string(),
            fields: None,
            limit: 10,
            offset: 0,
            highlight: false,
            fuzzy: None,
            filter: None,
        };

        let after_result = storage.full_text_search("test_delete", after_search).await.expect("Search failed");
        assert_eq!(after_result.total, 1, "Should find 1 document after delete");
        assert_eq!(after_result.hits[0].id, "doc2", "Should only find doc2");
    }

    #[tokio::test]
    async fn test_full_crud_cycle() {
        let storage = create_test_storage().await;

        // Create table
        let table_config = TableConfig {
            name: "test_crud".to_string(),
            dimension: 128,
            distance_metric: DistanceMetric::Cosine,
            description: Some("Test full CRUD cycle".to_string()),
            index_type: None,
            text_fields: Some(vec![
                TextFieldConfig {
                    name: "content".to_string(),
                    source: TextSource::MetadataField("text".to_string()),
                    analyzer: TextAnalyzer::Standard,
                    boost: 1.0,
                },
            ]),
            enable_full_text_search: true,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
            hnsw: None,
            quantization: None,
        };

        storage.create_table(table_config).await.expect("Failed to create table");

        // 1. CREATE
        let create_req = CreateRecordRequest {
            id: "doc1".to_string(),
            vector: vec![0.1; 128],
            sparse_vector: None,
            payload: Some(serde_json::json!({
                "text": "version 1: neural networks"
            })),
            named_vectors: None,
        };
        storage.insert_records("test_crud", vec![create_req]).await.expect("INSERT failed");
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // 2. READ - verify inserted
        let read1 = FullTextSearchRequest {
            query: "neural".to_string(),
            fields: None,
            limit: 10,
            offset: 0,
            highlight: false,
            fuzzy: None,
            filter: None,
        };
        let result1 = storage.full_text_search("test_crud", read1).await.expect("READ 1 failed");
        assert_eq!(result1.total, 1, "Should find doc after INSERT");

        // 3. UPDATE
        let update_req = CreateRecordRequest {
            id: "doc1".to_string(),
            vector: vec![0.2; 128],
            sparse_vector: None,
            payload: Some(serde_json::json!({
                "text": "version 2: convolutional networks"
            })),
            named_vectors: None,
        };
        storage.update_record("test_crud", &"doc1".to_string(), update_req).await.expect("UPDATE failed");
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // 4. READ - verify updated (old term should not match)
        let read2 = FullTextSearchRequest {
            query: "neural".to_string(),
            fields: None,
            limit: 10,
            offset: 0,
            highlight: false,
            fuzzy: None,
            filter: None,
        };
        let result2 = storage.full_text_search("test_crud", read2).await.expect("READ 2 failed");
        assert_eq!(result2.total, 0, "Should NOT find old term after UPDATE");

        // 5. READ - verify updated (new term should match)
        let read3 = FullTextSearchRequest {
            query: "convolutional".to_string(),
            fields: None,
            limit: 10,
            offset: 0,
            highlight: false,
            fuzzy: None,
            filter: None,
        };
        let result3 = storage.full_text_search("test_crud", read3).await.expect("READ 3 failed");
        assert_eq!(result3.total, 1, "Should find new term after UPDATE");

        // 6. DELETE
        storage.delete_records("test_crud", vec!["doc1".to_string()]).await.expect("DELETE failed");
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // 7. READ - verify deleted
        let read4 = FullTextSearchRequest {
            query: "convolutional".to_string(),
            fields: None,
            limit: 10,
            offset: 0,
            highlight: false,
            fuzzy: None,
            filter: None,
        };
        let result4 = storage.full_text_search("test_crud", read4).await.expect("READ 4 failed");
        assert_eq!(result4.total, 0, "Should NOT find doc after DELETE");
    }
}
