/// Shared test fixtures and utilities
///
/// This module provides common test data, mock storage instances,
/// and helper functions to reduce code duplication across tests.

use dxmt_vectordb::{
    storage::IntelligentIndexedStorage,
    config::{StorageConfig, MemoryConfig},
    types::{
        table::{TableConfig, TextFieldConfig, TextSource, TextAnalyzer},
        record::CreateRecordRequest,
        vector::DistanceMetric,
    },
};

/// Create a test storage instance with default memory configuration
pub async fn create_test_storage() -> IntelligentIndexedStorage {
    create_test_storage_with_size(1024).await
}

/// Create a test storage instance with specified memory size
pub async fn create_test_storage_with_size(max_size_mb: usize) -> IntelligentIndexedStorage {
    let config = StorageConfig {
        r#type: "memory".to_string(),
        memory: Some(MemoryConfig {
            max_size_mb,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    };
    IntelligentIndexedStorage::new(&config)
        .await
        .expect("Failed to create test storage")
}

/// Create a simple table configuration for testing
pub fn create_simple_table_config(name: &str, dimension: usize) -> TableConfig {
    TableConfig {
        name: name.to_string(),
        dimension,
        distance_metric: DistanceMetric::Cosine,
        description: Some(format!("Test table: {}", name)),
        index_type: None,
        text_fields: None,
        enable_full_text_search: false,
        memory_optimization: None,
        payload_schema: None,
        schema_version: 1,
        named_vectors: None,
    }
}

/// Create a table configuration with full-text search enabled
#[cfg(feature = "text-bm25")]
pub fn create_fts_table_config(name: &str, dimension: usize) -> TableConfig {
    TableConfig {
        name: name.to_string(),
        dimension,
        distance_metric: DistanceMetric::Cosine,
        description: Some(format!("Test FTS table: {}", name)),
        index_type: None,
        text_fields: Some(vec![
            TextFieldConfig {
                name: "content".to_string(),
                source: TextSource::MetadataField("text".to_string()),
                analyzer: TextAnalyzer::Standard,
                boost: 1.0,
            }
        ]),
        enable_full_text_search: true,
        memory_optimization: None,
        payload_schema: None,
        schema_version: 1,
        named_vectors: None,
    }
}

/// Generate test vectors with specified dimension
pub fn generate_test_vectors(count: usize, dimension: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            (0..dimension)
                .map(|j| ((i + j) as f32 * 0.1) % 1.0)
                .collect()
        })
        .collect()
}

/// Generate normalized test vectors (unit vectors)
pub fn generate_normalized_test_vectors(count: usize, dimension: usize) -> Vec<Vec<f32>> {
    generate_test_vectors(count, dimension)
        .into_iter()
        .map(|mut v| {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|x| *x /= norm);
            }
            v
        })
        .collect()
}

/// Create test records with sequential IDs
pub fn create_test_records(vectors: Vec<Vec<f32>>) -> Vec<CreateRecordRequest> {
    vectors
        .into_iter()
        .enumerate()
        .map(|(i, vector)| CreateRecordRequest {
            id: format!("test_{}", i),
            vector,
            sparse_vector: None,
            payload: Some(serde_json::json!({
                "index": i,
                "description": format!("Test record {}", i)
            })),
            named_vectors: None,
        })
        .collect()
}

/// Create test records with custom metadata
pub fn create_test_records_with_metadata(
    vectors: Vec<Vec<f32>>,
    metadata: Vec<serde_json::Value>,
) -> Vec<CreateRecordRequest> {
    vectors
        .into_iter()
        .zip(metadata)
        .enumerate()
        .map(|(i, (vector, payload))| CreateRecordRequest {
            id: format!("test_{}", i),
            vector,
            sparse_vector: None,
            payload: Some(payload),
            named_vectors: None,
        })
        .collect()
}

/// Sample dataset for SIFT-like testing (128D)
pub struct SiftSample {
    pub dimension: usize,
    pub train_vectors: Vec<Vec<f32>>,
    pub test_queries: Vec<Vec<f32>>,
}

impl SiftSample {
    pub fn new_small() -> Self {
        let dimension = 128;
        let train_vectors = generate_normalized_test_vectors(100, dimension);
        let test_queries = generate_normalized_test_vectors(10, dimension);

        Self {
            dimension,
            train_vectors,
            test_queries,
        }
    }

    pub fn new_medium() -> Self {
        let dimension = 128;
        let train_vectors = generate_normalized_test_vectors(1000, dimension);
        let test_queries = generate_normalized_test_vectors(100, dimension);

        Self {
            dimension,
            train_vectors,
            test_queries,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_test_storage() {
        let storage = create_test_storage().await;
        // Just verify it was created successfully
        assert!(storage.list_tables().await.is_ok());
    }

    #[test]
    fn test_generate_test_vectors() {
        let vectors = generate_test_vectors(10, 128);
        assert_eq!(vectors.len(), 10);
        assert!(vectors.iter().all(|v| v.len() == 128));
    }

    #[test]
    fn test_generate_normalized_test_vectors() {
        let vectors = generate_normalized_test_vectors(10, 128);
        assert_eq!(vectors.len(), 10);

        // Check that vectors are normalized (L2 norm ≈ 1.0)
        for v in vectors {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "Vector not normalized: norm = {}", norm);
        }
    }

    #[test]
    fn test_create_test_records() {
        let vectors = generate_test_vectors(5, 64);
        let records = create_test_records(vectors);

        assert_eq!(records.len(), 5);
        assert_eq!(records[0].id, "test_0");
        assert_eq!(records[4].id, "test_4");
    }

    #[test]
    fn test_sift_sample_small() {
        let sample = SiftSample::new_small();
        assert_eq!(sample.dimension, 128);
        assert_eq!(sample.train_vectors.len(), 100);
        assert_eq!(sample.test_queries.len(), 10);
    }
}
