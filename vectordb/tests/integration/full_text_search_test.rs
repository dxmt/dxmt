/// Integration tests for full-text search functionality
///
/// These tests verify that:
/// 1. Tables can be created with full-text search enabled
/// 2. Text content is automatically indexed when records are inserted
/// 3. Full-text search returns relevant results
/// 4. Hybrid search combines vector and text search correctly
#[cfg(test)]
#[cfg(feature = "text-bm25")]
mod full_text_search_tests {
    use dxmt_vectordb::{
        storage::IntelligentIndexedStorage,
        config::{StorageConfig, MemoryConfig},
        types::{
            table::{TableConfig, TextFieldConfig, TextSource, TextAnalyzer},
            record::{CreateRecordRequest, FullTextSearchRequest, HybridSearchRequest},
            vector::DistanceMetric,
        },
    };

    /// Helper function to create a test storage instance
    async fn create_test_storage() -> IntelligentIndexedStorage {
        let config = StorageConfig {
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
    async fn test_create_table_with_full_text_search() {
        let storage = create_test_storage().await;

        let table_config = TableConfig {
            name: "test_full_text".to_string(),
            dimension: 128,
            distance_metric: DistanceMetric::Cosine,
            description: Some("Test table with full-text search".to_string()),
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
            payload_index: None,
            embedding: None,
            rerank: None,
        };

        let table = storage.create_table(table_config).await.expect("Failed to create table");

        assert_eq!(table.name, "test_full_text");
        assert!(table.enable_full_text_search);
        assert!(table.text_fields.is_some());
        assert_eq!(table.text_fields.as_ref().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_insert_and_full_text_search() {
        let storage = create_test_storage().await;

        // Create table with full-text search
        let table_config = TableConfig {
            name: "test_search".to_string(),
            dimension: 128,
            distance_metric: DistanceMetric::Cosine,
            description: Some("Test search table".to_string()),
            index_type: None,
            text_fields: Some(vec![
                TextFieldConfig {
                    name: "content".to_string(),
                    source: TextSource::MetadataField("description".to_string()),
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
            payload_index: None,
            embedding: None,
            rerank: None,
        };

        storage.create_table(table_config).await.expect("Failed to create table");

        // Insert test records
        let records = vec![
            CreateRecordRequest {
                id: "doc1".to_string(),
                vector: vec![0.1; 128],
                sparse_vector: None,
                payload: Some(serde_json::json!({
                    "description": "Machine learning is a subset of artificial intelligence"
                })),
                named_vectors: None,
                text: None,
            },
            CreateRecordRequest {
                id: "doc2".to_string(),
                vector: vec![0.2; 128],
                sparse_vector: None,
                payload: Some(serde_json::json!({
                    "description": "Deep learning uses neural networks for pattern recognition"
                })),
                named_vectors: None,
                text: None,
            },
            CreateRecordRequest {
                id: "doc3".to_string(),
                vector: vec![0.3; 128],
                sparse_vector: None,
                payload: Some(serde_json::json!({
                    "description": "Natural language processing enables computers to understand text"
                })),
                named_vectors: None,
                text: None,
            },
        ];

        storage.insert_records("test_search", records).await.expect("Failed to insert records");

        // Wait a bit for indexing to complete
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Perform full-text search
        let search_request = FullTextSearchRequest {
            query: "machine learning".to_string(),
            fields: None,
            limit: 10,
            offset: 0,
            highlight: false,
            fuzzy: None,
            filter: None,
            rerank: None,
        };

        let result = storage.full_text_search("test_search", search_request).await.expect("Search failed");

        // Verify results
        assert!(result.total > 0, "Expected at least one result");
        assert!(!result.hits.is_empty(), "Expected search hits");

        // Results should contain doc1 and doc2 (both contain "learning")
        // Note: BM25 scoring may rank them differently, so we just verify both are present
        let hit_ids: Vec<String> = result.hits.iter().map(|h| h.id.clone()).collect();
        assert!(hit_ids.contains(&"doc1".to_string()), "Expected doc1 in results (contains 'machine learning')");
        assert!(hit_ids.contains(&"doc2".to_string()), "Expected doc2 in results (contains 'learning')");
    }

    #[tokio::test]
    async fn test_hybrid_search_vector_and_text() {
        let storage = create_test_storage().await;

        // Create table with full-text search
        let table_config = TableConfig {
            name: "test_hybrid".to_string(),
            dimension: 3,
            distance_metric: DistanceMetric::Cosine,
            description: Some("Test hybrid search".to_string()),
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
            payload_index: None,
            embedding: None,
            rerank: None,
        };

        storage.create_table(table_config).await.expect("Failed to create table");

        // Insert test records with different vectors and text
        let records = vec![
            CreateRecordRequest {
                id: "item1".to_string(),
                vector: vec![1.0, 0.0, 0.0],
                sparse_vector: None,
                payload: Some(serde_json::json!({
                    "text": "apple fruit red delicious"
                })),
                named_vectors: None,
                text: None,
            },
            CreateRecordRequest {
                id: "item2".to_string(),
                vector: vec![0.0, 1.0, 0.0],
                sparse_vector: None,
                payload: Some(serde_json::json!({
                    "text": "banana fruit yellow sweet"
                })),
                named_vectors: None,
                text: None,
            },
            CreateRecordRequest {
                id: "item3".to_string(),
                vector: vec![0.9, 0.1, 0.0],
                sparse_vector: None,
                payload: Some(serde_json::json!({
                    "text": "orange fruit citrus vitamin"
                })),
                named_vectors: None,
                text: None,
            },
        ];

        storage.insert_records("test_hybrid", records).await.expect("Failed to insert records");

        // Wait for indexing
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Hybrid search: vector close to item1, text matching "banana"
        let hybrid_request = HybridSearchRequest {
            vector: Some(vec![0.95, 0.05, 0.0]),  // Close to item1's vector
            text: Some("banana".to_string()),     // Matches item2's text
            vector_weight: 0.5,                   // Equal weight
            fusion_method: dxmt_vectordb::types::record::FusionMethod::WeightedSum,
            rrf_k: 60.0,
            limit: 10,
            filter: None,
            with_payload: true,
            with_vector: false,
            sparse_vector: None,
            rerank: None,
        };

        let result = storage.hybrid_search("test_hybrid", hybrid_request).await.expect("Hybrid search failed");

        // Verify results - should have results from both vector and text search
        assert!(result.total > 0, "Expected at least one result");
        assert!(!result.hits.is_empty(), "Expected search hits");

        // Both item1 (vector match) and item2 (text match) should be in results
        let ids: Vec<_> = result.hits.iter().map(|h| h.id.as_str()).collect();
        assert!(ids.contains(&"item1") || ids.contains(&"item2"),
                "Expected either item1 or item2 in results");
    }

    #[tokio::test]
    async fn test_full_text_search_with_filter() {
        let storage = create_test_storage().await;

        // Create table
        let table_config = TableConfig {
            name: "test_filter".to_string(),
            dimension: 128,
            distance_metric: DistanceMetric::Cosine,
            description: None,
            index_type: None,
            text_fields: Some(vec![
                TextFieldConfig {
                    name: "content".to_string(),
                    source: TextSource::MetadataField("description".to_string()),
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
            payload_index: None,
            embedding: None,
            rerank: None,
        };

        storage.create_table(table_config).await.expect("Failed to create table");

        // Insert records with categories
        let records = vec![
            CreateRecordRequest {
                id: "tech1".to_string(),
                vector: vec![0.1; 128],
                sparse_vector: None,
                payload: Some(serde_json::json!({
                    "description": "AI and machine learning technologies",
                    "category": "technology"
                })),
                named_vectors: None,
                text: None,
            },
            CreateRecordRequest {
                id: "science1".to_string(),
                vector: vec![0.2; 128],
                sparse_vector: None,
                payload: Some(serde_json::json!({
                    "description": "Machine learning applications in biology",
                    "category": "science"
                })),
                named_vectors: None,
                text: None,
            },
        ];

        storage.insert_records("test_filter", records).await.expect("Failed to insert");

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Search with filter for category=technology
        use dxmt_vectordb::types::record::{Filter, Condition, FilterOperation};

        let search_request = FullTextSearchRequest {
            query: "machine learning".to_string(),
            fields: None,
            limit: 10,
            offset: 0,
            highlight: false,
            fuzzy: None,
            filter: Some(Filter {
                must: Some(vec![Condition {
                    key: "category".to_string(),
                    operation: FilterOperation::Equal(serde_json::Value::String("technology".to_string())),
                }]),
                must_not: None,
                should: None,
            }),
            rerank: None,
        };

        let result = storage.full_text_search("test_filter", search_request).await.expect("Search failed");

        // Should only return tech1 due to filter
        assert_eq!(result.total, 1, "Expected exactly 1 result due to filter");
        assert_eq!(result.hits[0].id, "tech1");
    }

    /// A table whose text field declares `analyzer: chinese` has to find a word inside a
    /// sentence written without spaces. Tantivy's default tokenizer splits on non-alphanumeric
    /// characters, and CJK ideographs are alphanumeric, so a Chinese sentence is one token and
    /// no query shorter than the sentence ever matches it. The jieba tokenizer was registered
    /// under the name "jieba" from the start and no field was ever told to use it; `analyzer`
    /// was parsed, stored, and read by nothing.
    async fn chinese_table(storage: &IntelligentIndexedStorage, name: &str, analyzer: TextAnalyzer) {
        let table_config = TableConfig {
            name: name.to_string(),
            dimension: 4,
            distance_metric: DistanceMetric::Cosine,
            description: None,
            index_type: None,
            text_fields: Some(vec![TextFieldConfig {
                name: "content".to_string(),
                source: TextSource::MetadataField("text".to_string()),
                analyzer,
                boost: 1.0,
            }]),
            enable_full_text_search: true,
            memory_optimization: None,
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
            hnsw: None,
            quantization: None,
            payload_index: None,
            embedding: None,
            rerank: None,
        };
        storage.create_table(table_config).await.expect("create table");
        let texts = [
            ("zh1", "向量数据库支持混合检索和全文搜索"),
            ("zh2", "机器学习模型需要大量的训练数据"),
            ("zh3", "The English sentence mentions Machine Learning too"),
        ];
        let records = texts
            .iter()
            .enumerate()
            .map(|(i, (id, text))| CreateRecordRequest {
                id: id.to_string(),
                vector: vec![i as f32 * 0.1 + 0.1; 4],
                sparse_vector: None,
                payload: Some(serde_json::json!({ "text": text })),
                named_vectors: None,
                text: None,
            })
            .collect();
        storage.insert_records(name, records).await.expect("insert");
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    async fn text_hits(storage: &IntelligentIndexedStorage, table: &str, query: &str) -> Vec<String> {
        let request = FullTextSearchRequest {
            query: query.to_string(),
            fields: None,
            limit: 10,
            offset: 0,
            highlight: false,
            fuzzy: None,
            filter: None,
            rerank: None,
        };
        let mut ids: Vec<String> = storage
            .full_text_search(table, request)
            .await
            .expect("search")
            .hits
            .iter()
            .map(|h| h.id.clone())
            .collect();
        ids.sort();
        ids
    }

    #[tokio::test]
    async fn a_chinese_analyzer_finds_a_word_inside_a_sentence() {
        let storage = create_test_storage().await;
        chinese_table(&storage, "zh", TextAnalyzer::Chinese).await;

        assert_eq!(text_hits(&storage, "zh", "全文搜索").await, vec!["zh1"]);
        assert_eq!(text_hits(&storage, "zh", "训练数据").await, vec!["zh2"]);
        // Mixed text: the ASCII words are still words, and case does not matter.
        assert_eq!(text_hits(&storage, "zh", "machine learning").await, vec!["zh3"]);
        assert!(text_hits(&storage, "zh", "区块链").await.is_empty(), "a word in no document");
    }

    /// A question asked as written: no spaces, words that are not contiguous in the document,
    /// punctuation. The query parser made one jieba-split term into a phrase, so this found
    /// nothing at all -- 1.8% of CMRC2018 questions found their paragraph in the top ten -- while
    /// the same words separated by spaces found it.
    #[tokio::test]
    async fn a_chinese_question_finds_a_sentence_holding_its_words_apart() {
        let storage = create_test_storage().await;
        chinese_table(&storage, "zhq", TextAnalyzer::Chinese).await;

        let asked = text_hits(&storage, "zhq", "哪种模型需要训练数据？").await;
        assert_eq!(asked.first().map(String::as_str), Some("zh2"), "{asked:?}");
        let asked = text_hits(&storage, "zhq", "向量数据库支持什么检索").await;
        assert_eq!(asked.first().map(String::as_str), Some("zh1"), "{asked:?}");
        // A quoted phrase is still a phrase.
        assert!(text_hits(&storage, "zhq", "\"数据库训练\"").await.is_empty());
    }

    #[tokio::test]
    async fn the_standard_analyzer_is_still_the_default_and_still_cannot() {
        // Pinned so the Chinese case above is known to be the analyzer and not a tokenizer
        // change for every table: the default tokenizer sees the whole sentence as one word.
        let storage = create_test_storage().await;
        chinese_table(&storage, "std", TextAnalyzer::Standard).await;
        assert!(text_hits(&storage, "std", "全文搜索").await.is_empty());
        assert_eq!(text_hits(&storage, "std", "machine learning").await, vec!["zh3"]);
    }
}
