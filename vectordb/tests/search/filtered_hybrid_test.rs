//! A filter narrows keyword and hybrid search to the records it admits -- all of them, not
//! whichever happened to rank in the first few dozen of the whole table.
//!
//! Both searches took a fixed pool from the unfiltered ranking (three times the limit for
//! keywords, ten times for hybrid) and filtered that. A filter that admits a small share of the
//! table therefore found few or none of its records: on the memory API, where every recall is
//! restricted to a user's or a conversation's tags, a profile kept beside nine thousand other
//! memories was found by dense search and by neither keyword nor hybrid search -- `recall` with
//! `kind: profile` answered an empty list. Dense search pre-filters and was never affected.
//!
//! Here 400 records all contain the query word; three carry the tag `rare`, and their vectors
//! are the farthest from the query, so in the unfiltered rankings they come last.

#![cfg(feature = "text-bm25")]

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{CreateRecordRequest, FullTextSearchRequest, HybridSearchRequest},
        table::{TableConfig, TextAnalyzer, TextFieldConfig, TextSource},
        vector::DistanceMetric,
    },
};

async fn storage_with_rare_records() -> IntelligentIndexedStorage {
    let storage = IntelligentIndexedStorage::new(&StorageConfig {
        memory: Some(MemoryConfig {
            max_size_mb: 128,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    })
    .await
    .unwrap();
    storage
        .create_table(TableConfig {
            name: "t".to_string(),
            dimension: 4,
            distance_metric: DistanceMetric::Cosine,
            description: None,
            index_type: None,
            // `name` is the index's own field (`content`); the payload key is the source.
            text_fields: Some(vec![TextFieldConfig {
                name: "content".to_string(),
                source: TextSource::MetadataField("text".to_string()),
                analyzer: TextAnalyzer::Standard,
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
        })
        .await
        .unwrap();
    let records = (0..400)
        .map(|i| {
            let rare = i < 3;
            CreateRecordRequest {
                id: format!("r{i:03}"),
                // The common records sit near the query; the rare ones far from it.
                vector: if rare {
                    vec![0.0, 1.0, 0.1 * i as f32, 0.0]
                } else {
                    vec![1.0, 0.001 * i as f32, 0.0, 0.0]
                },
                sparse_vector: None,
                payload: Some(serde_json::json!({
                    "text": if rare {
                        format!("apple orchard note {i} with several more words in it")
                    } else {
                        format!("apple {i}")
                    },
                    "tag": if rare { "rare" } else { "common" },
                })),
                named_vectors: None,
                text: None,
            }
        })
        .collect();
    storage.insert_records("t", records).await.unwrap();
    // The keyword index commits in the background; wait until it holds all 400.
    for _ in 0..100 {
        let request: FullTextSearchRequest =
            serde_json::from_value(serde_json::json!({"query": "apple", "limit": 1000})).unwrap();
        if storage
            .full_text_search("t", request)
            .await
            .unwrap()
            .hits
            .len()
            == 400
        {
            return storage;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    panic!("the keyword index never caught up");
}

fn rare() -> serde_json::Value {
    serde_json::json!({"must": [{"key": "tag", "operation": {"type": "Equal", "value": "rare"}}]})
}

fn ids<T>(hits: &[T], id: impl Fn(&T) -> &str) -> Vec<String> {
    hits.iter().map(|h| id(h).to_string()).collect()
}

#[tokio::test]
async fn keyword_search_finds_every_record_a_narrow_filter_admits() {
    let storage = storage_with_rare_records().await;
    let request: FullTextSearchRequest =
        serde_json::from_value(serde_json::json!({"query": "apple", "limit": 3, "filter": rare()}))
            .unwrap();
    let all = storage.full_text_search("t", request).await.unwrap();
    let mut found = ids(&all.hits, |h| &h.id);
    assert_eq!(found.len(), 3, "a narrow filter lost records: {found:?}");
    let order = found.clone();
    found.sort();
    assert_eq!(found, vec!["r000", "r001", "r002"]);

    // `offset` counts records the filter admits, so the pages are the long answer cut up.
    let mut paged = Vec::new();
    for offset in 0..4 {
        let request: FullTextSearchRequest = serde_json::from_value(serde_json::json!({
            "query": "apple", "limit": 1, "offset": offset, "filter": rare()
        }))
        .unwrap();
        paged.extend(ids(
            &storage.full_text_search("t", request).await.unwrap().hits,
            |h| &h.id,
        ));
    }
    assert_eq!(paged, order, "offset is applied after the filter");
}

#[tokio::test]
async fn hybrid_search_finds_every_record_a_narrow_filter_admits() {
    let storage = storage_with_rare_records().await;
    let request: HybridSearchRequest = serde_json::from_value(serde_json::json!({
        "vector": [1.0, 0.0, 0.0, 0.0],
        "text": "apple",
        "limit": 3,
        "filter": rare(),
    }))
    .unwrap();
    let result = storage.hybrid_search("t", request).await.unwrap();
    let mut found = ids(&result.hits, |h| &h.id);
    found.sort();
    assert_eq!(
        found,
        vec!["r000", "r001", "r002"],
        "a narrow filter lost records"
    );
}
