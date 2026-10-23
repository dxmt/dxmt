//! Changing a table's text index after it holds data: `PATCH /tables/{t}` with `text_fields`
//! rebuilds it from the records without stopping reads or writes, and the result survives a
//! restart -- or is rebuilt at the next start if it was built for another config.
#![cfg(feature = "text-bm25")]

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{CreateRecordRequest, FullTextSearchRequest},
        table::{TableAlterRequest, TableConfig, TextAnalyzer, TextFieldConfig, TextSource},
        vector::DistanceMetric,
    },
};
use std::path::Path;
use std::sync::Arc;

fn config() -> StorageConfig {
    StorageConfig {
        memory: Some(MemoryConfig {
            max_size_mb: 128,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    }
}

async fn open(dir: &Path) -> IntelligentIndexedStorage {
    IntelligentIndexedStorage::new_with_wal(&config(), Some(dir.to_path_buf()))
        .await
        .unwrap()
}

fn fields(analyzer: TextAnalyzer) -> Vec<TextFieldConfig> {
    vec![TextFieldConfig {
        name: "content".to_string(),
        source: TextSource::MetadataField("text".to_string()),
        analyzer,
        boost: 1.0,
    }]
}

fn table(name: &str, text: Option<TextAnalyzer>) -> TableConfig {
    TableConfig {
        name: name.to_string(),
        dimension: 4,
        distance_metric: DistanceMetric::Cosine,
        description: None,
        index_type: None,
        enable_full_text_search: text.is_some(),
        text_fields: text.map(fields),
        memory_optimization: None,
        payload_schema: None,
        schema_version: 1,
        named_vectors: None,
        hnsw: None,
        quantization: None,
        payload_index: None,
        embedding: None,
        rerank: None,
    }
}

fn record(id: &str, text: &str) -> CreateRecordRequest {
    CreateRecordRequest {
        id: id.to_string(),
        vector: vec![0.5, 0.5, 0.5, 0.5],
        sparse_vector: None,
        named_vectors: None,
        payload: Some(serde_json::json!({ "text": text })),
        text: None,
    }
}

async fn hits(storage: &IntelligentIndexedStorage, table: &str, query: &str) -> Vec<String> {
    let mut ids: Vec<String> = storage
        .full_text_search(
            table,
            FullTextSearchRequest {
                query: query.to_string(),
                fields: None,
                limit: 10_000,
                offset: 0,
                highlight: false,
                fuzzy: None,
                filter: None,
                rerank: None,
            },
        )
        .await
        .expect("search")
        .hits
        .into_iter()
        .map(|h| h.id)
        .collect();
    ids.sort();
    ids
}

fn alter_text(analyzer: TextAnalyzer) -> TableAlterRequest {
    TableAlterRequest {
        text_fields: Some(fields(analyzer)),
        ..Default::default()
    }
}

async fn settle() {
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_analyzer_changes_on_a_table_that_holds_data_and_survives_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    {
        let storage = open(dir.path()).await;
        storage
            .create_table(table("t", Some(TextAnalyzer::Standard)))
            .await
            .unwrap();
        storage
            .insert_records(
                "t",
                vec![
                    record("zh", "向量数据库支持混合检索和全文搜索"),
                    record("en", "machine learning needs training data"),
                ],
            )
            .await
            .unwrap();
        settle().await;
        // The standard analyzer keeps a Chinese sentence as one token: a word inside it is not found.
        assert!(hits(&storage, "t", "全文搜索").await.is_empty());

        storage
            .alter_table("t", alter_text(TextAnalyzer::Chinese))
            .await
            .unwrap();
        assert_eq!(hits(&storage, "t", "全文搜索").await, vec!["zh"]);
        assert_eq!(hits(&storage, "t", "machine learning").await, vec!["en"]);
    }
    // Restart: the new config came back from the log, and the index built for it is opened,
    // not rebuilt.
    let storage = open(dir.path()).await;
    assert_eq!(hits(&storage, "t", "全文搜索").await, vec!["zh"]);
    let root = dir.path().join("fulltext");
    let generations: Vec<_> =
        std::fs::read_dir(root.read_dir().unwrap().next().unwrap().unwrap().path())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with('g'))
            .collect();
    assert_eq!(
        generations.len(),
        1,
        "old generations are removed: {generations:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_text_search_is_turned_on_for_a_table_created_without_it() {
    let dir = tempfile::tempdir().unwrap();
    let storage = open(dir.path()).await;
    storage.create_table(table("plain", None)).await.unwrap();
    storage
        .insert_records("plain", vec![record("a", "hnsw builds a graph index")])
        .await
        .unwrap();
    // Was accepted and ignored: `enable_full_text_search: true` flipped a flag and built nothing.
    storage
        .alter_table("plain", alter_text(TextAnalyzer::Standard))
        .await
        .unwrap();
    assert_eq!(hits(&storage, "plain", "graph").await, vec!["a"]);

    // And off again: text search says so rather than answering empty.
    storage
        .alter_table(
            "plain",
            TableAlterRequest {
                enable_full_text_search: Some(false),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let err = storage
        .full_text_search(
            "plain",
            FullTextSearchRequest {
                query: "graph".to_string(),
                fields: None,
                limit: 10,
                offset: 0,
                highlight: false,
                fuzzy: None,
                filter: None,
                rerank: None,
            },
        )
        .await
        .map(|_| ())
        .unwrap_err();
    assert!(
        err.to_string().to_lowercase().contains("full-text")
            || err.to_string().contains("not enabled"),
        "{err}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn writes_during_a_rebuild_are_in_the_new_index_once_each() {
    let dir = tempfile::tempdir().unwrap();
    let storage = Arc::new(open(dir.path()).await);
    storage
        .create_table(table("t", Some(TextAnalyzer::Standard)))
        .await
        .unwrap();
    let before: Vec<CreateRecordRequest> = (0..3000)
        .map(|i| record(&format!("old{i}"), &format!("common oldtok{i}")))
        .collect();
    for chunk in before.chunks(500) {
        storage.insert_records("t", chunk.to_vec()).await.unwrap();
    }

    // A writer inserting new records and rewriting old ones while the index is rebuilt.
    let writer = {
        let storage = storage.clone();
        tokio::spawn(async move {
            for i in 0..60 {
                storage
                    .insert_records(
                        "t",
                        vec![
                            record(&format!("new{i}"), &format!("common newtok{i}")),
                            record(&format!("old{i}"), &format!("common rewritten{i}")),
                        ],
                    )
                    .await
                    .unwrap();
                tokio::task::yield_now().await;
            }
        })
    };
    storage
        .alter_table("t", alter_text(TextAnalyzer::Chinese))
        .await
        .unwrap();
    writer.await.unwrap();
    settle().await;

    for i in 0..60 {
        assert_eq!(
            hits(&storage, "t", &format!("newtok{i}")).await,
            vec![format!("new{i}")]
        );
        assert_eq!(
            hits(&storage, "t", &format!("rewritten{i}")).await,
            vec![format!("old{i}")]
        );
        assert!(
            hits(&storage, "t", &format!("oldtok{i}")).await.is_empty(),
            "old{i}'s replaced text is still indexed"
        );
    }
    // One document per record: every record matches "common" exactly once.
    let all = hits(&storage, "t", "common").await;
    let mut dedup = all.clone();
    dedup.dedup();
    assert_eq!(all.len(), dedup.len(), "a record is indexed twice");
    assert_eq!(all.len(), 3060);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_index_built_for_another_config_is_rebuilt_at_the_next_start() {
    let dir = tempfile::tempdir().unwrap();
    {
        let storage = open(dir.path()).await;
        storage
            .create_table(table("t", Some(TextAnalyzer::Chinese)))
            .await
            .unwrap();
        storage
            .insert_records("t", vec![record("zh", "向量数据库支持混合检索和全文搜索")])
            .await
            .unwrap();
        settle().await;
        storage.snapshot().await.unwrap();
    }
    // As if a rebuild was cut short: the current generation says it was built for something else.
    let table_root = std::fs::read_dir(dir.path().join("fulltext"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let current = std::fs::read_to_string(table_root.join("CURRENT")).unwrap();
    std::fs::write(
        table_root
            .join(current.trim())
            .join("dxmt-text-config.json"),
        "{\"fields\":\"something else\"}",
    )
    .unwrap();

    let storage = open(dir.path()).await;
    assert_eq!(hits(&storage, "t", "全文搜索").await, vec!["zh"]);
    let now = std::fs::read_to_string(table_root.join("CURRENT")).unwrap();
    assert_ne!(now.trim(), current.trim(), "a new generation was built");
}

/// Records held in memory have no rows to page through; the rebuild copies them in one pass.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_rebuild_copies_records_held_in_memory() {
    let storage = IntelligentIndexedStorage::new(&config()).await.unwrap();
    storage.create_table(table("m", None)).await.unwrap();
    storage
        .insert_records("m", vec![record("a", "向量数据库支持全文搜索")])
        .await
        .unwrap();
    storage
        .alter_table("m", alter_text(TextAnalyzer::Chinese))
        .await
        .unwrap();
    assert_eq!(hits(&storage, "m", "全文搜索").await, vec!["a"]);
}
