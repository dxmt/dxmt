//! Paging by cursor while the table is written to.
//!
//! The property that matters: every record present for the whole scroll is returned exactly once,
//! and nothing is returned twice. Paging by `offset` has neither — the list is newest first, so a
//! record inserted mid-scroll pushes everything back one and the next page repeats a record.

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::{payload_index::ListCursor, IntelligentIndexedStorage},
    types::{
        record::{Condition, CreateRecordRequest, Filter, FilterOperation},
        table::TableConfig,
        vector::DistanceMetric,
    },
};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};

fn record(id: &str, keep: bool) -> CreateRecordRequest {
    CreateRecordRequest {
        id: id.to_string(),
        vector: vec![1.0, id.len() as f32 / 10.0],
        sparse_vector: None,
        named_vectors: None,
        payload: Some(json!({ "keep": keep })),
    }
}

async fn storage(wal: Option<std::path::PathBuf>) -> IntelligentIndexedStorage {
    let s = IntelligentIndexedStorage::new_with_wal(
        &StorageConfig {
            r#type: "memory".to_string(),
            memory: Some(MemoryConfig {
                max_size_mb: 128,
                eviction_policy: "lru".to_string(),
            }),
            ..Default::default()
        },
        wal,
    )
    .await
    .unwrap();
    s.create_table(TableConfig {
        name: "t".into(),
        dimension: 2,
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
    })
    .await
    .unwrap();
    s
}

async fn scroll_while_writing(s: &IntelligentIndexedStorage, filter: Option<&Filter>) {
    // Forty records; a few milliseconds apart so the order is by time, not only by id.
    for i in 0..40 {
        s.insert_records("t", vec![record(&format!("r{i:02}"), i % 4 != 0)])
            .await
            .unwrap();
        if i % 10 == 0 {
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
    }
    // The records the scroll has to find: all of them, or only the ones the filter keeps.
    let wanted = |keep: bool| filter.is_none() || keep;
    let before: BTreeSet<String> = (0..40)
        .filter(|i| wanted(i % 4 != 0))
        .map(|i| format!("r{i:02}"))
        .collect();

    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    let mut deleted = BTreeSet::new();
    let mut cursor: Option<ListCursor> = None;
    let mut page_no = 0;
    loop {
        let (page, _, next) = s
            .list_records_after("t", filter, cursor.as_ref(), 6, false)
            .await
            .unwrap();
        for r in &page {
            *seen.entry(r.id.clone()).or_default() += 1;
        }
        // Between pages: insert new records (newer than everything, so before the cursor),
        // update one not yet reached (keeps its created_at), delete one not yet reached.
        page_no += 1;
        s.insert_records("t", vec![record(&format!("new{page_no}"), true)])
            .await
            .unwrap();
        let unreached: Vec<&String> = before
            .iter()
            .filter(|id| !seen.contains_key(*id) && !deleted.contains(*id))
            .collect();
        if let Some(id) = unreached.first() {
            s.apply_update(
                "t",
                id,
                None,
                Some(json!({"keep": true, "touched": page_no})),
                false,
                false,
            )
            .await
            .unwrap();
        }
        if let Some(id) = unreached.get(1) {
            s.delete_records("t", vec![id.to_string()]).await.unwrap();
            deleted.insert(id.to_string());
        }
        match next {
            Some(n) => cursor = Some(n),
            None => break,
        }
    }

    let twice: Vec<_> = seen.iter().filter(|(_, n)| **n > 1).collect();
    assert!(twice.is_empty(), "returned more than once: {twice:?}");
    let throughout: BTreeSet<String> = before.difference(&deleted).cloned().collect();
    let missing: Vec<_> = throughout
        .iter()
        .filter(|id| !seen.contains_key(*id))
        .collect();
    assert!(
        missing.is_empty(),
        "present throughout and never returned: {missing:?}"
    );
    assert!(
        seen.keys().all(|id| !id.starts_with("new")),
        "a record inserted mid-scroll is newer than the cursor and belongs to no later page"
    );
}

fn keep() -> Filter {
    Filter {
        must: Some(vec![Condition {
            key: "keep".into(),
            operation: FilterOperation::Equal(json!(true)),
        }]),
        must_not: None,
        should: None,
    }
}

#[tokio::test]
async fn a_scroll_returns_each_record_once_in_memory() {
    scroll_while_writing(&storage(None).await, None).await;
}

#[tokio::test]
async fn a_scroll_returns_each_record_once_on_disk() {
    let dir = tempfile::tempdir().unwrap();
    scroll_while_writing(&storage(Some(dir.path().join("wal"))).await, None).await;
}

#[tokio::test]
async fn a_filtered_scroll_returns_each_match_once() {
    let dir = tempfile::tempdir().unwrap();
    scroll_while_writing(&storage(Some(dir.path().join("wal"))).await, Some(&keep())).await;
}

/// The last page says it is the last, rather than sending the caller for an empty one.
#[tokio::test]
async fn the_last_page_has_no_next_cursor() {
    let s = storage(None).await;
    s.insert_records(
        "t",
        (0..12).map(|i| record(&format!("r{i:02}"), true)).collect(),
    )
    .await
    .unwrap();
    let (first, total, next) = s
        .list_records_after("t", None, None, 6, false)
        .await
        .unwrap();
    assert_eq!((first.len(), total), (6, 12));
    let (second, _, next) = s
        .list_records_after("t", None, next.as_ref(), 6, false)
        .await
        .unwrap();
    assert_eq!(second.len(), 6);
    assert!(
        next.is_none(),
        "exactly two pages, and the second is the last"
    );
}
