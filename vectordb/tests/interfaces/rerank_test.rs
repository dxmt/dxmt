//! A search asks, the table's rerank provider reorders the page.
//!
//! The provider is a real HTTP server (`tests/fake_provider/`), speaking Cohere's `/rerank`
//! shape or text-embeddings-inference's, and it scores a document by how many query words it
//! contains. The embedding model is the same server's bag of words, so the dense order and
//! the reranked order can be made to disagree on purpose: a document that repeats one query
//! word many times is near in the bag-of-words space, and a document that contains every
//! query word once is what the reranker prefers.
//!
//! Two servers are run where the failure under test belongs to the reranker alone, so that
//! "the rerank provider answered 401" is not also "the embedding provider answered 401".

use std::sync::atomic::Ordering;

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{CreateRecordRequest, RerankRequest, SearchQuery},
        table::{EmbeddingConfig, RerankConfig, RerankFormat, TableAlterRequest, TableConfig},
        vector::DistanceMetric,
    },
    Error,
};

#[path = "../fake_provider/mod.rs"]
mod fake_provider;
use fake_provider::{overlap, Behaviour};

const DIM: usize = 32;

async fn serve(behaviour: Behaviour) -> (String, fake_provider::Provider) {
    fake_provider::serve(behaviour, DIM).await
}

fn storage_config() -> StorageConfig {
    StorageConfig {
        memory: Some(MemoryConfig {
            max_size_mb: 128,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    }
}

async fn open() -> IntelligentIndexedStorage {
    IntelligentIndexedStorage::new(&storage_config())
        .await
        .unwrap()
}

fn embedding(base_url: &str) -> EmbeddingConfig {
    EmbeddingConfig {
        model: "bag-of-words".to_string(),
        base_url: Some(base_url.to_string()),
        field: None,
        api_key_env: Some("DXMT_TEST_EMBEDDING_KEY_UNSET".to_string()),
        batch_size: None,
    }
}

fn reranker(base_url: &str, format: RerankFormat) -> RerankConfig {
    RerankConfig {
        model: "word-overlap".to_string(),
        base_url: base_url.to_string(),
        format,
        api_key_env: Some("DXMT_TEST_RERANK_KEY_UNSET".to_string()),
        field: None,
        top_n: None,
    }
}

fn table(
    name: &str,
    embedding: Option<EmbeddingConfig>,
    rerank: Option<RerankConfig>,
) -> TableConfig {
    TableConfig {
        name: name.to_string(),
        dimension: DIM,
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
        embedding,
        rerank,
    }
}

fn text_record(id: &str, text: &str) -> CreateRecordRequest {
    CreateRecordRequest {
        id: id.to_string(),
        vector: Vec::new(),
        text: Some(text.to_string()),
        sparse_vector: None,
        named_vectors: None,
        payload: None,
    }
}

const QUERY: &str = "borrow checker rejects aliasing";

/// A corpus where the dense order and the overlap order disagree. `near` repeats one query
/// word and nothing else, so the bag of words puts it first; `full` has every query word once,
/// among many others, so the bag of words puts it behind `near` and the overlap count puts
/// it first. `none` shares no word with the query.
fn corpus() -> Vec<CreateRecordRequest> {
    vec![
        text_record("near", "borrow borrow borrow borrow borrow"),
        text_record(
            "full",
            "the borrow checker rejects aliasing of a mutable reference across a call",
        ),
        text_record("two", "a checker that rejects a move after it happened"),
        text_record("none", "a database index answers a predicate quickly"),
    ]
}

fn query(k: usize, rerank: Option<RerankRequest>) -> SearchQuery {
    SearchQuery {
        text: Some(QUERY.to_string()),
        top_k: k,
        limit: k,
        with_payload: true,
        rerank,
        ..Default::default()
    }
}

fn ids(hits: &[dxmt_vectordb::types::record::SearchHit]) -> Vec<String> {
    hits.iter().map(|h| h.id.clone()).collect()
}

#[tokio::test]
async fn the_page_comes_back_in_the_providers_order_with_its_score() {
    let (embed, _) = serve(Behaviour::Normal).await;
    let (rank, provider) = serve(Behaviour::Normal).await;
    let storage = open().await;
    storage
        .create_table(table(
            "docs",
            Some(embedding(&embed)),
            Some(reranker(&rank, RerankFormat::Cohere)),
        ))
        .await
        .unwrap();
    storage.insert_records("docs", corpus()).await.unwrap();

    let plain = storage.search("docs", query(4, None)).await.unwrap().hits;
    assert_eq!(
        plain[0].id,
        "near",
        "the dense order should start at the repeater: {:?}",
        ids(&plain)
    );
    assert!(plain.iter().all(|h| h.rerank_score.is_none()));
    assert_eq!(
        provider.calls.load(Ordering::SeqCst),
        0,
        "no rerank was asked for"
    );

    let reranked = storage
        .search("docs", query(4, Some(RerankRequest::default())))
        .await
        .unwrap()
        .hits;
    assert_eq!(ids(&reranked)[0], "full", "ranked {:?}", ids(&reranked));
    assert_eq!(ids(&reranked)[3], "none");
    for hit in &reranked {
        let text = hit.payload.as_ref().unwrap()["text"].as_str().unwrap();
        assert_eq!(hit.rerank_score, Some(overlap(QUERY, text)), "{}", hit.id);
        assert!(hit.score.is_finite(), "the dense score stays on the hit");
    }
    assert_eq!(
        provider.calls.load(Ordering::SeqCst),
        1,
        "one call for the page"
    );
    assert_eq!(
        provider.seen.lock().unwrap().len(),
        4,
        "every candidate with text went to the provider"
    );
}

#[tokio::test]
async fn top_n_is_the_pool_and_the_page_is_cut_afterwards() {
    let (embed, _) = serve(Behaviour::Normal).await;
    let (rank, provider) = serve(Behaviour::Normal).await;
    let storage = open().await;
    storage
        .create_table(table(
            "docs",
            Some(embedding(&embed)),
            Some(reranker(&rank, RerankFormat::Cohere)),
        ))
        .await
        .unwrap();
    storage.insert_records("docs", corpus()).await.unwrap();

    // A page of one with a pool of three: the dense top three are `near`, `full`, `two` in
    // some order, and `full` wins the rerank. With a pool of one the page could only be `near`.
    let hits = storage
        .search(
            "docs",
            query(
                1,
                Some(RerankRequest {
                    top_n: Some(3),
                    field: None,
                }),
            ),
        )
        .await
        .unwrap()
        .hits;
    assert_eq!(ids(&hits), ["full"]);
    assert_eq!(
        provider.seen.lock().unwrap().len(),
        3,
        "the pool, not the page, was sent"
    );

    let narrow = storage
        .search(
            "docs",
            query(
                1,
                Some(RerankRequest {
                    top_n: Some(1),
                    field: None,
                }),
            ),
        )
        .await
        .unwrap()
        .hits;
    assert_eq!(
        ids(&narrow),
        ["near"],
        "a pool of one has nothing to reorder"
    );

    // The table's own `top_n` is the default behind a request that names none.
    storage
        .alter_table(
            "docs",
            TableAlterRequest {
                rerank: Some(RerankConfig {
                    top_n: Some(1),
                    ..reranker(&rank, RerankFormat::Cohere)
                }),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let table_default = storage
        .search("docs", query(1, Some(RerankRequest::default())))
        .await
        .unwrap()
        .hits;
    assert_eq!(ids(&table_default), ["near"]);
    assert_eq!(
        storage
            .get_table("docs")
            .await
            .unwrap()
            .rerank
            .unwrap()
            .top_n,
        Some(1),
        "PATCH keeps the config"
    );
}

#[tokio::test]
async fn offset_pages_through_the_reranked_order() {
    let (embed, _) = serve(Behaviour::Normal).await;
    let (rank, _) = serve(Behaviour::Normal).await;
    let storage = open().await;
    storage
        .create_table(table(
            "docs",
            Some(embedding(&embed)),
            Some(reranker(&rank, RerankFormat::Cohere)),
        ))
        .await
        .unwrap();
    storage.insert_records("docs", corpus()).await.unwrap();

    let whole = storage
        .search("docs", query(4, Some(RerankRequest::default())))
        .await
        .unwrap()
        .hits;
    let mut first = query(
        2,
        Some(RerankRequest {
            top_n: Some(4),
            field: None,
        }),
    );
    first.offset = 0;
    let mut second = first.clone();
    second.offset = 2;
    let first = storage.search("docs", first).await.unwrap().hits;
    let second = storage.search("docs", second).await.unwrap().hits;
    assert_eq!(
        [ids(&first), ids(&second)].concat(),
        ids(&whole),
        "two pages of the reranked order are the reranked order"
    );
}

#[tokio::test]
async fn tei_is_spoken_too_and_a_reversed_answer_is_read_by_index() {
    let (embed, _) = serve(Behaviour::Normal).await;
    let (tei, tei_provider) = serve(Behaviour::Normal).await;
    let (reversed, _) = serve(Behaviour::Reversed).await;
    let storage = open().await;
    storage
        .create_table(table(
            "tei",
            Some(embedding(&embed)),
            Some(reranker(&tei, RerankFormat::Tei)),
        ))
        .await
        .unwrap();
    storage
        .create_table(table(
            "reversed",
            Some(embedding(&embed)),
            Some(reranker(&reversed, RerankFormat::Cohere)),
        ))
        .await
        .unwrap();
    storage.insert_records("tei", corpus()).await.unwrap();
    storage.insert_records("reversed", corpus()).await.unwrap();

    let via_tei = storage
        .search("tei", query(4, Some(RerankRequest::default())))
        .await
        .unwrap()
        .hits;
    assert_eq!(ids(&via_tei)[0], "full", "{:?}", ids(&via_tei));
    assert_eq!(tei_provider.calls.load(Ordering::SeqCst), 1);

    // The reversed provider answers worst first, every entry carrying its `index`. A client
    // that trusted array position would hand back the worst page; this one sorts by score.
    let via_reversed = storage
        .search("reversed", query(4, Some(RerankRequest::default())))
        .await
        .unwrap()
        .hits;
    assert_eq!(ids(&via_reversed), ids(&via_tei));
}

#[tokio::test]
async fn a_hit_without_text_in_the_field_trails_unscored() {
    let (embed, _) = serve(Behaviour::Normal).await;
    let (rank, provider) = serve(Behaviour::Normal).await;
    let storage = open().await;
    storage
        .create_table(table(
            "docs",
            Some(embedding(&embed)),
            Some(reranker(&rank, RerankFormat::Cohere)),
        ))
        .await
        .unwrap();
    let mut records = corpus();
    // A record that arrived as a vector, with no text filed: nothing to show the provider.
    records.push(CreateRecordRequest {
        id: "mute".to_string(),
        vector: vec![1.0; DIM],
        text: None,
        sparse_vector: None,
        named_vectors: None,
        payload: Some(serde_json::json!({"note": "no text here"})),
    });
    storage.insert_records("docs", records).await.unwrap();

    let hits = storage
        .search("docs", query(5, Some(RerankRequest::default())))
        .await
        .unwrap()
        .hits;
    assert_eq!(hits.len(), 5);
    assert_eq!(hits[4].id, "mute", "{:?}", ids(&hits));
    assert_eq!(hits[4].rerank_score, None);
    assert!(hits[..4].iter().all(|h| h.rerank_score.is_some()));
    assert_eq!(
        provider.seen.lock().unwrap().len(),
        4,
        "the mute record was not sent"
    );

    // `field` on the request points the provider at another payload key.
    let by_note = storage
        .search(
            "docs",
            query(
                5,
                Some(RerankRequest {
                    top_n: None,
                    field: Some("note".to_string()),
                }),
            ),
        )
        .await
        .unwrap()
        .hits;
    assert_eq!(by_note[0].id, "mute", "{:?}", ids(&by_note));
    assert!(by_note[1..].iter().all(|h| h.rerank_score.is_none()));
}

#[tokio::test]
async fn a_search_with_nothing_to_rerank_against_is_refused() {
    let (embed, _) = serve(Behaviour::Normal).await;
    let (rank, provider) = serve(Behaviour::Normal).await;
    let storage = open().await;
    storage
        .create_table(table(
            "ranked",
            Some(embedding(&embed)),
            Some(reranker(&rank, RerankFormat::Cohere)),
        ))
        .await
        .unwrap();
    storage
        .create_table(table("unranked", Some(embedding(&embed)), None))
        .await
        .unwrap();
    storage.insert_records("ranked", corpus()).await.unwrap();
    storage.insert_records("unranked", corpus()).await.unwrap();

    // A vector-only search has no query text, and the provider scores text against text.
    let err = storage
        .search(
            "ranked",
            SearchQuery {
                vector: vec![1.0; DIM],
                top_k: 2,
                limit: 2,
                rerank: Some(RerankRequest::default()),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)), "{err}");
    assert!(err.to_string().contains("text"), "{err}");

    // A table with no provider.
    let err = storage
        .search("unranked", query(2, Some(RerankRequest::default())))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)), "{err}");
    assert!(err.to_string().contains("rerank"), "{err}");

    // Grouped search flattens the query and does not rerank, and says so.
    let err = storage
        .search_groups(
            "ranked",
            query(2, Some(RerankRequest::default())),
            "text",
            2,
            1,
        )
        .await
        .unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)), "{err}");
    assert_eq!(
        provider.calls.load(Ordering::SeqCst),
        0,
        "nothing reached the provider"
    );
}

#[tokio::test]
async fn a_provider_that_will_not_answer_fails_the_request() {
    let (embed, _) = serve(Behaviour::Normal).await;
    let (unauthorized, _) = serve(Behaviour::Unauthorized).await;
    let (flaky, flaky_provider) = serve(Behaviour::RateLimitOnce).await;
    let storage = open().await;
    for (name, base) in [
        ("unauthorized", unauthorized.as_str()),
        ("flaky", flaky.as_str()),
        ("gone", "http://127.0.0.1:1"),
    ] {
        storage
            .create_table(table(
                name,
                Some(embedding(&embed)),
                Some(reranker(base, RerankFormat::Cohere)),
            ))
            .await
            .unwrap();
        storage.insert_records(name, corpus()).await.unwrap();
    }

    // A 401 is the caller's configuration, and no retry would change it.
    let err = storage
        .search("unauthorized", query(4, Some(RerankRequest::default())))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::InvalidConfig(_)), "{err}");
    assert!(err.to_string().contains("401"), "{err}");

    // A 429 is retried, and the retry is what answers.
    let hits = storage
        .search("flaky", query(4, Some(RerankRequest::default())))
        .await
        .unwrap()
        .hits;
    assert_eq!(ids(&hits)[0], "full");
    assert_eq!(flaky_provider.calls.load(Ordering::SeqCst), 2);

    // Nothing listening: the request fails rather than returning the page unranked.
    let err = storage
        .search("gone", query(4, Some(RerankRequest::default())))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::ServiceUnavailable { .. }), "{err}");
    let plain = storage.search("gone", query(4, None)).await.unwrap().hits;
    assert_eq!(
        plain.len(),
        4,
        "the same search without `rerank` still answers"
    );
}

#[cfg(feature = "text-bm25")]
mod text_routes {
    //! `/search/text` and `/search/hybrid` take the same `rerank`; the query text is the
    //! request's own, so neither needs an embedding model for the rerank half.

    use super::*;
    use dxmt_vectordb::types::{
        record::{FullTextSearchRequest, FusionMethod, HybridSearchRequest},
        table::{TextAnalyzer, TextFieldConfig, TextSource},
    };

    fn text_table(
        name: &str,
        embedding: Option<EmbeddingConfig>,
        rerank: RerankConfig,
    ) -> TableConfig {
        let mut config = table(name, embedding, Some(rerank));
        config.enable_full_text_search = true;
        config.text_fields = Some(vec![TextFieldConfig {
            name: "content".to_string(),
            source: TextSource::MetadataField("text".to_string()),
            analyzer: TextAnalyzer::Standard,
            boost: 1.0,
        }]);
        config
    }

    #[tokio::test]
    async fn full_text_and_hybrid_pages_are_reranked() {
        let (embed, _) = serve(Behaviour::Normal).await;
        let (rank, provider) = serve(Behaviour::Normal).await;
        let storage = open().await;
        storage
            .create_table(text_table(
                "docs",
                Some(embedding(&embed)),
                reranker(&rank, RerankFormat::Cohere),
            ))
            .await
            .unwrap();
        storage.insert_records("docs", corpus()).await.unwrap();

        // BM25 on this corpus: `near` has "borrow" five times in a short document, which is
        // one query term with a high term frequency. Whatever BM25's order is, the rerank
        // puts `full` first, and the test asserts the reranked order against the provider's
        // own rule over the pool that was sent.
        let page = storage
            .full_text_search(
                "docs",
                FullTextSearchRequest {
                    query: QUERY.to_string(),
                    fields: None,
                    limit: 3,
                    offset: 0,
                    highlight: false,
                    fuzzy: None,
                    filter: None,
                    rerank: Some(RerankRequest::default()),
                },
            )
            .await
            .unwrap()
            .hits;
        assert_eq!(
            page[0].id,
            "full",
            "{:?}",
            page.iter().map(|h| &h.id).collect::<Vec<_>>()
        );
        assert!(page.iter().all(|h| h.rerank_score.is_some()));
        assert_eq!(page.len(), 3);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);

        let fused = storage
            .hybrid_search(
                "docs",
                HybridSearchRequest {
                    vector: None,
                    sparse_vector: None,
                    text: Some(QUERY.to_string()),
                    vector_weight: 0.5,
                    fusion_method: FusionMethod::Dbsf,
                    rrf_k: 60.0,
                    limit: 4,
                    filter: None,
                    with_payload: false,
                    with_vector: false,
                    rerank: Some(RerankRequest::default()),
                },
            )
            .await
            .unwrap()
            .hits;
        assert_eq!(ids(&fused)[0], "full", "{:?}", ids(&fused));
        assert!(
            fused.iter().all(|h| h.payload.is_none()),
            "with_payload=false is honoured after the rerank read the payload"
        );
        assert!(fused.iter().all(|h| h.rerank_score.is_some()));
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    }
}
