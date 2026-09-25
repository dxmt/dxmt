//! Text in, vectors out: the table's model runs on the server.
//!
//! The provider here is a real HTTP server speaking the OpenAI `/embeddings` protocol, not a
//! stub swapped in behind the client. That is the whole point of the test: the thing being
//! checked is that pointing `base_url` at *something that speaks that protocol* works, because
//! that is the only claim `EmbeddingConfig` makes — the same one that has to hold for
//! text-embeddings-inference, ollama and Infinity, none of which can be in a test suite.
//!
//! It also means the failure paths are real failures: a 429 is a 429 on the wire, an
//! unreachable provider is a refused connection.
//!
//! The model is a hashed bag of words. It is not an embedding model and nothing here pretends
//! it is — it only has to be deterministic and to put similar text nearer together, which is
//! enough to tell "the server embedded the query with the same thing it embedded the corpus
//! with" from "it did not".

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use axum::{extract::State, routing::post, Json, Router};
use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{CreateRecordRequest, SearchQuery},
        table::{EmbeddingConfig, TableConfig},
        vector::DistanceMetric,
    },
};
use serde_json::{json, Value};

const DIM: usize = 32;

/// How the fake provider should behave, so the failure paths can be exercised.
#[derive(Clone, Copy, PartialEq)]
enum Behaviour {
    Normal,
    /// 429 on the first call, then normal. Proves the retry is a retry and not a give-up.
    RateLimitOnce,
    /// A 401 that no number of attempts would fix.
    Unauthorized,
    /// Vectors of the wrong width, which must be caught before anything is stored.
    WrongDimension,
    /// Answers in reverse order with `index` set, which is legal and which a client that
    /// trusts array position gets silently wrong.
    Reversed,
}

#[derive(Clone)]
struct Provider {
    behaviour: Behaviour,
    calls: Arc<AtomicUsize>,
    /// Every text the provider was asked to embed, in order, so a test can assert what the
    /// server sent rather than inferring it from the result.
    seen: Arc<std::sync::Mutex<Vec<String>>>,
}

fn bag_of_words(text: &str, dimension: usize) -> Vec<f32> {
    let mut v = vec![0f32; dimension];
    for word in text.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
        if word.is_empty() {
            continue;
        }
        let mut h: u64 = 1469598103934665603;
        for b in word.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(1099511628211);
        }
        v[(h % dimension as u64) as usize] += 1.0;
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        v[0] = 1.0;
        return v;
    }
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

async fn embeddings(
    State(provider): State<Provider>,
    Json(body): Json<Value>,
) -> (axum::http::StatusCode, Json<Value>) {
    let n = provider.calls.fetch_add(1, Ordering::SeqCst);
    let inputs: Vec<String> = body["input"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    provider.seen.lock().unwrap().extend(inputs.iter().cloned());

    match provider.behaviour {
        Behaviour::RateLimitOnce if n == 0 => {
            return (
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                Json(json!({"error": {"message": "slow down"}})),
            );
        }
        Behaviour::Unauthorized => {
            return (
                axum::http::StatusCode::UNAUTHORIZED,
                Json(json!({"error": {"message": "invalid api key"}})),
            );
        }
        _ => {}
    }

    let width = if provider.behaviour == Behaviour::WrongDimension {
        DIM + 1
    } else {
        DIM
    };
    let mut data: Vec<Value> = inputs
        .iter()
        .enumerate()
        .map(|(i, text)| json!({"index": i, "embedding": bag_of_words(text, width)}))
        .collect();
    if provider.behaviour == Behaviour::Reversed {
        data.reverse();
    }
    (
        axum::http::StatusCode::OK,
        Json(json!({"object": "list", "data": data, "model": body["model"]})),
    )
}

/// A provider listening on a loopback port, and its base URL.
async fn serve(behaviour: Behaviour) -> (String, Provider) {
    let provider = Provider {
        behaviour,
        calls: Arc::new(AtomicUsize::new(0)),
        seen: Arc::new(std::sync::Mutex::new(Vec::new())),
    };
    let app = Router::new()
        .route("/embeddings", post(embeddings))
        .with_state(provider.clone());
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::Server::from_tcp(listener)
            .unwrap()
            .serve(app.into_make_service())
            .await
            .unwrap();
    });
    (format!("http://{addr}"), provider)
}

fn storage_config() -> StorageConfig {
    StorageConfig {
        r#type: "memory".to_string(),
        memory: Some(MemoryConfig {
            max_size_mb: 128,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    }
}

fn table(name: &str, embedding: Option<EmbeddingConfig>) -> TableConfig {
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
    }
}

fn config(base_url: &str) -> EmbeddingConfig {
    EmbeddingConfig {
        model: "bag-of-words".to_string(),
        base_url: Some(base_url.to_string()),
        field: None,
        api_key_env: Some("DXMT_TEST_EMBEDDING_KEY_UNSET".to_string()),
        batch_size: None,
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

async fn open() -> IntelligentIndexedStorage {
    IntelligentIndexedStorage::new(&storage_config())
        .await
        .unwrap()
}

#[tokio::test]
async fn text_goes_in_and_a_text_query_finds_it() {
    let (base, provider) = serve(Behaviour::Normal).await;
    let storage = open().await;
    storage
        .create_table(table("docs", Some(config(&base))))
        .await
        .unwrap();

    storage
        .insert_records(
            "docs",
            vec![
                text_record("rust", "rust borrows memory instead of collecting it"),
                text_record("go", "go collects garbage instead of borrowing"),
                text_record("sql", "a database index answers a predicate quickly"),
            ],
        )
        .await
        .unwrap();

    let hits = storage
        .search(
            "docs",
            SearchQuery {
                // Words that appear verbatim: this model has no stemming, so "borrowing"
                // would match the *go* document ("instead of borrowing") and not the rust one.
                text: Some("borrows memory".to_string()),
                top_k: 3,
                limit: 3,
                with_payload: true,
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .hits;

    assert_eq!(
        hits[0].id,
        "rust",
        "ranked {:?}",
        hits.iter().map(|h| &h.id).collect::<Vec<_>>()
    );

    // The text is in the payload, under the config's default field, so a caller gets the chunk
    // back with the hit instead of keeping a second copy of the corpus.
    assert_eq!(
        hits[0]
            .payload
            .as_ref()
            .and_then(|p| p.get("text"))
            .and_then(|v| v.as_str()),
        Some("rust borrows memory instead of collecting it")
    );

    // One batch for the three records, one for the query: the corpus is not embedded per
    // record, and the query is not embedded per candidate.
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    assert_eq!(provider.seen.lock().unwrap().len(), 4);
}

#[tokio::test]
async fn a_batch_larger_than_batch_size_is_split_and_stays_in_order() {
    let (base, provider) = serve(Behaviour::Normal).await;
    let storage = open().await;
    let mut embedding = config(&base);
    embedding.batch_size = Some(3);
    storage
        .create_table(table("split", Some(embedding)))
        .await
        .unwrap();

    // Each record's text names itself, so a vector attached to the wrong record shows up as a
    // query for "word-k" ranking something else first.
    let records: Vec<_> = (0..10)
        .map(|i| text_record(&format!("r{i}"), &format!("word{i} word{i} word{i}")))
        .collect();
    storage.insert_records("split", records).await.unwrap();

    assert_eq!(
        provider.calls.load(Ordering::SeqCst),
        4,
        "10 texts at 3 per request"
    );

    for i in 0..10 {
        let hits = storage
            .search(
                "split",
                SearchQuery {
                    text: Some(format!("word{i}")),
                    top_k: 1,
                    limit: 1,
                    ..Default::default()
                },
            )
            .await
            .unwrap()
            .hits;
        assert_eq!(
            hits[0].id,
            format!("r{i}"),
            "record {i} got another record's vector"
        );
    }
}

#[tokio::test]
async fn a_provider_answering_out_of_order_does_not_shuffle_the_corpus() {
    // The `index` field is what says which input a vector belongs to. A provider is allowed to
    // return them in any order, and one that does will attach every vector to the wrong record
    // in a client that trusts array position -- silently, since every record still has a
    // plausible vector.
    let (base, _) = serve(Behaviour::Reversed).await;
    let storage = open().await;
    storage
        .create_table(table("rev", Some(config(&base))))
        .await
        .unwrap();
    storage
        .insert_records(
            "rev",
            vec![
                text_record("a", "alpha alpha alpha"),
                text_record("b", "bravo bravo bravo"),
                text_record("c", "charlie charlie charlie"),
            ],
        )
        .await
        .unwrap();

    let hits = storage
        .search(
            "rev",
            SearchQuery {
                text: Some("alpha".to_string()),
                top_k: 1,
                limit: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .hits;
    assert_eq!(hits[0].id, "a");
}

#[tokio::test]
async fn a_rate_limit_is_retried_and_a_bad_key_is_not() {
    let (base, provider) = serve(Behaviour::RateLimitOnce).await;
    let storage = open().await;
    storage
        .create_table(table("retry", Some(config(&base))))
        .await
        .unwrap();
    storage
        .insert_records("retry", vec![text_record("a", "alpha")])
        .await
        .expect("a 429 followed by a 200 should succeed");
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);

    let (base, provider) = serve(Behaviour::Unauthorized).await;
    let storage = open().await;
    storage
        .create_table(table("nokey", Some(config(&base))))
        .await
        .unwrap();
    let err = storage
        .insert_records("nokey", vec![text_record("a", "alpha")])
        .await
        .unwrap_err();
    assert_eq!(err.status_code(), 400, "{err}");
    assert!(err.to_string().contains("401"), "{err}");
    assert_eq!(
        provider.calls.load(Ordering::SeqCst),
        1,
        "a 401 is the same on every attempt; retrying it only delays the answer"
    );
}

#[tokio::test]
async fn a_model_of_the_wrong_width_is_refused_before_anything_is_stored() {
    let (base, _) = serve(Behaviour::WrongDimension).await;
    let storage = open().await;
    storage
        .create_table(table("wide", Some(config(&base))))
        .await
        .unwrap();
    let err = storage
        .insert_records("wide", vec![text_record("a", "alpha")])
        .await
        .unwrap_err();
    let message = err.to_string();
    assert!(message.contains(&(DIM + 1).to_string()), "{message}");
    assert!(message.contains(&DIM.to_string()), "{message}");
    assert_eq!(storage.get_table("wide").await.unwrap().record_count, 0);
}

#[tokio::test]
async fn an_unreachable_provider_fails_the_write_rather_than_storing_a_placeholder() {
    let storage = open().await;
    // Port 1 is reserved; nothing listens there.
    storage
        .create_table(table("down", Some(config("http://127.0.0.1:1"))))
        .await
        .unwrap();
    let err = storage
        .insert_records("down", vec![text_record("a", "alpha")])
        .await
        .unwrap_err();
    assert_eq!(err.status_code(), 503, "{err}");
    assert_eq!(storage.get_table("down").await.unwrap().record_count, 0);
}

#[tokio::test]
async fn a_table_with_no_model_says_so_instead_of_ignoring_the_text() {
    let storage = open().await;
    storage.create_table(table("plain", None)).await.unwrap();

    let err = storage
        .insert_records("plain", vec![text_record("a", "alpha")])
        .await
        .unwrap_err();
    assert!(err.to_string().contains("no embedding model"), "{err}");

    let err = storage
        .search(
            "plain",
            SearchQuery {
                text: Some("alpha".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("no embedding model"), "{err}");
}

#[tokio::test]
async fn a_search_carrying_both_a_vector_and_text_is_refused() {
    // Preferring one of them silently is how a caller measures the wrong thing for a week.
    let (base, _) = serve(Behaviour::Normal).await;
    let storage = open().await;
    storage
        .create_table(table("both", Some(config(&base))))
        .await
        .unwrap();
    let err = storage
        .search(
            "both",
            SearchQuery {
                vector: vec![0.0; DIM],
                text: Some("alpha".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not both"), "{err}");
}

#[tokio::test]
async fn a_vector_still_goes_in_directly_on_a_table_that_has_a_model() {
    // Declaring a model must not take away the ability to send vectors: a corpus embedded
    // offline and a query embedded by the server is a normal arrangement.
    let (base, provider) = serve(Behaviour::Normal).await;
    let storage = open().await;
    storage
        .create_table(table("mixed", Some(config(&base))))
        .await
        .unwrap();
    storage
        .insert_records(
            "mixed",
            vec![CreateRecordRequest {
                id: "given".to_string(),
                vector: bag_of_words("alpha alpha", DIM),
                text: None,
                sparse_vector: None,
                named_vectors: None,
                payload: None,
            }],
        )
        .await
        .unwrap();
    assert_eq!(
        provider.calls.load(Ordering::SeqCst),
        0,
        "a record that brought its own vector must not be sent to the provider"
    );

    let hits = storage
        .search(
            "mixed",
            SearchQuery {
                text: Some("alpha".to_string()),
                top_k: 1,
                limit: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .hits;
    assert_eq!(hits[0].id, "given");
}

#[tokio::test]
async fn text_beside_a_vector_is_stored_but_not_embedded() {
    let (base, provider) = serve(Behaviour::Normal).await;
    let storage = open().await;
    storage
        .create_table(table("keep", Some(config(&base))))
        .await
        .unwrap();
    storage
        .insert_records(
            "keep",
            vec![CreateRecordRequest {
                id: "r".to_string(),
                vector: bag_of_words("alpha", DIM),
                text: Some("the chunk this vector came from".to_string()),
                sparse_vector: None,
                named_vectors: None,
                payload: Some(serde_json::json!({"page": 3})),
            }],
        )
        .await
        .unwrap();
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);

    let record = storage.get_record("keep", &"r".to_string()).await.unwrap();
    let payload = record.payload;
    assert_eq!(payload["text"], "the chunk this vector came from");
    assert_eq!(payload["page"], 3, "the text must not replace the payload");
}

#[tokio::test]
async fn a_record_with_neither_a_vector_nor_text_is_named() {
    let (base, _) = serve(Behaviour::Normal).await;
    let storage = open().await;
    storage
        .create_table(table("empty", Some(config(&base))))
        .await
        .unwrap();
    let err = storage
        .insert_records(
            "empty",
            vec![CreateRecordRequest {
                id: "lonely".to_string(),
                vector: Vec::new(),
                text: None,
                sparse_vector: None,
                named_vectors: None,
                payload: Some(serde_json::json!({"page": 3})),
            }],
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("lonely"), "{err}");
}

#[tokio::test]
async fn the_payload_field_is_embedded_when_the_record_sends_no_text() {
    // The other way to load a corpus: the text is already a payload field, under whatever name
    // the pipeline uses. Naming it in the table saves rewriting every record.
    let (base, provider) = serve(Behaviour::Normal).await;
    let storage = open().await;
    let mut embedding = config(&base);
    embedding.field = Some("body".to_string());
    storage
        .create_table(table("field", Some(embedding)))
        .await
        .unwrap();

    storage
        .insert_records(
            "field",
            vec![CreateRecordRequest {
                id: "doc".to_string(),
                vector: Vec::new(),
                text: None,
                sparse_vector: None,
                named_vectors: None,
                payload: Some(serde_json::json!({"body": "alpha bravo", "page": 1})),
            }],
        )
        .await
        .unwrap();
    assert_eq!(provider.seen.lock().unwrap()[0], "alpha bravo");
    assert_eq!(storage.get_table("field").await.unwrap().record_count, 1);
}

#[tokio::test]
async fn the_model_survives_a_reopen() {
    // The config is on the table, so it is in the snapshot and in the WAL's table record. If it
    // were not, a restart would turn a text table into one that refuses text -- and the first
    // symptom would be an insert failing hours later.
    let dir = tempfile::TempDir::new().unwrap();
    let wal = dir.path().join("wal");
    let (base, _) = serve(Behaviour::Normal).await;

    {
        let storage = IntelligentIndexedStorage::new_with_wal(&storage_config(), Some(wal.clone()))
            .await
            .unwrap();
        storage
            .create_table(table("persist", Some(config(&base))))
            .await
            .unwrap();
        storage
            .insert_records("persist", vec![text_record("a", "alpha bravo")])
            .await
            .unwrap();
    }

    let storage = IntelligentIndexedStorage::new_with_wal(&storage_config(), Some(wal))
        .await
        .unwrap();
    let reopened = storage.get_table("persist").await.unwrap();
    let embedding = reopened
        .embedding
        .expect("the model did not survive the reopen");
    assert_eq!(embedding.model, "bag-of-words");
    assert_eq!(embedding.base_url.as_deref(), Some(base.as_str()));

    // And it still works, which is the part a field comparison alone would not show.
    let hits = storage
        .search(
            "persist",
            SearchQuery {
                text: Some("alpha".to_string()),
                top_k: 1,
                limit: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .hits;
    assert_eq!(hits[0].id, "a");
}

#[tokio::test]
async fn the_endpoint_can_move_but_the_model_cannot() {
    let (base, _) = serve(Behaviour::Normal).await;
    let storage = open().await;
    storage
        .create_table(table("moved", Some(config(&base))))
        .await
        .unwrap();
    storage
        .insert_records("moved", vec![text_record("a", "alpha bravo")])
        .await
        .unwrap();

    // A provider that moved: same model, new address. This is the case that would otherwise
    // need the table recreated and the corpus re-embedded, for an operational change.
    let (moved_base, moved) = serve(Behaviour::Normal).await;
    let mut repointed = config(&moved_base);
    repointed.batch_size = Some(8);
    storage
        .alter_table(
            "moved",
            dxmt_vectordb::types::table::TableAlterRequest {
                embedding: Some(repointed),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    storage
        .search(
            "moved",
            SearchQuery {
                text: Some("alpha".to_string()),
                top_k: 1,
                limit: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(
        moved.calls.load(Ordering::SeqCst),
        1,
        "the query went to the new address"
    );

    let mut other_model = config(&moved_base);
    other_model.model = "something-else".to_string();
    let err = storage
        .alter_table(
            "moved",
            dxmt_vectordb::types::table::TableAlterRequest {
                embedding: Some(other_model),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("cannot be re-embedded"), "{err}");
}

#[tokio::test]
async fn a_loaded_table_cannot_be_given_a_model_after_the_fact() {
    let (base, _) = serve(Behaviour::Normal).await;
    let storage = open().await;
    storage.create_table(table("late", None)).await.unwrap();
    storage
        .insert_records(
            "late",
            vec![CreateRecordRequest {
                id: "a".to_string(),
                vector: bag_of_words("alpha", DIM),
                text: None,
                sparse_vector: None,
                named_vectors: None,
                payload: None,
            }],
        )
        .await
        .unwrap();

    let err = storage
        .alter_table(
            "late",
            dxmt_vectordb::types::table::TableAlterRequest {
                embedding: Some(config(&base)),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("different"), "{err}");

    // An empty table is free to acquire one: there is nothing to be inconsistent with.
    storage.create_table(table("fresh", None)).await.unwrap();
    storage
        .alter_table(
            "fresh",
            dxmt_vectordb::types::table::TableAlterRequest {
                embedding: Some(config(&base)),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    storage
        .insert_records("fresh", vec![text_record("a", "alpha")])
        .await
        .unwrap();
}
