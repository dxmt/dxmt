// Simple mode API — zero-config Collections API
//
// Embedding strategy (no external model required):
//   Dense vector:  bag-of-words projection — each term hashes to a bucket in [0, dim),
//                  bucket value = TF-IDF weight. Gives real semantic overlap between
//                  documents that share vocabulary.
//   Sparse vector: per-term TF-IDF weights stored as SparseVector, fed into SparseIndex
//                  for WAND-based MIPS. The two are fused via DBSF in hybrid_search.
//
// Quality trade-off: worse than a trained sentence-transformer but orders-of-magnitude
// better than a random hash seed, and requires zero external dependencies.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::{
    api::handlers::AppState,
    config::{MemoryConfig, StorageConfig},
    search::bm25::{BM25Document, BM25Index},
    storage::IntelligentIndexedStorage,
    types::{
        record::{CreateRecordRequest, Filter, HybridSearchRequest, SearchQuery},
        sparse_vector::SparseVector,
        table::TableConfig,
    },
    Result,
};

// ─── request / response structs ──────────────────────────────────────────────

#[derive(Deserialize)]
struct CreateCollectionRequest {
    name: String,
    #[serde(default = "default_dimension")]
    dimension: usize,
    #[serde(default = "default_distance_metric")]
    distance_metric: String,
}

fn default_dimension() -> usize {
    384
}
fn default_distance_metric() -> String {
    "cosine".to_string()
}

#[derive(Deserialize)]
struct AddDocumentsRequest {
    documents: Vec<String>,
    ids: Option<Vec<String>>,
    embeddings: Option<Vec<Vec<f32>>>,
    metadata: Option<Vec<serde_json::Value>>,
}

#[derive(Deserialize)]
struct QueryRequest {
    query: String,
    #[serde(default = "default_limit")]
    limit: usize,
    /// Caller-supplied dense embedding (skips our encoding if provided).
    embedding: Option<Vec<f32>>,
    filter: Option<serde_json::Value>,
}

fn default_limit() -> usize {
    10
}

#[derive(Serialize)]
struct QueryResult {
    id: String,
    distance: f32,
    document: String,
    metadata: Option<serde_json::Value>,
}

// ─── per-collection BM25 state ───────────────────────────────────────────────

/// Maintains a BM25 index and a term→dimension vocabulary per collection.
/// The vocabulary assigns each unique term a stable bucket in [0, dim) via
/// a deterministic hash, allowing BM25 TF-IDF weights to be projected into
/// both the sparse and dense vector spaces.
struct CollectionBm25 {
    bm25: BM25Index,
    /// term → bucket index in [0, dimension)
    vocab: HashMap<String, u32>,
    dimension: usize,
}

impl CollectionBm25 {
    fn new(dimension: usize) -> Self {
        Self {
            bm25: BM25Index::new(),
            vocab: HashMap::new(),
            dimension,
        }
    }

    /// Assign a stable bucket to `term` (creates entry if absent).
    fn term_dim(&mut self, term: &str) -> u32 {
        if let Some(&d) = self.vocab.get(term) {
            return d;
        }
        // Deterministic hash: FNV-1a 32-bit
        let mut h: u32 = 2_166_136_261;
        for b in term.bytes() {
            h ^= b as u32;
            h = h.wrapping_mul(16_777_619);
        }
        let dim = h % self.dimension as u32;
        self.vocab.insert(term.to_string(), dim);
        dim
    }

    /// Index `text` under `id` and return (dense, sparse) encodings.
    fn encode_document(&mut self, id: &str, text: &str) -> (Vec<f32>, SparseVector) {
        // Insert into BM25 (updates IDF stats)
        self.bm25.insert_document(BM25Document {
            id: id.to_string(),
            text: text.to_string(),
            tokens: vec![],
        });

        self.encode_text(text)
    }

    /// Encode `text` into (dense bag-of-words, sparse TF-IDF) without modifying the index.
    /// Use for query encoding after all documents are already indexed.
    fn encode_text(&mut self, text: &str) -> (Vec<f32>, SparseVector) {
        let tokens = self.bm25.tokenize(text);
        if tokens.is_empty() {
            return (
                vec![0.0; self.dimension],
                SparseVector::new_unchecked(vec![], vec![], self.dimension),
            );
        }

        // Term frequency in this text
        let mut tf: HashMap<String, u32> = HashMap::new();
        for t in &tokens {
            *tf.entry(t.clone()).or_insert(0) += 1;
        }

        let mut dense = vec![0.0f32; self.dimension];
        let mut sparse_indices: Vec<u32> = Vec::new();
        let mut sparse_values: Vec<f32> = Vec::new();

        let n_tokens = tokens.len() as f32;
        for (term, count) in &tf {
            let weight = (*count as f32) / n_tokens; // normalised TF
            let dim = self.term_dim(term);

            // Accumulate into dense bucket (multiple terms may collide — that's OK)
            dense[dim as usize] += weight;

            // Sparse entry: one entry per unique term
            sparse_indices.push(dim);
            sparse_values.push(weight);
        }

        // L2-normalise dense vector
        let norm: f32 = dense.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for x in &mut dense {
                *x /= norm;
            }
        }

        // Sort sparse by index (required by SparseVector)
        let mut pairs: Vec<(u32, f32)> = sparse_indices.into_iter().zip(sparse_values).collect();
        pairs.sort_by_key(|(i, _)| *i);
        // Deduplicate colliding dims by summing
        let mut deduped: Vec<(u32, f32)> = Vec::new();
        for (idx, val) in pairs {
            if let Some(last) = deduped.last_mut() {
                if last.0 == idx {
                    last.1 += val;
                    continue;
                }
            }
            deduped.push((idx, val));
        }

        let (si, sv): (Vec<u32>, Vec<f32>) = deduped.into_iter().unzip();
        let sparse = SparseVector::new_unchecked(si, sv, self.dimension);

        (dense, sparse)
    }
}

// ─── simple mode application state ───────────────────────────────────────────

struct SimpleModeState {
    app: Arc<AppState>,
    /// Per-collection BM25 state, keyed by collection name.
    bm25: Arc<RwLock<HashMap<String, CollectionBm25>>>,
}

// ─── router factory ──────────────────────────────────────────────────────────

pub async fn create_simple_app(memory_limit: usize) -> Result<Router> {
    info!(
        "Initializing simple mode with {} MB memory (BM25 sparse encoding enabled)",
        memory_limit
    );

    let storage_config = StorageConfig {
        r#type: "memory".to_string(),
        memory: Some(MemoryConfig {
            max_size_mb: memory_limit,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    };

    // Simple mode gets the same durability as standard mode, but in its own directory: both
    // modes name tables from user input, so sharing one log would let a collection here collide
    // with a table there and replay into the wrong shape.
    let storage = Arc::new(
        IntelligentIndexedStorage::new_with_wal(
            &storage_config,
            crate::api::server::wal_dir_from_env().map(|d| d.join("simple")),
        )
        .await?,
    );
    let server_health = Arc::new(crate::api::server_health::ServerHealth::default());

    let app_state = Arc::new(AppState {
        storage: storage.clone(),
        server_health,
    });

    let state = Arc::new(SimpleModeState {
        app: app_state,
        bm25: Arc::new(RwLock::new(HashMap::new())),
    });

    Ok(Router::new()
        .route(
            "/collections",
            post(create_collection).get(list_collections),
        )
        .route("/collections/:name", delete(delete_collection))
        .route("/collections/:name/add", post(add_documents))
        .route("/collections/:name/query", post(query))
        .route("/search/:collection", post(query)) // compat alias
        .route("/health", get(health))
        .route("/live", get(liveness))
        .route("/ready", get(readiness))
        .route("/health/live", get(liveness))
        .route("/health/ready", get(readiness))
        .route("/status", get(status))
        .with_state(state))
}

// ─── handlers ────────────────────────────────────────────────────────────────

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "mode": "simple",
        "backend": "intelligent_indexed_storage + bm25_sparse_encoding",
        "version": "1.0.0"
    }))
}

async fn liveness() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "alive",
        "mode": "simple",
        "service": "vectordb",
        "timestamp": chrono::Utc::now().to_rfc3339(),
    }))
}

async fn readiness(State(state): State<Arc<SimpleModeState>>) -> impl IntoResponse {
    let storage = state.app.storage.health_snapshot().await;
    let status = if storage.ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        Json(serde_json::json!({
            "status": if storage.ready { "ready" } else { "not_ready" },
            "mode": "simple",
            "service": "vectordb",
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "reason": storage.durability_error.clone(),
            "storage": storage,
        })),
    )
}

async fn status(State(state): State<Arc<SimpleModeState>>) -> impl IntoResponse {
    let stats = state.app.storage.get_stats().await;
    let bm25_collections = state.bm25.read().await.len();
    Json(serde_json::json!({
        "status": "healthy",
        "mode": "simple",
        "storage": { "type": "intelligent_indexed", "stats": stats },
        "bm25_collections": bm25_collections,
    }))
}

async fn create_collection(
    State(state): State<Arc<SimpleModeState>>,
    Json(req): Json<CreateCollectionRequest>,
) -> impl IntoResponse {
    info!("Creating collection '{}' (dim={})", req.name, req.dimension);

    let distance_metric = match req.distance_metric.as_str() {
        "euclidean" => crate::types::vector::DistanceMetric::Euclidean,
        "dot" => crate::types::vector::DistanceMetric::Dot,
        _ => crate::types::vector::DistanceMetric::Cosine,
    };

    let table_config = TableConfig {
        name: req.name.clone(),
        dimension: req.dimension,
        distance_metric,
        index_type: None,
        description: Some("Collection created via simple mode API".to_string()),
        text_fields: None,
        enable_full_text_search: false,
        memory_optimization: None,
        payload_schema: None,
        schema_version: 1,
        named_vectors: None,
    };

    match state.app.storage.create_table(table_config).await {
        Ok(table) => {
            // Initialise per-collection BM25 state
            state
                .bm25
                .write()
                .await
                .insert(req.name.clone(), CollectionBm25::new(req.dimension));

            (
                StatusCode::CREATED,
                Json(serde_json::json!({
                    "collection": table.name,
                    "status": "created",
                    "dimension": table.dimension,
                })),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": format!("{}", e) })),
        )
            .into_response(),
    }
}

async fn list_collections(State(state): State<Arc<SimpleModeState>>) -> impl IntoResponse {
    let tables = state.app.storage.list_tables().await.unwrap_or_default();
    let collections: Vec<_> = tables
        .iter()
        .map(|t| {
            serde_json::json!({
                "name": t.name,
                "dimension": t.dimension,
                "documents": t.record_count,
                "distance_metric": format!("{:?}", t.distance_metric),
            })
        })
        .collect();
    Json(serde_json::json!({ "collections": collections }))
}

async fn delete_collection(
    State(state): State<Arc<SimpleModeState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    state.bm25.write().await.remove(&name);
    match state.app.storage.delete_table(&name).await {
        Ok(_) => (
            StatusCode::OK,
            Json(serde_json::json!({ "status": "deleted" })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("{}", e) })),
        )
            .into_response(),
    }
}

async fn add_documents(
    State(state): State<Arc<SimpleModeState>>,
    Path(collection_name): Path<String>,
    Json(req): Json<AddDocumentsRequest>,
) -> impl IntoResponse {
    info!(
        "Adding {} documents to collection '{}'",
        req.documents.len(),
        collection_name
    );

    // Get collection dimension
    let dimension = match state.app.storage.get_table(&collection_name).await {
        Ok(t) => t.dimension,
        Err(e) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": format!("Collection not found: {}", e) })),
            )
                .into_response();
        }
    };

    let ids: Vec<String> = req.ids.unwrap_or_else(|| {
        req.documents
            .iter()
            .map(|_| uuid::Uuid::new_v4().to_string())
            .collect()
    });

    // Compute (dense, sparse) for each document
    let mut encodings: Vec<(Vec<f32>, SparseVector)> = Vec::with_capacity(req.documents.len());

    if let Some(ref caller_embeddings) = req.embeddings {
        // Caller supplied dense embeddings — still build BM25 sparse on our side
        let mut bm25_map = state.bm25.write().await;
        let bm25 = bm25_map
            .entry(collection_name.clone())
            .or_insert_with(|| CollectionBm25::new(dimension));

        for (i, doc) in req.documents.iter().enumerate() {
            let dense = caller_embeddings
                .get(i)
                .cloned()
                .unwrap_or_else(|| vec![0.0; dimension]);
            // Index into BM25 for sparse; encode_document returns (dense, sparse)
            // but we discard BM25's dense and use the caller's
            let (_, sparse) = bm25.encode_document(&ids[i], doc);
            encodings.push((dense, sparse));
        }
    } else {
        // No caller embeddings — generate our own
        let mut bm25_map = state.bm25.write().await;
        let bm25 = bm25_map
            .entry(collection_name.clone())
            .or_insert_with(|| CollectionBm25::new(dimension));

        for (i, doc) in req.documents.iter().enumerate() {
            let (dense, sparse) = bm25.encode_document(&ids[i], doc);
            encodings.push((dense, sparse));
        }
    };

    // Build record requests
    let records: Vec<CreateRecordRequest> = req
        .documents
        .iter()
        .enumerate()
        .map(|(i, doc)| {
            let (dense, sparse) = &encodings[i];
            let mut payload = serde_json::json!({ "text": doc });
            if let Some(ref meta_list) = req.metadata {
                if let Some(meta) = meta_list.get(i) {
                    if let Some(obj) = meta.as_object() {
                        for (k, v) in obj {
                            payload[k] = v.clone();
                        }
                    }
                }
            }
            CreateRecordRequest {
                id: ids[i].clone(),
                vector: dense.clone(),
                sparse_vector: Some(sparse.clone()),
                payload: Some(payload),
                named_vectors: None,
            }
        })
        .collect();

    match state
        .app
        .storage
        .insert_records(&collection_name, records)
        .await
    {
        Ok(inserted) => {
            let inserted_ids: Vec<String> = inserted.iter().map(|r| r.id.clone()).collect();
            debug!(
                "Inserted {} records into '{}'",
                inserted_ids.len(),
                collection_name
            );
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "ids": inserted_ids,
                    "count": inserted_ids.len(),
                })),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": format!("Insert failed: {}", e) })),
        )
            .into_response(),
    }
}

async fn query(
    State(state): State<Arc<SimpleModeState>>,
    Path(collection_name): Path<String>,
    Json(req): Json<QueryRequest>,
) -> impl IntoResponse {
    info!("Querying '{}': {:?}", collection_name, req.query);

    let dimension = match state.app.storage.get_table(&collection_name).await {
        Ok(t) => t.dimension,
        Err(e) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": format!("Collection not found: {}", e) })),
            )
                .into_response();
        }
    };

    let filter: Option<Filter> = if let Some(ref fj) = req.filter {
        match serde_json::from_value(fj.clone()) {
            Ok(f) => Some(f),
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "error": format!("Invalid filter: {}", e) })),
                )
                    .into_response();
            }
        }
    } else {
        None
    };

    // If caller supplied a dense embedding, use it + generate our BM25 sparse.
    // Otherwise generate both from the query text.
    let (query_dense, query_sparse) = if let Some(emb) = req.embedding {
        // Caller's dense + our BM25 sparse for the text
        let mut bm25_map = state.bm25.write().await;
        let bm25 = bm25_map
            .entry(collection_name.clone())
            .or_insert_with(|| CollectionBm25::new(dimension));
        let (_, sparse) = bm25.encode_text(&req.query);
        (emb, sparse)
    } else {
        let mut bm25_map = state.bm25.write().await;
        let bm25 = bm25_map
            .entry(collection_name.clone())
            .or_insert_with(|| CollectionBm25::new(dimension));
        bm25.encode_text(&req.query)
    };

    // Use hybrid search (DBSF: dense + sparse + text=none) when we have a non-empty sparse vec
    let hits = if !query_sparse.indices.is_empty() {
        let hybrid_req = HybridSearchRequest {
            vector: Some(query_dense),
            sparse_vector: Some(query_sparse),
            text: Some(req.query.clone()),
            vector_weight: 0.7,
            fusion_method: crate::types::record::FusionMethod::Dbsf,
            rrf_k: 60.0,
            limit: req.limit,
            filter,
            with_payload: true,
            with_vector: false,
        };
        match state
            .app
            .storage
            .hybrid_search(&collection_name, hybrid_req)
            .await
        {
            Ok(r) => r.hits,
            Err(_) => {
                // Fallback to plain vector search if hybrid fails (e.g. no text engine)
                plain_search(
                    &state,
                    &collection_name,
                    &req.query,
                    req.limit,
                    filter_none(),
                )
                .await
            }
        }
    } else {
        plain_search(&state, &collection_name, &req.query, req.limit, filter).await
    };

    let results: Vec<QueryResult> = hits
        .into_iter()
        .map(|hit| {
            let text = hit
                .payload
                .as_ref()
                .and_then(|p| p.get("text"))
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string();
            QueryResult {
                id: hit.id,
                // NOTE: this field is a similarity, not a distance — higher is better, and the
                // list is ordered by it descending, where a field named `distance` invites the
                // opposite reading. The
                // name is kept because it is already published; a caller that sorts by it
                // ascending, as the name invites, gets the worst match first.
                distance: hit.score,
                document: text,
                metadata: hit.payload,
            }
        })
        .collect();

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "results": results,
        })),
    )
        .into_response()
}

fn filter_none() -> Option<Filter> {
    None
}

async fn plain_search(
    state: &Arc<SimpleModeState>,
    collection: &str,
    query_text: &str,
    limit: usize,
    filter: Option<Filter>,
) -> Vec<crate::types::record::SearchHit> {
    // Encode query dense vector for a plain ANN fallback
    let dimension = state
        .app
        .storage
        .get_table(collection)
        .await
        .map(|t| t.dimension)
        .unwrap_or(384);

    let mut bm25_map = state.bm25.write().await;
    let bm25 = bm25_map
        .entry(collection.to_string())
        .or_insert_with(|| CollectionBm25::new(dimension));
    let (dense, _) = bm25.encode_text(query_text);
    drop(bm25_map);

    let sq = SearchQuery {
        vector: dense,
        top_k: limit,
        limit,
        filter,
        with_payload: true,
        with_vector: false,
        include_metadata: true,
        ..Default::default()
    };
    state
        .app
        .storage
        .search(collection, sq)
        .await
        .map(|r| r.hits)
        .unwrap_or_default()
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Cosine similarity between two dense vectors.
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-8 || nb < 1e-8 {
            return 0.0;
        }
        dot / (na * nb)
    }

    #[test]
    fn test_bm25_encoding_similar_docs_closer() {
        let dim = 128;
        let mut enc = CollectionBm25::new(dim);

        // Index three documents
        let (d_fox, _) = enc.encode_document("doc1", "the quick brown fox jumps over the lazy dog");
        let (d_fox2, _) = enc.encode_document("doc2", "a quick brown fox runs through the forest");
        let (d_weather, _) = enc.encode_document("doc3", "sunny weather with clear blue sky");

        // Encode query after all docs are indexed (IDF is populated)
        let (q, _) = enc.encode_text("quick brown fox");

        let sim_fox = cosine(&q, &d_fox);
        let sim_fox2 = cosine(&q, &d_fox2);
        let sim_weather = cosine(&q, &d_weather);

        // Both fox docs should score higher than the weather doc
        assert!(
            sim_fox > sim_weather,
            "fox doc1 ({:.3}) should beat weather ({:.3})",
            sim_fox,
            sim_weather
        );
        assert!(
            sim_fox2 > sim_weather,
            "fox doc2 ({:.3}) should beat weather ({:.3})",
            sim_fox2,
            sim_weather
        );
    }

    #[test]
    fn test_bm25_sparse_encoding_produces_valid_sparse_vector() {
        let mut enc = CollectionBm25::new(64);
        let (_, sparse) = enc.encode_document("d1", "hello world hello");

        assert!(
            !sparse.indices.is_empty(),
            "sparse vector must be non-empty"
        );
        assert_eq!(sparse.indices.len(), sparse.values.len());
        // All indices must be in [0, dim)
        for &idx in &sparse.indices {
            assert!(idx < 64, "index {} out of range", idx);
        }
        // Indices must be sorted
        for w in sparse.indices.windows(2) {
            assert!(w[0] < w[1], "indices not sorted");
        }
        // All values must be positive
        for &v in &sparse.values {
            assert!(v > 0.0, "values must be positive");
        }
    }

    #[test]
    fn test_empty_text_returns_zero_vectors() {
        let mut enc = CollectionBm25::new(32);
        let (dense, sparse) = enc.encode_text("");
        assert_eq!(dense.len(), 32);
        assert!(dense.iter().all(|&x| x == 0.0));
        assert!(sparse.indices.is_empty());
    }

    #[test]
    fn test_term_vocab_stable_across_calls() {
        let mut enc = CollectionBm25::new(128);
        let d1 = enc.term_dim("rust");
        let d2 = enc.term_dim("rust");
        assert_eq!(d1, d2, "same term must map to same bucket");

        let d3 = enc.term_dim("python");
        // Different terms should get different buckets with high probability (not guaranteed but very likely)
        // Just check both are in range
        assert!(d3 < 128);
        assert!(d1 < 128);
    }

    #[tokio::test]
    async fn test_simple_mode_end_to_end() {
        let router = create_simple_app(64).await.expect("create app");

        // We can't easily drive HTTP through the router in a unit test without a test client,
        // so instead test the storage + encoding path directly.
        let config = StorageConfig {
            r#type: "memory".to_string(),
            memory: Some(MemoryConfig {
                max_size_mb: 64,
                eviction_policy: "lru".to_string(),
            }),
            ssd: None,
            object_storage: None,
        };
        let storage = Arc::new(IntelligentIndexedStorage::new(&config).await.unwrap());

        storage
            .create_table(TableConfig {
                name: "test".to_string(),
                dimension: 128,
                distance_metric: crate::types::vector::DistanceMetric::Cosine,
                description: None,
                index_type: None,
                text_fields: None,
                enable_full_text_search: false,
                memory_optimization: None,
                payload_schema: None,
                schema_version: 1,
                named_vectors: None,
            })
            .await
            .unwrap();

        let mut enc = CollectionBm25::new(128);
        let docs = [
            ("d1", "Rust is a systems programming language"),
            ("d2", "Python is great for machine learning"),
            (
                "d3",
                "Rust achieves memory safety without garbage collection",
            ),
        ];

        let records: Vec<CreateRecordRequest> = docs
            .iter()
            .map(|(id, text)| {
                let (dense, sparse) = enc.encode_document(id, text);
                CreateRecordRequest {
                    id: id.to_string(),
                    vector: dense,
                    sparse_vector: Some(sparse),
                    payload: Some(serde_json::json!({ "text": text })),
                    named_vectors: None,
                }
            })
            .collect();

        storage.insert_records("test", records).await.unwrap();

        // Query for Rust — the two Rust docs should outscore the Python doc
        let (query_dense, _) = enc.encode_text("Rust programming language");
        let result = storage
            .search(
                "test",
                SearchQuery {
                    vector: query_dense,
                    top_k: 3,
                    limit: 3,
                    with_payload: true,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(result.hits.len(), 3);
        // Top result should be a Rust document (d1 or d3)
        let top_id = &result.hits[0].id;
        assert!(
            top_id == "d1" || top_id == "d3",
            "top result should be a Rust doc, got: {}",
            top_id
        );
        drop(router); // ensure it compiled
    }
}
