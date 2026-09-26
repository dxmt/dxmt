//! An OpenAI-compatible `/embeddings` server, on loopback, for tests.
//!
//! Shared rather than duplicated because two test binaries need it and because it is the thing
//! under test as much as the code is: `EmbeddingConfig` makes exactly one claim -- that
//! `base_url` may point at anything speaking this protocol -- and the only way to check that
//! claim is to point it at something that does. Stubbing the client out instead would test the
//! parts we wrote and none of the part we promised.
//!
//! The model is a hashed bag of words. It is not an embedding model and nothing here pretends
//! it is; it only has to be deterministic and to put similar text nearer together, which is
//! enough to tell "the query was embedded with the same thing as the corpus" from "it was not".

#![allow(dead_code)] // each test binary uses a subset

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use axum::{extract::State, routing::post, Json, Router};
use serde_json::{json, Value};

/// How the provider should behave, so the failure paths are real failures on the wire.
#[derive(Clone, Copy, PartialEq)]
pub enum Behaviour {
    Normal,
    /// 429 on the first call, then normal. Proves the retry is a retry and not a give-up.
    RateLimitOnce,
    /// A 401 that no number of attempts would fix.
    Unauthorized,
    /// Vectors one wider than asked for, which must be caught before anything is stored.
    WrongDimension,
    /// Answers in reverse order with `index` set, which is legal and which a client that
    /// trusts array position gets silently wrong.
    Reversed,
}

#[derive(Clone)]
pub struct Provider {
    behaviour: Behaviour,
    dimension: usize,
    pub calls: Arc<AtomicUsize>,
    /// Every text the provider was asked to embed, in order, so a test can assert what the
    /// server sent rather than inferring it from the result.
    pub seen: Arc<std::sync::Mutex<Vec<String>>>,
}

impl Provider {
    pub fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

/// The deterministic stand-in model. Public so a test can compute the vector it expects.
pub fn bag_of_words(text: &str, dimension: usize) -> Vec<f32> {
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

    let width = provider.dimension + usize::from(provider.behaviour == Behaviour::WrongDimension);
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

fn bind(behaviour: Behaviour, dimension: usize) -> (std::net::TcpListener, String, Provider) {
    let provider = Provider {
        behaviour,
        dimension,
        calls: Arc::new(AtomicUsize::new(0)),
        seen: Arc::new(std::sync::Mutex::new(Vec::new())),
    };
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    (listener, url, provider)
}

fn router(provider: Provider) -> Router {
    Router::new()
        .route("/embeddings", post(embeddings))
        .with_state(provider)
}

/// Start it on the caller's runtime. For `#[tokio::test]`.
pub async fn serve(behaviour: Behaviour, dimension: usize) -> (String, Provider) {
    let (listener, url, provider) = bind(behaviour, dimension);
    let app = router(provider.clone());
    tokio::spawn(async move {
        axum::Server::from_tcp(listener)
            .unwrap()
            .serve(app.into_make_service())
            .await
            .unwrap();
    });
    (url, provider)
}

/// Start it on a runtime of its own, in a thread. For a plain `#[test]` -- in particular one
/// that spawns the server binary as a child process, which then reaches this over loopback
/// like any other provider.
///
/// The thread is never joined; it ends with the test binary. A test process that has already
/// decided its result has nothing to gain from an orderly shutdown here.
pub fn serve_on_a_thread(behaviour: Behaviour, dimension: usize) -> (String, Provider) {
    let (listener, url, provider) = bind(behaviour, dimension);
    let app = router(provider.clone());
    std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                axum::Server::from_tcp(listener)
                    .unwrap()
                    .serve(app.into_make_service())
                    .await
                    .unwrap();
            });
    });
    (url, provider)
}
