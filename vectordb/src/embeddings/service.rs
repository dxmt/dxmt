//! Turning text into a vector, on the server, through an OpenAI-compatible `/embeddings` API.
//!
//! This is the only file in `src/embeddings/` that a request reaches. The rest of the directory
//! is four overlapping provider abstractions that nothing calls; see the note in `CLAUDE.md`.
//!
//! **Why one provider shape and not a trait with five implementations.** Every embedding service
//! worth pointing at speaks `POST {base}/embeddings` with `{"model": …, "input": [...]}` —
//! OpenAI, Jina, Voyage, text-embeddings-inference, Infinity, ollama, vLLM, LM Studio,
//! llama.cpp. `base_url` is the entire difference. A provider enum would add code to maintain
//! and reach nothing new, and the directory next to this file is what that looks like after a
//! year.
//!
//! **What this deliberately does not do:**
//!
//! * **No cache.** A cache keyed by text would help a benchmark that embeds the same query
//!   repeatedly and nothing else; the write path embeds each chunk once by construction. The
//!   dead `openai.rs` has a 1,000-entry LRU, which is the kind of thing that looks like a
//!   feature and is a way to serve a stale vector after a model is swapped under a name.
//! * **No fallback vector.** If the provider will not answer, the request fails. A zero or
//!   hashed vector would be stored, would rank badly forever, and would look exactly like a
//!   working system — which is what simple mode's "auto-embedding" was, and why it went.
//! * **No key in the table.** The table names an *environment variable*; the value is read here,
//!   at call time. So a key is not in a create-table body, a snapshot, an export, or any
//!   response, and rotating one does not need a table rewrite.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::types::table::EmbeddingConfig;
use crate::{Error, Result};

/// Total budget for one call to the provider, retries included.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
/// Attempts per batch. Three covers a rate-limit window and a restarting local server; more
/// than that turns a provider outage into a request that hangs instead of one that fails.
const MAX_ATTEMPTS: u32 = 3;
const FIRST_BACKOFF: Duration = Duration::from_millis(500);

/// One `reqwest::Client` for the process: the connection pool and the TLS session cache live
/// on the client, so building one per call would mean a fresh handshake per batch.
fn http() -> &'static reqwest::Client {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .expect("a reqwest client with no proxy or TLS override cannot fail to build")
    })
}

#[derive(Serialize)]
struct EmbeddingsRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingDatum>,
}

#[derive(Deserialize)]
struct EmbeddingDatum {
    /// Position of the input this vector belongs to.
    ///
    /// **Not assumed to be the array position.** The OpenAI specification says to sort by it,
    /// and a provider that batches internally can and does return them out of order; ignoring
    /// it attaches every vector to the wrong record, and nothing downstream can tell.
    #[serde(default)]
    index: usize,
    embedding: Vec<f32>,
}

/// Embed `texts`, returning one vector per input, in input order.
///
/// `expected_dimension` is the table's width. A model that answers at a different width is an
/// error here rather than a dimension mismatch from the index later, because at this point we
/// can still say which model produced which width.
pub async fn embed_texts(
    config: &EmbeddingConfig,
    texts: &[String],
    expected_dimension: Option<usize>,
) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    // Through the helper, like every other environment read here: for a name outside the
    // `DXMT_` namespace -- which this one usually is, since the table names it -- it behaves
    // exactly as `std::env::var`, and `tests/env_prefix_test.rs` is what keeps that uniform.
    let api_key = crate::config::env::var(config.api_key_env())
        .ok()
        .filter(|k| !k.is_empty());
    let url = format!("{}/embeddings", config.base_url());

    let mut out = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(config.batch_size()) {
        let mut vectors = embed_chunk(&url, config, api_key.as_deref(), chunk).await?;
        if vectors.len() != chunk.len() {
            return Err(Error::Internal(format!(
                "the embedding provider returned {} vectors for {} inputs (model {}, {})",
                vectors.len(),
                chunk.len(),
                config.model,
                url
            )));
        }
        if let Some(expected) = expected_dimension {
            if let Some(actual) = vectors.first().map(|v| v.len()) {
                if actual != expected {
                    return Err(Error::InvalidConfig(format!(
                        "model {} returns {}-dimension vectors but this table is {}-dimension. \
                         A table's width is fixed at creation: create the table with \
                         \"dimension\": {}, or name a model of the table's width.",
                        config.model, actual, expected, actual
                    )));
                }
            }
        }
        out.append(&mut vectors);
    }
    Ok(out)
}

async fn embed_chunk(
    url: &str,
    config: &EmbeddingConfig,
    api_key: Option<&str>,
    texts: &[String],
) -> Result<Vec<Vec<f32>>> {
    let body = EmbeddingsRequest {
        model: &config.model,
        input: texts,
    };
    let mut backoff = FIRST_BACKOFF;
    let mut last: Option<Error> = None;

    for attempt in 1..=MAX_ATTEMPTS {
        let mut request = http().post(url).json(&body);
        if let Some(key) = api_key {
            request = request.bearer_auth(key);
        }
        let outcome = match request.send().await {
            Ok(response) => classify(response, url, &config.model).await,
            // A transport error is a provider that is not there: DNS, refused, TLS, timeout.
            Err(e) => Err(Retryable::Yes(Error::ServiceUnavailable {
                message: format!("embedding provider at {url} is unreachable: {e}"),
            })),
        };
        match outcome {
            Ok(vectors) => return Ok(vectors),
            Err(Retryable::No(e)) => return Err(e),
            Err(Retryable::Yes(e)) => {
                last = Some(e);
                if attempt < MAX_ATTEMPTS {
                    tokio::time::sleep(backoff).await;
                    backoff *= 2;
                }
            }
        }
    }
    Err(last.unwrap_or_else(|| Error::Internal("embedding retry loop fell through".into())))
}

enum Retryable {
    Yes(Error),
    No(Error),
}

async fn classify(
    response: reqwest::Response,
    url: &str,
    model: &str,
) -> std::result::Result<Vec<Vec<f32>>, Retryable> {
    let status = response.status();
    if status.is_success() {
        let parsed: EmbeddingsResponse = response.json().await.map_err(|e| {
            // A 200 that is not an embeddings response is almost always the wrong `base_url` —
            // one pointing at the host root, or at a page. Retrying will not change it.
            Retryable::No(Error::InvalidConfig(format!(
                "{url} answered 200 but not with an embeddings response ({e}). Check that \
                 base_url is the API root, the part before /embeddings."
            )))
        })?;
        let mut data = parsed.data;
        data.sort_by_key(|d| d.index);
        return Ok(data.into_iter().map(|d| d.embedding).collect());
    }

    // The body carries the provider's own message, which is the useful half of the error:
    // "model not found", "quota exceeded", "context length". Truncated so a provider that
    // answers an error with an HTML page cannot fill the log.
    let detail = response
        .text()
        .await
        .map(|t| t.chars().take(400).collect::<String>())
        .unwrap_or_default();
    let message = format!("embedding provider {url} (model {model}) returned {status}: {detail}");

    Err(match status.as_u16() {
        // Rate limited, or the provider is briefly unwell: worth another attempt.
        429 | 500 | 502 | 503 | 504 => Retryable::Yes(Error::ServiceUnavailable { message }),
        // A bad key, a missing model, an input too long — every attempt gets the same answer,
        // so the caller gets it now rather than after three of them.
        _ => Retryable::No(Error::InvalidConfig(message)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(base: &str) -> EmbeddingConfig {
        EmbeddingConfig {
            model: "test-model".into(),
            base_url: Some(base.into()),
            field: None,
            api_key_env: Some("DXMT_TEST_NO_SUCH_KEY".into()),
            batch_size: Some(2),
        }
    }

    #[test]
    fn a_response_out_of_order_is_put_back_in_input_order() {
        let body = serde_json::json!({
            "data": [
                {"index": 1, "embedding": [1.0, 1.0]},
                {"index": 0, "embedding": [0.0, 0.0]}
            ]
        });
        let parsed: EmbeddingsResponse = serde_json::from_value(body).unwrap();
        let mut data = parsed.data;
        data.sort_by_key(|d| d.index);
        assert_eq!(data[0].embedding, vec![0.0, 0.0]);
        assert_eq!(data[1].embedding, vec![1.0, 1.0]);
    }

    #[test]
    fn the_defaults_are_the_openai_ones_and_a_trailing_slash_does_not_double_up() {
        let bare = EmbeddingConfig {
            model: "m".into(),
            base_url: None,
            field: None,
            api_key_env: None,
            batch_size: None,
        };
        assert_eq!(bare.base_url(), "https://api.openai.com/v1");
        assert_eq!(bare.field(), "text");
        assert_eq!(bare.api_key_env(), "OPENAI_API_KEY");
        assert_eq!(bare.batch_size(), 64);
        assert_eq!(
            config("http://localhost:8080/v1/").base_url(),
            "http://localhost:8080/v1"
        );
    }

    #[tokio::test]
    async fn an_unreachable_provider_is_an_error_not_an_empty_vector() {
        // Port 1 is reserved and nothing listens on it.
        let err = embed_texts(
            &config("http://127.0.0.1:1/v1"),
            &["hello".to_string()],
            None,
        )
        .await
        .unwrap_err();
        assert_eq!(err.status_code(), 503, "got {err}");
    }

    #[tokio::test]
    async fn no_inputs_makes_no_request() {
        // Would fail with a connection error if it called the provider at all.
        let vectors = embed_texts(&config("http://127.0.0.1:1/v1"), &[], None)
            .await
            .unwrap();
        assert!(vectors.is_empty());
    }
}
