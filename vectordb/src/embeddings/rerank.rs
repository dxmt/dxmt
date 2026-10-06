//! Rerank a page with a cross-encoder served by someone else.
//!
//! The same shape as [`super::service`]: the table names a provider, the key lives in the
//! server's environment, and a provider that will not answer fails the request rather than
//! returning the page unranked -- a page that looks reranked and is not is the failure that
//! shows nowhere. Two wire formats cover the field: Cohere's v2 (`relevance_score`, also
//! Jina's and Voyage's) and HuggingFace text-embeddings-inference's (`score`).

use crate::types::table::{RerankConfig, RerankFormat};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

const MAX_ATTEMPTS: u32 = 3;
const FIRST_BACKOFF: Duration = Duration::from_millis(500);

#[derive(Serialize)]
struct CohereRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: &'a [String],
    top_n: usize,
}

#[derive(Deserialize)]
struct CohereResponse {
    results: Vec<CohereResult>,
}

#[derive(Deserialize)]
struct CohereResult {
    index: usize,
    relevance_score: f32,
}

#[derive(Serialize)]
struct TeiRequest<'a> {
    query: &'a str,
    texts: &'a [String],
}

#[derive(Deserialize)]
struct TeiResult {
    index: usize,
    score: f32,
}

enum Retryable {
    Yes(Error),
    No(Error),
}

/// `(document index, score)` for the best `top_n` of `documents`, best first.
pub async fn rerank(
    config: &RerankConfig,
    query: &str,
    documents: &[String],
    top_n: usize,
) -> Result<Vec<(usize, f32)>> {
    if documents.is_empty() || top_n == 0 {
        return Ok(Vec::new());
    }
    let api_key = crate::config::env::var(config.api_key_env())
        .ok()
        .filter(|k| !k.is_empty());
    let url = format!("{}/rerank", config.base_url.trim_end_matches('/'));
    let top_n = top_n.min(documents.len());
    let mut backoff = FIRST_BACKOFF;
    let mut last: Option<Error> = None;
    for attempt in 1..=MAX_ATTEMPTS {
        let mut request = super::service::http().post(&url);
        if let Some(key) = api_key.as_deref() {
            request = request.bearer_auth(key);
        }
        let request = match config.format {
            RerankFormat::Cohere => request.json(&CohereRequest {
                model: &config.model,
                query,
                documents,
                top_n,
            }),
            RerankFormat::Tei => request.json(&TeiRequest {
                query,
                texts: documents,
            }),
        };
        let outcome = match request.send().await {
            Ok(response) => classify(response, &url, config).await,
            Err(e) => Err(Retryable::Yes(Error::ServiceUnavailable {
                message: format!("rerank provider at {url} is unreachable: {e}"),
            })),
        };
        match outcome {
            Ok(mut scored) => {
                scored.retain(|(i, _)| *i < documents.len());
                scored.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.0.cmp(&b.0))
                });
                scored.dedup_by_key(|(i, _)| *i);
                scored.truncate(top_n);
                return Ok(scored);
            }
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
    Err(last.unwrap_or_else(|| Error::Internal("rerank retry loop fell through".into())))
}

async fn classify(
    response: reqwest::Response,
    url: &str,
    config: &RerankConfig,
) -> std::result::Result<Vec<(usize, f32)>, Retryable> {
    let status = response.status();
    if status.is_success() {
        let body = response.text().await.map_err(|e| {
            Retryable::Yes(Error::ServiceUnavailable {
                message: format!("rerank provider at {url}: {e}"),
            })
        })?;
        let parsed = match config.format {
            RerankFormat::Cohere => serde_json::from_str::<CohereResponse>(&body).map(|r| {
                r.results
                    .into_iter()
                    .map(|r| (r.index, r.relevance_score))
                    .collect()
            }),
            RerankFormat::Tei => serde_json::from_str::<Vec<TeiResult>>(&body)
                .map(|r| r.into_iter().map(|r| (r.index, r.score)).collect()),
        };
        return parsed.map_err(|e| {
            Retryable::No(Error::InvalidConfig(format!(
                "{url} answered 200 but not with a {:?} rerank response ({e}). Check that \
                 base_url is the API root, the part before /rerank, and that `format` \
                 matches the provider.",
                config.format
            )))
        });
    }
    let detail = response
        .text()
        .await
        .map(|t| t.chars().take(400).collect::<String>())
        .unwrap_or_default();
    let message = format!(
        "rerank provider {url} (model {}) returned {status}: {detail}",
        config.model
    );
    Err(match status.as_u16() {
        429 | 500 | 502 | 503 | 504 => Retryable::Yes(Error::ServiceUnavailable { message }),
        _ => Retryable::No(Error::InvalidConfig(message)),
    })
}
