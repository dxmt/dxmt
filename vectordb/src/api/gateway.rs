//! A memory gateway: an OpenAI-compatible `POST /v1/chat/completions` in front of any chat model,
//! which puts what DXMT remembers about the user in front of each request and keeps the exchange
//! afterwards.
//!
//! Every way of giving an agent memory so far asks the agent to change: an MCP server it must call,
//! a plugin for its framework. An application that already talks to an OpenAI-compatible endpoint
//! -- most of them -- gets memory here by changing one URL: `base_url = http://dxmt:9999/v1`.
//! OpenViking ships the same idea as a separate gateway process; here it is the server itself.
//!
//! Per request:
//! 1. The user is the request's `user` field (the OpenAI parameter for exactly this), or the
//!    `X-DXMT-User` header. Their memories carry the tag `user:<id>`; without one, the memory is
//!    the server's shared memory.
//! 2. The last user message is the query for `context` (profile, standing preferences and
//!    instructions, relevant memories, inside `DXMT_GATEWAY_CONTEXT_TOKENS`), and the block goes in
//!    as a system message after the request's own system messages. `X-DXMT-Memory: off` skips it.
//!    A failure to build it (no embedding model) is logged and the request goes on without it:
//!    memory must never be the reason a chat request fails.
//! 3. The request goes to `DXMT_GATEWAY_UPSTREAM` unchanged otherwise, streaming or not, with the
//!    key from the variable `DXMT_GATEWAY_API_KEY_ENV` names (default `OPENAI_API_KEY`), under the
//!    same rules as an embedding key.
//! 4. Afterwards, off the response path: with `DXMT_GATEWAY_SAVE=extract` (the default when the
//!    server has a memory model) the exchange goes through `extract`; with `turns`, it is kept as
//!    one memory of kind `turn`; `off` keeps nothing.
//!
//! Off unless `DXMT_GATEWAY_UPSTREAM` is set.

use std::sync::{Arc, Mutex};

use axum::{
    body::{Bytes, StreamBody},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Extension,
};
use futures::StreamExt;

use crate::storage::agent_memory::{ContextRequest, RememberRequest};
use crate::storage::memory_derive::{DeriveModel, ExtractRequest};
use crate::storage::IntelligentIndexedStorage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Save {
    Off,
    Turns,
    Extract,
}

#[derive(Debug, Clone)]
struct Gateway {
    upstream: String,
    api_key_env: String,
    context_tokens: usize,
    save: Save,
}

fn var(name: &str) -> Option<String> {
    crate::config::env::var(name)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

impl Gateway {
    fn from_env() -> Option<Self> {
        let upstream = var("DXMT_GATEWAY_UPSTREAM")?;
        let save = match var("DXMT_GATEWAY_SAVE").as_deref() {
            Some("off") => Save::Off,
            Some("turns") => Save::Turns,
            Some("extract") => Save::Extract,
            _ if DeriveModel::from_env().is_some() => Save::Extract,
            _ => Save::Turns,
        };
        Some(Self {
            upstream: upstream.trim_end_matches('/').to_string(),
            api_key_env: var("DXMT_GATEWAY_API_KEY_ENV").unwrap_or_else(|| "OPENAI_API_KEY".into()),
            context_tokens: var("DXMT_GATEWAY_CONTEXT_TOKENS")
                .and_then(|v| v.parse().ok())
                .unwrap_or(1200),
            save,
        })
    }
}

fn openai_error(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        axum::Json(
            serde_json::json!({"error": {"message": message.into(), "type": "dxmt_gateway"}}),
        ),
    )
        .into_response()
}

fn off() -> Response {
    openai_error(
        StatusCode::NOT_FOUND,
        "the memory gateway is off: set DXMT_GATEWAY_UPSTREAM to the chat model's OpenAI-compatible \
         base URL (e.g. https://api.openai.com/v1) and restart",
    )
}

/// The text of a message's `content`: a string, or the text parts of a list of parts.
fn content_text(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn last_user_text(messages: &[serde_json::Value]) -> String {
    messages
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        .map(|m| content_text(m.get("content").unwrap_or(&serde_json::Value::Null)))
        .unwrap_or_default()
}

/// The memory block as a system message, after the request's own leading system messages.
fn insert_memory(messages: &mut Vec<serde_json::Value>, block: &str) {
    let at = messages
        .iter()
        .take_while(|m| {
            matches!(
                m.get("role").and_then(|r| r.as_str()),
                Some("system") | Some("developer")
            )
        })
        .count();
    messages.insert(
        at,
        serde_json::json!({
            "role": "system",
            "content": format!(
                "What you remember about this user from earlier conversations (DXMT memory). Use \
                 it where it helps; do not mention it unless asked.\n\n{block}"
            ),
        }),
    );
}

/// The assistant's text in a streamed response so far: the `delta.content` of each `data:` line.
fn collect_deltas(buffer: &mut String, chunk: &[u8], text: &Mutex<String>) {
    buffer.push_str(&String::from_utf8_lossy(chunk));
    while let Some(end) = buffer.find('\n') {
        let line: String = buffer.drain(..=end).collect();
        let Some(data) = line.trim().strip_prefix("data:") else {
            continue;
        };
        let Ok(event) = serde_json::from_str::<serde_json::Value>(data.trim()) else {
            continue;
        };
        if let Some(delta) = event
            .pointer("/choices/0/delta/content")
            .and_then(|c| c.as_str())
        {
            if let Ok(mut t) = text.lock() {
                t.push_str(delta);
            }
        }
    }
}

async fn keep(
    storage: Arc<IntelligentIndexedStorage>,
    save: Save,
    tags: Option<Vec<String>>,
    user_text: String,
    reply: String,
) {
    if save == Save::Off || user_text.trim().is_empty() {
        return;
    }
    let exchange = format!("User: {}\nAssistant: {}", user_text.trim(), reply.trim());
    let result = if save == Save::Extract && DeriveModel::from_env().is_some() {
        storage
            .extract(ExtractRequest {
                text: exchange,
                when: Some(chrono::Utc::now().format("%-d %B %Y").to_string()),
                tags: tags.clone(),
                source: Some("gateway".to_string()),
                id_prefix: None,
            })
            .await
            .map(|_| ())
    } else {
        storage
            .remember(RememberRequest {
                text: exchange,
                kind: Some("turn".to_string()),
                tags: tags.clone(),
                source: Some("gateway".to_string()),
                ..Default::default()
            })
            .await
            .map(|_| ())
    };
    if let Err(e) = result {
        tracing::warn!("memory gateway: the exchange was not kept: {e}");
    }
}

/// `POST /v1/chat/completions`.
pub async fn chat_completions(
    Extension(storage): Extension<Arc<IntelligentIndexedStorage>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let Some(gateway) = Gateway::from_env() else {
        return off();
    };
    let Ok(mut request) = serde_json::from_slice::<serde_json::Value>(&body) else {
        return openai_error(StatusCode::BAD_REQUEST, "the body is not JSON");
    };
    let user = request_user(&headers, request.get("user").and_then(|u| u.as_str()));
    let tags = user.as_ref().map(|u| vec![format!("user:{u}")]);
    let Some(messages) = request.get("messages").and_then(|m| m.as_array()) else {
        return openai_error(StatusCode::BAD_REQUEST, "the body has no `messages` array");
    };
    let user_text = last_user_text(messages);
    let memory_on = headers
        .get("x-dxmt-memory")
        .and_then(|v| v.to_str().ok())
        .is_none_or(|v| !v.eq_ignore_ascii_case("off"));

    let mut used = 0usize;
    if memory_on && !user_text.trim().is_empty() {
        match storage
            .context(ContextRequest {
                query: user_text.clone(),
                tags: tags.clone(),
                max_tokens: Some(gateway.context_tokens),
            })
            .await
        {
            Ok(block) => {
                let text = block["context"].as_str().unwrap_or("");
                used = block["memories"].as_array().map_or(0, |m| m.len());
                if !text.is_empty() {
                    if let Some(messages) =
                        request.get_mut("messages").and_then(|m| m.as_array_mut())
                    {
                        insert_memory(messages, text);
                    }
                }
            }
            Err(e) => tracing::warn!("memory gateway: no memory for this request: {e}"),
        }
    }
    let streaming = request
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);

    let url = format!("{}/chat/completions", gateway.upstream);
    let upstream = match send(&gateway, &url, &request).await {
        Ok(r) => r,
        Err(e) => return openai_error(StatusCode::BAD_GATEWAY, e.to_string()),
    };
    let status =
        StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let content_type = upstream
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or(if streaming {
            "text/event-stream"
        } else {
            "application/json"
        })
        .to_string();
    let mut out = HeaderMap::new();
    if let Ok(v) = HeaderValue::from_str(&content_type) {
        out.insert(header::CONTENT_TYPE, v);
    }
    out.insert("x-dxmt-memories", HeaderValue::from(used));
    let save = if status.is_success() {
        gateway.save
    } else {
        Save::Off
    };

    if streaming {
        let text = Arc::new(Mutex::new(String::new()));
        let mut buffer = String::new();
        let seen = text.clone();
        let stream = upstream
            .bytes_stream()
            .inspect(move |chunk| {
                if let Ok(chunk) = chunk {
                    collect_deltas(&mut buffer, chunk, &seen);
                }
            })
            .chain(futures::stream::once(async move {
                let reply = text.lock().map(|t| t.clone()).unwrap_or_default();
                tokio::spawn(keep(storage, save, tags, user_text, reply));
                Ok(Bytes::new())
            }));
        return (status, out, StreamBody::new(stream)).into_response();
    }

    let bytes = match upstream.bytes().await {
        Ok(b) => b,
        Err(e) => return openai_error(StatusCode::BAD_GATEWAY, e.to_string()),
    };
    let reply = serde_json::from_slice::<serde_json::Value>(&bytes)
        .ok()
        .and_then(|v| v.pointer("/choices/0/message/content").map(content_text))
        .unwrap_or_default();
    tokio::spawn(keep(storage, save, tags, user_text, reply));
    (status, out, bytes).into_response()
}

/// `GET /v1/models`: the upstream's, so a client that lists models before choosing one works.
pub async fn models() -> Response {
    let Some(gateway) = Gateway::from_env() else {
        return off();
    };
    let url = format!("{}/models", gateway.upstream);
    let mut request = crate::embeddings::service::http().get(&url);
    match crate::embeddings::policy::provider_key(&gateway.api_key_env, &url) {
        Ok(Some(key)) => request = request.bearer_auth(key),
        Ok(None) => {}
        Err(e) => return openai_error(StatusCode::BAD_GATEWAY, e.to_string()),
    }
    match request.send().await {
        Ok(r) => {
            let status = StatusCode::from_u16(r.status().as_u16()).unwrap_or(StatusCode::OK);
            let bytes = r.bytes().await.unwrap_or_default();
            (status, [(header::CONTENT_TYPE, "application/json")], bytes).into_response()
        }
        Err(e) => openai_error(StatusCode::BAD_GATEWAY, e.to_string()),
    }
}

/// Whose memory: the request's `user` (the OpenAI parameter for an end-user id), else the
/// `X-DXMT-User` header.
fn request_user(headers: &HeaderMap, body_user: Option<&str>) -> Option<String> {
    body_user
        .map(str::to_string)
        .or_else(|| {
            headers
                .get("x-dxmt-user")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string)
        })
        .map(|u| u.trim().to_string())
        .filter(|u| !u.is_empty())
}

async fn send(
    gateway: &Gateway,
    url: &str,
    request: &serde_json::Value,
) -> crate::Result<reqwest::Response> {
    let key = crate::embeddings::policy::provider_key(&gateway.api_key_env, url)?;
    let mut builder = crate::embeddings::service::http()
        .post(url)
        .timeout(std::time::Duration::from_secs(600))
        .json(request);
    if let Some(key) = key {
        builder = builder.bearer_auth(key);
    }
    builder
        .send()
        .await
        .map_err(|e| crate::Error::Internal(format!("the chat model at {url}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_last_user_message_is_the_query_whatever_its_shape() {
        let messages = vec![
            serde_json::json!({"role": "system", "content": "be brief"}),
            serde_json::json!({"role": "user", "content": "first"}),
            serde_json::json!({"role": "assistant", "content": "ok"}),
            serde_json::json!({"role": "user", "content": [{"type": "text", "text": "my dog"}, {"type": "image_url"}]}),
        ];
        assert_eq!(last_user_text(&messages), "my dog");
    }

    #[test]
    fn memory_goes_after_the_requests_own_system_messages() {
        let mut messages = vec![
            serde_json::json!({"role": "system", "content": "be brief"}),
            serde_json::json!({"role": "user", "content": "hi"}),
        ];
        insert_memory(&mut messages, "## About the user\nLikes tea");
        assert_eq!(messages[0]["content"], "be brief");
        assert_eq!(messages[1]["role"], "system");
        assert!(messages[1]["content"]
            .as_str()
            .unwrap()
            .contains("Likes tea"));
        assert_eq!(messages[2]["role"], "user");
    }

    #[test]
    fn a_streamed_reply_is_reassembled_across_chunk_boundaries() {
        let text = Mutex::new(String::new());
        let mut buffer = String::new();
        collect_deltas(
            &mut buffer,
            b"data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\ndata: {\"choi",
            &text,
        );
        collect_deltas(
            &mut buffer,
            b"ces\":[{\"delta\":{\"content\":\"lo\"}}]}\n\ndata: [DONE]\n\n",
            &text,
        );
        assert_eq!(*text.lock().unwrap(), "Hello");
    }
}
