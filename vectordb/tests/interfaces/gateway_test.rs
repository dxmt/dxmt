//! The memory gateway: an OpenAI-compatible chat endpoint that puts the user's memory in front of
//! each request and keeps the exchange. The upstream "model" is the fake provider's chat route,
//! which records the messages it was sent and echoes the last user message, streaming or not.
#![cfg(feature = "server-embedding")]

use std::process::{Command, Stdio};
use std::time::Duration;

#[path = "../fake_provider/mod.rs"]
mod fake_provider;
#[path = "../common/http.rs"]
mod http;

struct Kill(std::process::Child);
impl Drop for Kill {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn serve(wal: &std::path::Path, base_url: &str, gateway: bool) -> (Kill, u16) {
    let port = free_port();
    let mut command = Command::new(env!("CARGO_BIN_EXE_vectordb"));
    command
        .args(["serve", "--host", "127.0.0.1", "--port", &port.to_string()])
        .env("DXMT_WAL_DIR", wal)
        .env("DXMT_EMBEDDING_MODEL", "bag-of-words")
        .env("DXMT_EMBEDDING_BASE_URL", base_url)
        .env("DXMT_EMBEDDING_AUTODETECT", "0")
        .env("DXMT_SNAPSHOT_INTERVAL_SECS", "0")
        .env("DXMT_GATEWAY_SAVE", "turns")
        .env("DXMT_GATEWAY_API_KEY_ENV", "DXMT_TEST_GATEWAY_KEY_UNSET")
        .env_remove("DXMT_API_KEY")
        .env_remove("DXMT_MEMORY_LLM_MODEL")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if gateway {
        command.env("DXMT_GATEWAY_UPSTREAM", base_url);
    } else {
        command.env_remove("DXMT_GATEWAY_UPSTREAM");
    }
    let server = Kill(command.spawn().unwrap());
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    while http::call(port, "GET", "/live", &[], None).is_none() {
        assert!(std::time::Instant::now() < deadline, "server never came up");
        std::thread::sleep(Duration::from_millis(100));
    }
    (server, port)
}

fn post(port: u16, path: &str, headers: &[(&str, &str)], body: serde_json::Value) -> http::Reply {
    http::request(
        port,
        "POST",
        path,
        headers,
        Some(body.to_string().as_bytes()),
        Duration::from_secs(30),
    )
    .expect("no answer")
}

fn last_upstream_request(provider: &fake_provider::Provider) -> Vec<serde_json::Value> {
    let seen = provider.seen.lock().unwrap();
    let last = seen
        .iter()
        .rev()
        .find(|s| s.starts_with('['))
        .expect("the upstream saw no chat request");
    serde_json::from_str(last).unwrap()
}

fn wait_for_turn(port: u16, user: &str, words: &str) -> serde_json::Value {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let reply = post(
            port,
            "/api/v1/memories/recall",
            &[],
            serde_json::json!({"query": words, "kind": "turn", "tags": [format!("user:{user}")]}),
        );
        let body: serde_json::Value = serde_json::from_str(&reply.text()).unwrap();
        if let Some(hit) = body["data"]["memories"].as_array().and_then(|m| {
            m.iter()
                .find(|m| m["text"].as_str().unwrap_or("").contains(words))
        }) {
            return hit.clone();
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the exchange was never kept: {body}"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}

#[test]
fn the_gateway_adds_the_users_memory_and_keeps_the_exchange() {
    let (base_url, provider) =
        fake_provider::serve_on_a_thread(fake_provider::Behaviour::Normal, 32);
    let tmp = tempfile::TempDir::new().unwrap();
    let (_server, port) = serve(&tmp.path().join("wal"), &base_url, true);

    for (text, user) in [
        ("Nate has a dog named Max", "nate"),
        ("Ann has a cat named Tom", "ann"),
    ] {
        let reply = post(
            port,
            "/api/v1/memories",
            &[],
            serde_json::json!({"text": text, "kind": "fact", "tags": [format!("user:{user}")]}),
        );
        assert_eq!(reply.status, 200, "{}", reply.text());
    }

    // Not streaming: the user's memory goes in after the request's own system message, nobody
    // else's does, and the model's answer comes back unchanged.
    let reply = post(
        port,
        "/v1/chat/completions",
        &[],
        serde_json::json!({
            "model": "any", "user": "nate",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "what is my dog named"}
            ]
        }),
    );
    assert_eq!(reply.status, 200, "{}", reply.text());
    let answer: serde_json::Value = serde_json::from_str(&reply.text()).unwrap();
    assert_eq!(
        answer["choices"][0]["message"]["content"],
        "echo: what is my dog named"
    );
    assert!(reply.header("x-dxmt-memories").unwrap_or("0") != "0");
    let sent = last_upstream_request(&provider);
    assert_eq!(sent[0]["content"], "be brief", "{sent:?}");
    let memory = sent[1]["content"].as_str().unwrap();
    assert_eq!(sent[1]["role"], "system");
    assert!(memory.contains("Nate has a dog named Max"), "{memory}");
    assert!(
        !memory.contains("Ann"),
        "another user's memory leaked: {memory}"
    );
    assert_eq!(sent[2]["content"], "what is my dog named");

    // ... and the exchange is kept as the user's turn.
    let turn = wait_for_turn(port, "nate", "what is my dog named");
    assert!(turn["text"]
        .as_str()
        .unwrap()
        .contains("Assistant: echo: what is my dog named"));

    // Streaming: the events pass through as sent, and the reassembled reply is kept.
    let reply = post(
        port,
        "/v1/chat/completions",
        &[("x-dxmt-user", "ann")],
        serde_json::json!({
            "model": "any", "stream": true,
            "messages": [{"role": "user", "content": "tell me about my cat"}]
        }),
    );
    assert_eq!(reply.status, 200);
    let events = reply.text();
    assert!(events.contains("data: [DONE]"), "{events}");
    let sent = last_upstream_request(&provider);
    assert!(sent[0]["content"]
        .as_str()
        .unwrap()
        .contains("Ann has a cat named Tom"));
    let turn = wait_for_turn(port, "ann", "tell me about my cat");
    assert!(
        turn["text"]
            .as_str()
            .unwrap()
            .contains("Assistant: echo: tell me about my cat"),
        "{turn}"
    );

    // X-DXMT-Memory: off sends the request as it came.
    let reply = post(
        port,
        "/v1/chat/completions",
        &[("x-dxmt-memory", "off")],
        serde_json::json!({"model": "any", "user": "nate", "messages": [{"role": "user", "content": "dog"}]}),
    );
    assert_eq!(reply.status, 200);
    assert_eq!(last_upstream_request(&provider).len(), 1);
}

#[test]
fn without_an_upstream_the_gateway_says_what_to_set() {
    let (base_url, _provider) =
        fake_provider::serve_on_a_thread(fake_provider::Behaviour::Normal, 32);
    let tmp = tempfile::TempDir::new().unwrap();
    let (_server, port) = serve(&tmp.path().join("wal"), &base_url, false);
    let reply = post(
        port,
        "/v1/chat/completions",
        &[],
        serde_json::json!({"model": "any", "messages": [{"role": "user", "content": "hi"}]}),
    );
    assert_eq!(reply.status, 404);
    assert!(
        reply.text().contains("DXMT_GATEWAY_UPSTREAM"),
        "{}",
        reply.text()
    );
}
