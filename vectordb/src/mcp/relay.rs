//! `vectordb mcp` as a relay: stdin and stdout on one side, a running server's `POST /mcp` on
//! the other.
//!
//! An MCP client starts its own `vectordb mcp` process, and a store can be open in only one
//! process at a time. So the second agent on a machine -- or the first, while `vectordb serve`
//! is running -- used to fail to start. Relaying through the server instead means every agent
//! reads and writes the same memory, however many there are and whichever way they connect.

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// Whether a DXMT server answers at `url`.
pub async fn server_answers(url: &str) -> bool {
    let Ok(client) = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
    else {
        return false;
    };
    client
        .get(format!("{}/live", url.trim_end_matches('/')))
        .send()
        .await
        .is_ok_and(|r| r.status().is_success())
}

/// Copy each JSON-RPC line from stdin to `{url}/mcp` and each reply to stdout, until stdin
/// closes. A request the server cannot be reached for is answered with a JSON-RPC error, so the
/// client sees why rather than waiting forever.
pub async fn relay_stdio(url: &str) -> crate::Result<()> {
    let endpoint = format!("{}/mcp", url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .build()
        .map_err(|e| crate::Error::Internal(format!("MCP relay: {e}")))?;
    let key = crate::config::env::var(crate::api::auth::KEY_VARIABLE)
        .ok()
        .filter(|k| !k.trim().is_empty());
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut stdout = tokio::io::stdout();
    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let mut request = client
            .post(&endpoint)
            .header("content-type", "application/json")
            .body(line.clone());
        if let Some(key) = &key {
            request = request.header(crate::api::auth::HEADER, key);
        }
        let reply = match request.send().await {
            Ok(r) if r.status() == reqwest::StatusCode::ACCEPTED => continue,
            Ok(r) if r.status().is_success() => r.text().await.unwrap_or_default(),
            Ok(r) => error_reply(&line, &format!("{endpoint} answered {}", r.status())),
            Err(e) => error_reply(&line, &format!("{endpoint} could not be reached: {e}")),
        };
        stdout.write_all(reply.trim_end().as_bytes()).await?;
        stdout.write_all(b"\n").await?;
        stdout.flush().await?;
    }
    Ok(())
}

fn error_reply(request: &str, message: &str) -> String {
    let id = serde_json::from_str::<serde_json::Value>(request)
        .ok()
        .and_then(|v| v.get("id").cloned())
        .unwrap_or(serde_json::Value::Null);
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": -32603, "message": message },
    })
    .to_string()
}
