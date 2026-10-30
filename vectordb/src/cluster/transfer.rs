//! Moving a table between shards, the node's half: freezing a table against writes, and
//! carrying it through the bucket (`transfers/...`) with the export/import a single server
//! already has. The router drives a move (`router::ClusterRouter::move_table`).
//!
//! Freezing has to happen where the write lands, not at the routers: a router that has not yet
//! reread the placement would still send writes to the old shard, and a write accepted there
//! after the export would be lost. So the node refuses them itself (503 `TABLE_MOVING`), and a
//! freeze waits for the writes already past that check to finish before the export starts.

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, Method, Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use parking_lot::Mutex;

use super::ClusterStore;
use crate::storage::IntelligentIndexedStorage;

/// Sent by a router on the request that drops a moved table from its old shard: the one write
/// a frozen table still takes.
pub const MOVE_DROP_HEADER: &str = "x-dxmt-move-drop";

#[derive(Default)]
struct TableGate {
    frozen: bool,
    in_flight: usize,
}

/// Which tables are frozen, and how many writes to each are running.
#[derive(Default)]
pub struct Freezer {
    tables: Mutex<HashMap<String, TableGate>>,
}

impl Freezer {
    /// Refuse new writes to `table`, then wait until those already running are done.
    pub async fn freeze(&self, table: &str, wait: std::time::Duration) -> bool {
        self.tables
            .lock()
            .entry(table.to_string())
            .or_default()
            .frozen = true;
        let deadline = std::time::Instant::now() + wait;
        loop {
            if self
                .tables
                .lock()
                .get(table)
                .is_none_or(|g| g.in_flight == 0)
            {
                return true;
            }
            if std::time::Instant::now() > deadline {
                return false;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }

    pub fn unfreeze(&self, table: &str) {
        let mut tables = self.tables.lock();
        if let Some(gate) = tables.get_mut(table) {
            gate.frozen = false;
            if gate.in_flight == 0 {
                tables.remove(table);
            }
        }
    }

    /// Admit a write to `table`, or `false` when it is frozen.
    fn enter(&self, table: &str) -> bool {
        let mut tables = self.tables.lock();
        let gate = tables.entry(table.to_string()).or_default();
        if gate.frozen {
            return false;
        }
        gate.in_flight += 1;
        true
    }

    fn leave(&self, table: &str) {
        let mut tables = self.tables.lock();
        if let Some(gate) = tables.get_mut(table) {
            gate.in_flight = gate.in_flight.saturating_sub(1);
            if gate.in_flight == 0 && !gate.frozen {
                tables.remove(table);
            }
        }
    }
}

/// The table a request is about, and whether it writes: `None` for anything else.
fn table_write(method: &Method, path: &str) -> Option<String> {
    let rest = path.trim_end_matches('/').strip_prefix("/api/v1/tables/")?;
    let table = rest.split('/').next().filter(|t| !t.is_empty())?;
    if table == "import" {
        return None;
    }
    let writes = *method != Method::GET
        && *method != Method::HEAD
        && !crate::api::admission::is_query(method, path);
    writes.then(|| super::percent_decode(table))
}

/// Middleware: refuse writes to a frozen table, and count the ones admitted.
pub async fn freeze_guard(
    State(freezer): State<Arc<Freezer>>,
    request: Request<Body>,
    next: Next<Body>,
) -> Response {
    let Some(table) = table_write(request.method(), request.uri().path()) else {
        return next.run(request).await;
    };
    // The move's own drop of the old copy.
    let dropping_moved = request.method() == Method::DELETE
        && request
            .headers()
            .get(MOVE_DROP_HEADER)
            .and_then(|v| v.to_str().ok())
            == Some(super::router_secret().as_str());
    if dropping_moved {
        let response = next.run(request).await;
        freezer.unfreeze(&table);
        return response;
    }
    if !freezer.enter(&table) {
        let body = serde_json::json!({
            "success": false,
            "data": null,
            "error": {
                "code": "TABLE_MOVING",
                "message": format!("table '{table}' is being moved to another shard; retry shortly"),
                "details": null,
            },
        });
        let mut response = (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response();
        response
            .headers_mut()
            .insert("retry-after", axum::http::HeaderValue::from_static("1"));
        return response;
    }
    let response = next.run(request).await;
    freezer.leave(&table);
    response
}

/// What the node's transfer routes need.
#[derive(Clone)]
pub struct NodeApi {
    pub storage: Arc<IntelligentIndexedStorage>,
    pub cluster: Arc<ClusterStore>,
    pub freezer: Arc<Freezer>,
}

/// `/cluster/node/{freeze,unfreeze,export,import}`, for routers only.
pub fn routes<S: Clone + Send + Sync + 'static>(api: NodeApi) -> axum::Router<S> {
    axum::Router::new()
        .route("/cluster/node/freeze", axum::routing::post(freeze))
        .route("/cluster/node/unfreeze", axum::routing::post(unfreeze))
        .route("/cluster/node/export", axum::routing::post(export))
        .route("/cluster/node/import", axum::routing::post(import))
        .with_state(api)
}

fn from_router(headers: &HeaderMap) -> bool {
    headers
        .get(super::ROUTER_HEADER)
        .and_then(|v| v.to_str().ok())
        == Some(super::router_secret().as_str())
}

fn not_a_router() -> Response {
    answer(
        StatusCode::FORBIDDEN,
        Err("only a router may ask this".to_string()),
    )
}

fn answer(status: StatusCode, result: Result<serde_json::Value, String>) -> Response {
    let body = match result {
        Ok(data) => serde_json::json!({ "success": true, "data": data, "error": null }),
        Err(message) => serde_json::json!({
            "success": false,
            "data": null,
            "error": { "code": "TRANSFER_FAILED", "message": message, "details": null },
        }),
    };
    (status, Json(body)).into_response()
}

#[derive(serde::Deserialize)]
struct TableBody {
    table: String,
}

async fn freeze(
    State(api): State<NodeApi>,
    headers: HeaderMap,
    Json(body): Json<TableBody>,
) -> Response {
    if !from_router(&headers) {
        return not_a_router();
    }
    if api
        .freezer
        .freeze(&body.table, std::time::Duration::from_secs(60))
        .await
    {
        answer(
            StatusCode::OK,
            Ok(serde_json::json!({ "frozen": body.table })),
        )
    } else {
        api.freezer.unfreeze(&body.table);
        answer(
            StatusCode::SERVICE_UNAVAILABLE,
            Err(format!(
                "writes to '{}' did not finish within 60 s",
                body.table
            )),
        )
    }
}

async fn unfreeze(
    State(api): State<NodeApi>,
    headers: HeaderMap,
    Json(body): Json<TableBody>,
) -> Response {
    if !from_router(&headers) {
        return not_a_router();
    }
    api.freezer.unfreeze(&body.table);
    answer(
        StatusCode::OK,
        Ok(serde_json::json!({ "unfrozen": body.table })),
    )
}

/// A scratch file for a transfer, under the backup root (the one place export and import may
/// read and write).
fn scratch(storage: &IntelligentIndexedStorage) -> Result<std::path::PathBuf, String> {
    let root = storage
        .backup_root_dir()
        .ok_or("this node has no data directory to stage a transfer in")?;
    let dir = root.join("transfers");
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    Ok(dir.join(format!("{}.dxmt", uuid::Uuid::new_v4())))
}

#[derive(serde::Deserialize)]
struct ExportBody {
    table: String,
    key: String,
}

/// Export a table and put the file in the bucket at `key`.
async fn export(
    State(api): State<NodeApi>,
    headers: HeaderMap,
    Json(body): Json<ExportBody>,
) -> Response {
    if !from_router(&headers) {
        return not_a_router();
    }
    let result = async {
        let path = scratch(&api.storage)?;
        let summary = api
            .storage
            .export_table(&body.table, &path)
            .await
            .map_err(|e| e.to_string());
        let uploaded = match summary {
            Ok(summary) => {
                let bytes = tokio::fs::read(&path).await.map_err(|e| e.to_string())?;
                let size = bytes.len();
                api.cluster
                    .store()
                    .put_object(&body.key, bytes::Bytes::from(bytes), None)
                    .await
                    .map_err(|e| format!("uploading the export: {e}"))
                    .map(|()| serde_json::json!({ "export": summary, "bytes": size }))
            }
            Err(e) => Err(e),
        };
        let _ = tokio::fs::remove_file(&path).await;
        uploaded
    }
    .await;
    match result {
        Ok(data) => answer(StatusCode::OK, Ok(data)),
        Err(e) => answer(StatusCode::INTERNAL_SERVER_ERROR, Err(e)),
    }
}

#[derive(serde::Deserialize)]
struct ImportBody {
    key: String,
    name: String,
}

/// Fetch an export from the bucket and import it under `name`.
async fn import(
    State(api): State<NodeApi>,
    headers: HeaderMap,
    Json(body): Json<ImportBody>,
) -> Response {
    if !from_router(&headers) {
        return not_a_router();
    }
    let result = async {
        let bytes = api
            .cluster
            .store()
            .get_object(&body.key)
            .await
            .map_err(|e| format!("fetching the export: {e}"))?;
        let path = scratch(&api.storage)?;
        tokio::fs::write(&path, &bytes)
            .await
            .map_err(|e| e.to_string())?;
        let imported = api
            .storage
            .import_table(&path, Some(&body.name), false)
            .await
            .map_err(|e| e.to_string());
        let _ = tokio::fs::remove_file(&path).await;
        imported
    }
    .await;
    match result {
        Ok(data) => answer(StatusCode::OK, Ok(data)),
        Err(e) => answer(StatusCode::INTERNAL_SERVER_ERROR, Err(e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn which_requests_are_table_writes() {
        assert_eq!(
            table_write(&Method::POST, "/api/v1/tables/t/records").as_deref(),
            Some("t")
        );
        assert_eq!(
            table_write(&Method::DELETE, "/api/v1/tables/t").as_deref(),
            Some("t")
        );
        assert_eq!(table_write(&Method::POST, "/api/v1/tables/t/search"), None);
        assert_eq!(
            table_write(&Method::GET, "/api/v1/tables/t/records/x"),
            None
        );
        assert_eq!(table_write(&Method::POST, "/api/v1/tables"), None);
    }

    #[tokio::test]
    async fn a_freeze_waits_for_the_writes_already_running() {
        let freezer = Arc::new(Freezer::default());
        assert!(freezer.enter("t"));
        let f = Arc::clone(&freezer);
        let freezing =
            tokio::spawn(async move { f.freeze("t", std::time::Duration::from_secs(5)).await });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(!freezer.enter("t"), "frozen: no new writes");
        assert!(!freezing.is_finished(), "still waiting for the running one");
        freezer.leave("t");
        assert!(freezing.await.unwrap());
        freezer.unfreeze("t");
        assert!(freezer.enter("t"));
    }
}
