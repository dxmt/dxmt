//! `vectordb router`: the entry point to a cluster. It serves `/api/v1` like a single server
//! and forwards each request to the shard that holds its table.
//!
//! It keeps nothing of its own. Its view of the cluster -- the placement and the live nodes --
//! is read from the bucket every `DXMT_ROUTER_REFRESH_MS` (default 1,000) and again whenever a
//! request names a table it does not know, so any number of routers can run side by side and a
//! restarted one is ready as soon as it has read the bucket once.
//!
//! What it does with a request:
//! - a table's routes go to its shard's primary; a read may go to a replica instead when the
//!   request says `X-DXMT-Read: replica` (replicas trail the primary by their poll interval);
//! - creating a table claims the name in the placement first (`creating`), on the shard holding
//!   the fewest tables, then creates it there and marks it `ready` -- or releases the claim if
//!   the shard refused;
//! - dropping one drops it on its shard, then removes it from the placement;
//! - listing tables and taking snapshots ask every shard;
//! - the knowledge-base table (`kb`), which a server makes on its first use, is placed the
//!   first time a request names it;
//! - anything that spans tables in a way one shard cannot answer -- aliases, DXQL, backups,
//!   imports -- is refused with 501 rather than answered for one shard as if it were all.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, HeaderValue, Method, Request, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use parking_lot::RwLock;

use super::nodes::{self, NodeRecord, ShardView};
use super::placement::{self, Placement, State as PlacementState};
use super::ClusterStore;

/// Above this, a request body is refused rather than buffered (a node's own limit is 500 MB).
const MAX_BODY: usize = 500 * 1024 * 1024;

#[derive(Debug, Clone, Default)]
struct View {
    placement: Placement,
    shards: BTreeMap<String, ShardView>,
    loaded: bool,
}

pub struct ClusterRouter {
    cluster: Arc<ClusterStore>,
    view: RwLock<View>,
    http: reqwest::Client,
    next_replica: AtomicUsize,
    secret: String,
    kb_table: Option<String>,
}

fn envelope(status: StatusCode, code: &str, message: impl Into<String>) -> Response {
    let body = serde_json::json!({
        "success": false,
        "data": null,
        "error": { "code": code, "message": message.into(), "details": null },
        "timestamp": chrono::Utc::now().to_rfc3339(),
    });
    (status, axum::Json(body)).into_response()
}

fn success(data: serde_json::Value) -> Response {
    let body = serde_json::json!({
        "success": true,
        "data": data,
        "error": null,
        "timestamp": chrono::Utc::now().to_rfc3339(),
    });
    (StatusCode::OK, axum::Json(body)).into_response()
}

/// Headers a proxy must not pass on (RFC 9110 §7.6.1), plus the ones the client library sets.
fn hop_by_hop(name: &str) -> bool {
    matches!(
        name,
        "connection"
            | "keep-alive"
            | "proxy-connection"
            | "transfer-encoding"
            | "te"
            | "trailer"
            | "upgrade"
            | "host"
            | "content-length"
    )
}

impl ClusterRouter {
    pub async fn new(cluster: Arc<ClusterStore>) -> crate::Result<Arc<Self>> {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(
                crate::config::env::var("DXMT_ROUTER_TIMEOUT_SECS")
                    .ok()
                    .and_then(|v| v.trim().parse().ok())
                    .unwrap_or(300),
            ))
            .build()
            .map_err(|e| crate::Error::Internal(format!("router HTTP client: {e}")))?;
        let router = Arc::new(Self {
            cluster,
            view: RwLock::new(View::default()),
            http,
            next_replica: AtomicUsize::new(0),
            secret: super::router_secret(),
            kb_table: crate::storage::knowledge_base::kb_table_name(),
        });
        router.refresh().await?;
        Ok(router)
    }

    /// Read the placement and the nodes from the bucket.
    pub async fn refresh(&self) -> crate::Result<()> {
        let (placement, _) = placement::load(&self.cluster).await?;
        let records = nodes::list(&self.cluster).await?;
        let mut shards = nodes::shards(&records, super::now(), nodes::ttl());
        // A shard that holds tables and has no live node is still a shard: it should be
        // reported as unavailable, not forgotten.
        for placed in placement.tables.values() {
            shards.entry(placed.shard.clone()).or_default();
        }
        *self.view.write() = View {
            placement,
            shards,
            loaded: true,
        };
        Ok(())
    }

    pub fn spawn_refresh(self: &Arc<Self>) {
        let router = Arc::clone(self);
        let every = std::time::Duration::from_millis(
            crate::config::env::var("DXMT_ROUTER_REFRESH_MS")
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(1_000),
        );
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(every);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                ticker.tick().await;
                if let Err(e) = router.refresh().await {
                    tracing::warn!("router refresh: {e}");
                }
            }
        });
    }

    /// The axum app: one handler for everything, behind the API key when one is set.
    pub fn app(self: Arc<Self>, api_key: Option<crate::api::auth::ApiKey>) -> axum::Router {
        axum::Router::new()
            .fallback(dispatch)
            .with_state(self)
            .layer(axum::middleware::from_fn_with_state(
                api_key,
                crate::api::auth::require_key,
            ))
            .layer(tower_http::trace::TraceLayer::new_for_http())
    }

    fn snapshot_view(&self) -> View {
        self.view.read().clone()
    }

    /// The shard a table is on, reading the bucket again once if it is not known.
    async fn shard_of(&self, table: &str) -> Option<String> {
        if let Some(shard) = self.view.read().placement.shard_of(table) {
            return Some(shard.to_string());
        }
        let _ = self.refresh().await;
        self.view
            .read()
            .placement
            .shard_of(table)
            .map(str::to_string)
    }

    fn primary(&self, shard: &str) -> Option<NodeRecord> {
        self.view
            .read()
            .shards
            .get(shard)
            .and_then(|s| s.primary.clone())
    }

    fn replica(&self, shard: &str) -> Option<NodeRecord> {
        let view = self.view.read();
        let replicas = &view.shards.get(shard)?.replicas;
        if replicas.is_empty() {
            return None;
        }
        let i = self.next_replica.fetch_add(1, Ordering::Relaxed) % replicas.len();
        Some(replicas[i].clone())
    }

    fn unavailable(shard: &str) -> Response {
        let mut response = envelope(
            StatusCode::SERVICE_UNAVAILABLE,
            "SHARD_UNAVAILABLE",
            format!("shard '{shard}' has no ready primary right now; retry shortly"),
        );
        response
            .headers_mut()
            .insert("retry-after", HeaderValue::from_static("2"));
        response
    }

    /// Send a request to a node and hand back its answer, marked with where it was served.
    async fn forward(
        &self,
        node: &NodeRecord,
        method: &Method,
        path_and_query: &str,
        headers: &HeaderMap,
        body: Bytes,
    ) -> Response {
        match self
            .try_forward(node, method, path_and_query, headers, body)
            .await
        {
            Ok(response) => response,
            Err(e) => self.unreachable(node, e).await,
        }
    }

    async fn unreachable(&self, node: &NodeRecord, e: reqwest::Error) -> Response {
        tracing::warn!("forwarding to {}: {e}", node.url);
        // The node may be gone; the next refresh will say so.
        let _ = self.refresh().await;
        envelope(
            StatusCode::BAD_GATEWAY,
            "SHARD_UNREACHABLE",
            format!("shard '{}' did not answer: {e}", node.shard),
        )
    }

    /// [`Self::forward`], with a failure to reach the node returned rather than answered.
    async fn try_forward(
        &self,
        node: &NodeRecord,
        method: &Method,
        path_and_query: &str,
        headers: &HeaderMap,
        body: Bytes,
    ) -> Result<Response, reqwest::Error> {
        let url = format!("{}{}", node.url, path_and_query);
        let Ok(method) = reqwest::Method::from_bytes(method.as_str().as_bytes()) else {
            return Ok(envelope(
                StatusCode::METHOD_NOT_ALLOWED,
                "INVALID_INPUT",
                "method",
            ));
        };
        let mut request = self.http.request(method, &url);
        for (name, value) in headers {
            if !hop_by_hop(name.as_str()) && name.as_str() != super::ROUTER_HEADER {
                request = request.header(name.as_str(), value.as_bytes());
            }
        }
        request = request.header(super::ROUTER_HEADER, self.secret.as_str());
        let answer = request.body(body).send().await?;
        let status =
            StatusCode::from_u16(answer.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        let mut out = HeaderMap::new();
        for (name, value) in answer.headers() {
            if !hop_by_hop(name.as_str()) {
                if let (Ok(n), Ok(v)) = (
                    axum::http::HeaderName::from_bytes(name.as_str().as_bytes()),
                    HeaderValue::from_bytes(value.as_bytes()),
                ) {
                    out.append(n, v);
                }
            }
        }
        let bytes = answer.bytes().await.unwrap_or_default();
        if let Ok(v) = HeaderValue::from_str(&node.shard) {
            out.insert("x-dxmt-shard", v);
        }
        if let Ok(v) = HeaderValue::from_str(&node.node) {
            out.insert("x-dxmt-node", v);
        }
        let mut response = (status, bytes).into_response();
        response.headers_mut().extend(out);
        Ok(response)
    }

    /// A table's request: to its shard's primary, or a replica for a read that asks for one.
    async fn table_request(
        &self,
        table: &str,
        method: &Method,
        path_and_query: &str,
        headers: &HeaderMap,
        body: Bytes,
    ) -> Response {
        let shard = match self.shard_of(table).await {
            Some(shard) => shard,
            None if self.kb_table.as_deref() == Some(table) => match self.place(table).await {
                Ok(shard) => shard,
                Err(e) => {
                    return envelope(StatusCode::SERVICE_UNAVAILABLE, "NO_SHARDS", e.to_string())
                }
            },
            None => {
                return envelope(
                    StatusCode::NOT_FOUND,
                    "TABLE_NOT_FOUND",
                    format!("Table not found: {table}"),
                )
            }
        };
        let path = path_and_query.split('?').next().unwrap_or(path_and_query);
        let wants_replica = headers
            .get("x-dxmt-read")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.eq_ignore_ascii_case("replica"));
        let is_read = *method == Method::GET || crate::api::admission::is_query(method, path);
        if wants_replica && is_read {
            if let Some(replica) = self.replica(&shard) {
                match self
                    .try_forward(&replica, method, path_and_query, headers, body.clone())
                    .await
                {
                    // A replica trails its primary: a table or record it has not applied yet
                    // is a 404 there and not an answer. The primary has it.
                    Ok(response) if response.status() != StatusCode::NOT_FOUND => return response,
                    Ok(_) => {}
                    Err(e) => {
                        tracing::warn!("replica {} unreachable: {e}", replica.url);
                        let _ = self.refresh().await;
                    }
                }
            }
        }
        // The primary; once more after reading the cluster again if it could not be reached
        // at all -- nothing was sent, so nothing is repeated, and a failover may have moved it.
        for attempt in 0..2 {
            let Some(primary) = self.primary(&shard) else {
                return Self::unavailable(&shard);
            };
            match self
                .try_forward(&primary, method, path_and_query, headers, body.clone())
                .await
            {
                Ok(response) => return response,
                Err(e) if e.is_connect() && attempt == 0 => {
                    tracing::warn!(
                        "primary {} unreachable: {e}; reading the cluster again",
                        primary.url
                    );
                    let _ = self.refresh().await;
                    if self.primary(&shard).is_some_and(|p| p.node == primary.node) {
                        // Not replaced yet: give the failover a moment before the retry.
                        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                        let _ = self.refresh().await;
                    }
                }
                Err(e) => return self.unreachable(&primary, e).await,
            }
        }
        Self::unavailable(&shard)
    }

    /// The shard with the fewest tables among those with a ready primary.
    fn least_loaded(placement: &Placement, shards: &BTreeMap<String, ShardView>) -> Option<String> {
        let live: Vec<&str> = shards
            .iter()
            .filter(|(_, v)| v.primary.is_some())
            .map(|(k, _)| k.as_str())
            .collect();
        placement
            .load_by_shard(live)
            .into_iter()
            .min_by_key(|(shard, n)| (*n, shard.clone()))
            .map(|(shard, _)| shard)
    }

    /// Put a table the router has not created -- the knowledge-base table -- on a shard.
    async fn place(&self, table: &str) -> crate::Result<String> {
        let shards = self.snapshot_view().shards;
        let table = table.to_string();
        placement::update(&self.cluster, |p| {
            if let Some(placed) = p.tables.get(&table) {
                return Ok(placed.shard.clone());
            }
            let shard =
                Self::least_loaded(p, &shards).ok_or_else(|| crate::Error::ServiceUnavailable {
                    message: "no shard has a ready primary".to_string(),
                })?;
            p.tables.insert(
                table.clone(),
                placement::Placed {
                    shard: shard.clone(),
                    state: PlacementState::Ready,
                    since: super::now(),
                },
            );
            Ok(shard)
        })
        .await
    }

    async fn create_table(&self, headers: &HeaderMap, body: Bytes) -> Response {
        let Some(name) = serde_json::from_slice::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| v.get("name").and_then(|n| n.as_str()).map(str::to_string))
        else {
            return envelope(
                StatusCode::BAD_REQUEST,
                "INVALID_INPUT",
                "a table needs a `name`",
            );
        };
        let shards = self.snapshot_view().shards;
        let claimed = placement::update(&self.cluster, |p| {
            if let Some(placed) = p.tables.get(&name) {
                let abandoned = placed.state == PlacementState::Creating
                    && super::now() - placed.since > placement::ABANDONED_AFTER;
                if !abandoned {
                    return Err(crate::Error::TableAlreadyExists(name.clone()));
                }
            }
            let shard =
                Self::least_loaded(p, &shards).ok_or_else(|| crate::Error::ServiceUnavailable {
                    message: "no shard has a ready primary".to_string(),
                })?;
            p.tables.insert(
                name.clone(),
                placement::Placed {
                    shard: shard.clone(),
                    state: PlacementState::Creating,
                    since: super::now(),
                },
            );
            Ok(shard)
        })
        .await;
        let shard = match claimed {
            Ok(shard) => shard,
            Err(crate::Error::TableAlreadyExists(_)) => {
                return envelope(
                    StatusCode::CONFLICT,
                    "TABLE_ALREADY_EXISTS",
                    format!("Table already exists: {name}"),
                )
            }
            Err(e) => return envelope(StatusCode::SERVICE_UNAVAILABLE, "NO_SHARDS", e.to_string()),
        };
        let Some(primary) = self.primary(&shard) else {
            self.release(&name, &shard).await;
            return Self::unavailable(&shard);
        };
        let response = self
            .forward(&primary, &Method::POST, "/api/v1/tables", headers, body)
            .await;
        if response.status().is_success() {
            let name2 = name.clone();
            let shard2 = shard.clone();
            let marked = placement::update(&self.cluster, |p| {
                if let Some(placed) = p.tables.get_mut(&name2) {
                    if placed.shard == shard2 {
                        placed.state = PlacementState::Ready;
                        placed.since = super::now();
                    }
                }
                Ok(())
            })
            .await;
            if let Err(e) = marked {
                // The table exists and is placed; only its state says `creating`, which every
                // route treats as placed. Logged, not returned: the caller's table is there.
                tracing::warn!("table '{name}' created on '{shard}' but not marked ready: {e}");
            }
            let _ = self.refresh().await;
        } else {
            self.release(&name, &shard).await;
        }
        response
    }

    /// Give a claimed name back after the shard refused to create it.
    async fn release(&self, name: &str, shard: &str) {
        let (name, shard) = (name.to_string(), shard.to_string());
        let _ = placement::update(&self.cluster, |p| {
            if p.tables.get(&name).is_some_and(|placed| {
                placed.shard == shard && placed.state == PlacementState::Creating
            }) {
                p.tables.remove(&name);
            }
            Ok(())
        })
        .await;
    }

    async fn drop_table(&self, table: &str, path_and_query: &str, headers: &HeaderMap) -> Response {
        let Some(shard) = self.shard_of(table).await else {
            return envelope(
                StatusCode::NOT_FOUND,
                "TABLE_NOT_FOUND",
                format!("Table not found: {table}"),
            );
        };
        let Some(primary) = self.primary(&shard) else {
            return Self::unavailable(&shard);
        };
        let response = self
            .forward(
                &primary,
                &Method::DELETE,
                path_and_query,
                headers,
                Bytes::new(),
            )
            .await;
        // Gone from the shard, or never there: either way it is not placed any more.
        if response.status().is_success() || response.status() == StatusCode::NOT_FOUND {
            let (table, shard) = (table.to_string(), shard.clone());
            if let Err(e) = placement::update(&self.cluster, |p| {
                if p.tables
                    .get(&table)
                    .is_some_and(|placed| placed.shard == shard)
                {
                    p.tables.remove(&table);
                }
                Ok(())
            })
            .await
            {
                tracing::warn!("table dropped but still placed: {e}");
            }
            let _ = self.refresh().await;
        }
        response
    }

    /// Ask every shard's primary the same thing; `(shard, status, body)` per shard, and the
    /// shards that have none.
    async fn ask_every_shard(
        &self,
        method: &Method,
        path: &str,
        headers: &HeaderMap,
    ) -> (Vec<(String, u16, serde_json::Value)>, Vec<String>) {
        let view = self.snapshot_view();
        let mut answers = Vec::new();
        let mut unavailable = Vec::new();
        for (shard, members) in &view.shards {
            let Some(primary) = &members.primary else {
                unavailable.push(shard.clone());
                continue;
            };
            let response = self
                .forward(primary, method, path, headers, Bytes::new())
                .await;
            let status = response.status().as_u16();
            let bytes = hyper::body::to_bytes(response.into_body())
                .await
                .unwrap_or_default();
            let body = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
            answers.push((shard.clone(), status, body));
        }
        (answers, unavailable)
    }

    async fn list_tables(&self, query: Option<&str>, headers: &HeaderMap) -> Response {
        let param = |name: &str| {
            query.and_then(|q| {
                q.split('&')
                    .filter_map(|kv| kv.split_once('='))
                    .find(|(k, _)| *k == name)
                    .and_then(|(_, v)| v.parse::<usize>().ok())
            })
        };
        let (offset, limit) = (param("offset").unwrap_or(0), param("limit").unwrap_or(50));
        let (answers, unavailable) = self
            .ask_every_shard(&Method::GET, "/api/v1/tables?limit=1000000", headers)
            .await;
        let mut tables: Vec<serde_json::Value> = Vec::new();
        for (shard, status, body) in answers {
            if status != 200 {
                return envelope(
                    StatusCode::BAD_GATEWAY,
                    "SHARD_ERROR",
                    format!("shard '{shard}' answered {status} when asked for its tables"),
                );
            }
            if let Some(list) = body["data"]["tables"].as_array() {
                tables.extend(list.iter().cloned());
            }
        }
        tables.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
        let total = tables.len();
        let page: Vec<_> = tables.into_iter().skip(offset).take(limit).collect();
        let mut data = serde_json::json!({
            "tables": page, "total": total, "offset": offset, "limit": limit,
        });
        if !unavailable.is_empty() {
            data["unavailable_shards"] = serde_json::json!(unavailable);
        }
        success(data)
    }

    fn status(&self) -> serde_json::Value {
        let view = self.snapshot_view();
        let mut tables_by_shard: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for (table, placed) in &view.placement.tables {
            tables_by_shard
                .entry(placed.shard.as_str())
                .or_default()
                .push(table.as_str());
        }
        let shards: Vec<serde_json::Value> = view
            .shards
            .iter()
            .map(|(id, v)| {
                serde_json::json!({
                    "shard": id,
                    "primary": v.primary.as_ref().map(|p| serde_json::json!({
                        "node": p.node, "url": p.url, "epoch": p.epoch,
                    })),
                    "replicas": v.replicas.iter().map(|r| serde_json::json!({
                        "node": r.node, "url": r.url,
                    })).collect::<Vec<_>>(),
                    "tables": tables_by_shard.get(id.as_str()).map(Vec::len).unwrap_or(0),
                })
            })
            .collect();
        serde_json::json!({
            "loaded": view.loaded,
            "tables": view.placement.tables.len(),
            "shards": shards,
            "unavailable_shards": view.shards.iter()
                .filter(|(_, v)| v.primary.is_none()).map(|(k, _)| k).collect::<Vec<_>>(),
        })
    }
}

async fn dispatch(State(router): State<Arc<ClusterRouter>>, request: Request<Body>) -> Response {
    let (parts, body) = request.into_parts();
    let method = parts.method.clone();
    let path = parts.uri.path().to_string();
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|p| p.as_str().to_string())
        .unwrap_or_else(|| path.clone());
    let declared_length = parts
        .headers
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok());
    let headers = parts.headers;
    if declared_length.is_some_and(|len| len > MAX_BODY) {
        return envelope(
            StatusCode::PAYLOAD_TOO_LARGE,
            "INVALID_INPUT",
            "request body too large",
        );
    }
    let body = match hyper::body::to_bytes(body).await {
        Ok(b) if b.len() <= MAX_BODY => b,
        Ok(_) => {
            return envelope(
                StatusCode::PAYLOAD_TOO_LARGE,
                "INVALID_INPUT",
                "request body too large",
            )
        }
        Err(e) => return envelope(StatusCode::BAD_REQUEST, "INVALID_INPUT", e.to_string()),
    };

    let trimmed = path.trim_end_matches('/');
    match (&method, trimmed) {
        (&Method::GET, "/live") => success(serde_json::json!({ "status": "live" })),
        (&Method::GET, "/ready") => {
            if router.view.read().loaded {
                success(serde_json::json!({ "status": "ready" }))
            } else {
                envelope(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "NOT_READY",
                    "no view of the cluster yet",
                )
            }
        }
        (&Method::GET, "/health") | (&Method::GET, "/api/v1/health") => {
            success(serde_json::json!({ "role": "router", "cluster": router.status() }))
        }
        (&Method::GET, "/cluster/status") | (&Method::GET, "/api/v1/cluster/status") => {
            success(router.status())
        }
        (&Method::GET, "/api/v1/tables") => router.list_tables(parts.uri.query(), &headers).await,
        (&Method::POST, "/api/v1/tables") => router.create_table(&headers, body).await,
        (&Method::POST, "/api/v1/snapshot") => {
            let (answers, unavailable) = router
                .ask_every_shard(&Method::POST, "/api/v1/snapshot", &headers)
                .await;
            let shards: serde_json::Map<String, serde_json::Value> = answers
                .into_iter()
                .map(|(s, status, body)| (s, serde_json::json!({ "status": status, "body": body })))
                .collect();
            success(serde_json::json!({ "shards": shards, "unavailable_shards": unavailable }))
        }
        _ => {
            if let Some(rest) = trimmed.strip_prefix("/api/v1/tables/") {
                let table = rest.split('/').next().unwrap_or_default();
                if table == "import" {
                    return envelope(
                        StatusCode::NOT_IMPLEMENTED,
                        "NOT_AVAILABLE_THROUGH_ROUTER",
                        "importing a table through a router is not supported yet",
                    );
                }
                let table = percent_decode(table);
                if method == Method::DELETE && !rest.contains('/') {
                    return router.drop_table(&table, &path_and_query, &headers).await;
                }
                return router
                    .table_request(&table, &method, &path_and_query, &headers, body)
                    .await;
            }
            envelope(
                StatusCode::NOT_IMPLEMENTED,
                "NOT_AVAILABLE_THROUGH_ROUTER",
                format!(
                    "{method} {path} spans the cluster in a way one shard cannot answer (aliases, \
                     DXQL, backups, metrics); send it to a node directly"
                ),
            )
        }
    }
}

fn percent_decode(segment: &str) -> String {
    let bytes = segment.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Some(b) = segment
                .get(i + 1..i + 3)
                .and_then(|hex| u8::from_str_radix(hex, 16).ok())
            {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| segment.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percent_decoding_a_table_segment() {
        assert_eq!(percent_decode("my%2Dtable"), "my-table");
        assert_eq!(percent_decode("plain"), "plain");
        assert_eq!(percent_decode("bad%zz"), "bad%zz");
    }

    #[test]
    fn the_least_loaded_shard_with_a_primary() {
        let rec = |s: &str| NodeRecord {
            node: s.into(),
            shard: s.into(),
            role: nodes::Role::Primary,
            url: String::new(),
            epoch: 1,
            ready: true,
            heartbeat_at: chrono::Utc::now(),
        };
        let mut shards = BTreeMap::new();
        shards.insert(
            "a".to_string(),
            ShardView {
                primary: Some(rec("a")),
                replicas: vec![],
            },
        );
        shards.insert(
            "b".to_string(),
            ShardView {
                primary: Some(rec("b")),
                replicas: vec![],
            },
        );
        shards.insert("c".to_string(), ShardView::default());
        let mut p = Placement::default();
        p.tables.insert(
            "x".into(),
            placement::Placed {
                shard: "a".into(),
                state: PlacementState::Ready,
                since: chrono::Utc::now(),
            },
        );
        assert_eq!(
            ClusterRouter::least_loaded(&p, &shards).as_deref(),
            Some("b")
        );
    }
}
