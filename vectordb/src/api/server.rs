use axum::{
    extract::DefaultBodyLimit,
    routing::{delete, get, post, put},
    Router,
};

use std::sync::Arc;
use tower_http::{cors::CorsLayer, trace::TraceLayer};

use crate::{
    api::handlers::{
        alter_table, branch_table, calibrate_recall, count_records, create_backup, create_snapshot,
        create_table, delete_alias, delete_records, delete_table, drop_index,
        drop_payload_index_field, execute_dxql, export_table, facets, full_text_search, get_record,
        get_status, get_table, get_table_stats, health_check, health_metrics, hybrid_search,
        hybrid_search_combined, import_table, index_payload_field, insert_records,
        insert_records_optimized, knn_graph, list_aliases, list_records, list_tables, liveness,
        payload_index_report, prometheus_metrics, readiness, rebuild_index, repair_reachability,
        search_groups, search_vectors, set_alias, sparse_search, update_record, AppState,
    },
    config::AppConfig,
    error::Result,
    storage::IntelligentIndexedStorage,
};

#[cfg(feature = "api-dashboard")]
use crate::api::dashboard;
#[cfg(feature = "api-graphql")]
use crate::api::graphql;
#[cfg(feature = "api-swagger")]
use crate::api::swagger;
#[cfg(feature = "api-ws")]
use crate::api::websocket;

/// WAL 目录：`DXMT_WAL_DIR` 覆盖，默认 `$DXMT_ROOT_DIR/store`（`DXMT_ROOT_DIR` 默认 `~/.dxmt`），
/// 显式设为空串关闭持久化。
///
/// 层次由 [`crate::config::layout::Layout`] 决定，不在这里各算各的——五处路径配置互不相干、
/// 其中三处相对工作目录，是这个函数以前的样子。
///
/// 只在服务端入口读环境变量，不放进 `IntelligentIndexedStorage::new`——那个构造函数有
/// 四十多个调用点（单测、bench、example），让它们各自往磁盘写日志会互相回放污染。
pub fn wal_dir_from_env() -> Option<std::path::PathBuf> {
    if let Ok(dir) = crate::config::env::var("DXMT_WAL_DIR") {
        if dir.trim().is_empty() {
            tracing::warn!("DXMT_WAL_DIR is empty: WAL disabled, data will not survive a restart");
            return None;
        }
        tracing::info!("data directory: {dir} (DXMT_WAL_DIR)");
        return Some(std::path::PathBuf::from(dir));
    }

    let layout = crate::config::layout::Layout::from_env();
    let store = layout.store();
    if let Some(note) = &store.note {
        tracing::warn!("{note}");
    }
    // Which database this process is about to open, said once and plainly. The point of taking
    // the default off the working directory is that the answer stops depending on where you
    // were standing; it is worth being able to read it.
    tracing::info!("data directory: {}", store.path.display());
    // Creating it, and finding out whether it can be written to at all, is `Layout::prepare`
    // at startup -- before this, and before anything is served.
    Some(store.path)
}

/// Which optional interfaces to serve.
///
/// A struct rather than a run of booleans. There are four of them now, all the same type, and
/// `create_app_with_storage(swagger, graphql, websocket, dashboard)` gives the compiler nothing
/// to catch a transposed pair with -- the symptom would be a server that serves the wrong two
/// of four, which no test asserts against because every test passes the same values.
///
/// Compiling an interface in and serving it are separate decisions throughout: the cargo
/// feature decides whether the code exists, and these decide whether it answers. Asking for one
/// the binary was not built with is an error rather than a silent no-op.
#[derive(Clone, Copy, Debug, Default)]
pub struct Interfaces {
    pub swagger: bool,
    pub graphql: bool,
    pub websocket: bool,
    pub dashboard: bool,
    /// Whether a missing dashboard bundle is an error rather than a warning.
    ///
    /// The console is served by default, and a binary built without running the bundle build
    /// carries the routes and none of the bytes. Refusing to start there would mean a checkout
    /// without Node cannot run the database at all — but someone who passed `--dashboard` by
    /// name is asking for the console specifically, and should be told it is absent rather
    /// than handed a server that quietly has no console.
    pub dashboard_required: bool,
}

/// 创建应用程序路由器
pub async fn create_app() -> Result<Router> {
    create_app_with_features(Interfaces::default(), None).await
}

/// Warn when the runtime cannot get round to a one-second timer.
///
/// This task does nothing but wake up every second and check how late it was. Lateness is not a
/// property of this task — it is the scheduling delay every other task on the runtime is also
/// paying, requests included, so it separates two failure modes that look identical from
/// outside: a request hanging because something holds a lock, and a request hanging because no
/// worker thread is free to poll it. Only the second one shows up here.
///
/// Cheap enough to leave on: one wakeup a second, one comparison, and it logs nothing at all
/// unless the runtime is already in trouble.
fn spawn_stall_detector(threshold: std::time::Duration) {
    if threshold.is_zero() {
        return;
    }
    tokio::spawn(async move {
        let period = std::time::Duration::from_secs(1);
        loop {
            // `sleep` rather than `interval`: an interval has missed-tick semantics — it decides
            // for itself what to do about deadlines that passed while nothing polled it — and
            // that is precisely the case being measured here. A version built on `interval` did
            // not fire at all against a 20-second SIGSTOP, and its silence then got mistaken for
            // evidence that the runtime was healthy. A sleep is unambiguous: it asked for one
            // second, so anything past that is scheduling delay.
            let expected = std::time::Instant::now() + period;
            tokio::time::sleep(period).await;
            let late = std::time::Instant::now().saturating_duration_since(expected);
            if late >= threshold {
                tracing::warn!(
                    "Runtime stalled: a 1s timer fired {:.1}s late. Every task on this runtime \
                     waited that long, so requests did too. Something is holding a worker thread \
                     without yielding — index builds and searches are the usual candidates.",
                    late.as_secs_f64()
                );
            }
        }
    });
}

/// How late a one-second tick has to be before it is reported, in seconds; 0 disables. Default 3.
fn stall_threshold_from_env() -> std::time::Duration {
    let secs = crate::config::env::var("DXMT_STALL_WARN_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    std::time::Duration::from_secs(secs)
}

/// Snapshot interval in seconds; 0 disables. Default 300.
///
/// Without a periodic snapshot the WAL only shrinks when someone calls POST /snapshot, so
/// recovery time grows without bound on a long-lived server.
fn snapshot_interval_from_env() -> u64 {
    crate::config::env::var("DXMT_SNAPSHOT_INTERVAL_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300)
}

/// Eviction sweep interval in seconds; 0 disables. Default 600.
fn eviction_interval_from_env() -> u64 {
    crate::config::env::var("DXMT_EVICTION_INTERVAL_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(600)
}

/// Drop cold tables from memory on a timer, leaving them in object storage.
///
/// This is what bounds memory to the working set rather than the whole dataset. A query on an
/// evicted table rehydrates it transparently, so the only cost of being wrong is a slow first
/// query.
fn spawn_eviction_sweep(storage: Arc<IntelligentIndexedStorage>, interval: u64) {
    if interval == 0 {
        tracing::info!("Cold-table eviction disabled (DXMT_EVICTION_INTERVAL_SECS=0)");
        return;
    }
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval));
        ticker.tick().await;
        loop {
            ticker.tick().await;
            match storage.evict_cold_tables().await {
                Ok(evicted) if !evicted.is_empty() => {
                    tracing::info!("Evicted {} cold table(s)", evicted.len())
                }
                Ok(_) => {}
                // Eviction snapshots first, so a failure here means the snapshot failed and
                // nothing was dropped. Data is intact either way.
                Err(e) => tracing::warn!("Eviction sweep failed: {}", e),
            }
        }
    });
    tracing::info!("Cold-table eviction sweep every {}s", interval);
}

/// Row-file reclamation interval in seconds; 0 disables. Default 600.
fn row_compaction_interval_from_env() -> u64 {
    crate::config::env::var("DXMT_ROW_COMPACTION_INTERVAL_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(600)
}

/// Give back the disk behind row-file rows that updates and deletes left behind.
///
/// Without this a table's files grow with the number of writes rather than with the amount of
/// data: an update appends a new row and the old one keeps its blocks forever. A soak at 200
/// rows/s over a fixed 400,000-row key space grew the vector file to 5.2 GB holding 1.2 GB of
/// live rows in 2.4 hours, and would have filled the disk long before it ran out of anything
/// else.
fn spawn_row_reclamation(storage: Arc<IntelligentIndexedStorage>, interval: u64) {
    if interval == 0 {
        tracing::info!("Row-file reclamation disabled (DXMT_ROW_COMPACTION_INTERVAL_SECS=0)");
        return;
    }
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval));
        ticker.tick().await;
        loop {
            ticker.tick().await;
            match storage.reclaim_dead_rows().await {
                Ok(reclaimed) if !reclaimed.is_empty() => {
                    let total: u64 = reclaimed.iter().map(|(_, bytes)| bytes).sum();
                    tracing::info!(
                        "Reclaimed {:.1} MB from {} table(s) of dead row-file space",
                        total as f64 / 1e6,
                        reclaimed.len()
                    )
                }
                Ok(_) => {}
                // Nothing is lost by a failed reclamation: the rows were already dead and the
                // live ones were never touched.
                Err(e) => tracing::warn!("Row-file reclamation failed: {}", e),
            }
        }
    });
    tracing::info!("Row-file reclamation every {}s", interval);
}

/// Index re-evaluation interval in seconds; 0 disables. Default 900.
fn reindex_interval_from_env() -> u64 {
    crate::config::env::var("DXMT_REINDEX_INTERVAL_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(900)
}

/// Re-check periodically whether each table's index still suits its size.
///
/// Without this the algorithm is whatever CREATE TABLE picked from a fixed estimate, for the
/// life of the table — an "adaptive index" that adapts once, before there is any data to adapt
/// to.
fn spawn_index_reevaluation(storage: Arc<IntelligentIndexedStorage>, interval: u64) {
    if interval == 0 {
        tracing::info!("Automatic index re-evaluation disabled (DXMT_REINDEX_INTERVAL_SECS=0)");
        return;
    }
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval));
        ticker.tick().await;
        loop {
            ticker.tick().await;
            match storage.reevaluate_indexes().await {
                Ok(rebuilt) if !rebuilt.is_empty() => {
                    tracing::info!(
                        "Rebuilt {} index(es) after re-evaluation: {:?}",
                        rebuilt.len(),
                        rebuilt
                    )
                }
                Ok(_) => {}
                Err(e) => tracing::warn!("Index re-evaluation failed: {}", e),
            }
        }
    });
    tracing::info!("Index re-evaluation every {}s", interval);
}

/// Snapshot every `interval` seconds for as long as the process runs.
fn spawn_periodic_snapshots(storage: Arc<IntelligentIndexedStorage>, interval: u64) {
    if interval == 0 {
        tracing::info!("Periodic snapshots disabled (DXMT_SNAPSHOT_INTERVAL_SECS=0)");
        return;
    }
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval));
        // The first tick fires immediately; skip it so startup does not snapshot what it has
        // just finished restoring.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            match storage.snapshot().await {
                Ok(tables) => tracing::debug!("Periodic snapshot: {} table(s)", tables),
                // Never fatal: the WAL still holds everything a failed snapshot would have
                // covered, so this costs startup time on the next boot and nothing else.
                Err(e) => tracing::warn!("Periodic snapshot failed: {}", e),
            }
        }
    });
    tracing::info!("Periodic snapshots every {}s", interval);
}

/// 创建带有功能开关的应用程序路由器
pub async fn create_app_with_features(
    interfaces: Interfaces,
    api_key: Option<crate::api::auth::ApiKey>,
) -> Result<Router> {
    create_app_with_storage(interfaces, api_key)
        .await
        .map(|(router, _)| router)
}

/// As [`create_app_with_features`], but also hands back the storage.
///
/// `main` needs it to snapshot on shutdown. Returning only a `Router` left no way to reach it,
/// which is why a clean SIGTERM used to leave the next start replaying the entire log.
/// The key is a parameter rather than something this function reads for itself. Every test
/// that builds a router would otherwise inherit whatever the developer happens to have in
/// their shell, and a `DXMT_API_KEY` left over from an afternoon of manual testing would turn
/// every one of them red for a reason none of them mention.
pub async fn create_app_with_storage(
    interfaces: Interfaces,
    api_key: Option<crate::api::auth::ApiKey>,
) -> Result<(Router, Arc<IntelligentIndexedStorage>)> {
    let Interfaces {
        swagger: enable_swagger,
        graphql: enable_graphql,
        websocket: enable_websocket,
        dashboard: enable_dashboard,
        dashboard_required: _,
    } = interfaces;

    #[cfg(not(feature = "api-swagger"))]
    if enable_swagger {
        return Err(crate::Error::InvalidConfig(
            "Swagger was requested, but this binary was built without `api-swagger`".to_string(),
        ));
    }
    #[cfg(not(feature = "api-graphql"))]
    if enable_graphql {
        return Err(crate::Error::InvalidConfig(
            "GraphQL was requested, but this binary was built without `api-graphql`".to_string(),
        ));
    }
    #[cfg(not(feature = "api-ws"))]
    if enable_websocket {
        return Err(crate::Error::InvalidConfig(
            "WebSocket was requested, but this binary was built without `api-ws`".to_string(),
        ));
    }
    #[cfg(not(feature = "api-dashboard"))]
    if enable_dashboard {
        return Err(crate::Error::InvalidConfig(
            "The dashboard was requested, but this binary was built without `api-dashboard`"
                .to_string(),
        ));
    }
    // Compiled in is not the same as present. `#[allow_missing]` lets `dashboard/dist/` be
    // absent so that a clean checkout -- and CI's feature-compilation job -- can build this
    // without Node, which means a binary can carry the routes and none of the bytes. Serving
    // that is a white page with no explanation, so refuse instead and name the command.
    #[cfg(feature = "api-dashboard")]
    let enable_dashboard = if enable_dashboard && !dashboard::assets_are_embedded() {
        if interfaces.dashboard_required {
            return Err(crate::Error::InvalidConfig(
                "The dashboard was requested, but no assets are embedded in this binary. Run \
                 `make dashboard-build` and rebuild."
                    .to_string(),
            ));
        }
        tracing::warn!(
            "No web console in this binary: `dashboard/dist/` was absent when it was built. \
             Run `make dashboard-build` and rebuild to get /dashboard; everything else is \
             unaffected."
        );
        false
    } else {
        enable_dashboard
    };

    // Everything this server writes, created and proved writable before anything is served.
    //
    // Failing here rather than on the first write is the whole point: an unwritable `backups/`
    // would otherwise be discovered by whoever takes the first backup, which may be weeks
    // after the server came up and is the worst moment to learn it.
    //
    // Skipped when the WAL is explicitly off -- `DXMT_WAL_DIR=""` says this instance keeps
    // nothing, and demanding a writable directory for it would be asking for something it
    // will never use.
    if crate::config::env::var("DXMT_WAL_DIR")
        .map(|dir| !dir.trim().is_empty())
        .unwrap_or(true)
    {
        crate::config::layout::Layout::from_env()
            .prepare()
            .map_err(crate::Error::InvalidConfig)?;
    }

    // 加载配置
    let config = AppConfig::load()?;

    // 创建优化的存储实例。带 WAL：进程重启后要能恢复。
    // DXMT_WAL_DIR 可覆盖目录；设成空串则彻底关掉持久化（只建议用于压测对比）。
    let storage = Arc::new(
        IntelligentIndexedStorage::new_with_wal(&config.storage, wal_dir_from_env()).await?,
    );

    // 应用状态
    // Initialize server health monitor
    let server_health = Arc::new(crate::api::server_health::ServerHealth::default());

    let base_state = AppState {
        storage: storage.clone(),
        server_health,
    };

    // 创建API v1路由
    let api_v1_routes = Router::new()
        // 表管理API
        .route("/tables", post(create_table).get(list_tables))
        .route(
            "/tables/:table_name",
            get(get_table).delete(delete_table).patch(alter_table),
        )
        // 记录操作API
        .route("/tables/:table_name/records", post(insert_records))
        .route(
            "/tables/:table_name/records/:record_id",
            get(get_record).put(update_record),
        )
        .route("/tables/:table_name/records", delete(delete_records))
        .route("/tables/:table_name/records/list", post(list_records))
        .route("/tables/:table_name/records/count", post(count_records))
        .route("/tables/:table_name/facets", post(facets))
        // The nearest-neighbour structure of a sample, without its vectors — so a client can
        // lay a table out without the wire deciding how many points it may draw.
        .route("/tables/:table_name/graph", post(knn_graph))
        .route("/tables/:table_name/stats", get(get_table_stats))
        // 批量操作API
        .route(
            "/tables/:table_name/records/bulk",
            post(insert_records_optimized),
        )
        // 搜索API
        .route("/tables/:table_name/search", post(search_vectors))
        .route("/tables/:table_name/search/groups", post(search_groups))
        // Sparse retrieval on its own, in the default build: the sparse index is pure Rust and
        // needs no full-text engine behind it.
        .route("/tables/:table_name/search/sparse", post(sparse_search))
        .route("/tables/:table_name/hybrid_search", post(hybrid_search))
        // An alias for /search, on purpose and pinned by a test: both claim to answer "what
        // is nearest to this vector" and an alias that quietly starts answering differently is
        // drift no unit test sees. It does not batch anything.
        .route("/tables/:table_name/batch_search", post(search_vectors))
        // 索引管理API
        .route("/tables/:table_name/index", delete(drop_index))
        // The payload (filter) index, which is a different index from the vector one above.
        .route(
            "/tables/:table_name/payload_index",
            get(payload_index_report),
        )
        .route(
            "/tables/:table_name/payload_index/:field",
            put(index_payload_field).delete(drop_payload_index_field),
        )
        .route("/tables/:table_name/index/rebuild", post(rebuild_index))
        .route(
            "/tables/:table_name/recall/calibrate",
            post(calibrate_recall),
        )
        .route(
            "/tables/:table_name/index/repair",
            post(repair_reachability),
        )
        // 快照：把所有表落盘，下次启动只回放快照之后的 WAL
        .route("/snapshot", post(create_snapshot))
        // 表分支API
        .route("/tables/:table_name/branch", post(branch_table))
        // DXQL query API
        .route("/query", post(execute_dxql))
        // 全文搜索API
        .route("/tables/:table_name/search/text", post(full_text_search))
        .route(
            "/tables/:table_name/search/hybrid",
            post(hybrid_search_combined),
        )
        // 备份：写一份自洽的备份目录。恢复是离线操作，见 docs/operations/backup-restore.md
        .route("/backup", post(create_backup))
        // One table, rather than the whole instance: written to a file, read back on this
        // server or another one.
        .route("/tables/:table_name/export", post(export_table))
        .route("/tables/import", post(import_table))
        // Aliases: another name for a table, and the way to swap one for another without a
        // client-side deploy.
        .route("/aliases", get(list_aliases))
        .route("/aliases/:alias", put(set_alias).delete(delete_alias))
        // Keep system probes available under the SDK base URL as well as at the root.
        .route("/health", get(health_check))
        .route("/health/live", get(liveness))
        .route("/health/ready", get(readiness))
        .route("/health/metrics", get(health_metrics))
        // Prometheus scrapes `/metrics` at the root, which is where Qdrant, Weaviate and
        // everything else that speaks this format put it; a scrape config should not need a
        // per-database path.
        .route("/metrics", get(prometheus_metrics))
        .with_state(base_state.clone());

    // 创建系统路由（在根路径）
    let system_routes = Router::new()
        .route("/status", get(get_status))
        .route("/health", get(health_check))
        .route("/live", get(liveness))
        .route("/ready", get(readiness))
        .route("/health/live", get(liveness))
        .route("/health/ready", get(readiness))
        .route("/health/metrics", get(health_metrics))
        .route("/metrics", get(prometheus_metrics))
        .with_state(base_state.clone());

    // 合并路由
    #[allow(unused_mut)]
    let mut app = Router::new()
        .merge(system_routes)
        .nest("/api/v1", api_v1_routes);

    // 条件性添加 GraphQL 路由
    #[cfg(feature = "api-graphql")]
    if enable_graphql {
        let graphql_routes = graphql::create_graphql_router();
        app = app.merge(graphql_routes);
    }

    // 条件性添加 WebSocket 路由
    #[cfg(feature = "api-ws")]
    if enable_websocket {
        let websocket_routes = websocket::create_websocket_router();
        app = app.merge(websocket_routes);
    }

    match &api_key {
        Some(key) => tracing::info!(
            "API key required ({}); /health, /live and /ready stay open for probes",
            key.describe()
        ),
        None => tracing::info!(
            "no API key configured: every endpoint is open. Set {} to require one.",
            crate::api::auth::KEY_VARIABLE
        ),
    }

    let app = app
        // 增加请求体大小限制到500MB
        .layer(DefaultBodyLimit::max(500 * 1024 * 1024))
        // Authentication, and the position is the whole of it. `layer` wraps inward-out, so
        // the last one applied is the outermost: written after `CorsLayer` this ran *before*
        // it, and a browser's `OPTIONS` preflight got a bare 401 with no CORS headers on it.
        // The real request then never leaves the browser, and the failure looks like a CORS
        // misconfiguration rather than a missing key. Manual verification found that; the
        // unit and integration tests, which use curl and send no preflight, did not.
        //
        // Inside the body limit too, so an unauthenticated client cannot make us buffer 500 MB
        // before we tell it no.
        .layer(axum::middleware::from_fn_with_state(
            api_key,
            crate::api::auth::require_key,
        ))
        // 添加安全中间件
        .layer(
            CorsLayer::new()
                .allow_origin(
                    "http://localhost:3000"
                        .parse::<axum::http::HeaderValue>()
                        .map_err(|_| crate::Error::Internal("Operation failed".to_string()))?,
                )
                .allow_origin(
                    "http://127.0.0.1:3000"
                        .parse::<axum::http::HeaderValue>()
                        .map_err(|_| crate::Error::Internal("Operation failed".to_string()))?,
                )
                .allow_methods([
                    axum::http::Method::GET,
                    axum::http::Method::POST,
                    axum::http::Method::PUT,
                    axum::http::Method::DELETE,
                ])
                .allow_headers([
                    axum::http::header::CONTENT_TYPE,
                    axum::http::header::AUTHORIZATION,
                    axum::http::HeaderName::from_static(crate::api::auth::HEADER),
                ])
                .max_age(std::time::Duration::from_secs(3600)),
        )
        .layer(TraceLayer::new_for_http());

    // Outside the layer stack, and not for the usual "it costs nothing" reason.
    //
    // The first attempt merged this before the layers, so the bundle would get `TraceLayer` and
    // compression for free. It got compression, and lost its `Vary: accept-encoding` -- the
    // `CorsLayer` replaces that header wholesale with its own three values, which a test
    // caught. `Vary` is not decoration next to `immutable`: without it a shared cache can hand
    // a gzip-encoded body to a client that never asked for one.
    //
    // So the dashboard sits outside and carries its own compression and tracing. Nothing is
    // lost, and the response headers are entirely its own.
    #[cfg(feature = "api-dashboard")]
    let app = if enable_dashboard {
        app.merge(dashboard::create_dashboard_router())
    } else {
        app
    };
    #[cfg(not(feature = "api-dashboard"))]
    let app = app;

    #[cfg(feature = "api-swagger")]
    let app = if enable_swagger {
        app.merge(swagger::create_swagger_ui())
    } else {
        app
    };
    #[cfg(not(feature = "api-swagger"))]
    let app = app;

    let app = app.with_state(base_state);

    spawn_periodic_snapshots(storage.clone(), snapshot_interval_from_env());
    spawn_eviction_sweep(storage.clone(), eviction_interval_from_env());
    spawn_index_reevaluation(storage.clone(), reindex_interval_from_env());
    IntelligentIndexedStorage::spawn_derived_warmup(storage.clone());
    spawn_row_reclamation(storage.clone(), row_compaction_interval_from_env());
    spawn_stall_detector(stall_threshold_from_env());

    Ok((app, storage))
}

// integration tests live in tests/
