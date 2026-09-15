use axum::{
    extract::DefaultBodyLimit,
    routing::{delete, get, post},
    Router,
};

use std::sync::Arc;
use tower_http::{cors::CorsLayer, trace::TraceLayer};

use crate::{
    api::{
        // batch_handlers::batch_search_vectors,  // Removed
        // bulk_handlers::bulk_insert_records,  // Removed
        handlers::{
            alter_table, branch_table, create_backup, create_snapshot, create_table,
            delete_records, delete_table, drop_index, execute_dxql, full_text_search, get_record,
            get_status, get_table, get_table_stats, health_check, health_metrics, hybrid_search,
            hybrid_search_combined, insert_records, insert_records_optimized, list_records,
            list_tables, liveness, readiness, rebuild_index, search_vectors, update_record,
            AppState,
        },
    },
    config::AppConfig,
    error::Result,
    storage::IntelligentIndexedStorage,
};

#[cfg(feature = "api-graphql")]
use crate::api::graphql;
#[cfg(feature = "api-swagger")]
use crate::api::swagger;
#[cfg(feature = "api-ws")]
use crate::api::websocket;
#[cfg(feature = "multimodal")]
use crate::{
    api::multimodal::{
        create_multimodal_record, cross_modal_search, generate_embedding,
        validate_multimodal_content, MultimodalAppState,
    },
    embeddings::EmbeddingManager,
};

/// WAL 目录：`DB_WAL_DIR` 覆盖，默认 `./data/wal`，显式设为空串关闭持久化。
///
/// 只在服务端入口读环境变量，不放进 `IntelligentIndexedStorage::new`——那个构造函数有
/// 四十多个调用点（单测、bench、example），让它们各自往磁盘写日志会互相回放污染。
pub(crate) fn wal_dir_from_env() -> Option<std::path::PathBuf> {
    match std::env::var("DB_WAL_DIR") {
        Ok(dir) if dir.trim().is_empty() => {
            tracing::warn!("DB_WAL_DIR is empty: WAL disabled, data will not survive a restart");
            None
        }
        Ok(dir) => Some(std::path::PathBuf::from(dir)),
        Err(_) => Some(std::path::PathBuf::from("./data/wal")),
    }
}

/// 创建应用程序路由器
pub async fn create_app() -> Result<Router> {
    create_app_with_features(false, false, false).await
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
    let secs = std::env::var("DB_STALL_WARN_SECS")
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
    std::env::var("DB_SNAPSHOT_INTERVAL_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300)
}

/// Eviction sweep interval in seconds; 0 disables. Default 600.
fn eviction_interval_from_env() -> u64 {
    std::env::var("DB_EVICTION_INTERVAL_SECS")
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
        tracing::info!("Cold-table eviction disabled (DB_EVICTION_INTERVAL_SECS=0)");
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
    std::env::var("DB_ROW_COMPACTION_INTERVAL_SECS")
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
        tracing::info!("Row-file reclamation disabled (DB_ROW_COMPACTION_INTERVAL_SECS=0)");
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
    std::env::var("DB_REINDEX_INTERVAL_SECS")
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
        tracing::info!("Automatic index re-evaluation disabled (DB_REINDEX_INTERVAL_SECS=0)");
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
        tracing::info!("Periodic snapshots disabled (DB_SNAPSHOT_INTERVAL_SECS=0)");
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
    enable_swagger: bool,
    enable_graphql: bool,
    enable_websocket: bool,
) -> Result<Router> {
    create_app_with_storage(enable_swagger, enable_graphql, enable_websocket)
        .await
        .map(|(router, _)| router)
}

/// As [`create_app_with_features`], but also hands back the storage.
///
/// `main` needs it to snapshot on shutdown. Returning only a `Router` left no way to reach it,
/// which is why a clean SIGTERM used to leave the next start replaying the entire log.
pub async fn create_app_with_storage(
    enable_swagger: bool,
    enable_graphql: bool,
    enable_websocket: bool,
) -> Result<(Router, Arc<IntelligentIndexedStorage>)> {
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

    // 加载配置
    let config = AppConfig::load()?;

    // 创建优化的存储实例。带 WAL：进程重启后要能恢复。
    // DB_WAL_DIR 可覆盖目录；设成空串则彻底关掉持久化（只建议用于压测对比）。
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

    #[cfg(feature = "multimodal")]
    let multimodal_state = MultimodalAppState {
        basestate: base_state.clone(),
        embeddingmanager: Arc::new(EmbeddingManager::new()),
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
        .route("/tables/:table_name/stats", get(get_table_stats))
        // 批量操作API
        // .route("/tables/:table_name/bulk_insert", post(bulk_insert_records))  // Removed
        .route(
            "/tables/:table_name/records/bulk",
            post(insert_records_optimized),
        )
        // 搜索API
        .route("/tables/:table_name/search", post(search_vectors))
        .route("/tables/:table_name/hybrid_search", post(hybrid_search))
        .route(
            "/tables/:table_name/batch_search",
            post(search_vectors), // Use standard search
        )
        // 索引管理API
        .route("/tables/:table_name/index", delete(drop_index))
        .route("/tables/:table_name/index/rebuild", post(rebuild_index))
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
        // Keep system probes available under the SDK base URL as well as at the root.
        .route("/health", get(health_check))
        .route("/health/live", get(liveness))
        .route("/health/ready", get(readiness))
        .route("/health/metrics", get(health_metrics))
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
        .with_state(base_state.clone());

    #[cfg(feature = "multimodal")]
    let multimodal_routes = Router::new()
        // .route("/embeddings/functions", get(list_embedding_functions))
        // .route("/embeddings/functions/:name", post(load_embedding_function))
        .route("/embeddings/embed", post(generate_embedding))
        .route(
            "/tables/:table_name/records/multimodal",
            post(create_multimodal_record),
        )
        .route(
            "/tables/:table_name/search/cross-modal",
            post(cross_modal_search),
        )
        .route("/multimodal/validate", post(validate_multimodal_content))
        .with_state(multimodal_state);

    // 合并路由
    #[allow(unused_mut)]
    let mut app = Router::new()
        .merge(system_routes)
        .nest("/api/v1", api_v1_routes);

    #[cfg(feature = "multimodal")]
    {
        app = app.nest("/api/v1", multimodal_routes);
    }

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

    let app = app
        // 增加请求体大小限制到500MB
        .layer(DefaultBodyLimit::max(500 * 1024 * 1024))
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
                ])
                .max_age(std::time::Duration::from_secs(3600)),
        )
        .layer(TraceLayer::new_for_http());

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
    spawn_row_reclamation(storage.clone(), row_compaction_interval_from_env());
    spawn_stall_detector(stall_threshold_from_env());

    Ok((app, storage))
}

// integration tests live in tests/
