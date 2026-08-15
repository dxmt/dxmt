use axum::{
    extract::DefaultBodyLimit,
    routing::{delete, get, post},
    Router,
};

use std::sync::Arc;
use tower_http::{cors::CorsLayer, trace::TraceLayer};

use crate::{
    api::{
        graphql,
        // batch_handlers::batch_search_vectors,  // Removed
        // bulk_handlers::bulk_insert_records,  // Removed
        handlers::{
            alter_table, branch_table, create_backup, create_snapshot, create_table,
            delete_records, delete_table, drop_index, execute_dxql, full_text_search, get_record,
            get_status, get_table, health_check, health_metrics, hybrid_search,
            hybrid_search_combined, insert_records, insert_records_optimized, list_records,
            list_tables, rebuild_index, search_vectors, update_record, AppState,
        },
        multimodal::{
            create_multimodal_record, cross_modal_search, generate_embedding,
            validate_multimodal_content, MultimodalAppState,
        },
        swagger,
        websocket,
    },
    config::AppConfig,
    embeddings::EmbeddingManager,
    error::Result,
    storage::IntelligentIndexedStorage,
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
    // 加载配置
    let config = AppConfig::load()?;

    // 创建优化的存储实例。带 WAL：进程重启后要能恢复。
    // DB_WAL_DIR 可覆盖目录；设成空串则彻底关掉持久化（只建议用于压测对比）。
    let storage = Arc::new(
        IntelligentIndexedStorage::new_with_wal(&config.storage, wal_dir_from_env()).await?,
    );

    // 创建嵌入管理器
    let embedding_manager = Arc::new(EmbeddingManager::new());

    // 应用状态
    // Initialize server health monitor
    let server_health = Arc::new(crate::api::server_health::ServerHealth::default());

    let base_state = AppState {
        storage: storage.clone(),
        server_health,
    };

    // 多模态应用状态
    let multimodal_state = MultimodalAppState {
        basestate: base_state.clone(),
        embeddingmanager: embedding_manager,
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
        .with_state(base_state.clone());

    // 创建系统路由（在根路径）
    let system_routes = Router::new()
        .route("/status", get(get_status))
        .route("/health", get(health_check))
        .route("/health/metrics", get(health_metrics))
        .with_state(base_state.clone());

    // 创建多模态路由（使用多模态状态）
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
    let mut app = Router::new()
        .merge(system_routes)
        .nest("/api/v1", api_v1_routes)
        .nest("/api/v1", multimodal_routes);

    // 条件性添加 GraphQL 路由
    if enable_graphql {
        let graphql_routes = graphql::create_graphql_router();
        app = app.merge(graphql_routes);
    }

    // 条件性添加 WebSocket 路由
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

    // 条件性添加 Swagger UI 路由
    let app = if enable_swagger {
        app.merge(swagger::create_swagger_ui())
    } else {
        app
    };

    let app = app.with_state(base_state);

    spawn_periodic_snapshots(storage.clone(), snapshot_interval_from_env());
    spawn_eviction_sweep(storage.clone(), eviction_interval_from_env());

    Ok((app, storage))
}

// integration tests live in tests/
