use axum::{
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use tower_http::{cors::CorsLayer, trace::TraceLayer};

use crate::{
    api::{
        cluster::{get_cluster_status, handle_announce, handle_heartbeat},
        graphql,
        handlers::{
            create_backup, create_table, delete_records, delete_table, get_record, get_status,
            get_table, health_check, insert_records, list_tables, search_vectors, update_record,
            AppState,
        },
        multimodal::{
            create_multimodal_record, cross_modal_search, generate_embedding,
            validate_multimodal_content, MultimodalAppState,
        },
        swagger, websocket,
    },
    cluster::SimpleDiscovery,
    config::AppConfig,
    embeddings::EmbeddingManager,
};

/// 创建带集群功能的应用程序路由器
pub async fn create_app_with_cluster(config: &AppConfig) -> crate::Result<Router> {
    // 创建优化的存储实例
    let storage = Arc::new(crate::storage::IntelligentIndexedStorage::new(&config.storage).await?);

    // Initialize server health monitor
    let server_health = Arc::new(crate::api::server_health::ServerHealth::default());

    // 创建应用状态
    let base_state = AppState {
        storage: storage.clone(),
        server_health,
    };

    // 创建多模态状态
    let embedding_manager = Arc::new(EmbeddingManager::new());
    let multimodal_state = MultimodalAppState {
        basestate: base_state.clone(),
        embeddingmanager: embedding_manager,
    };

    // 创建系统路由（在根路径）
    let system_routes = Router::new()
        .route("/health", get(health_check))
        .route("/status", get(get_status))
        .with_state(base_state.clone());

    // 创建API v1路由
    let api_v1_routes = Router::new()
        .route("/tables", get(list_tables).post(create_table))
        .route("/tables/:table_name", get(get_table).delete(delete_table))
        .route(
            "/tables/:table_name/records",
            post(insert_records).delete(delete_records),
        )
        .route(
            "/tables/:table_name/records/:record_id",
            get(get_record).put(update_record),
        )
        .route("/tables/:table_name/search", post(search_vectors))
        // Backup is a single endpoint now; the catalogue routes returned 500 unconditionally.
        .route("/backup", post(create_backup))
        .with_state(base_state.clone());

    // 初始化应用
    let mut app = Router::new()
        .merge(system_routes)
        .nest("/api/v1", api_v1_routes);

    // 如果启用了集群模式，添加集群相关路由
    let _discovery = if let Some(cluster_config) = &config.cluster {
        if cluster_config.enabled {
            // 创建服务发现实例
            let discovery = Arc::new(SimpleDiscovery::new(
                crate::cluster::ClusterConfig::default(),
            ));

            // 创建集群路由（带有自己的状态）
            let cluster_routes = Router::new()
                .route("/cluster/announce", post(handle_announce))
                .route("/cluster/heartbeat", post(handle_heartbeat))
                .route("/cluster/status", get(get_cluster_status))
                .with_state(Arc::clone(&discovery));

            // 合并集群路由
            app = app.merge(cluster_routes);

            Some(discovery)
        } else {
            None
        }
    } else {
        None
    };

    // 创建多模态路由（带有自己的状态）
    let multimodal_routes = Router::new()
        // .route("/embedding-functions", get(list_embedding_functions))
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
    app = app.nest("/api/v1", multimodal_routes);

    // GraphQL 与 WebSocket 路由
    let graphql_routes = graphql::create_graphql_router();
    let websocket_routes = websocket::create_websocket_router();
    app = app.merge(graphql_routes).merge(websocket_routes);

    // 合并Swagger UI路由
    app = app.merge(swagger::create_swagger_ui());

    // 添加中间件
    let app = app
        .layer(
            CorsLayer::new()
                .allow_origin(
                    "http://localhost:3000"
                        .parse::<axum::http::HeaderValue>()
                        .unwrap(),
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
        .layer(TraceLayer::new_for_http())
        // 合并Swagger UI路由
        .merge(swagger::create_swagger_ui())
        // 应用主状态到所有需要AppState的路由
        .with_state(base_state);

    Ok(app)
}
