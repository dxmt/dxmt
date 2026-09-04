use clap::{Parser, Subcommand};
use std::net::SocketAddr;
use tracing::info;

/// Swap the system allocator for jemalloc when the `jemalloc` feature is on.
///
/// Measured on Cohere 1M x 768, 16 cores, snapshots disabled: glibc held 16,171 bytes of RSS
/// per record while the structs account for about 6.8 KB, and constraining it with
/// MALLOC_ARENA_MAX=2 brought that to 11,450 at the cost of 31% of insert throughput. The
/// gap is per-arena free lists that never get reused, not anything the database is storing.
#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// jemalloc returns freed pages to the kernel on a decay clock that is normally advanced by
/// allocation activity, so a server that goes idle after a burst keeps the burst's dirty
/// pages. A background thread runs the decay on wall-clock time instead. Measured on the
/// same 1M load: 8.5 KB per record after a 3,000-query burst instead of 11.5 KB, and the
/// same throughput. jemalloc reads this symbol before `main` runs; it replaces the
/// `_RJEM_MALLOC_CONF` environment variable, which an operator would otherwise have to know.
#[cfg(feature = "jemalloc")]
#[allow(non_upper_case_globals)]
#[export_name = "_rjem_malloc_conf"]
pub static malloc_conf: &[u8] = b"background_thread:true\0";

use dxmt_vectordb::{
    api::{create_app_with_storage, simple_core::create_simple_app},
    config::AppConfig,
    mcp::server::McpServer,
    storage::IntelligentIndexedStorage,
    Result,
};

#[cfg(feature = "cluster")]
use dxmt_vectordb::api::create_app_with_cluster;

/// Unified VectorDB CLI
#[derive(Parser)]
#[command(name = "vectordb")]
#[command(about = "Unified VectorDB CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the VectorDB server
    Serve {
        /// Host to bind
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        /// Port to bind
        #[arg(long, default_value = "9999")]
        port: u16,
        /// Enable simple mode (zero-config collection API)
        #[arg(
            long,
            help = "Run in simple mode with zero configuration and auto-tuned performance"
        )]
        simple: bool,
        /// Memory limit for simple mode in MB
        #[arg(
            long,
            default_value = "1024",
            help = "Memory limit for caching in simple mode"
        )]
        memory_limit: usize,
        /// Serve the Swagger UI (standard mode only; requires the `api-swagger` feature)
        #[arg(long)]
        swagger: bool,
        /// Serve the GraphQL endpoint at /graphql (standard mode only)
        #[arg(long)]
        graphql: bool,
        /// Serve the WebSocket endpoint (standard mode only)
        #[arg(long)]
        websocket: bool,
    },
    /// Speak the Model Context Protocol over stdin/stdout
    ///
    /// Launch this from an MCP client (Claude Desktop, Cline, the MCP inspector) rather than by
    /// hand — it reads JSON-RPC from stdin and writes replies to stdout, so a terminal session
    /// just sits there waiting for input.
    Mcp {
        /// Directory for the write-ahead log; empty disables persistence
        #[arg(long)]
        wal_dir: Option<String>,
    },
}

/// Resolve when the process is asked to stop, so the server can drain and then snapshot.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => {
                tracing::warn!("Cannot listen for SIGTERM ({}); Ctrl-C only", e);
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => info!("Received Ctrl-C, shutting down"),
        _ = terminate => info!("Received SIGTERM, shutting down"),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // MCP owns stdout: it is the JSON-RPC channel. A log line written there lands in the middle
    // of a message and the client drops the connection over a parse error that names no cause.
    if matches!(cli.command, Commands::Mcp { .. }) {
        tracing_subscriber::fmt()
            .with_writer(std::io::stderr)
            .init();
    } else {
        tracing_subscriber::fmt::init();
    }

    match cli.command {
        Commands::Serve {
            host,
            port,
            simple,
            memory_limit,
            swagger,
            graphql,
            websocket,
        } => {
            let bind_addr: SocketAddr = format!("{}:{}", host, port)
                .parse()
                .expect("Invalid bind address");

            if simple {
                // Simple mode: zero-config with auto-tuned performance
                info!(
                    "Starting DXMT VectorDB in SIMPLE MODE on {} (Zero-config, Auto-tuned)",
                    bind_addr
                );
                info!("Memory limit: {} MB", memory_limit);
                info!("Using auto-optimized search algorithms for best performance");

                let app = create_simple_app(memory_limit).await?;

                info!("✓ Simple mode API endpoints:");
                info!("  POST   /collections          - Create collection");
                info!("  GET    /collections          - List collections");
                info!("  POST   /collections/{{name}}  - Add documents (auto-embedding)");
                info!("  POST   /search/{{collection}} - Search (auto-embedding)");
                info!("  DELETE /collections/{{name}}  - Delete collection");

                axum::Server::bind(&bind_addr)
                    .serve(app.into_make_service())
                    .await
                    .map_err(|e| dxmt_vectordb::Error::Internal(format!("server error: {}", e)))?;
            } else {
                // Standard mode with full configuration
                info!("Starting DXMT VectorDB on {}...", bind_addr);

                let config = AppConfig::load()?;
                let cluster_enabled = config.cluster.as_ref().map(|c| c.enabled).unwrap_or(false);
                let (app, storage) = if cluster_enabled {
                    #[cfg(feature = "cluster")]
                    {
                        (create_app_with_cluster(&config).await?, None)
                    }
                    #[cfg(not(feature = "cluster"))]
                    {
                        return Err(dxmt_vectordb::Error::InvalidConfig(
                            "Cluster mode is configured, but this binary was built without the \
                             `cluster` feature"
                                .to_string(),
                        ));
                    }
                } else {
                    // These are off unless asked for: each one widens the exposed
                    // surface, and `create_app()` has always passed false for all three.
                    let (app, storage) =
                        create_app_with_storage(swagger, graphql, websocket).await?;
                    (app, Some(storage))
                };

                info!("REST API mounted under /api/v1; /health and /status are at the root");
                if swagger {
                    info!("  Swagger UI     GET  /api/docs  (spec at /api/docs/openapi.json)");
                }
                if graphql {
                    info!("  GraphQL        POST /graphql");
                }
                if websocket {
                    info!("  WebSocket      GET  /ws  (upgrade)");
                }

                info!("VectorDB server listening on {}", bind_addr);

                axum::Server::bind(&bind_addr)
                    .serve(app.into_make_service())
                    .with_graceful_shutdown(shutdown_signal())
                    .await
                    .map_err(|e| dxmt_vectordb::Error::Internal(format!("server error: {}", e)))?;

                // A clean shutdown used to discard everything written since the last snapshot.
                // Not a correctness problem — the WAL still held it — but it made the next
                // start replay the whole log for no reason.
                if let Some(storage) = storage {
                    match storage.snapshot().await {
                        Ok(tables) => info!("Shutdown snapshot written for {} table(s)", tables),
                        Err(e) => tracing::warn!("Shutdown snapshot failed: {}", e),
                    }
                }
            }
        }

        Commands::Mcp { wal_dir } => {
            let config = AppConfig::load()?;
            let wal_dir = match wal_dir {
                Some(dir) if dir.trim().is_empty() => None,
                Some(dir) => Some(std::path::PathBuf::from(dir)),
                None => Some(std::path::PathBuf::from("./data/wal")),
            };

            info!("Starting DXMT VectorDB MCP server on stdio");
            let storage = std::sync::Arc::new(
                IntelligentIndexedStorage::new_with_wal(&config.storage, wal_dir).await?,
            );

            McpServer::new(storage)
                .serve_stdio()
                .await
                .map_err(|e| dxmt_vectordb::Error::Internal(format!("mcp server error: {}", e)))?;
        }
    }

    Ok(())
}
