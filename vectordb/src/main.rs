use clap::{Parser, Subcommand};
use std::net::SocketAddr;
use tracing::{info, warn};

/// Swap the system allocator for jemalloc when the `jemalloc` feature is on.
///
/// Measured on Cohere 1M x 768, 16 cores, snapshots disabled: glibc held 16,171 bytes of RSS
/// per record while the structs account for about 6.8 KB, and constraining it with
/// MALLOC_ARENA_MAX=2 brought that to 11,450 at the cost of 31% of insert throughput. The
/// gap is per-arena free lists that never get reused, not anything the database is storing.
#[cfg(all(feature = "jemalloc", not(windows)))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// jemalloc returns freed pages to the kernel on a decay clock that is normally advanced by
/// allocation activity, so a server that goes idle after a burst keeps the burst's dirty
/// pages. A background thread runs the decay on wall-clock time instead. Measured on the
/// same 1M load: 8.5 KB per record after a 3,000-query burst instead of 11.5 KB, and the
/// same throughput. jemalloc reads this symbol before `main` runs; it replaces the
/// `_RJEM_MALLOC_CONF` environment variable, which an operator would otherwise have to know.
#[cfg(all(feature = "jemalloc", not(windows)))]
#[allow(non_upper_case_globals)]
#[export_name = "_rjem_malloc_conf"]
pub static malloc_conf: &[u8] = b"background_thread:true\0";

use dxmt_vectordb::{
    api::{create_app_with_storage, Interfaces},
    config::AppConfig,
    mcp::server::McpServer,
    storage::IntelligentIndexedStorage,
    Result,
};

/// Unified VectorDB CLI
#[derive(Parser)]
#[command(name = "vectordb")]
#[command(about = "Unified VectorDB CLI", long_about = None)]
// `vectordb --version` answered `error: unexpected argument '--version' found` right up to
// the 1.0.0 tag. It is the first thing anyone runs against a binary they were handed, and the
// only way to tell two builds apart in a bug report. From the manifest, so it cannot drift.
#[command(version = env!("CARGO_PKG_VERSION"))]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the VectorDB server
    Serve {
        /// Host to bind
        /// This machine only by default. Every route is open unless DXMT_API_KEY is set, so
        /// serving the network (`0.0.0.0`) is a choice to make with a key in place.
        #[arg(long, default_value = "127.0.0.1", env = "DXMT_HOST")]
        host: String,
        /// Port to bind
        #[arg(long, default_value = "9999", env = "DXMT_PORT")]
        port: u16,
        /// Serve the Swagger UI at /api/docs. On by default; `--no-swagger` turns it off.
        ///
        /// Accepted and redundant, so a script that passes it keeps working.
        #[arg(long)]
        swagger: bool,
        /// Do not serve the Swagger UI.
        #[arg(long, conflicts_with = "swagger")]
        no_swagger: bool,
        /// Serve the GraphQL endpoint at /graphql
        #[arg(long)]
        graphql: bool,
        /// Serve the WebSocket endpoint
        #[arg(long)]
        websocket: bool,
        /// Serve the read-only web console at /dashboard. On by default when the bundle was
        /// built into this binary; `--no-dashboard` turns it off.
        ///
        /// Passing it explicitly also makes a missing bundle an error rather than a warning:
        /// someone who asked for the console by name should be told it is not there, where a
        /// default-on console that is simply absent should not stop the database serving.
        #[arg(long)]
        dashboard: bool,
        /// Do not serve the web console.
        #[arg(long, conflicts_with = "dashboard")]
        no_dashboard: bool,
        /// Serve gRPC on this port, alongside HTTP (standard mode only; requires `api-grpc`).
        /// 0 disables, which is the default.
        #[arg(long, default_value = "0", env = "DXMT_GRPC_PORT")]
        grpc_port: u16,
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

    /// Restore a backup into the data directory, with the server stopped.
    ///
    /// The backup is checked against its manifest first; whatever the data directory held is
    /// moved into a sibling directory rather than deleted; the restored store is opened once
    /// to prove it recovers, and its tables are listed.
    Restore {
        /// A directory written by `POST /api/v1/backup` or a scheduled backup
        backup: std::path::PathBuf,
        /// The data directory to restore into; defaults to the one `serve` would open
        /// (`DXMT_WAL_DIR`, else `$DXMT_ROOT_DIR/store`)
        #[arg(long)]
        wal_dir: Option<std::path::PathBuf>,
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

fn main() -> Result<()> {
    // `.env` first, so the variables clap reads (`DXMT_HOST`, `DXMT_PORT`, `DXMT_GRPC_PORT`)
    // and the legacy names below see it too. A variable already set in the environment wins.
    let _ = dotenv::dotenv();
    // Before `Cli::parse`, because clap reads some variables itself and takes one name each.
    let legacy = dxmt_vectordb::config::env::adopt_legacy_names();

    let cli = Cli::parse();

    // MCP owns stdout: it is the JSON-RPC channel. A log line written there lands in the middle
    // of a message and the client drops the connection over a parse error that names no cause.
    if matches!(cli.command, Commands::Mcp { .. }) {
        tracing_subscriber::fmt()
            .with_writer(std::io::stderr)
            .init();
    } else {
        // Warnings by default, not errors only: with nothing set, a durability failure or a
        // snapshot that could not be written went unsaid. Progress lines stay behind RUST_LOG.
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
            )
            .init();
    }

    // After the subscriber exists, or the warning goes nowhere.
    if !legacy.is_empty() {
        for (was, now) in &legacy {
            warn!("{was} is deprecated and will be removed; use {now} instead");
        }
        warn!(
            "{} environment variable(s) still use the DB_ prefix. It is not a namespace — \
             DB_HOST is this server's bind address and also what every compose file calls its \
             Postgres.",
            legacy.len()
        );
    }

    // Before the runtime, because its size is one of the settings.
    let config = AppConfig::load()?;
    let mut runtime = tokio::runtime::Builder::new_multi_thread();
    runtime.enable_all();
    match config.server.workers {
        Some(0) => {
            return Err(dxmt_vectordb::Error::InvalidConfig(
                "DXMT_WORKER_THREADS must be at least 1".to_string(),
            ))
        }
        Some(threads) => {
            info!("Running {} worker threads (DXMT_WORKER_THREADS)", threads);
            runtime.worker_threads(threads);
        }
        // One per core, tokio's default.
        None => {}
    }
    runtime
        .build()
        .map_err(|e| dxmt_vectordb::Error::Internal(format!("cannot start the runtime: {e}")))?
        .block_on(run(cli.command, config))
}

async fn run(command: Commands, config: AppConfig) -> Result<()> {
    match command {
        Commands::Serve {
            host,
            port,
            swagger,
            no_swagger,
            graphql,
            websocket,
            dashboard,
            no_dashboard,
            grpc_port,
        } => {
            let bind_addr: SocketAddr = format!("{}:{}", host, port)
                .parse()
                .expect("Invalid bind address");

            // On unless turned off. A database you cannot look at, and whose API you cannot
            // read, is harder to evaluate than one you can; both are read-only surfaces and
            // both sit behind `DXMT_API_KEY` when one is set, like every other route.
            let serve_swagger = !no_swagger;
            let serve_dashboard = !no_dashboard;

            {
                // The simple mode that used to be the other half of this `if` is gone: it was
                // a second data model (`/collections/*`) over the same storage, with its own
                // wire shapes and no tests, and its "auto-embedding" was a hashed bag of
                // words — lexical retrieval wearing a dense vector's shape. Peers ship several
                // protocols over one model; none ships two models.
                info!("Starting DXMT VectorDB on {}...", bind_addr);

                let cluster_enabled = config.cluster.as_ref().map(|c| c.enabled).unwrap_or(false);
                let (app, storage) = if cluster_enabled {
                    {
                        return Err(dxmt_vectordb::Error::InvalidConfig(
                            "Cluster mode is configured, but this build has no cluster mode: \
                             DXMT VectorDB is a single-node server"
                                .to_string(),
                        ));
                    }
                } else {
                    // These are off unless asked for: each one widens the exposed surface,
                    // and `create_app()` passes the default, which is all of them off.
                    let (app, storage) = create_app_with_storage(
                        Interfaces {
                            swagger: serve_swagger,
                            graphql,
                            websocket,
                            dashboard: serve_dashboard,
                            // Only a dashboard asked for by name turns a missing bundle into
                            // a refusal to start.
                            dashboard_required: dashboard,
                        },
                        dxmt_vectordb::api::auth::ApiKey::from_env(),
                    )
                    .await?;
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
                if dashboard {
                    info!("  Dashboard      GET  /dashboard  (read-only console, unauthenticated)");
                }

                // gRPC gets its own port rather than sharing the HTTP one. Multiplexing both
                // over one listener is possible and costs a tower layer that has to sniff every
                // connection; two ports is what the comparable databases do, and it keeps the
                // HTTP path exactly as it was.
                if grpc_port != 0 {
                    #[cfg(feature = "api-grpc")]
                    {
                        let Some(storage) = storage.clone() else {
                            return Err(dxmt_vectordb::Error::InvalidConfig(
                                "gRPC needs a storage handle, and this server was built \
                                 without one"
                                    .to_string(),
                            ));
                        };
                        let grpc_addr: std::net::SocketAddr =
                            format!("{}:{}", host, grpc_port).parse().map_err(|e| {
                                dxmt_vectordb::Error::InvalidConfig(format!(
                                    "bad gRPC address: {e}"
                                ))
                            })?;
                        info!("  gRPC           {} (proto/dxmt.proto)", grpc_addr);
                        let service =
                            dxmt_vectordb::api::grpc::GrpcService::new(storage).into_server();
                        tokio::spawn(async move {
                            if let Err(e) = tonic::transport::Server::builder()
                                .add_service(service)
                                .serve(grpc_addr)
                                .await
                            {
                                tracing::error!("gRPC server stopped: {}", e);
                            }
                        });
                    }
                    #[cfg(not(feature = "api-grpc"))]
                    {
                        return Err(dxmt_vectordb::Error::InvalidConfig(
                            "--grpc-port needs the `api-grpc` feature: rebuild with \
                             `cargo build --features api-grpc`"
                                .to_string(),
                        ));
                    }
                }

                info!("VectorDB server listening on {}", bind_addr);
                // On stdout, whatever RUST_LOG says: where everything is and how to put the
                // first note in. Without it a server started with no configuration printed
                // nothing at all.
                if let Some(storage) = &storage {
                    // `api::dashboard` exists only with the feature; the minimal build has no
                    // console to point at.
                    #[cfg(feature = "api-dashboard")]
                    let console =
                        serve_dashboard && dxmt_vectordb::api::dashboard::assets_are_embedded();
                    #[cfg(not(feature = "api-dashboard"))]
                    let console = false;
                    print!(
                        "{}",
                        dxmt_vectordb::api::quickstart::banner(
                            storage,
                            dxmt_vectordb::api::quickstart::resolved_default_embedding(),
                            &host,
                            port,
                            serve_swagger,
                            console,
                            dxmt_vectordb::api::auth::ApiKey::from_env().is_some(),
                        )
                        .await
                    );
                }

                axum::Server::bind(&bind_addr)
                    .serve(app.into_make_service())
                    .with_graceful_shutdown(shutdown_signal())
                    .await
                    .map_err(|e| dxmt_vectordb::Error::Internal(format!("server error: {}", e)))?;

                // A clean shutdown used to discard everything written since the last snapshot.
                // Not a correctness problem — the WAL still held it — but it made the next
                // start replay the whole log for no reason. Without a WAL there is nothing to
                // snapshot, and saying it failed on every stop was noise.
                if let Some(storage) = storage.filter(|s| s.is_persistent()) {
                    match storage.snapshot().await {
                        Ok(tables) => info!("Shutdown snapshot written for {} table(s)", tables),
                        Err(e) => tracing::warn!("Shutdown snapshot failed: {}", e),
                    }
                    // And the payload and sparse indexes, so the next start reads them instead
                    // of rebuilding them -- 23-27 s of every core at 10M rows.
                    if let Err(e) = storage.persist_derived().await {
                        tracing::warn!("Derived indexes not persisted: {}", e);
                    }
                }
            }
        }

        Commands::Restore { backup, wal_dir } => {
            let Some(store) = wal_dir.or_else(dxmt_vectordb::api::server::wal_dir_from_env) else {
                return Err(dxmt_vectordb::Error::InvalidConfig(
                    "no data directory to restore into: DXMT_WAL_DIR is empty; pass --wal-dir"
                        .to_string(),
                ));
            };
            let report =
                IntelligentIndexedStorage::restore_backup(&config.storage, &backup, &store).await?;
            println!("Restored {} into {}", backup.display(), store.display());
            if let Some(aside) = &report.moved_aside {
                println!("The previous contents are in {}", aside.display());
            }
            for (table, records) in &report.tables {
                println!("  {table}: {records} records");
            }
        }

        Commands::Mcp { wal_dir } => {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// No release before this read `DXMT_HOST` or `DXMT_PORT`, although the README listed both.
    /// A flag still wins over the variable. One test, because the environment is the process's.
    #[test]
    fn serve_takes_its_address_from_the_environment_unless_a_flag_says_otherwise() {
        let address = |args: &[&str]| match Cli::try_parse_from(args).unwrap().command {
            Commands::Serve { host, port, .. } => (host, port),
            Commands::Mcp { .. } | Commands::Restore { .. } => unreachable!(),
        };
        std::env::set_var("DXMT_HOST", "127.0.0.1");
        std::env::set_var("DXMT_PORT", "19876");
        assert_eq!(
            address(&["vectordb", "serve"]),
            ("127.0.0.1".to_string(), 19876)
        );
        assert_eq!(
            address(&["vectordb", "serve", "--port", "9999"]),
            ("127.0.0.1".to_string(), 9999)
        );
        std::env::remove_var("DXMT_HOST");
        std::env::remove_var("DXMT_PORT");
        assert_eq!(
            address(&["vectordb", "serve"]),
            ("127.0.0.1".to_string(), 9999)
        );
    }
}
