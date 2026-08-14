use clap::{Parser, Subcommand};
use std::net::SocketAddr;
use tracing::info;

use dxmt_vectordb::{
	api::{create_app, create_app_with_cluster, simple_core::create_simple_app},
	config::AppConfig,
	Result,
};

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
		/// Enable simple mode (zero-config, Chroma-like API)
		#[arg(long, help = "Run in simple mode with zero configuration and auto-tuned performance")]
		simple: bool,
		/// Memory limit for simple mode in MB
		#[arg(long, default_value = "1024", help = "Memory limit for caching in simple mode")]
		memory_limit: usize,
	},
}

#[tokio::main]
async fn main() -> Result<()> {
	tracing_subscriber::fmt::init();

	let cli = Cli::parse();

	match cli.command {
		Commands::Serve { host, port, simple, memory_limit } => {
			let bind_addr: SocketAddr = format!("{}:{}", host, port)
				.parse()
				.expect("Invalid bind address");

			if simple {
				// Simple mode: zero-config with auto-tuned performance
				info!("Starting DXMT VectorDB in SIMPLE MODE on {} (Zero-config, Auto-tuned)", bind_addr);
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
				let app = if config.cluster.as_ref().map(|c| c.enabled).unwrap_or(false) {
					create_app_with_cluster(&config).await?
				} else {
					create_app().await?
				};

				info!("VectorDB server listening on {}", bind_addr);

				axum::Server::bind(&bind_addr)
					.serve(app.into_make_service())
					.await
					.map_err(|e| dxmt_vectordb::Error::Internal(format!("server error: {}", e)))?;
			}
		}
	}

	Ok(())
}
