pub mod metrics;
pub mod tracing;

use crate::config::Config;

use crate::Result;

/// Initialize observability (metrics + tracing)
pub async fn init_observability(config: &Config) -> crate::Result<()> {
    // Initialize distributed tracing
    if config.observability.tracing_enabled {

        tracing::inittracing(config)?;

    }
    // Initialize metrics collection
    if config.observability.metrics_enabled {

        metrics::init_metrics(config)?;

    }
    Ok(())
}

/// Shutdown observability gracefully
pub fn shutdown_observability() {
    tracing::shutdowntracing();
    metrics::shutdown_metrics()
}
