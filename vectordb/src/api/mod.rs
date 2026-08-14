pub mod batch_operations;
pub mod cluster;
pub mod cross_modal_batch;
pub mod graphql;
pub mod handlers;
pub mod multimodal;
pub mod rate_limiter;
pub mod response;
pub mod server;
pub mod server_health;  // Server health monitoring and backpressure
pub mod simple_core;  // Uses IntelligentIndexedStorage with auto-config
pub mod server_with_cluster;
pub mod streaming_search;
pub mod swagger;
pub mod websocket;

pub use server::{create_app, create_app_with_features};
pub use server_with_cluster::create_app_with_cluster;
pub use simple_core::create_simple_app;
