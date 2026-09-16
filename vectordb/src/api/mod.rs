#[cfg(feature = "api-graphql")]
pub mod graphql;
#[cfg(feature = "api-grpc")]
pub mod grpc;
pub mod handlers;
#[cfg(feature = "multimodal")]
pub mod multimodal;
pub mod response;
pub mod server;
pub mod server_health; // Server health monitoring and backpressure
pub mod simple_core; // Uses IntelligentIndexedStorage with auto-config
#[cfg(feature = "api-swagger")]
pub mod swagger;
#[cfg(feature = "api-ws")]
pub mod websocket;

pub use server::{create_app, create_app_with_features, create_app_with_storage};
#[cfg(feature = "cluster")]
pub use server_with_cluster::create_app_with_cluster;
pub use simple_core::create_simple_app;
