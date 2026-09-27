pub mod auth;
#[cfg(feature = "api-dashboard")]
pub mod dashboard;
pub mod extract;
#[cfg(feature = "api-graphql")]
pub mod graphql;
#[cfg(feature = "api-grpc")]
pub mod grpc;
pub mod handlers;
pub mod response;
pub mod server;
pub mod server_health; // Server health monitoring and backpressure
#[cfg(feature = "api-swagger")]
pub mod swagger;
#[cfg(feature = "api-ws")]
pub mod websocket;

pub use server::{create_app, create_app_with_features, create_app_with_storage, Interfaces};
