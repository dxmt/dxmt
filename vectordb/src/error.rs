//! Unified error handling for VectorDB
//! Following best practices from TiKV and other top-tier projects

use serde::{Serialize, Deserialize};

/// 应用程序错误类型
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Internal error: {0}")]
    Internal(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Not found: {0}")]
    NotFound(String),
    
    #[error("Unauthorized")]
    Unauthorized,
    
    #[error("Configuration error: {0}")]
    InvalidConfig(String),
    
    #[error("Storage error: {0}")]
    Storage(String),
    
    #[error("Network error: {0}")]
    Network(String),
    
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Bincode error: {0}")]
    Bincode(#[from] Box<bincode::ErrorKind>),
    
    #[error("Object storage error: {0}")]
    ObjectStorage(#[from] crate::storage::object::ObjectStorageError),

    #[error("Hyper error: {0}")]
    Hyper(#[from] hyper::Error),

    // Legacy compatibility variants
    #[error("Table not found: {0}")]
    TableNotFound(String),
    
    #[error("Record not found: {0}")]
    RecordNotFound(String),
    
    #[error("Node not found: {0}")]
    NodeNotFound(String),
    
    #[error("Table already exists: {0}")]
    TableAlreadyExists(String),
    
    #[error("Record already exists: {0}")]
    RecordAlreadyExists(String),
    
    #[error("Invalid vector: {0}")]
    InvalidVector(String),
    
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Not implemented: {0}")]
    NotImplemented(String),
    
    #[error("Timeout: {0}")]
    Timeout(String),
    
    #[error("Not leader")]
    NotLeader,
    
    #[error("No available nodes")]
    NoAvailableNodes,
    
    #[error("Empty vector")]
    EmptyVector,
    
    // Structured error variants for compatibility
    #[error("Validation error: {message}")]
    ValidationError { message: String },
    
    #[error("Serialization error: {message}")]
    SerializationError { message: String },
    
    #[error("Compression error: {message}")]
    CompressionError { message: String },
    
    #[error("Decompression error: {message}")]
    DecompressionError { message: String },
    
    #[error("Remote error: {message}")]
    RemoteError { message: String },
    
    #[error("Service unavailable: {message}")]
    ServiceUnavailable { message: String },
}

impl From<async_graphql::Error> for Error {
    fn from(err: async_graphql::Error) -> Self {
        Error::Internal(format!("{:?}", err))
    }
}

impl From<&str> for Error {
    fn from(msg: &str) -> Self {
        Error::Internal(msg.to_string())
    }
}

impl From<String> for Error {
    fn from(msg: String) -> Self {
        Error::Internal(msg)
    }
}

/// Result type alias using our Error
pub type Result<T> = std::result::Result<T, Error>;

/// Error code for API responses
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorCode {
    // Client errors (4xx)
    NotFound,
    AlreadyExists,
    InvalidInput,
    InvalidConfig,
    DimensionMismatch,
    Unauthorized,

    // Server errors (5xx)
    Internal,
    Storage,
    Timeout,
    NotImplemented,
    ClusterError,
}

impl Error {
    /// Get the error code for API responses
    pub fn error_code(&self) -> ErrorCode {
        match self {
            crate::Error::NotFound(_)
            | crate::Error::TableNotFound(_)
            | crate::Error::RecordNotFound(_)
            | crate::Error::NodeNotFound(_) => {
                ErrorCode::NotFound
            }
            crate::Error::TableAlreadyExists(_ ) | crate::Error::RecordAlreadyExists(_) => {
                ErrorCode::AlreadyExists
            }
            crate::Error::InvalidVector(_) | crate::Error::InvalidInput(_ ) => ErrorCode::InvalidInput,
            crate::Error::InvalidConfig(_) => ErrorCode::InvalidConfig,
            crate::Error::DimensionMismatch { .. } => ErrorCode::DimensionMismatch,
            crate::Error::Storage(_) | crate::Error::Io(_) => ErrorCode::Storage,
            crate::Error::Timeout(_) => ErrorCode::Timeout,
            crate::Error::NotImplemented(_ ) => ErrorCode::NotImplemented,
            crate::Error::NotLeader | crate::Error::NoAvailableNodes => {
                ErrorCode::ClusterError
            }
            crate::Error::ValidationError { .. } => ErrorCode::InvalidInput,
            crate::Error::SerializationError { .. } | crate::Error::EmptyVector => ErrorCode::InvalidInput,
            crate::Error::CompressionError { .. } | crate::Error::DecompressionError { .. } => ErrorCode::Storage,
            crate::Error::RemoteError { .. } | crate::Error::ServiceUnavailable { .. } => ErrorCode::ClusterError,
            crate::Error::Unauthorized => ErrorCode::Unauthorized,
            _ => ErrorCode::Internal,
        }
    }

    /// Get HTTP status code
    pub fn status_code(&self) -> u16 {
        match self.error_code() {
            ErrorCode::NotFound => 404,
            ErrorCode::AlreadyExists => 409,
            ErrorCode::InvalidInput | ErrorCode::InvalidConfig | ErrorCode::DimensionMismatch => {
                400
            }
            ErrorCode::Unauthorized => 401,
            ErrorCode::NotImplemented => 501,
            _ => 500,
        }
    }

    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            crate::Error::Timeout(_) | crate::Error::NotLeader | crate::Error::NoAvailableNodes
        )
    }
}

/// Context trait for adding context to errors
pub trait ErrorContext<T> {
    /// Add context to an error
    fn context<C>(self, context: C) -> crate::Result<T>
    where
        C: std::fmt::Display + Send + Sync + 'static;

    /// Add context with a closure (lazy evaluation)
    fn with_context<C, F>(self, f: F) -> crate::Result<T>
    where
        C: std::fmt::Display + Send + Sync + 'static,
        F: FnOnce() -> C;
}

impl<T, E> ErrorContext<T> for std::result::Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn context<C>(self, context: C) -> crate::Result<T>
    where
        C: std::fmt::Display + Send + Sync + 'static,
    {
        self.map_err(|e| crate::Error::Internal(format!("{}: {}", context, e)))
    }

    fn with_context<C, F>(self, f: F) -> crate::Result<T>
    where
        C: std::fmt::Display + Send + Sync + 'static,
        F: FnOnce() -> C,
    {
        self.map_err(|e| crate::Error::Internal(format!("{}: {}", f(), e)))
    }
}

/// Macro for creating errors with format strings
#[macro_export]
macro_rules! error {
    ($kind:ident, $($arg:tt)*) => {
        $crate::error::Error::$kind(format!($($arg)*))
    }
}

/// Macro for returning errors
#[macro_export]
macro_rules! bail {
    ($kind:ident, $($arg:tt)*) => {
        return Err($crate::error!($kind, $($arg)*))
    }
}

/// Macro for ensuring conditions
#[macro_export]
macro_rules! ensure {
    ($cond:expr, $kind:ident, $($arg:tt)*) => {
        if !$cond {
            $crate::bail!($kind, $($arg)*)
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code() {
        let err = crate::Error::TableNotFound("test".to_string());
        assert_eq!(err.error_code(), ErrorCode::NotFound);
        assert_eq!(err.status_code(), 404);
    }

    #[test]
    fn test_not_found_generic_maps_404() {
        let err = crate::Error::NotFound("something".to_string());
        assert_eq!(err.error_code(), ErrorCode::NotFound);
        assert_eq!(err.status_code(), 404, "generic NotFound should return 404, not 500");
    }

    #[test]
    fn test_unauthorized_status_code() {
        let err = crate::Error::Unauthorized;
        assert_eq!(err.error_code(), ErrorCode::Unauthorized);
        assert_eq!(err.status_code(), 401);
    }

    #[test]
    fn test_dimension_mismatch_status_code() {
        let err = crate::Error::DimensionMismatch { expected: 128, actual: 64 };
        assert_eq!(err.error_code(), ErrorCode::DimensionMismatch);
        assert_eq!(err.status_code(), 400);
    }

    #[test]
    fn test_retryable() {
        assert!(crate::Error::Timeout("timeout".to_string()).is_retryable());
        assert!(!crate::Error::TableNotFound("test".to_string()).is_retryable())
    }

    #[test]
    fn test_error_macro() {
        let err = error!(TableNotFound, "table_{}", 123);
        assert_eq!(err.to_string(), "Table not found: table_123");
    }
}
