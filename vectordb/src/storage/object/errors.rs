use thiserror::Error;

/// 对象存储操作错误
#[derive(Debug, Error, Clone)]
pub enum ObjectStorageError {
    #[error("Object not found: {key}")]
    ObjectNotFound { key: String },

    #[error("Bucket not found: {bucket}")]
    BucketNotFound { bucket: String },

    #[error("Access denied: {message}")]
    AccessDenied { message: String },

    #[error("Invalid credentials")]
    InvalidCredentials,

    #[error("Network error: {message}")]
    NetworkError { message: String },

    #[error("Timeout error: operation timed out after {timeout_ms}ms")]
    TimeoutError { timeout_ms: u64 },

    #[error("Invalid object key: {key}")]
    InvalidKey { key: String },

    #[error("Object too large: {size} bytes (max: {max_size} bytes)")]
    ObjectTooLarge { size: u64, max_size: u64 },

    #[error("Invalid multipart upload: {message}")]
    InvalidMultipartUpload { message: String },

    #[error("Checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },

    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },

    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    #[error("Serialization error: {message}")]
    SerializationError { message: String },

    #[error("Compression error: {message}")]
    CompressionError { message: String },

    #[error("Internal error: {message}")]
    InternalError { message: String },

    #[error("AWS SDK error: {message}")]
    AwsError { message: String },

    #[error("HTTP error: {status_code} - {message}")]
    HttpError { status_code: u16, message: String },

    #[error("IO error: {message}")]
    IoError { message: String },

    /// A conditional write whose condition did not hold: the object is not the version the
    /// writer read, or exists when it was to be created.
    #[error("Precondition failed: {key} changed since it was read")]
    PreconditionFailed { key: String },

    /// This backend cannot make a write conditional on the object's version.
    #[error("Conditional writes are not supported by this object store")]
    ConditionalWritesUnsupported,
}

impl From<std::io::Error> for ObjectStorageError {
    fn from(err: std::io::Error) -> Self {
        ObjectStorageError::IoError {
            message: err.to_string(),
        }
    }
}

#[cfg(feature = "s3-storage")]
impl From<reqwest::Error> for ObjectStorageError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            ObjectStorageError::TimeoutError { timeout_ms: 30000 }
        } else if err.is_connect() {
            ObjectStorageError::NetworkError {
                message: err.to_string(),
            }
        } else if let Some(status) = err.status() {
            ObjectStorageError::HttpError {
                status_code: status.as_u16(),
                message: err.to_string(),
            }
        } else {
            ObjectStorageError::NetworkError {
                message: err.to_string(),
            }
        }
    }
}

impl From<serde_json::Error> for ObjectStorageError {
    fn from(err: serde_json::Error) -> Self {
        ObjectStorageError::SerializationError {
            message: err.to_string(),
        }
    }
}
