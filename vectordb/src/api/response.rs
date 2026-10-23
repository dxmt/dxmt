use crate::error::Error;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};

/// 标准API响应格式
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct ApiResponse {
    pub success: bool,
    pub data: Option<serde_json::Value>,
    pub error: Option<ApiError>,
    pub timestamp: String,
}

/// API错误信息
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
pub struct ApiError {
    pub code: String,
    pub message: String,
    pub details: Option<serde_json::Value>,
}

impl ApiResponse {
    /// 创建成功响应
    pub fn success<T: Serialize>(data: T) -> Self {
        let value = serde_json::to_value(data)
            .unwrap_or_else(|e| serde_json::json!({"serialization_error": e.to_string()}));
        ApiResponse {
            success: true,
            data: Some(value),
            error: None,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// 创建错误响应
    pub fn error(error: ApiError) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(error),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// 创建成功响应并指定状态码
    pub fn success_with_status<T: Serialize>(
        data: T,
        _status: StatusCode,
    ) -> std::result::Result<ApiResponse, StatusCode> {
        // Since StatusCode is already handled in IntoResponse implementation,
        // we can just return a normal success response
        Ok(Self::success(data))
    }
}

impl ApiError {
    pub fn new(code: &str, message: &str) -> Self {
        Self {
            code: code.to_string(),
            message: message.to_string(),
            details: None,
        }
    }

    pub fn with_details(code: &str, message: &str, details: serde_json::Value) -> Self {
        Self {
            code: code.to_string(),
            message: message.to_string(),
            details: Some(details),
        }
    }
}

impl From<Error> for ApiError {
    fn from(err: Error) -> Self {
        let code = match &err {
            crate::Error::TableNotFound(_) => "TABLE_NOT_FOUND",
            crate::Error::TableAlreadyExists(_) => "TABLE_ALREADY_EXISTS",
            crate::Error::RecordNotFound(_) => "RECORD_NOT_FOUND",
            crate::Error::RecordAlreadyExists(_) => "RECORD_ALREADY_EXISTS",
            crate::Error::InvalidVector(_) => "INVALID_VECTOR",
            crate::Error::InvalidInput(_) => "INVALID_INPUT",
            crate::Error::InvalidConfig(_) => "INVALID_CONFIG",
            crate::Error::DimensionMismatch { .. } => "DIMENSION_MISMATCH",
            crate::Error::Storage(_) => "STORAGE_ERROR",
            crate::Error::Network(_) => "NETWORK_ERROR",
            crate::Error::Serialization(_) => "SERIALIZATION_ERROR",
            crate::Error::NotImplemented(_) => "NOT_IMPLEMENTED",
            crate::Error::Timeout(_) => "TIMEOUT",
            crate::Error::ReadOnlyReplica(_) => "READ_ONLY_REPLICA",
            crate::Error::NotLeader => "NOT_LEADER",
            crate::Error::NoAvailableNodes => "NO_AVAILABLE_NODES",
            crate::Error::EmptyVector => "EMPTY_VECTOR",
            crate::Error::ValidationError { .. } => "VALIDATION_ERROR",
            crate::Error::SerializationError { .. } => "SERIALIZATION_ERROR",
            crate::Error::CompressionError { .. } => "COMPRESSION_ERROR",
            crate::Error::DecompressionError { .. } => "DECOMPRESSION_ERROR",
            crate::Error::RemoteError { .. } => "REMOTE_ERROR",
            crate::Error::ServiceUnavailable { .. } => "SERVICE_UNAVAILABLE",
            crate::Error::Internal(_) => "INTERNAL_ERROR",
            crate::Error::NotFound(_) => "NOT_FOUND",
            crate::Error::Unauthorized => "UNAUTHORIZED",
            crate::Error::Io(_) => "IO_ERROR",
            crate::Error::Bincode(_) => "BINCODE_ERROR",
            crate::Error::ObjectStorage(_) => "OBJECT_STORAGE_ERROR",
            crate::Error::NodeNotFound(_) => "NODE_NOT_FOUND",
            crate::Error::Hyper(_) => "HYPER_ERROR",
        };

        ApiError::new(code, &err.to_string())
    }
}

impl IntoResponse for ApiResponse {
    fn into_response(self) -> Response {
        let status_code = if self.success {
            StatusCode::OK
        } else {
            match self.error.as_ref().map(|e| e.code.as_str()) {
                Some("NOT_FOUND")
                | Some("TABLE_NOT_FOUND")
                | Some("RECORD_NOT_FOUND")
                | Some("NODE_NOT_FOUND") => StatusCode::NOT_FOUND,
                Some("TABLE_ALREADY_EXISTS") | Some("RECORD_ALREADY_EXISTS") => {
                    StatusCode::CONFLICT
                }
                Some("DIMENSION_MISMATCH")
                | Some("INVALID_VECTOR")
                | Some("INVALID_CONFIG")
                | Some("INVALID_INPUT")
                | Some("INVALID_TABLE_NAME")
                | Some("VALIDATION_ERROR")
                | Some("BAD_REQUEST")
                | Some("EMPTY_VECTOR") => StatusCode::BAD_REQUEST,
                Some("UNAUTHORIZED") => StatusCode::UNAUTHORIZED,
                Some("FULL_TEXT_NOT_ENABLED")
                | Some("FEATURE_DISABLED")
                | Some("NOT_IMPLEMENTED") => StatusCode::NOT_IMPLEMENTED,
                Some("SERVICE_UNAVAILABLE") => StatusCode::SERVICE_UNAVAILABLE,
                Some("TIMEOUT") => StatusCode::REQUEST_TIMEOUT,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            }
        };

        (status_code, Json(self)).into_response()
    }
}

/// 便捷宏：创建成功响应
#[macro_export]
macro_rules! successresponse {
    ($data:expr) => {
        $crate::api::response::ApiResponse::success($data)
    };
}

/// 便捷宏：创建错误响应
#[macro_export]
macro_rules! errorresponse {
    ($code:expr, $message:expr) => {
        $crate::api::response::ApiResponse::error($crate::api::response::ApiError::new(
            $code, $message,
        ))
    };
    ($err:expr) => {
        $crate::api::response::ApiResponse::error($crate::api::response::ApiError::from($err))
    };
}

/// 便捷宏：创建带详情的错误响应
#[macro_export]
macro_rules! errorresponse_with_details {
    ($code:expr, $message:expr, $details:expr) => {
        $crate::api::response::ApiResponse::error($crate::api::response::ApiError::with_details(
            $code, $message, $details,
        ))
    };
}
