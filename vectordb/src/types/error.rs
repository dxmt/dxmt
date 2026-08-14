//! This module re-exports the error types from the main error module
//! to maintain backward compatibility while using a unified error system

// Re-export main error types
pub use crate::error::{Error, Result};

// Compatibility types
pub type ErrorCode = String;
pub type ErrorContext = String;

// Compatibility constructors
impl Error {
    pub fn storage<T: Into<String>>(message: T) -> Self {
        Self::Storage(message.into())
    }

    pub fn config<T: Into<String>>(message: T) -> Self {
        Self::InvalidConfig(message.into())
    }

    pub fn internal<T: Into<String>>(message: T) -> Self {
        Self::Internal(message.into())
    }

    pub fn validation<T: Into<String>>(message: T) -> Self {
        Self::InvalidInput(message.into())
    }

    pub fn cluster<T: Into<String>>(message: T) -> Self {
        Self::Internal(message.into())
    }
}
