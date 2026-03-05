use std::fmt;

/// Error types for the MLX Rust framework.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("MLX error: {0}")]
    Mlx(String),

    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    #[error("DType mismatch: expected {expected:?}, got {got:?}")]
    DTypeMismatch {
        expected: crate::DType,
        got: crate::DType,
    },

    #[error("Null pointer returned from mlx-c")]
    NullPointer,

    #[error("{0}")]
    Message(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Safetensors error: {0}")]
    SafeTensors(String),
}

impl Error {
    pub fn msg(s: impl fmt::Display) -> Self {
        Error::Message(s.to_string())
    }
}

/// A specialized `Result` type for MLX operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Convenience macro for creating an error with a formatted message.
#[macro_export]
macro_rules! bail {
    ($($arg:tt)*) => {
        return Err($crate::Error::Message(format!($($arg)*)))
    };
}
