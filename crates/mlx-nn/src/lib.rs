//! mlx-nn: Neural network building blocks for the MLX Rust framework.
//!
//! Provides reusable, composable layers that implement the `Module` trait:
//! - [`Linear`] — fully-connected layer with quantization support
//! - [`Embedding`] — token embedding lookup
//! - [`RmsNorm`] — RMS normalization
//! - [`RoPE`] — rotary position embedding
//! - [`KvCache`] — key-value cache for autoregressive decoding
//! - [`Activation`] — activation function enum
//! - [`VarBuilder`] — hierarchical weight loading from safetensors

pub mod activation;
pub mod attention;
pub mod embedding;
pub mod kv_cache;
pub mod layer_norm;
pub mod linear;
pub mod rms_norm;
pub mod rope;
pub mod var_builder;

// Re-exports
pub use activation::Activation;
pub use attention::{causal_attention, repeat_kv, scaled_dot_product_attention};
pub use embedding::Embedding;
pub use kv_cache::{kv_cache_stats, reset_kv_cache_stats, KvCache, KvCacheStats};
pub use layer_norm::LayerNorm;
pub use linear::{Linear, QuantConfig};
pub use rms_norm::RmsNorm;
pub use rope::{RoPE, RopeScaling};
pub use var_builder::VarBuilder;

// Re-export Module from mlx-core
pub use mlx_core::{Module, ModuleT};
