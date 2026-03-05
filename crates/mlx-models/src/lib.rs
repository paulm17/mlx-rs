//! mlx-models: Model implementations for the MLX Rust framework.
//!
//! Each model follows candle's pattern: Config → Layer → Block → Model.
//! Models are loaded via `VarBuilder` with `pp()` prefix scoping.

pub mod llama;
pub mod lfm2_moe;
pub mod qwen3;
pub mod qwen3_5;
pub mod qwen3_moe;

pub use llama::{Llama, LlamaConfig};
pub use lfm2_moe::{Lfm2Moe, Lfm2MoeConfig};
pub use qwen3::{Qwen3, Qwen3Config};
pub use qwen3_5::{Qwen35, Qwen35Config};
pub use qwen3_moe::{Qwen3Moe, Qwen3MoeConfig};
