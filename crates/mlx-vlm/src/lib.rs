//! mlx-vlm: Vision-language model pipeline for the MLX Rust framework.
//!
//! Provides model loading, image preprocessing, and multimodal generation
//! for Gemma4 and future VLM architectures.

pub mod generate;
pub mod loader;
pub mod processing;

pub use loader::{load_gemma4_vlm, VlmComponents};
pub use generate::{VlmGenerationPipeline, VlmGenerateOptions};
pub use processing::{Gemma4ImageProcessor, ProcessedImage};
