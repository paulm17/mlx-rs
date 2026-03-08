//! mlx-lm: Language model generation pipeline for the MLX Rust framework.
//!
//! Provides tokenization, sampling, model loading, and text generation.

pub mod chat_template;
pub mod generate;
pub mod loader;
pub mod sampler;
pub mod server;
pub mod tokenizer;

pub use chat_template::{
    ChatTemplate, ChatTemplateError, ChatTemplateOptions, Conversation, Message,
};
pub use generate::{
    CausalLM, EmbeddingModel, EmbeddingPooling, GenerationMetrics, GenerationPipeline, ModelRuntime,
};
pub use loader::load_model;
pub use sampler::Sampler;
pub use server::{run_server, run_server_from_toml_path, ServerConfig};
pub use tokenizer::Tokenizer;
