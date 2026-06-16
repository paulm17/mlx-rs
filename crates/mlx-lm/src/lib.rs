pub mod config;
pub mod loader;
pub mod runtime;
pub mod sampler;
pub mod server;
pub mod types;

pub use loader::resolve_model_path;
pub use sampler::Sampler;
pub use server::{run_server, run_server_from_toml_path, ServerConfig};
pub use types::{
    ChatMessage, EmbeddingData, EmbeddingOutput, EmbeddingUsage, GenerateOutput,
    GenerationMetrics, GenerationOptions, LoadedModelInfo, StopReason,
};

pub struct GenerationPipeline;

impl GenerationPipeline {
    pub fn new() -> anyhow::Result<Self> {
        anyhow::bail!("GenerationPipeline not yet implemented; llama.cpp backend coming in milestone 1.7")
    }
}
