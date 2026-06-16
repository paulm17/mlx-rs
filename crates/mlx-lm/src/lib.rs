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

use config::LlamaCppConfig;
use runtime::Runtime;

pub struct GenerationPipeline {
    runtime: Runtime,
}

impl GenerationPipeline {
    pub fn new(model_path: &str, config: LlamaCppConfig) -> anyhow::Result<Self> {
        let runtime = Runtime::new(model_path, config)?;
        Ok(Self { runtime })
    }

    pub fn generate(&mut self, prompt: &str, options: &GenerationOptions) -> anyhow::Result<GenerateOutput> {
        self.runtime.generate(prompt, options)
    }

    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    pub fn runtime_mut(&mut self) -> &mut Runtime {
        &mut self.runtime
    }
}
