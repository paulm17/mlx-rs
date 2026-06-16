pub mod backend;
pub mod config;
pub mod llamacpp;
pub mod loader;
pub mod registry;
pub mod runner;
pub mod runtime;
pub mod sampler;
pub mod server;
pub mod subprocess;
pub mod types;

pub use backend::Backend;
pub use llamacpp::LlamaCppBackend;
pub use loader::resolve_model_path;
pub use registry::{create_backend, detect_format, register_safetensors_backend, ModelFormat};
pub use runner::{CompletionOptions, CompletionRequest, CompletionResponse, RunnerClient};
pub use sampler::Sampler;
pub use server::{run_server, run_server_from_toml_path, ServerConfig};
pub use types::{
    ChatMessage, EmbeddingData, EmbeddingOutput, EmbeddingUsage, GenerateOutput,
    GenerationMetrics, GenerationOptions, LoadedModelInfo, StopReason,
};

use config::LlamaCppConfig;

pub struct GenerationPipeline {
    backend: Box<dyn Backend>,
}

impl GenerationPipeline {
    pub fn new(model_path: &str, config: LlamaCppConfig) -> anyhow::Result<Self> {
        let backend = create_backend(model_path, config)?;
        Ok(Self { backend })
    }

    pub fn generate(&mut self, prompt: &str, options: &GenerationOptions) -> anyhow::Result<GenerateOutput> {
        self.backend.generate(prompt, options)
    }

    pub fn generate_stream<F>(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
        on_token: F,
    ) -> anyhow::Result<GenerationMetrics>
    where
        F: FnMut(&str) -> bool + Send + 'static,
    {
        self.backend.generate_stream(prompt, options, Box::new(on_token))
    }

    pub fn apply_chat_template(&self, messages: &[ChatMessage]) -> anyhow::Result<String> {
        self.backend.apply_chat_template(messages)
    }

    pub fn backend(&self) -> &dyn Backend {
        &*self.backend
    }

    pub fn backend_mut(&mut self) -> &mut dyn Backend {
        &mut *self.backend
    }
}
