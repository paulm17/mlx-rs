use anyhow::Result;

use crate::backend::Backend;
use crate::config::LlamaCppConfig;
use crate::runtime::Runtime;
use crate::types::{
    AppliedChatTemplate, ChatMessage, ChatTemplateOptions, EmbeddingData, EmbeddingOutput,
    EmbeddingUsage, GenerateOutput, GenerationMetrics, GenerationOptions, LoadedModelInfo,
};

pub struct LlamaCppBackend {
    runtime: Runtime,
}

impl LlamaCppBackend {
    pub fn new(model_path: &str, config: LlamaCppConfig) -> Result<Self> {
        let runtime = Runtime::new(model_path, config)?;
        Ok(Self { runtime })
    }

    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    pub fn runtime_mut(&mut self) -> &mut Runtime {
        &mut self.runtime
    }
}

impl Backend for LlamaCppBackend {
    fn model_info(&self) -> LoadedModelInfo {
        self.runtime.model_info()
    }

    fn tokenize(&self, text: &str, add_bos: bool) -> Result<Vec<i32>> {
        self.runtime.tokenize(text, add_bos)
    }

    fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        self.runtime.detokenize(tokens)
    }

    fn detokenize_piece(&self, token_id: i32) -> Result<String> {
        self.runtime.detokenize_piece(token_id)
    }

    fn is_eog(&self, token_id: i32) -> bool {
        self.runtime.is_eog(token_id)
    }

    fn token_eos(&self) -> i32 {
        self.runtime.token_eos()
    }

    fn embeddings_enabled(&self) -> bool {
        self.runtime.embeddings_enabled()
    }

    fn supports_chat_template(&self) -> bool {
        true
    }

    fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
        self.runtime.apply_chat_template(messages)
    }

    fn apply_chat_template_with_options(
        &self,
        messages: &[ChatMessage],
        options: &ChatTemplateOptions,
    ) -> Result<AppliedChatTemplate> {
        self.runtime
            .apply_chat_template_with_options(messages, options)
    }

    fn parse_chat_response(
        &self,
        template: &AppliedChatTemplate,
        text: &str,
        is_partial: bool,
    ) -> Result<String> {
        self.runtime.parse_chat_response(template, text, is_partial)
    }

    fn generate(&mut self, prompt: &str, options: &GenerationOptions) -> Result<GenerateOutput> {
        self.runtime.generate(prompt, options)
    }

    fn generate_stream(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
        mut on_token: Box<dyn FnMut(&str) -> bool + Send>,
    ) -> Result<GenerationMetrics> {
        self.runtime
            .generate_with_callback(prompt, options, move |piece| on_token(piece))
    }

    fn generate_stream_output(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
        mut on_token: Box<dyn FnMut(&str) -> bool + Send>,
    ) -> Result<GenerateOutput> {
        self.runtime
            .generate_with_callback_output(prompt, options, move |piece| on_token(piece))
    }

    fn embed(&mut self, text: &str) -> Result<EmbeddingOutput> {
        let embedding = self.runtime.embed(text)?;

        Ok(EmbeddingOutput {
            object: "list".to_string(),
            data: vec![EmbeddingData {
                object: "embedding".to_string(),
                index: 0,
                embedding,
            }],
            usage: EmbeddingUsage {
                prompt_tokens: self.runtime.tokenize(text, true)?.len(),
                total_tokens: self.runtime.tokenize(text, true)?.len(),
            },
        })
    }

    fn memory_info(&self) -> Option<String> {
        Some(format!(
            "context_length={}, embedding_dim={}, vocab_size={}",
            self.runtime.context_length(),
            self.runtime.embedding_dimension(),
            self.runtime.vocab_size(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_backend() -> Option<LlamaCppBackend> {
        let model_path = match std::env::var("MLX_RS_TEST_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("Skipping backend test: MLX_RS_TEST_GGUF not set");
                return None;
            }
        };
        LlamaCppBackend::new(
            &model_path,
            LlamaCppConfig {
                n_ctx: Some(512),
                n_gpu_layers: Some(0),
                ..Default::default()
            },
        )
        .ok()
    }

    #[test]
    fn test_backend_model_info() {
        let backend = match test_backend() {
            Some(b) => b,
            None => return,
        };
        let info = backend.model_info();
        assert!(info.context_length.is_some());
        assert!(info.embedding_dimension.is_some());
        assert!(info.vocab_size.is_some());
    }

    #[test]
    fn test_backend_tokenize_detokenize() {
        let backend = match test_backend() {
            Some(b) => b,
            None => return,
        };
        let tokens = backend.tokenize("Hello, world!", false).unwrap();
        assert!(!tokens.is_empty());
        let text = backend.detokenize(&tokens).unwrap();
        assert_eq!(text, "Hello, world!");
    }

    #[test]
    fn test_backend_eog() {
        let backend = match test_backend() {
            Some(b) => b,
            None => return,
        };
        let eos = backend.token_eos();
        assert!(backend.is_eog(eos));
        assert!(!backend.is_eog(0));
    }

    #[test]
    fn test_backend_chat_template() {
        let backend = match test_backend() {
            Some(b) => b,
            None => return,
        };
        assert!(backend.supports_chat_template());
        let result = backend.apply_chat_template(&[ChatMessage::user("Hello")]);
        assert!(result.is_ok());
        assert!(!result.unwrap().trim().is_empty());
    }

    #[test]
    fn test_backend_generate() {
        let mut backend = match test_backend() {
            Some(b) => b,
            None => return,
        };
        let options = GenerationOptions {
            max_tokens: Some(8),
            temperature: 0.0,
            ..Default::default()
        };
        let output = backend.generate("Hello", &options).unwrap();
        assert!(output.metrics.prompt_tokens > 0);
    }

    #[test]
    fn test_backend_generate_stream() {
        let mut backend = match test_backend() {
            Some(b) => b,
            None => return,
        };
        let options = GenerationOptions {
            max_tokens: Some(8),
            temperature: 0.0,
            ..Default::default()
        };
        let chunks: std::sync::Arc<std::sync::Mutex<Vec<String>>> =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let chunks_clone = chunks.clone();
        let metrics = backend
            .generate_stream(
                "Hello",
                &options,
                Box::new(move |piece| {
                    chunks_clone.lock().unwrap().push(piece.to_string());
                    true
                }),
            )
            .unwrap();
        assert!(metrics.prompt_tokens > 0);
    }

    #[test]
    fn test_backend_memory_info() {
        let backend = match test_backend() {
            Some(b) => b,
            None => return,
        };
        let info = backend.memory_info();
        assert!(info.is_some());
        assert!(info.unwrap().contains("context_length"));
    }
}
