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

    fn embedding_token_limit(&self) -> Option<usize> {
        Some(self.runtime.embedding_token_limit())
    }

    fn embedding_batch_token_limit(&self) -> Option<usize> {
        Some(self.runtime.embedding_batch_token_limit())
    }

    fn embedding_sequence_limit(&self) -> Option<usize> {
        Some(self.runtime.embedding_sequence_limit())
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

    fn generate_chat_stream_output(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
        template: &AppliedChatTemplate,
        on_delta: Box<dyn FnMut(&str) -> bool + Send>,
    ) -> Result<GenerateOutput> {
        self.runtime
            .generate_chat_with_callback_output(prompt, options, template, on_delta)
    }

    fn embed(&mut self, text: &str) -> Result<EmbeddingOutput> {
        let embedding = self.runtime.embed(text)?;
        let token_count = self
            .runtime
            .tokenize(text, true)?
            .len()
            .min(self.runtime.embedding_token_limit());

        Ok(EmbeddingOutput {
            object: "list".to_string(),
            data: vec![EmbeddingData {
                object: "embedding".to_string(),
                index: 0,
                embedding,
            }],
            usage: EmbeddingUsage {
                prompt_tokens: token_count,
                total_tokens: token_count,
            },
        })
    }

    fn embed_batch(&mut self, texts: &[String]) -> Result<EmbeddingOutput> {
        let embeddings = self.runtime.embed_batch(texts)?;
        let token_limit = self.runtime.embedding_token_limit();
        let total_tokens = texts
            .iter()
            .map(|text| {
                self.runtime
                    .tokenize(text, true)
                    .map(|tokens| tokens.len().min(token_limit))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .sum();
        Ok(EmbeddingOutput {
            object: "list".to_string(),
            data: embeddings
                .into_iter()
                .enumerate()
                .map(|(index, embedding)| EmbeddingData {
                    object: "embedding".to_string(),
                    index,
                    embedding,
                })
                .collect(),
            usage: EmbeddingUsage {
                prompt_tokens: total_tokens,
                total_tokens,
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
    fn test_real_embedding_32_sequence_boundary_and_usage() {
        let model_path = match std::env::var("MLX_RS_TEST_EMBED_GGUF") {
            Ok(path) => path,
            Err(_) => {
                eprintln!("Skipping 32-sequence boundary test: MLX_RS_TEST_EMBED_GGUF not set");
                return;
            }
        };
        let mut backend = LlamaCppBackend::new(
            &model_path,
            LlamaCppConfig {
                n_ctx: Some(4096),
                n_ubatch: Some(512),
                n_seq_max: Some(32),
                n_gpu_layers: Some(0),
                embedding: Some(true),
                pooling: Some("cls".to_string()),
                ..Default::default()
            },
        )
        .expect("Failed to load real embedding model");

        // BERT WordPiece encodes this as [CLS] plus one `word` token per
        // repetition: 510 repetitions make an exact 512-token request,
        // including the BOS token.
        let long = "word ".repeat(510);
        let long_tokens = backend
            .tokenize(&long, true)
            .expect("long input tokenization failed")
            .len();
        assert!(
            long_tokens == 512,
            "boundary input encoded to {long_tokens} tokens, expected exactly 512"
        );
        assert_eq!(backend.runtime().embedding_token_limit(), 128);

        let texts = vec![long, "short mixed sequence".to_string()];
        let output = backend
            .embed_batch(&texts)
            .expect("512-token mixed embedding batch must remain safe");
        assert_eq!(output.data.len(), 2);
        assert_eq!(output.data[0].index, 0);
        assert_eq!(output.data[1].index, 1);
        assert_eq!(output.data[0].embedding.len(), 1024);
        assert_eq!(output.data[1].embedding.len(), 1024);
        let short_tokens = backend
            .tokenize(&texts[1], true)
            .expect("short input tokenization failed")
            .len();
        assert_eq!(
            output.usage.total_tokens,
            128 + short_tokens.min(backend.runtime().embedding_token_limit())
        );
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
