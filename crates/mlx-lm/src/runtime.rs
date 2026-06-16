use std::num::NonZeroU32;
use std::path::Path;
use std::sync::OnceLock;

use anyhow::Result;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};

use crate::config::LlamaCppConfig;
use crate::types::LoadedModelInfo;

fn ensure_backend() -> &'static LlamaBackend {
    static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
    BACKEND.get_or_init(|| LlamaBackend::init().expect("Failed to initialize llama.cpp backend"))
}

pub struct Runtime {
    model: LlamaModel,
    context: LlamaContext<'static>,
    model_path: String,
    embeddings_enabled: bool,
}

impl Runtime {
    pub fn new(model_path: &str, config: LlamaCppConfig) -> Result<Self> {
        let path = Path::new(model_path);
        let backend = ensure_backend();

        // Model params
        let mut model_params = LlamaModelParams::default();
        if let Some(n_gpu_layers) = config.n_gpu_layers {
            model_params = model_params.with_n_gpu_layers(n_gpu_layers);
        }
        if let Some(use_mmap) = config.use_mmap {
            model_params = model_params.with_use_mmap(use_mmap);
        }
        if let Some(use_mlock) = config.use_mlock {
            model_params = model_params.with_use_mlock(use_mlock);
        }

        let model = LlamaModel::load_from_file(backend, path, &model_params)
            .map_err(|e| anyhow::anyhow!("Failed to load model from {}: {}", model_path, e))?;

        // Context params
        let embeddings_enabled = config.embedding.unwrap_or(false);
        let mut ctx_params = LlamaContextParams::default()
            .with_embeddings(embeddings_enabled);

        if let Some(n_ctx) = config.n_ctx {
            ctx_params = ctx_params.with_n_ctx(NonZeroU32::new(n_ctx));
        }
        if let Some(n_batch) = config.n_batch {
            ctx_params = ctx_params.with_n_batch(n_batch);
        }
        if let Some(n_ubatch) = config.n_ubatch {
            ctx_params = ctx_params.with_n_ubatch(n_ubatch);
        }
        if let Some(n_threads) = config.n_threads {
            ctx_params = ctx_params.with_n_threads(n_threads as i32);
        }
        if let Some(n_threads_batch) = config.n_threads_batch {
            ctx_params = ctx_params.with_n_threads_batch(n_threads_batch as i32);
        }
        if let Some(pooling) = &config.pooling {
            let pooling_type = match pooling.as_str() {
                "mean" => LlamaPoolingType::Mean,
                "cls" => LlamaPoolingType::Cls,
                "last" => LlamaPoolingType::Last,
                "none" => LlamaPoolingType::None,
                _ => LlamaPoolingType::Unspecified,
            };
            ctx_params = ctx_params.with_pooling_type(pooling_type);
        }

        // Safety: model lives as long as this Runtime, and context borrows from model.
        // We transmute the lifetime to 'static, which is safe because we never move
        // the model out of this struct.
        let model_ref: &'static LlamaModel = unsafe { std::mem::transmute(&model) };
        let context = model_ref
            .new_context(backend, ctx_params)
            .map_err(|e| anyhow::anyhow!("Failed to create context: {}", e))?;

        Ok(Self {
            model,
            context,
            model_path: model_path.to_string(),
            embeddings_enabled,
        })
    }

    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    pub fn context_length(&self) -> u32 {
        self.context.n_ctx()
    }

    pub fn embedding_dimension(&self) -> i32 {
        self.model.n_embd()
    }

    pub fn vocab_size(&self) -> i32 {
        self.model.n_vocab()
    }

    pub fn embeddings_enabled(&self) -> bool {
        self.embeddings_enabled
    }

    pub fn n_ctx_train(&self) -> u32 {
        self.model.n_ctx_train()
    }

    pub fn model_info(&self) -> LoadedModelInfo {
        LoadedModelInfo {
            model_path: self.model_path.clone(),
            context_length: Some(self.context.n_ctx() as usize),
            embedding_dimension: Some(self.model.n_embd() as usize),
            vocab_size: Some(self.model.n_vocab() as usize),
        }
    }

    pub fn model(&self) -> &LlamaModel {
        &self.model
    }

    pub fn context(&mut self) -> &mut LlamaContext<'static> {
        &mut self.context
    }

    pub fn tokenize(&self, text: &str, add_bos: bool) -> Result<Vec<i32>> {
        let bos = if add_bos { AddBos::Always } else { AddBos::Never };
        let tokens = self.model
            .str_to_token(text, bos)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        Ok(tokens.into_iter().map(|t| t.0).collect())
    }

    pub fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut result = String::new();
        for &token_id in tokens {
            let token = llama_cpp_2::token::LlamaToken(token_id);
            let piece = self.model
                .token_to_piece(token, &mut decoder, false, None)
                .map_err(|e| anyhow::anyhow!("Detokenization failed for token {}: {}", token_id, e))?;
            result.push_str(&piece);
        }
        Ok(result)
    }

    pub fn is_eog(&self, token_id: i32) -> bool {
        let token = llama_cpp_2::token::LlamaToken(token_id);
        self.model.is_eog_token(token)
    }

    pub fn token_eos(&self) -> i32 {
        self.model.token_eos().0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_not_implemented_without_model() {
        let config = LlamaCppConfig::default();
        // LlamaModel::load_from_file panics on nonexistent paths, so use catch_unwind
        let result = std::panic::catch_unwind(|| Runtime::new("/nonexistent/model.gguf", config));
        assert!(result.is_err());
    }

    #[test]
    fn test_backend_init() {
        // Should not panic
        let _backend = ensure_backend();
    }

    #[test]
    fn test_runtime_with_real_model() {
        let model_path = match std::env::var("MLX_RS_TEST_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("Skipping real model test: MLX_RS_TEST_GGUF not set");
                return;
            }
        };

        let config = LlamaCppConfig {
            n_ctx: Some(512),
            n_gpu_layers: Some(0), // CPU only for test
            ..Default::default()
        };

        let runtime = Runtime::new(&model_path, config).expect("Failed to load model");
        assert!(runtime.context_length() > 0);
        assert!(runtime.embedding_dimension() > 0);
        assert!(runtime.vocab_size() > 0);
        assert_eq!(runtime.model_path(), model_path);
        assert!(!runtime.embeddings_enabled());

        let info = runtime.model_info();
        assert_eq!(info.model_path, model_path);
        assert!(info.context_length.is_some());
        assert!(info.embedding_dimension.is_some());
        assert!(info.vocab_size.is_some());
    }

    #[test]
    fn test_runtime_with_embeddings() {
        let model_path = match std::env::var("MLX_RS_TEST_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("Skipping embeddings test: MLX_RS_TEST_GGUF not set");
                return;
            }
        };

        let config = LlamaCppConfig {
            n_ctx: Some(512),
            embedding: Some(true),
            ..Default::default()
        };

        let runtime = Runtime::new(&model_path, config).expect("Failed to load model");
        assert!(runtime.embeddings_enabled());
    }

    #[test]
    fn test_tokenize_detokenize_roundtrip() {
        let model_path = match std::env::var("MLX_RS_TEST_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("Skipping tokenize test: MLX_RS_TEST_GGUF not set");
                return;
            }
        };

        let config = LlamaCppConfig {
            n_ctx: Some(512),
            ..Default::default()
        };

        let runtime = Runtime::new(&model_path, config).expect("Failed to load model");
        let text = "Hello, world!";
        let tokens = runtime.tokenize(text, false).expect("Tokenization failed");
        assert!(!tokens.is_empty());

        let detokenized = runtime.detokenize(&tokens).expect("Detokenization failed");
        assert_eq!(detokenized, text);
    }

    #[test]
    fn test_tokenize_with_bos() {
        let model_path = match std::env::var("MLX_RS_TEST_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("Skipping tokenize test: MLX_RS_TEST_GGUF not set");
                return;
            }
        };

        let config = LlamaCppConfig {
            n_ctx: Some(512),
            ..Default::default()
        };

        let runtime = Runtime::new(&model_path, config).expect("Failed to load model");
        let text = "Hello";
        let tokens_no_bos = runtime.tokenize(text, false).expect("Tokenization failed");
        let tokens_with_bos = runtime.tokenize(text, true).expect("Tokenization failed");
        assert!(tokens_with_bos.len() > tokens_no_bos.len());
    }

    #[test]
    fn test_eog_detection() {
        let model_path = match std::env::var("MLX_RS_TEST_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("Skipping EOG test: MLX_RS_TEST_GGUF not set");
                return;
            }
        };

        let config = LlamaCppConfig {
            n_ctx: Some(512),
            ..Default::default()
        };

        let runtime = Runtime::new(&model_path, config).expect("Failed to load model");
        let eos = runtime.token_eos();
        assert!(runtime.is_eog(eos));
        // A regular token should not be EOG
        assert!(!runtime.is_eog(0));
    }
}
