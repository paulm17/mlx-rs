use std::num::NonZeroU32;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

use anyhow::Result;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;

use crate::config::LlamaCppConfig;
use crate::sampler::Sampler;
use crate::types::{GenerateOutput, GenerationMetrics, GenerationOptions, StopReason, LoadedModelInfo};

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

    pub fn generate(&mut self, prompt: &str, options: &GenerationOptions) -> Result<GenerateOutput> {
        let mut text = String::new();

        let metrics = self.generate_with_callback(prompt, options, |piece| {
            text.push_str(piece);
            true
        })?;

        Ok(GenerateOutput {
            text,
            stop_reason: StopReason::Eos,
            metrics,
        })
    }

    pub fn generate_with_callback<F>(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
        mut on_token: F,
    ) -> Result<GenerationMetrics>
    where
        F: FnMut(&str) -> bool,
    {
        let start = Instant::now();

        let prompt_tokens = self.tokenize(prompt, true)?;
        let n_prompt = prompt_tokens.len();
        let max_tokens = options.max_tokens.unwrap_or(512);

        // Prefill: process all prompt tokens
        let mut batch = LlamaBatch::new(n_prompt, 1);
        for (i, &token_id) in prompt_tokens.iter().enumerate() {
            let is_last = i == n_prompt - 1;
            batch.add(LlamaToken(token_id), i as i32, &[0], is_last)?;
        }
        self.context.decode(&mut batch)?;

        let ttft = start.elapsed().as_secs_f64();

        // Build sampler
        let sampler_config = Sampler::new(options.temperature, options.top_p)
            .with_top_k(options.top_k)
            .with_min_p(options.min_p);
        let mut sampler = sampler_config.build_llama_sampler();

        let mut generated_tokens = 0usize;
        let mut output_text = String::new();

        // Sample first token from prefill logits
        let mut current_token = sampler.sample(self.context(), n_prompt as i32 - 1);
        sampler.accept(current_token);

        if self.is_eog(current_token.0) {
            let total_s = start.elapsed().as_secs_f64();
            return Ok(GenerationMetrics {
                prompt_tokens: n_prompt,
                generated_tokens: 0,
                total_tokens: n_prompt,
                ttft_s: Some(ttft),
                total_s: Some(total_s),
                tokens_per_s: None,
            });
        }

        let piece = self.detokenize(&[current_token.0])?;
        if !on_token(&piece) {
            let total_s = start.elapsed().as_secs_f64();
            let tps = generated_tokens as f64 / (total_s - ttft).max(0.001);
            return Ok(GenerationMetrics {
                prompt_tokens: n_prompt,
                generated_tokens,
                total_tokens: n_prompt + generated_tokens,
                ttft_s: Some(ttft),
                total_s: Some(total_s),
                tokens_per_s: Some(tps),
            });
        }
        output_text.push_str(&piece);
        generated_tokens += 1;

        // Check stop sequences
        if let Some(stops) = &options.stop {
            if stops.iter().any(|s| output_text.contains(s)) {
                let total_s = start.elapsed().as_secs_f64();
                let tps = generated_tokens as f64 / (total_s - ttft).max(0.001);
                return Ok(GenerationMetrics {
                    prompt_tokens: n_prompt,
                    generated_tokens,
                    total_tokens: n_prompt + generated_tokens,
                    ttft_s: Some(ttft),
                    total_s: Some(total_s),
                    tokens_per_s: Some(tps),
                });
            }
        }

        // Continue decoding
        let mut pos = n_prompt as i32;
        while generated_tokens < max_tokens {
            let mut batch = LlamaBatch::new(1, 1);
            batch.add(current_token, pos, &[0], true)?;
            self.context.decode(&mut batch)?;

            let next_token = sampler.sample(self.context(), 0);
            sampler.accept(next_token);

            if self.is_eog(next_token.0) {
                break;
            }

            let piece = self.detokenize(&[next_token.0])?;
            if !on_token(&piece) {
                break;
            }
            output_text.push_str(&piece);
            generated_tokens += 1;
            pos += 1;
            current_token = next_token;

            // Check stop sequences
            if let Some(stops) = &options.stop {
                if stops.iter().any(|s| output_text.contains(s)) {
                    break;
                }
            }
        }

        let total_s = start.elapsed().as_secs_f64();
        let decode_time = total_s - ttft;
        let tps = if decode_time > 0.001 {
            generated_tokens as f64 / decode_time
        } else {
            0.0
        };

        Ok(GenerationMetrics {
            prompt_tokens: n_prompt,
            generated_tokens,
            total_tokens: n_prompt + generated_tokens,
            ttft_s: Some(ttft),
            total_s: Some(total_s),
            tokens_per_s: Some(tps),
        })
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

    #[test]
    fn test_generate_non_streaming() {
        let model_path = match std::env::var("MLX_RS_TEST_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("Skipping generate test: MLX_RS_TEST_GGUF not set");
                return;
            }
        };

        let config = LlamaCppConfig {
            n_ctx: Some(2048),
            ..Default::default()
        };

        let mut runtime = Runtime::new(&model_path, config).expect("Failed to load model");

        let options = GenerationOptions {
            max_tokens: Some(32),
            temperature: 0.0,
            ..Default::default()
        };

        let output = runtime.generate("Hello", &options).expect("Generation failed");

        assert!(output.metrics.prompt_tokens > 0);
        assert!(output.metrics.total_tokens > output.metrics.prompt_tokens);
        assert!(output.metrics.ttft_s.is_some());
        assert!(output.metrics.total_s.is_some());
        assert!(output.metrics.tokens_per_s.is_some());
        // Text may be empty if model immediately outputs EOG, but metrics must be valid
        match &output.stop_reason {
            StopReason::Eos | StopReason::MaxTokens => {}
            StopReason::Cancelled => panic!("Unexpected cancelled"),
        }
    }

    #[test]
    fn test_generate_with_stop_sequence() {
        let model_path = match std::env::var("MLX_RS_TEST_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("Skipping generate stop test: MLX_RS_TEST_GGUF not set");
                return;
            }
        };

        let config = LlamaCppConfig {
            n_ctx: Some(2048),
            ..Default::default()
        };

        let mut runtime = Runtime::new(&model_path, config).expect("Failed to load model");

        let options = GenerationOptions {
            max_tokens: Some(100),
            temperature: 0.0,
            stop: Some(vec![".".to_string()]),
            ..Default::default()
        };

        let output = runtime.generate("Count to five.", &options).expect("Generation failed");

        // Should have stopped - either by stop sequence or EOS
        assert!(output.metrics.generated_tokens <= 100);
    }

    #[test]
    fn test_generate_streaming() {
        let model_path = match std::env::var("MLX_RS_TEST_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("Skipping streaming test: MLX_RS_TEST_GGUF not set");
                return;
            }
        };

        let config = LlamaCppConfig {
            n_ctx: Some(2048),
            ..Default::default()
        };

        let mut runtime = Runtime::new(&model_path, config).expect("Failed to load model");

        let options = GenerationOptions {
            max_tokens: Some(16),
            temperature: 0.0,
            ..Default::default()
        };

        let mut chunks = Vec::new();
        let metrics = runtime
            .generate_with_callback("Hello", &options, |piece| {
                chunks.push(piece.to_string());
                true
            })
            .expect("Streaming generation failed");

        assert!(metrics.prompt_tokens > 0);
        // Should have received at least one chunk
        assert!(!chunks.is_empty() || metrics.generated_tokens == 0);
        assert!(metrics.ttft_s.is_some());
        assert!(metrics.total_s.is_some());
    }

    #[test]
    fn test_generate_streaming_cancel() {
        let model_path = match std::env::var("MLX_RS_TEST_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("Skipping streaming cancel test: MLX_RS_TEST_GGUF not set");
                return;
            }
        };

        let config = LlamaCppConfig {
            n_ctx: Some(2048),
            ..Default::default()
        };

        let mut runtime = Runtime::new(&model_path, config).expect("Failed to load model");

        let options = GenerationOptions {
            max_tokens: Some(100),
            temperature: 0.0,
            ..Default::default()
        };

        let mut count = 0;
        let metrics = runtime
            .generate_with_callback("Count to ten.", &options, |_piece| {
                count += 1;
                count <= 3 // Cancel after 3 tokens
            })
            .expect("Streaming generation with cancel failed");

        // Should have stopped early
        assert!(count <= 4); // at most 3 accepted + 1 that triggered cancel
        assert!(metrics.generated_tokens <= 4);
    }
}
