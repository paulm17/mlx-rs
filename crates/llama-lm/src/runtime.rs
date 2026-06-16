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
use crate::types::{
    GenerateOutput, GenerationMetrics, GenerationOptions, LoadedModelInfo, StopReason,
};

fn ensure_backend() -> &'static LlamaBackend {
    static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
    BACKEND.get_or_init(|| LlamaBackend::init().expect("Failed to initialize llama.cpp backend"))
}

pub struct Runtime {
    context: LlamaContext<'static>,
    model: Box<LlamaModel>,
    model_path: String,
    embeddings_enabled: bool,
}

// Safety: Runtime is only used within a Mutex on a single-threaded tokio runtime.
// The LlamaContext pointers are never accessed concurrently.
unsafe impl Send for Runtime {}

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

        let model = Box::new(
            LlamaModel::load_from_file(backend, path, &model_params)
                .map_err(|e| anyhow::anyhow!("Failed to load model from {}: {}", model_path, e))?,
        );

        // Context params
        let embeddings_enabled = config.embedding.unwrap_or(false);
        let mut ctx_params = LlamaContextParams::default().with_embeddings(embeddings_enabled);

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

        // Safety: the model allocation is stable behind Box, and Runtime declares
        // context before model so the context is dropped first.
        let model_ref: &'static LlamaModel = unsafe { std::mem::transmute(&*model) };
        let context = model_ref
            .new_context(backend, ctx_params)
            .map_err(|e| anyhow::anyhow!("Failed to create context: {}", e))?;

        Ok(Self {
            context,
            model,
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
        let bos = if add_bos {
            AddBos::Always
        } else {
            AddBos::Never
        };
        let tokens = self
            .model
            .str_to_token(text, bos)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        Ok(tokens.into_iter().map(|t| t.0).collect())
    }

    pub fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut result = String::new();
        for &token_id in tokens {
            result.push_str(&self.detokenize_piece_with_decoder(token_id, &mut decoder)?);
        }
        Ok(result)
    }

    pub fn detokenize_piece(&self, token_id: i32) -> Result<String> {
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        self.detokenize_piece_with_decoder(token_id, &mut decoder)
    }

    fn detokenize_piece_with_decoder(
        &self,
        token_id: i32,
        decoder: &mut encoding_rs::Decoder,
    ) -> Result<String> {
        let token = llama_cpp_2::token::LlamaToken(token_id);
        self.model
            .token_to_piece(token, decoder, false, None)
            .map_err(|e| anyhow::anyhow!("Detokenization failed for token {}: {}", token_id, e))
    }

    pub fn is_eog(&self, token_id: i32) -> bool {
        let token = llama_cpp_2::token::LlamaToken(token_id);
        self.model.is_eog_token(token)
    }

    pub fn token_eos(&self) -> i32 {
        self.model.token_eos().0
    }

    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        if !self.embeddings_enabled {
            anyhow::bail!("Embeddings not enabled for this model");
        }

        let tokens = self.tokenize(text, true)?;
        let n_tokens = tokens.len();

        self.context.clear_kv_cache();

        let mut batch = LlamaBatch::new(n_tokens, 1);
        for (i, &token_id) in tokens.iter().enumerate() {
            let is_last = i == n_tokens - 1;
            batch.add(LlamaToken(token_id), i as i32, &[0], is_last)?;
        }
        self.context.decode(&mut batch)?;

        let embedding = self
            .context
            .embeddings_seq_ith(0)
            .map_err(|e| anyhow::anyhow!("Failed to get embeddings: {}", e))?;

        // Normalize the embedding vector
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            Ok(embedding.iter().map(|x| x / norm).collect())
        } else {
            Ok(embedding.to_vec())
        }
    }

    pub fn apply_chat_template(&self, messages: &[crate::types::ChatMessage]) -> Result<String> {
        use llama_cpp_2::model::LlamaChatMessage;

        // Try to get the native chat template from the model
        let template = self.model.chat_template(None).ok();

        let chat_messages: Vec<LlamaChatMessage> = messages
            .iter()
            .map(|m| LlamaChatMessage::new(m.role.clone(), m.content.clone()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("Failed to create chat message: {}", e))?;

        let tmpl = if let Some(ref tmpl) = template {
            let result = self.model.apply_chat_template(tmpl, &chat_messages, true);
            if let Ok(result) = result {
                return Ok(result);
            }
            Some(
                tmpl.to_str()
                    .map(|s| s.to_string())
                    .unwrap_or_default(),
            )
        } else {
            None
        };

        Ok(render_fallback_chat_template(tmpl.as_deref(), messages))
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
    ) -> Result<GenerateOutput> {
        self.generate_inner(prompt, options, |_| true)
    }

    pub fn generate_with_callback<F>(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
        on_token: F,
    ) -> Result<GenerationMetrics>
    where
        F: FnMut(&str) -> bool,
    {
        Ok(self.generate_inner(prompt, options, on_token)?.metrics)
    }

    pub fn generate_with_callback_output<F>(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
        on_token: F,
    ) -> Result<GenerateOutput>
    where
        F: FnMut(&str) -> bool,
    {
        self.generate_inner(prompt, options, on_token)
    }

    fn generate_inner<F>(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
        mut on_token: F,
    ) -> Result<GenerateOutput>
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

        if max_tokens == 0 {
            return Ok(GenerateOutput {
                text: String::new(),
                stop_reason: StopReason::MaxTokens,
                metrics: self.generation_metrics(start, ttft, n_prompt, 0),
            });
        }

        // Build sampler
        let sampler_config = Sampler::new(options.temperature, options.top_p)
            .with_top_k(options.top_k)
            .with_min_p(options.min_p);
        let mut sampler = sampler_config.build_llama_sampler();

        let mut generated_tokens = 0usize;
        let mut output_text = String::new();
        let mut token_decoder = encoding_rs::UTF_8.new_decoder();

        // Sample first token from prefill logits
        let mut current_token = sampler.sample(self.context(), n_prompt as i32 - 1);
        sampler.accept(current_token);

        if self.is_eog(current_token.0) {
            return Ok(GenerateOutput {
                text: output_text,
                stop_reason: StopReason::Eos,
                metrics: self.generation_metrics(start, ttft, n_prompt, generated_tokens),
            });
        }

        let piece = self.detokenize_piece_with_decoder(current_token.0, &mut token_decoder)?;
        if !on_token(&piece) {
            return Ok(GenerateOutput {
                text: output_text,
                stop_reason: StopReason::Cancelled,
                metrics: self.generation_metrics(start, ttft, n_prompt, generated_tokens),
            });
        }
        output_text.push_str(&piece);
        generated_tokens += 1;

        // Check stop sequences
        if let Some(stops) = &options.stop {
            if stops.iter().any(|s| output_text.contains(s)) {
                return Ok(GenerateOutput {
                    text: output_text,
                    stop_reason: StopReason::Eos,
                    metrics: self.generation_metrics(start, ttft, n_prompt, generated_tokens),
                });
            }
        }

        // Continue decoding
        let mut pos = n_prompt as i32;
        let mut stop_reason = StopReason::MaxTokens;
        while generated_tokens < max_tokens {
            let mut batch = LlamaBatch::new(1, 1);
            batch.add(current_token, pos, &[0], true)?;
            self.context.decode(&mut batch)?;

            let next_token = sampler.sample(self.context(), 0);
            sampler.accept(next_token);

            if self.is_eog(next_token.0) {
                stop_reason = StopReason::Eos;
                break;
            }

            let piece = self.detokenize_piece_with_decoder(next_token.0, &mut token_decoder)?;
            if !on_token(&piece) {
                stop_reason = StopReason::Cancelled;
                break;
            }
            output_text.push_str(&piece);
            generated_tokens += 1;
            pos += 1;
            current_token = next_token;

            // Check stop sequences
            if let Some(stops) = &options.stop {
                if stops.iter().any(|s| output_text.contains(s)) {
                    stop_reason = StopReason::Eos;
                    break;
                }
            }
        }

        Ok(GenerateOutput {
            text: output_text,
            stop_reason,
            metrics: self.generation_metrics(start, ttft, n_prompt, generated_tokens),
        })
    }

    fn generation_metrics(
        &self,
        start: Instant,
        ttft: f64,
        prompt_tokens: usize,
        generated_tokens: usize,
    ) -> GenerationMetrics {
        let total_s = start.elapsed().as_secs_f64();
        let decode_time = total_s - ttft;
        let tps = if decode_time > 0.001 {
            generated_tokens as f64 / decode_time
        } else {
            0.0
        };

        GenerationMetrics {
            prompt_tokens,
            generated_tokens,
            total_tokens: prompt_tokens + generated_tokens,
            ttft_s: Some(ttft),
            total_s: Some(total_s),
            tokens_per_s: Some(tps),
        }
    }
}

fn render_fallback_chat_template(tmpl: Option<&str>, messages: &[crate::types::ChatMessage]) -> String {
    let is_gemma_family = tmpl.map_or(false, |t| t.contains("<|turn|>") || t.contains("<start_of_turn>"));

    if is_gemma_family {
        render_gemma_chat_template(messages)
    } else {
        render_llama_chat_template(messages)
    }
}

fn render_gemma_chat_template(messages: &[crate::types::ChatMessage]) -> String {
    let system_prompt = messages
        .iter()
        .filter(|msg| msg.role == "system")
        .map(|msg| msg.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    let mut prompt = String::new();
    let mut system_appended = false;

    for msg in messages {
        match msg.role.as_str() {
            "system" => {}
            "user" => {
                prompt.push_str("<|turn>user\n");
                if !system_prompt.is_empty() && !system_appended {
                    prompt.push_str(&system_prompt);
                    prompt.push_str("\n\n");
                    system_appended = true;
                }
                prompt.push_str(&msg.content);
                prompt.push_str("<turn|>\n");
            }
            "assistant" => {
                prompt.push_str("<|turn>model\n");
                prompt.push_str(&msg.content);
                prompt.push_str("<turn|>\n");
            }
            _ => {}
        }
    }

    prompt.push_str("<|turn>model\n");
    prompt
}

fn render_llama_chat_template(messages: &[crate::types::ChatMessage]) -> String {
    let system_prompt = messages
        .iter()
        .filter(|msg| msg.role == "system")
        .map(|msg| msg.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    let mut prompt = String::new();
    let mut first_user = true;

    for msg in messages {
        match msg.role.as_str() {
            "system" => {}
            "user" => {
                if first_user {
                    prompt.push_str("[INST] ");
                    if !system_prompt.is_empty() {
                        prompt.push_str("<<SYS>>\n");
                        prompt.push_str(&system_prompt);
                        prompt.push_str("\n<</SYS>>\n\n");
                    }
                    prompt.push_str(&msg.content);
                    prompt.push_str(" [/INST]");
                    first_user = false;
                } else {
                    prompt.push_str(" [INST] ");
                    prompt.push_str(&msg.content);
                    prompt.push_str(" [/INST]");
                }
            }
            "assistant" => {
                if !prompt.is_empty() {
                    prompt.push(' ');
                }
                prompt.push_str(&msg.content);
            }
            _ => {}
        }
    }

    if prompt.is_empty() && !system_prompt.is_empty() {
        format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n [/INST]", system_prompt)
    } else {
        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;

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
    fn test_fallback_chat_template_simple_user() {
        let prompt = render_fallback_chat_template(None, &[ChatMessage::user("Hello")]);
        assert_eq!(prompt, "[INST] Hello [/INST]");
    }

    #[test]
    fn test_fallback_chat_template_system_user() {
        let prompt = render_fallback_chat_template(None, &[
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello"),
        ]);
        assert!(prompt.contains("[INST]"));
        assert!(prompt.contains("<<SYS>>\nYou are helpful.\n<</SYS>>"));
        assert!(prompt.ends_with("Hello [/INST]"));
    }

    #[test]
    fn test_fallback_chat_template_assistant_history() {
        let prompt = render_fallback_chat_template(None, &[
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there."),
            ChatMessage::user("How are you?"),
        ]);
        assert_eq!(
            prompt,
            "[INST] Hello [/INST] Hi there. [INST] How are you? [/INST]"
        );
    }

    #[test]
    fn test_fallback_chat_template_ignores_unsupported_tool_role() {
        let prompt = render_fallback_chat_template(None, &[
            ChatMessage::user("Hello"),
            ChatMessage {
                role: "tool".to_string(),
                content: "ignored".to_string(),
            },
        ]);
        assert_eq!(prompt, "[INST] Hello [/INST]");
    }

    #[test]
    fn test_fallback_chat_template_gemma_family() {
        let prompt = render_fallback_chat_template(
            Some("{%- macro format_parameters(properties...) %}...<|turn|>user\n..."),
            &[ChatMessage::user("Hello")],
        );
        assert!(prompt.contains("<|turn>user\n"));
        assert!(prompt.contains("<|turn>model\n"));
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
    fn test_apply_chat_template_with_real_model() {
        let model_path = match std::env::var("MLX_RS_TEST_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("Skipping real chat template test: MLX_RS_TEST_GGUF not set");
                return;
            }
        };

        let runtime = Runtime::new(
            &model_path,
            LlamaCppConfig {
                n_ctx: Some(512),
                n_gpu_layers: Some(0),
                ..Default::default()
            },
        )
        .expect("Failed to load model");

        let prompt = runtime
            .apply_chat_template(&[
                ChatMessage::system("You are helpful."),
                ChatMessage::user("Hello"),
            ])
            .expect("Chat template should render");

        assert!(prompt.contains("Hello"));
        assert!(!prompt.trim().is_empty());
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
    fn test_embed_with_real_embedding_model() {
        let model_path = match std::env::var("MLX_RS_TEST_EMBED_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("Skipping real embedding test: MLX_RS_TEST_EMBED_GGUF not set");
                return;
            }
        };

        let config = LlamaCppConfig {
            n_ctx: Some(512),
            embedding: Some(true),
            pooling: Some("mean".to_string()),
            ..Default::default()
        };

        let mut runtime =
            Runtime::new(&model_path, config).expect("Failed to load embedding model");
        let embedding = runtime.embed("hello world").expect("Embedding failed");
        assert_eq!(embedding.len(), runtime.embedding_dimension() as usize);

        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3 || norm == 0.0);

        let second = runtime
            .embed("goodbye world")
            .expect("Second embedding failed");
        assert_eq!(second.len(), embedding.len());
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

        let first_piece = runtime
            .detokenize_piece(tokens[0])
            .expect("Piece detokenization failed");
        assert!(!first_piece.is_empty());
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

        let output = runtime
            .generate("Hello", &options)
            .expect("Generation failed");

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

        let output = runtime
            .generate("Count to five.", &options)
            .expect("Generation failed");

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
