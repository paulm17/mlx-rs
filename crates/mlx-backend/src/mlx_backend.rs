use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use backend_trait::types::{
    ChatMessage, EmbeddingOutput, GenerateOutput,
    GenerationMetrics, GenerationOptions, LoadedModelInfo, StopReason,
};

use crate::array::Array;
use crate::cache::PrefixCache;
use crate::chat_template::ChatTemplate;
use crate::llama::LayerCache;
use crate::manifest::ModelManifest;
use crate::model::Model;
use crate::registry;
use crate::sampler::Sampler;

const DEFAULT_CACHE_CAPACITY: usize = 64;
const DEFAULT_PREFILL_CHUNK_SIZE: usize = 2048;

pub struct MlxBackend {
    model: Box<dyn Model>,
    tokenizer: tokenizers::Tokenizer,
    model_path: String,
    chat_template: ChatTemplate,
    prefix_cache: PrefixCache,
    prefill_chunk_size: usize,
    #[allow(dead_code)]
    compile_enabled: bool,
    max_position_embeddings: i32,
    hidden_size: i32,
    vocab_size: i32,
    eog_token_ids: std::collections::HashSet<i32>,
}

fn choose_next_token(logits: &Array, sampler: &Sampler) -> anyhow::Result<i32> {
    crate::ops::eval(&[logits])?;
    if sampler.temperature <= 0.0 {
        let idx = crate::ops::argmax_axis(logits, -1, false)?;
        Ok(idx.item_i32()?)
    } else {
        // Convert to f32 before reading data — data_f32() reinterprets raw bytes,
        // so f16/bf16 arrays must be cast first.
        let logits_f32 = crate::ops::astype(logits, crate::ffi::MlxDtype::Float32)?;
        crate::ops::eval(&[&logits_f32])?;
        let data = logits_f32.data_f32()?;
        Ok(sampler.sample(data) as i32)
    }
}

impl MlxBackend {
    pub fn load(dir: &Path, mlx_config: &llama_lm::MlxConfig) -> Result<Self> {
        // Check MLX availability before loading files
        crate::loader::check_init()
            .context("MLX-C runtime (libmlxc.dylib) not found; safetensors models require \
                      the MLX-C runtime. Set LLAMA_RS_BUILD_MLX=1 to auto-build it, \
                      or set MLX_RS_MLX_LIBRARY to the directory containing libmlxc.dylib")?;

        // Initialize GPU/CPU streams for the current thread (required for eval())
        crate::ops::init_streams();

        let manifest = ModelManifest::open(dir)
            .with_context(|| format!("failed to open model at {}", dir.display()))?;

        let config = manifest.config().clone();
        let architecture = registry::detect_architecture(&config);

        let tensors = manifest.load_all_tensors()
            .context("failed to load tensors")?;

        let model = registry::create_model(&architecture, tensors, &config)
            .context("failed to construct model")?;

        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            tokenizers::Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?
        } else {
            anyhow::bail!("tokenizer.json not found in {}", dir.display());
        };

        let chat_template = ChatTemplate::load(dir, Some(&architecture))
            .unwrap_or_else(|_| ChatTemplate::default_for_architecture(Some(&architecture)));

        let mut eog_token_ids = Self::eog_token_ids_from_tokenizer(&tokenizer);
        if let Ok(config_str) = std::fs::read_to_string(dir.join("tokenizer_config.json")) {
            if let Ok(config_json) = serde_json::from_str::<serde_json::Value>(&config_str) {
                if let Some(eos_ids) = config_json.get("eos_token_id") {
                    match eos_ids {
                        serde_json::Value::Number(n) => {
                            if let Some(id) = n.as_i64() {
                                eog_token_ids.insert(id as i32);
                            }
                        }
                        serde_json::Value::Array(arr) => {
                            for v in arr {
                                if let Some(id) = v.as_i64() {
                                    eog_token_ids.insert(id as i32);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        let num_layers = model.num_layers();
        let max_position_embeddings = model.max_position_embeddings();
        let hidden_size = model.hidden_size();
        let vocab_size = model.vocab_size();
        let cache_capacity = mlx_config.cache_limit.unwrap_or(DEFAULT_CACHE_CAPACITY);
        let prefix_cache = PrefixCache::new(num_layers, cache_capacity);
        let prefill_chunk_size = mlx_config.prefill_chunk_size.unwrap_or(DEFAULT_PREFILL_CHUNK_SIZE);
        let compile_enabled = mlx_config.compile.unwrap_or(true);

        Ok(Self {
            model,
            tokenizer,
            model_path: dir.display().to_string(),
            chat_template,
            prefix_cache,
            prefill_chunk_size,
            compile_enabled,
            max_position_embeddings,
            hidden_size,
            vocab_size,
            eog_token_ids,
        })
    }

    /// Collect EOG token IDs from special token strings in both the base vocab
    /// and the added tokens vocabulary (which covers special tokens not in base vocab).
    fn eog_token_ids_from_tokenizer(tokenizer: &tokenizers::Tokenizer) -> std::collections::HashSet<i32> {
        let vocab = tokenizer.get_vocab(true);
        let added_vocab = tokenizer.get_added_vocabulary().get_vocab();
        let mut ids = std::collections::HashSet::new();
        let eog_strings = [
            "<eos>",
            "<turn|>",
            "<|end_of_turn|>",
            "<|end_of_text|>",
            "<|eot_id|>",
            "</s>",
            "<|im_end|>",
        ];
        for token in &eog_strings {
            if let Some(&id) = vocab.get(*token) {
                ids.insert(id as i32);
            }
            if let Some(&id) = added_vocab.get(*token) {
                ids.insert(id as i32);
            }
        }
        ids
    }
}

fn make_positions(start: usize, _length: usize) -> Result<Array> {
    Array::from_data_i32(&[start as i32], &[1])
}

fn make_input_ids(tokens: &[i32]) -> Result<Array> {
    let n = tokens.len();
    Array::from_data_i32(tokens, &[1, n])
}

fn prefill_chunked(
    model: &dyn Model,
    tokens: &[i32],
    start_pos: usize,
    caches: &mut [LayerCache],
    chunk_size: usize,
) -> anyhow::Result<()> {
    if tokens.is_empty() {
        return Ok(());
    }
    let mut pos = start_pos;
    for chunk in tokens.chunks(chunk_size) {
        let input_ids = make_input_ids(chunk)?;
        let positions = make_positions(pos, chunk.len())?;
        let logits = model.forward(&input_ids, caches, &positions)?;
        crate::ops::eval(&[&logits])?;
        pos += chunk.len();
        let _ = crate::memory::clear_cache();
    }
    Ok(())
}

impl backend_trait::Backend for MlxBackend {
    fn model_info(&self) -> LoadedModelInfo {
        LoadedModelInfo {
            model_path: self.model_path.clone(),
            context_length: Some(self.max_position_embeddings as usize),
            embedding_dimension: Some(self.hidden_size as usize),
            vocab_size: Some(self.vocab_size as usize),
        }
    }

    fn tokenize(&self, text: &str, add_bos: bool) -> Result<Vec<i32>> {
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        let mut ids: Vec<i32> = encoding.get_ids().iter().map(|&x| x as i32).collect();
        if add_bos {
            if let Some(&bos_id) = self.tokenizer.get_vocab(true).get(self.chat_template.bos_token()) {
                if ids.first() != Some(&(bos_id as i32)) {
                    ids.insert(0, bos_id as i32);
                }
            }
        }
        Ok(ids)
    }

    fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        let u32_ids: Vec<u32> = tokens.iter().map(|&x| x as u32).collect();
        self.tokenizer.decode(&u32_ids, true)
            .map_err(|e| anyhow::anyhow!("detokenization failed: {e}"))
    }

    fn detokenize_piece(&self, token_id: i32) -> Result<String> {
        let id = token_id as u32;
        self.tokenizer.decode(&[id], false)
            .map_err(|e| anyhow::anyhow!("detokenization failed: {e}"))
    }

    fn is_eog(&self, token_id: i32) -> bool {
        self.eog_token_ids.contains(&token_id)
    }

    fn token_eos(&self) -> i32 {
        let vocab = self.tokenizer.get_vocab(true);
        // Gemma models use <eos>; many models use <|end_of_text|> or </s>.
        vocab.get("<eos>")
            .or_else(|| vocab.get("<|end_of_text|>"))
            .or_else(|| vocab.get("</s>"))
            .map(|&id| id as i32)
            .unwrap_or(2)
    }

    fn embeddings_enabled(&self) -> bool {
        false
    }

    fn supports_chat_template(&self) -> bool {
        true
    }

    fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
        self.chat_template.render(messages)
    }

    fn generate(&mut self, prompt: &str, options: &GenerationOptions) -> Result<GenerateOutput> {
        let tokens = self.tokenize(prompt, true)?;
        let prompt_len = tokens.len();
        let max_tokens = options.max_tokens.unwrap_or(512);
        let start = Instant::now();

        let sampler = Sampler::new(options.temperature, options.top_p, options.top_k, options.min_p);

        let (prefix_len, cached_caches) = self.prefix_cache.find(&tokens);
        let mut caches: Vec<LayerCache> = if let Some(cached) = cached_caches {
            cached.clone()
        } else {
            self.model.new_caches()
        };

        let prefill_tokens = if prefix_len > 0 { &tokens[prefix_len..] } else { &tokens };
        let mut all_tokens = tokens.clone();

        prefill_chunked(self.model.as_ref(), prefill_tokens, prefix_len, &mut caches, self.prefill_chunk_size)?;

        let mut generated = 0;
        let mut text = String::new();

        for _step in 0..max_tokens {
            let last_token = *all_tokens.last().unwrap();
            let input_ids = make_input_ids(&[last_token])?;
            let pos = all_tokens.len() - 1;
            let positions = make_positions(pos, 1)?;
            let logits = self.model.forward(&input_ids, &mut caches, &positions)?;
            let next_token = choose_next_token(&logits, &sampler)?;

            if self.is_eog(next_token) {
                break;
            }

            all_tokens.push(next_token);
            generated += 1;

            let piece = self.detokenize_piece(next_token)?;
            text.push_str(&piece);
        }

        if generated > 0 {
            self.prefix_cache.insert(&all_tokens, caches);
        }
        let _ = crate::memory::clear_cache();

        let elapsed = start.elapsed().as_secs_f64();
        Ok(GenerateOutput {
            text,
            stop_reason: if generated >= max_tokens { StopReason::MaxTokens } else { StopReason::Eos },
            metrics: GenerationMetrics {
                prompt_tokens: prompt_len,
                generated_tokens: generated,
                total_tokens: prompt_len + generated,
                ttft_s: Some(elapsed),
                total_s: Some(elapsed),
                tokens_per_s: if elapsed > 0.0 { Some(generated as f64 / elapsed) } else { None },
            },
        })
    }

    fn generate_stream(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
        mut on_token: Box<dyn FnMut(&str) -> bool + Send>,
    ) -> Result<GenerationMetrics> {
        let tokens = self.tokenize(prompt, true)?;
        let prompt_len = tokens.len();
        let max_tokens = options.max_tokens.unwrap_or(512);
        let start = Instant::now();

        let sampler = Sampler::new(options.temperature, options.top_p, options.top_k, options.min_p);

        let (prefix_len, cached_caches) = self.prefix_cache.find(&tokens);
        let mut caches: Vec<LayerCache> = if let Some(cached) = cached_caches {
            cached.clone()
        } else {
            self.model.new_caches()
        };

        let prefill_tokens = if prefix_len > 0 { &tokens[prefix_len..] } else { &tokens };
        let mut all_tokens = tokens.clone();

        prefill_chunked(self.model.as_ref(), prefill_tokens, prefix_len, &mut caches, self.prefill_chunk_size)?;

        let mut generated = 0;

        for _ in 0..max_tokens {
            let last_token = *all_tokens.last().unwrap();
            let input_ids = make_input_ids(&[last_token])?;
            let positions = make_positions(all_tokens.len() - 1, 1)?;

            let logits = self.model.forward(&input_ids, &mut caches, &positions)?;
            let next_token = choose_next_token(&logits, &sampler)?;

            if self.is_eog(next_token) {
                break;
            }

            all_tokens.push(next_token);
            generated += 1;

            let piece = self.detokenize_piece(next_token)?;
            if !on_token(&piece) {
                break;
            }
        }

        if generated > 0 {
            self.prefix_cache.insert(&all_tokens, caches);
        }
        let _ = crate::memory::clear_cache();

        let elapsed = start.elapsed().as_secs_f64();
        Ok(GenerationMetrics {
            prompt_tokens: prompt_len,
            generated_tokens: generated,
            total_tokens: prompt_len + generated,
            ttft_s: Some(elapsed),
            total_s: Some(elapsed),
            tokens_per_s: if elapsed > 0.0 { Some(generated as f64 / elapsed) } else { None },
        })
    }

    fn generate_stream_output(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
        mut on_token: Box<dyn FnMut(&str) -> bool + Send>,
    ) -> Result<GenerateOutput> {
        let tokens = self.tokenize(prompt, true)?;
        let prompt_len = tokens.len();
        let max_tokens = options.max_tokens.unwrap_or(512);
        let start = Instant::now();

        let sampler = Sampler::new(options.temperature, options.top_p, options.top_k, options.min_p);

        let (prefix_len, cached_caches) = self.prefix_cache.find(&tokens);
        let mut caches: Vec<LayerCache> = if let Some(cached) = cached_caches {
            cached.clone()
        } else {
            self.model.new_caches()
        };

        let prefill_tokens = if prefix_len > 0 { &tokens[prefix_len..] } else { &tokens };
        let mut all_tokens = tokens.clone();

        prefill_chunked(self.model.as_ref(), prefill_tokens, prefix_len, &mut caches, self.prefill_chunk_size)?;

        let mut generated = 0;
        let mut text = String::new();

        for _ in 0..max_tokens {
            let last_token = *all_tokens.last().unwrap();
            let input_ids = make_input_ids(&[last_token])?;
            let positions = make_positions(all_tokens.len() - 1, 1)?;

            let logits = self.model.forward(&input_ids, &mut caches, &positions)?;
            let next_token = choose_next_token(&logits, &sampler)?;

            if self.is_eog(next_token) {
                break;
            }

            all_tokens.push(next_token);
            generated += 1;

            let piece = self.detokenize_piece(next_token)?;
            text.push_str(&piece);
            if !on_token(&piece) {
                break;
            }
        }

        if generated > 0 {
            self.prefix_cache.insert(&all_tokens, caches);
        }
        let _ = crate::memory::clear_cache();

        let elapsed = start.elapsed().as_secs_f64();
        Ok(GenerateOutput {
            text,
            stop_reason: if generated >= max_tokens { StopReason::MaxTokens } else { StopReason::Eos },
            metrics: GenerationMetrics {
                prompt_tokens: prompt_len,
                generated_tokens: generated,
                total_tokens: prompt_len + generated,
                ttft_s: Some(elapsed),
                total_s: Some(elapsed),
                tokens_per_s: if elapsed > 0.0 { Some(generated as f64 / elapsed) } else { None },
            },
        })
    }

    fn embed(&mut self, _text: &str) -> Result<EmbeddingOutput> {
        anyhow::bail!("embeddings not yet supported for MLX backend")
    }

    fn memory_info(&self) -> Option<String> {
        crate::memory::active_memory().ok().map(|active| {
            let cache = crate::memory::cache_memory().unwrap_or(0);
            let peak = crate::memory::peak_memory().unwrap_or(0);
            format!("active={} cache={} peak={}", active, cache, peak)
        })
    }
}
