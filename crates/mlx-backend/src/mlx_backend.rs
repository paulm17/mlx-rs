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
use crate::llama::{argmax, KvCache};
use crate::manifest::ModelManifest;
use crate::model::Model;
use crate::registry;

pub struct MlxBackend {
    model: Box<dyn Model>,
    tokenizer: tokenizers::Tokenizer,
    model_path: String,
    chat_template: ChatTemplate,
    prefix_cache: PrefixCache,
    max_position_embeddings: i32,
    hidden_size: i32,
    vocab_size: i32,
}

impl MlxBackend {
    pub fn load(dir: &Path) -> Result<Self> {
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

        let chat_template = ChatTemplate::load(dir)
            .unwrap_or_else(|_| ChatTemplate::default_llama3());

        let num_layers = model.num_layers();
        let max_position_embeddings = model.max_position_embeddings();
        let hidden_size = model.hidden_size();
        let vocab_size = model.vocab_size();
        let prefix_cache = PrefixCache::new(num_layers, 64);

        Ok(Self {
            model,
            tokenizer,
            model_path: dir.display().to_string(),
            chat_template,
            prefix_cache,
            max_position_embeddings,
            hidden_size,
            vocab_size,
        })
    }
}

fn make_positions(start: usize, length: usize) -> Result<Array> {
    let data: Vec<i32> = (start..start + length).map(|x| x as i32).collect();
    Array::from_data_i32(&data, &[length])
}

fn make_input_ids(tokens: &[i32]) -> Result<Array> {
    let n = tokens.len();
    Array::from_data_i32(tokens, &[1, n])
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
            if let Some(bos_id) = self.tokenizer.get_vocab(true).get("<|begin_of_text|>") {
                if ids.first() != Some(&(*bos_id as i32)) {
                    ids.insert(0, *bos_id as i32);
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
        token_id == self.token_eos()
    }

    fn token_eos(&self) -> i32 {
        let vocab = self.tokenizer.get_vocab(true);
        vocab.get("<|end_of_text|>")
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

        let (prefix_len, cached_caches) = self.prefix_cache.find(&tokens);
        let mut caches: Vec<KvCache> = if let Some(cached) = cached_caches {
            cached.clone()
        } else {
            (0..self.model.num_layers()).map(|_| KvCache::new()).collect()
        };

        let prefill_tokens = if prefix_len > 0 { &tokens[prefix_len..] } else { &tokens };
        let mut all_tokens = tokens.clone();

        if !prefill_tokens.is_empty() {
            let input_ids = make_input_ids(prefill_tokens)?;
            let positions = make_positions(prefix_len, prefill_tokens.len())?;
            self.model.forward(&input_ids, &mut caches, &positions)?;
        }

        let eos_token = self.token_eos();
        let mut generated = 0;
        let mut text = String::new();

        for _ in 0..max_tokens {
            let last_token = *all_tokens.last().unwrap();
            let input_ids = make_input_ids(&[last_token])?;
            let positions = make_positions(all_tokens.len() - 1, 1)?;

            let logits = self.model.forward(&input_ids, &mut caches, &positions)?;
            let next_token = argmax(&logits)?;

            if next_token == eos_token {
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

        let (prefix_len, cached_caches) = self.prefix_cache.find(&tokens);
        let mut caches: Vec<KvCache> = if let Some(cached) = cached_caches {
            cached.clone()
        } else {
            (0..self.model.num_layers()).map(|_| KvCache::new()).collect()
        };

        let prefill_tokens = if prefix_len > 0 { &tokens[prefix_len..] } else { &tokens };
        let mut all_tokens = tokens.clone();

        if !prefill_tokens.is_empty() {
            let input_ids = make_input_ids(prefill_tokens)?;
            let positions = make_positions(prefix_len, prefill_tokens.len())?;
            self.model.forward(&input_ids, &mut caches, &positions)?;
        }

        let eos_token = self.token_eos();
        let mut generated = 0;

        for _ in 0..max_tokens {
            let last_token = *all_tokens.last().unwrap();
            let input_ids = make_input_ids(&[last_token])?;
            let positions = make_positions(all_tokens.len() - 1, 1)?;

            let logits = self.model.forward(&input_ids, &mut caches, &positions)?;
            let next_token = argmax(&logits)?;

            if next_token == eos_token {
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

        let (prefix_len, cached_caches) = self.prefix_cache.find(&tokens);
        let mut caches: Vec<KvCache> = if let Some(cached) = cached_caches {
            cached.clone()
        } else {
            (0..self.model.num_layers()).map(|_| KvCache::new()).collect()
        };

        let prefill_tokens = if prefix_len > 0 { &tokens[prefix_len..] } else { &tokens };
        let mut all_tokens = tokens.clone();

        if !prefill_tokens.is_empty() {
            let input_ids = make_input_ids(prefill_tokens)?;
            let positions = make_positions(prefix_len, prefill_tokens.len())?;
            self.model.forward(&input_ids, &mut caches, &positions)?;
        }

        let eos_token = self.token_eos();
        let mut generated = 0;
        let mut text = String::new();

        for _ in 0..max_tokens {
            let last_token = *all_tokens.last().unwrap();
            let input_ids = make_input_ids(&[last_token])?;
            let positions = make_positions(all_tokens.len() - 1, 1)?;

            let logits = self.model.forward(&input_ids, &mut caches, &positions)?;
            let next_token = argmax(&logits)?;

            if next_token == eos_token {
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
