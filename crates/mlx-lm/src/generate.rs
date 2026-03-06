use anyhow::Result;
use mlx_core::{async_eval, Array};
use mlx_nn::{kv_cache_stats, reset_kv_cache_stats, KvCacheStats};
use std::time::Instant;

use crate::sampler::Sampler;
use crate::tokenizer::Tokenizer;

fn trace_memory_enabled() -> bool {
    matches!(
        std::env::var("MLX_TRACE_MEMORY").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn bytes_to_gib(v: usize) -> f64 {
    (v as f64) / 1024.0 / 1024.0 / 1024.0
}

fn trace_memory(stage: &str, generated: usize) {
    if !trace_memory_enabled() {
        return;
    }
    let mem = mlx_core::metal::memory_info();
    eprintln!(
        "[mem] stage={stage} generated={generated} active_gib={:.3} cache_gib={:.3} peak_gib={:.3}",
        bytes_to_gib(mem.active_memory),
        bytes_to_gib(mem.cache_memory),
        bytes_to_gib(mem.peak_memory)
    );
}

fn trace_generation_enabled() -> bool {
    matches!(
        std::env::var("MLX_TRACE_GENERATION").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

#[derive(Debug, Clone)]
pub struct GenerationProfile {
    pub clear_cache_s: f64,
    pub tokenize_s: f64,
    pub prefill_forward_s: f64,
    pub prefill_eval_s: f64,
    pub prefill_normalize_s: f64,
    pub first_sample_s: f64,
    pub first_decode_s: f64,
    pub decode_forward_s: f64,
    pub decode_eval_s: f64,
    pub decode_normalize_s: f64,
    pub sample_s: f64,
    pub decode_text_s: f64,
    pub eval_calls: usize,
    pub sample_calls: usize,
    pub cpu_logits_extractions: usize,
    pub decoded_pieces: usize,
    pub kv_cache_allocations: usize,
    pub kv_cache_growths: usize,
    pub moe_router_host_s: f64,
    pub moe_routing_build_s: f64,
    pub moe_expert_forward_s: f64,
    pub moe_shared_expert_s: f64,
    pub moe_single_token_fast_path_hits: usize,
}

impl GenerationProfile {
    fn new() -> Self {
        Self {
            clear_cache_s: 0.0,
            tokenize_s: 0.0,
            prefill_forward_s: 0.0,
            prefill_eval_s: 0.0,
            prefill_normalize_s: 0.0,
            first_sample_s: 0.0,
            first_decode_s: 0.0,
            decode_forward_s: 0.0,
            decode_eval_s: 0.0,
            decode_normalize_s: 0.0,
            sample_s: 0.0,
            decode_text_s: 0.0,
            eval_calls: 0,
            sample_calls: 0,
            cpu_logits_extractions: 0,
            decoded_pieces: 0,
            kv_cache_allocations: 0,
            kv_cache_growths: 0,
            moe_router_host_s: 0.0,
            moe_routing_build_s: 0.0,
            moe_expert_forward_s: 0.0,
            moe_shared_expert_s: 0.0,
            moe_single_token_fast_path_hits: 0,
        }
    }

    fn finish_kv_stats(&mut self, stats: KvCacheStats) {
        self.kv_cache_allocations = stats.allocations;
        self.kv_cache_growths = stats.growths;
    }
}

/// Trait for causal language models that support autoregressive generation.
pub trait CausalLM {
    /// Forward pass: given input token IDs, return last-token logits.
    fn forward_last_token_logits(&mut self, input_ids: &Array) -> mlx_core::Result<Array>;
    /// Clear the KV cache.
    fn clear_cache(&mut self);
}

#[derive(Debug, Clone)]
pub struct GenerationMetrics {
    pub ttft_s: f64,
    pub total_s: f64,
    pub tokens: usize,
    pub stop_reason: &'static str,
    pub last_token_id: Option<u32>,
    pub profile: Option<GenerationProfile>,
}

// Implement for supported dense decoder models.
impl CausalLM for mlx_models::Llama {
    fn forward_last_token_logits(&mut self, input_ids: &Array) -> mlx_core::Result<Array> {
        self.forward_last_token_logits(input_ids)
    }
    fn clear_cache(&mut self) {
        self.clear_cache();
    }
}

impl CausalLM for mlx_models::Qwen3 {
    fn forward_last_token_logits(&mut self, input_ids: &Array) -> mlx_core::Result<Array> {
        self.forward(input_ids)
    }
    fn clear_cache(&mut self) {
        self.clear_cache();
    }
}

impl CausalLM for mlx_models::Qwen35 {
    fn forward_last_token_logits(&mut self, input_ids: &Array) -> mlx_core::Result<Array> {
        self.forward(input_ids)
    }
    fn clear_cache(&mut self) {
        self.clear_cache();
    }
}

impl CausalLM for mlx_models::Qwen3Moe {
    fn forward_last_token_logits(&mut self, input_ids: &Array) -> mlx_core::Result<Array> {
        self.forward(input_ids)
    }
    fn clear_cache(&mut self) {
        self.clear_cache();
    }
}

impl CausalLM for mlx_models::Lfm2Moe {
    fn forward_last_token_logits(&mut self, input_ids: &Array) -> mlx_core::Result<Array> {
        self.forward(input_ids)
    }
    fn clear_cache(&mut self) {
        self.clear_cache();
    }
}

// Allow Box<dyn CausalLM> to be used as CausalLM
impl CausalLM for Box<dyn CausalLM> {
    fn forward_last_token_logits(&mut self, input_ids: &Array) -> mlx_core::Result<Array> {
        (**self).forward_last_token_logits(input_ids)
    }
    fn clear_cache(&mut self) {
        (**self).clear_cache();
    }
}

impl<T: CausalLM + ?Sized> CausalLM for &mut T {
    fn forward_last_token_logits(&mut self, input_ids: &Array) -> mlx_core::Result<Array> {
        (**self).forward_last_token_logits(input_ids)
    }
    fn clear_cache(&mut self) {
        (**self).clear_cache();
    }
}

fn has_runaway_repeat(generated_tokens: &[u32]) -> bool {
    // Detect contiguous repeated chunks near the tail (e.g. phrase loops).
    for &w in &[16usize, 24, 32] {
        if generated_tokens.len() < w * 3 {
            continue;
        }
        let n = generated_tokens.len();
        let a = &generated_tokens[n - w..n];
        let b = &generated_tokens[n - 2 * w..n - w];
        let c = &generated_tokens[n - 3 * w..n - 2 * w];
        if a == b && b == c {
            return true;
        }
    }
    false
}

fn sampled_token_array_to_u32(token: &Array) -> Result<u32> {
    let token = match token.ndim() {
        0 => token.clone(),
        1 => token.squeeze(0)?,
        2 => token.squeeze(0)?.squeeze(0)?,
        _ => anyhow::bail!("unexpected sampled token rank {}", token.ndim()),
    };
    match token.item_u32() {
        Ok(v) => Ok(v),
        Err(_) => Ok(token.item_i32()? as u32),
    }
}

fn sampled_token_array_to_input(token: &Array) -> Result<Array> {
    match token.ndim() {
        2 => Ok(token.clone()),
        1 => Ok(token.reshape(&[1, token.shape_raw()[0]])?),
        0 => Ok(token.reshape(&[1, 1])?),
        _ => anyhow::bail!("unexpected sampled token rank {}", token.ndim()),
    }
}

/// Text generation pipeline.
pub struct GenerationPipeline<M> {
    model: M,
    tokenizer: Tokenizer,
    sampler: Sampler,
}

impl<M: CausalLM> GenerationPipeline<M> {
    pub fn new(model: M, tokenizer: Tokenizer, sampler: Sampler) -> Self {
        Self {
            model,
            tokenizer,
            sampler,
        }
    }

    /// Generate text from a prompt.
    ///
    /// If `max_tokens` is `None`, generation is uncapped and stops only on EOS.
    pub fn generate(&mut self, prompt: &str, max_tokens: Option<usize>) -> Result<String> {
        self.generate_with_callback(prompt, max_tokens, |_token, _piece| {})
    }

    /// Generate text from a prompt and invoke a callback for each decoded piece.
    ///
    /// If `max_tokens` is `None`, generation is uncapped and stops only on EOS.
    pub fn generate_with_callback<F>(
        &mut self,
        prompt: &str,
        max_tokens: Option<usize>,
        mut on_piece: F,
    ) -> Result<String>
    where
        F: FnMut(u32, &str),
    {
        let (output, _) = self.generate_with_metrics(prompt, max_tokens, |token, piece| {
            on_piece(token, piece);
        })?;
        Ok(output)
    }

    pub fn generate_with_metrics<F>(
        &mut self,
        prompt: &str,
        max_tokens: Option<usize>,
        mut on_piece: F,
    ) -> Result<(String, GenerationMetrics)>
    where
        F: FnMut(u32, &str),
    {
        let t0 = Instant::now();
        let mut ttft_s: Option<f64> = None;
        let mut stop_reason: &'static str = "unknown";
        let profiling = trace_generation_enabled();
        let mut profile = profiling.then(GenerationProfile::new);

        if profiling {
            reset_kv_cache_stats();
            mlx_models::reset_moe_profile_stats();
        }
        let stage_t0 = Instant::now();
        self.model.clear_cache();
        if let Some(p) = profile.as_mut() {
            p.clear_cache_s += stage_t0.elapsed().as_secs_f64();
        }
        trace_memory("after_clear_cache", 0);
        let stage_t0 = Instant::now();
        let input_ids = self.tokenizer.encode(prompt)?;
        if let Some(p) = profile.as_mut() {
            p.tokenize_s += stage_t0.elapsed().as_secs_f64();
        }
        // Repetition controls should apply only to generated tokens, not prompt tokens.
        let mut history_tokens: Vec<u32> = Vec::new();

        // Create input array [1, seq_len]
        let prompt_i32: Vec<i32> = input_ids.iter().map(|&x| x as i32).collect();
        let input = Array::from_slice_i32(&prompt_i32)?
            .reshape(&[1, prompt_i32.len() as i32])?;

        // Prefill: run the full prompt through the model
        let stage_t0 = Instant::now();
        let logits = self.model.forward_last_token_logits(&input)
            .map_err(|e| anyhow::anyhow!("forward failed: {e}"))?;
        if let Some(p) = profile.as_mut() {
            p.prefill_forward_s += stage_t0.elapsed().as_secs_f64();
        }
        if trace_memory_enabled() {
            let stage_t0 = Instant::now();
            logits.eval()?;
            if let Some(p) = profile.as_mut() {
                p.prefill_eval_s += stage_t0.elapsed().as_secs_f64();
                p.eval_calls += 1;
            }
        }
        trace_memory("after_prefill_forward", 0);

        let mut output = String::new();
        let mut token_count = 0usize;
        let mut generated_tokens = Vec::new();
        let mut last_token_id: Option<u32>;
        if self.sampler.is_greedy() {
            let stage_t0 = Instant::now();
            let mut token_arr = self.sampler.sample_raw_last_token_logits_array(&logits)?;
            async_eval(&[&token_arr])?;
            if let Some(p) = profile.as_mut() {
                p.first_sample_s += stage_t0.elapsed().as_secs_f64();
                p.sample_calls += 1;
            }

            let mut generated = 0usize;
            loop {
                let can_schedule_next = max_tokens.is_none_or(|limit| generated + 1 < limit);
                let next_token_arr = if can_schedule_next {
                    let input = sampled_token_array_to_input(&token_arr)?;
                    let stage_t0 = Instant::now();
                    let logits = self.model.forward_last_token_logits(&input)
                        .map_err(|e| anyhow::anyhow!("forward failed: {e}"))?;
                    if let Some(p) = profile.as_mut() {
                        p.decode_forward_s += stage_t0.elapsed().as_secs_f64();
                    }

                    let stage_t0 = Instant::now();
                    let next = self.sampler.sample_raw_last_token_logits_array(&logits)?;
                    async_eval(&[&next])?;
                    if let Some(p) = profile.as_mut() {
                        p.sample_s += stage_t0.elapsed().as_secs_f64();
                        p.sample_calls += 1;
                    }
                    Some(next)
                } else {
                    None
                };

                let stage_t0 = Instant::now();
                let token = sampled_token_array_to_u32(&token_arr)?;
                last_token_id = Some(token);
                if let Some(p) = profile.as_mut() {
                    if generated == 0 {
                        p.first_sample_s += stage_t0.elapsed().as_secs_f64();
                    } else {
                        p.sample_s += stage_t0.elapsed().as_secs_f64();
                    }
                }
                generated += 1;

                if self.tokenizer.is_stop_token(token) {
                    stop_reason = "stop";
                    break;
                }

                generated_tokens.push(token);
                if has_runaway_repeat(&generated_tokens) {
                    stop_reason = "repeat_guard";
                    break;
                }

                let stage_t0 = Instant::now();
                let piece = self.tokenizer.decode(&[token])?;
                if let Some(p) = profile.as_mut() {
                    if generated == 1 {
                        p.first_decode_s += stage_t0.elapsed().as_secs_f64();
                    } else {
                        p.decode_text_s += stage_t0.elapsed().as_secs_f64();
                    }
                    p.decoded_pieces += 1;
                }
                on_piece(token, &piece);
                output.push_str(&piece);
                history_tokens.push(token);
                token_count += 1;
                if ttft_s.is_none() {
                    ttft_s = Some(t0.elapsed().as_secs_f64());
                }

                if let Some(limit) = max_tokens {
                    if generated >= limit {
                        stop_reason = "length";
                        break;
                    }
                }

                if generated % 32 == 0 {
                    trace_memory("decode", generated);
                }

                token_arr = match next_token_arr {
                    Some(next) => next,
                    None => {
                        stop_reason = "length";
                        break;
                    }
                };
            }
            trace_memory("after_first_token", 1);
        } else {
            let stage_t0 = Instant::now();
            let mut token = self.sampler.sample_raw_last_token_logits(&logits, &history_tokens)?;
            last_token_id = Some(token);
            if let Some(p) = profile.as_mut() {
                p.first_sample_s += stage_t0.elapsed().as_secs_f64();
                p.sample_calls += 1;
                if self.sampler.uses_host_sampling() {
                    p.cpu_logits_extractions += 1;
                }
            }
            if !self.tokenizer.is_stop_token(token) {
                let stage_t0 = Instant::now();
                let piece = self.tokenizer.decode(&[token])?;
                if let Some(p) = profile.as_mut() {
                    p.first_decode_s += stage_t0.elapsed().as_secs_f64();
                    p.decoded_pieces += 1;
                }
                on_piece(token, &piece);
                output.push_str(&piece);
                history_tokens.push(token);
                generated_tokens.push(token);
                token_count += 1;
                if ttft_s.is_none() {
                    ttft_s = Some(t0.elapsed().as_secs_f64());
                }
                trace_memory("after_first_token", 1);
            }

            // Autoregressive decode loop. Uncapped when max_tokens is None.
            let mut generated = 1usize;
            loop {
                if let Some(limit) = max_tokens {
                    if generated >= limit {
                        stop_reason = "length";
                        break;
                    }
                }
                if self.tokenizer.is_stop_token(token) {
                    stop_reason = "stop";
                    break;
                }
                let input = Array::from_int(token as i32)?
                    .reshape(&[1, 1])?;

                let stage_t0 = Instant::now();
                let logits = self.model.forward_last_token_logits(&input)
                    .map_err(|e| anyhow::anyhow!("forward failed: {e}"))?;
                if let Some(p) = profile.as_mut() {
                    p.decode_forward_s += stage_t0.elapsed().as_secs_f64();
                }

                let stage_t0 = Instant::now();
                token = self.sampler.sample_raw_last_token_logits(&logits, &history_tokens)?;
                last_token_id = Some(token);
                if let Some(p) = profile.as_mut() {
                    p.sample_s += stage_t0.elapsed().as_secs_f64();
                    p.sample_calls += 1;
                    if self.sampler.uses_host_sampling() {
                        p.cpu_logits_extractions += 1;
                    }
                }
                generated += 1;

                if self.tokenizer.is_stop_token(token) {
                    break;
                }

                generated_tokens.push(token);
                if has_runaway_repeat(&generated_tokens) {
                    stop_reason = "repeat_guard";
                    break;
                }

                let stage_t0 = Instant::now();
                let piece = self.tokenizer.decode(&[token])?;
                if let Some(p) = profile.as_mut() {
                    p.decode_text_s += stage_t0.elapsed().as_secs_f64();
                    p.decoded_pieces += 1;
                }
                on_piece(token, &piece);
                output.push_str(&piece);
                history_tokens.push(token);
                token_count += 1;
                if generated % 32 == 0 {
                    trace_memory("decode", generated);
                }
            }
        }
        trace_memory("end", token_count);
        if let Some(p) = profile.as_mut() {
            p.finish_kv_stats(kv_cache_stats());
            let moe = mlx_models::moe_profile_stats();
            p.moe_router_host_s = moe.router_host_s;
            p.moe_routing_build_s = moe.routing_build_s;
            p.moe_expert_forward_s = moe.expert_forward_s;
            p.moe_shared_expert_s = moe.shared_expert_s;
            p.moe_single_token_fast_path_hits = moe.single_token_fast_path_hits;
        }
        let metrics = GenerationMetrics {
            ttft_s: ttft_s.unwrap_or(0.0),
            total_s: t0.elapsed().as_secs_f64(),
            tokens: token_count,
            stop_reason,
            last_token_id,
            profile,
        };
        Ok((output, metrics))
    }
}
