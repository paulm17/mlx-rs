use anyhow::Result;
use mlx_core::Array;
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

/// Normalize last-token logits to 1-D vocab scores for sampling.
fn normalized_last_token_logits(logits: &Array) -> mlx_core::Result<Array> {
    let shape = logits.shape_raw();
    match shape.len() {
        3 => {
            // [1, 1, vocab] → [vocab]
            logits.squeeze(0)?.squeeze(0)?.contiguous()
        }
        2 => {
            // [1, vocab] → [vocab]
            logits.squeeze(0)?.contiguous()
        }
        _ => logits.contiguous(),
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

        self.model.clear_cache();
        trace_memory("after_clear_cache", 0);
        let input_ids = self.tokenizer.encode(prompt)?;
        // Repetition controls should apply only to generated tokens, not prompt tokens.
        let mut history_tokens: Vec<u32> = Vec::new();

        // Create input array [1, seq_len]
        let prompt_i32: Vec<i32> = input_ids.iter().map(|&x| x as i32).collect();
        let input = Array::from_slice_i32(&prompt_i32)?
            .reshape(&[1, prompt_i32.len() as i32])?;

        // Prefill: run the full prompt through the model
        let logits = self.model.forward_last_token_logits(&input)
            .map_err(|e| anyhow::anyhow!("forward failed: {e}"))?;
        logits.eval()?;
        trace_memory("after_prefill_forward", 0);

        let logits = normalized_last_token_logits(&logits)?;
        logits.eval()?;
        let mut token = self.sampler.sample_with_history(&logits, &history_tokens)?;
        let mut output = String::new();
        let mut token_count = 0usize;
        let mut generated_tokens = Vec::new();
        if !self.tokenizer.is_stop_token(token) {
            let piece = self.tokenizer.decode(&[token])?;
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
                    break;
                }
            }
            if self.tokenizer.is_stop_token(token) {
                break;
            }
            let input = Array::from_int(token as i32)?
                .reshape(&[1, 1])?;

            let logits = self.model.forward_last_token_logits(&input)
                .map_err(|e| anyhow::anyhow!("forward failed: {e}"))?;
            logits.eval()?;

            let logits = normalized_last_token_logits(&logits)?;
            logits.eval()?;

            token = self.sampler.sample_with_history(&logits, &history_tokens)?;
            generated += 1;

            if self.tokenizer.is_stop_token(token) {
                break;
            }

            generated_tokens.push(token);
            if has_runaway_repeat(&generated_tokens) {
                break;
            }

            let piece = self.tokenizer.decode(&[token])?;
            on_piece(token, &piece);
            output.push_str(&piece);
            history_tokens.push(token);
            token_count += 1;
            if generated % 32 == 0 {
                trace_memory("decode", generated);
            }
        }
        trace_memory("end", generated);
        let metrics = GenerationMetrics {
            ttft_s: ttft_s.unwrap_or(0.0),
            total_s: t0.elapsed().as_secs_f64(),
            tokens: token_count,
        };
        Ok((output, metrics))
    }
}
