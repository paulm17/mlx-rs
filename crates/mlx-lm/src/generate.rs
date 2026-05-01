use anyhow::Result;
use mlx_core::{async_eval, Array};
use mlx_nn::{kv_cache_stats, reset_kv_cache_stats, KvCacheStats};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
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
    pub moe_device_router_shadow_checks: usize,
    pub moe_device_router_shadow_mismatches: usize,
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
            moe_device_router_shadow_checks: 0,
            moe_device_router_shadow_mismatches: 0,
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

    /// Multimodal forward pass. Defaults to text-only forward.
    /// `pixel_values` shape: [1, 3, H, W], or `None` for text-only.
    fn forward_last_token_logits_multimodal(
        &mut self,
        input_ids: &Array,
        _pixel_values: Option<&Array>,
    ) -> mlx_core::Result<Array> {
        self.forward_last_token_logits(input_ids)
    }

    /// Clear the KV cache.
    fn clear_cache(&mut self);
}

/// Trait for models that can expose hidden states for embedding extraction.
pub trait EmbeddingModel {
    /// Forward pass. Returns hidden states for all input tokens.
    fn forward_hidden_states(&mut self, _input_ids: &Array) -> Result<Array> {
        anyhow::bail!("embeddings are not supported by the loaded model")
    }

    /// Forward pass with an optional attention mask for padded embedding batches.
    fn forward_hidden_states_masked(
        &mut self,
        input_ids: &Array,
        _attention_mask: Option<&Array>,
    ) -> Result<Array> {
        self.forward_hidden_states(input_ids)
    }

    fn embedding_pooling(&self) -> EmbeddingPooling {
        EmbeddingPooling::LastToken
    }

    fn supports_padded_embedding_batching(&self) -> bool {
        false
    }
}

pub trait ModelRuntime: CausalLM + EmbeddingModel {}

impl<T: CausalLM + EmbeddingModel> ModelRuntime for T {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingPooling {
    LastToken,
    Mean,
}

#[derive(Debug, Clone)]
pub struct GenerationMetrics {
    pub ttft_s: f64,
    pub total_s: f64,
    pub tokens: usize,
    pub stop_reason: &'static str,
    pub last_token_id: Option<u32>,
    pub generated_token_ids: Vec<u32>,
    pub tail_token_ids: Vec<u32>,
    pub stop_token_ids: Vec<u32>,
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

impl EmbeddingModel for mlx_models::Llama {
    fn forward_hidden_states(&mut self, input_ids: &Array) -> Result<Array> {
        self.forward_hidden_states(input_ids)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
    }
}

impl CausalLM for mlx_models::Gemma3 {
    fn forward_last_token_logits(&mut self, input_ids: &Array) -> mlx_core::Result<Array> {
        self.forward_last_token_logits(input_ids)
    }
    fn clear_cache(&mut self) {
        self.clear_cache();
    }
}

impl EmbeddingModel for mlx_models::Gemma3 {
    fn forward_hidden_states(&mut self, input_ids: &Array) -> Result<Array> {
        self.forward_hidden_states(input_ids)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
    }
}

impl CausalLM for mlx_models::Gemma4 {
    fn forward_last_token_logits(&mut self, input_ids: &Array) -> mlx_core::Result<Array> {
        self.forward_last_token_logits(input_ids, None)
    }
    fn forward_last_token_logits_multimodal(
        &mut self,
        input_ids: &Array,
        pixel_values: Option<&Array>,
    ) -> mlx_core::Result<Array> {
        self.forward_last_token_logits(input_ids, pixel_values)
    }
    fn clear_cache(&mut self) {
        mlx_models::Gemma4::clear_cache(self);
    }
}

impl EmbeddingModel for mlx_models::Gemma4 {
    fn forward_hidden_states(&mut self, input_ids: &Array) -> Result<Array> {
        self.forward_hidden_states(input_ids)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
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

impl EmbeddingModel for mlx_models::Qwen3 {
    fn forward_hidden_states(&mut self, input_ids: &Array) -> Result<Array> {
        self.forward_hidden_states(input_ids)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
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

impl EmbeddingModel for mlx_models::Qwen35 {
    fn forward_hidden_states(&mut self, input_ids: &Array) -> Result<Array> {
        self.forward_hidden_states(input_ids)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
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

impl EmbeddingModel for mlx_models::Qwen3Moe {
    fn forward_hidden_states(&mut self, input_ids: &Array) -> Result<Array> {
        self.forward_hidden_states(input_ids)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
    }
}

impl CausalLM for mlx_models::Qwen3MoePythonPort {
    fn forward_last_token_logits(&mut self, input_ids: &Array) -> mlx_core::Result<Array> {
        self.forward(input_ids)
    }
    fn clear_cache(&mut self) {
        self.clear_cache();
    }
}

impl EmbeddingModel for mlx_models::Qwen3MoePythonPort {
    fn forward_hidden_states(&mut self, input_ids: &Array) -> Result<Array> {
        self.forward_hidden_states(input_ids)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
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

impl EmbeddingModel for mlx_models::Lfm2Moe {
    fn forward_hidden_states(&mut self, input_ids: &Array) -> Result<Array> {
        self.forward_hidden_states(input_ids)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
    }
}

impl CausalLM for mlx_models::Lfm2MoePythonPort {
    fn forward_last_token_logits(&mut self, input_ids: &Array) -> mlx_core::Result<Array> {
        self.forward(input_ids)
    }
    fn clear_cache(&mut self) {
        self.clear_cache();
    }
}

impl EmbeddingModel for mlx_models::Lfm2MoePythonPort {
    fn forward_hidden_states(&mut self, input_ids: &Array) -> Result<Array> {
        self.forward_hidden_states(input_ids)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
    }
}

impl CausalLM for mlx_models::Bert {
    fn forward_last_token_logits(&mut self, _input_ids: &Array) -> mlx_core::Result<Array> {
        Err(mlx_core::Error::Message(
            "text generation is not supported by encoder-only models".to_string(),
        ))
    }

    fn clear_cache(&mut self) {
        self.reset_state();
    }
}

impl EmbeddingModel for mlx_models::Bert {
    fn forward_hidden_states(&mut self, input_ids: &Array) -> Result<Array> {
        self.encode(input_ids)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
    }

    fn forward_hidden_states_masked(
        &mut self,
        input_ids: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array> {
        self.encode_masked(input_ids, attention_mask)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
    }

    fn embedding_pooling(&self) -> EmbeddingPooling {
        EmbeddingPooling::Mean
    }

    fn supports_padded_embedding_batching(&self) -> bool {
        true
    }
}

// Allow Box<dyn CausalLM> to be used as CausalLM
impl CausalLM for Box<dyn CausalLM> {
    fn forward_last_token_logits(&mut self, input_ids: &Array) -> mlx_core::Result<Array> {
        (**self).forward_last_token_logits(input_ids)
    }
    fn forward_last_token_logits_multimodal(
        &mut self,
        input_ids: &Array,
        pixel_values: Option<&Array>,
    ) -> mlx_core::Result<Array> {
        (**self).forward_last_token_logits_multimodal(input_ids, pixel_values)
    }
    fn clear_cache(&mut self) {
        (**self).clear_cache();
    }
}

impl CausalLM for Box<dyn ModelRuntime> {
    fn forward_last_token_logits(&mut self, input_ids: &Array) -> mlx_core::Result<Array> {
        (**self).forward_last_token_logits(input_ids)
    }
    fn forward_last_token_logits_multimodal(
        &mut self,
        input_ids: &Array,
        pixel_values: Option<&Array>,
    ) -> mlx_core::Result<Array> {
        (**self).forward_last_token_logits_multimodal(input_ids, pixel_values)
    }
    fn clear_cache(&mut self) {
        (**self).clear_cache();
    }
}

impl EmbeddingModel for Box<dyn ModelRuntime> {
    fn forward_hidden_states(&mut self, input_ids: &Array) -> Result<Array> {
        (**self).forward_hidden_states(input_ids)
    }

    fn forward_hidden_states_masked(
        &mut self,
        input_ids: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array> {
        (**self).forward_hidden_states_masked(input_ids, attention_mask)
    }

    fn embedding_pooling(&self) -> EmbeddingPooling {
        (**self).embedding_pooling()
    }

    fn supports_padded_embedding_batching(&self) -> bool {
        (**self).supports_padded_embedding_batching()
    }
}

impl<T: CausalLM + ?Sized> CausalLM for &mut T {
    fn forward_last_token_logits(&mut self, input_ids: &Array) -> mlx_core::Result<Array> {
        (**self).forward_last_token_logits(input_ids)
    }
    fn forward_last_token_logits_multimodal(
        &mut self,
        input_ids: &Array,
        pixel_values: Option<&Array>,
    ) -> mlx_core::Result<Array> {
        (**self).forward_last_token_logits_multimodal(input_ids, pixel_values)
    }
    fn clear_cache(&mut self) {
        (**self).clear_cache();
    }
}

impl<T: EmbeddingModel + ?Sized> EmbeddingModel for &mut T {
    fn forward_hidden_states(&mut self, input_ids: &Array) -> Result<Array> {
        (**self).forward_hidden_states(input_ids)
    }

    fn forward_hidden_states_masked(
        &mut self,
        input_ids: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array> {
        (**self).forward_hidden_states_masked(input_ids, attention_mask)
    }

    fn supports_padded_embedding_batching(&self) -> bool {
        (**self).supports_padded_embedding_batching()
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

fn strip_thinking_blocks(text: &str) -> String {
    let mut result = text.to_string();
    // Strip complete <think>...</think> blocks (including with attributes like <think type="reasoning">)
    loop {
        if let Some(start) = result.find("<think>") {
            if let Some(end) = result[start..].find("</think>") {
                result.replace_range(start..start + end + "</think>".len(), "");
                continue;
            } else {
                result.truncate(start);
                break;
            }
        }
        break;
    }
    // Strip any orphaned </think> tags
    while let Some(start) = result.find("</think>") {
        result.replace_range(start..start + "</think>".len(), "");
    }
    result.trim().to_string()
}

/// Build a reverse ByteLevel mapping (Unicode character → raw byte value).
/// For GPT-2-style BPE tokenizers, token strings use a specific character-to-byte
/// encoding where some byte values map to Latin-1 Supplement / Extended-A chars.
fn build_byte_reverse_map() -> HashMap<char, u8> {
    // Printable ASCII (0x21-0x7E) and Latin-1 Supplement (0xA1-0xAC, 0xAE-0xFF) map directly.
    let bs: Vec<u32> = (0x21u32..=0x7Eu32)
        .chain(0xA1u32..=0xACu32)
        .chain(0xAEu32..=0xFFu32)
        .collect();

    let mut byte_to_char = [0u32; 256];
    let mut n = 0u32;
    for b in 0u32..256u32 {
        if bs.contains(&b) {
            byte_to_char[b as usize] = b;
        } else {
            byte_to_char[b as usize] = 256 + n;
            n += 1;
        }
    }

    let mut map = HashMap::new();
    for b in 0u32..256u32 {
        if let Some(ch) = char::from_u32(byte_to_char[b as usize]) {
            map.insert(ch, b as u8);
        }
    }
    map
}

/// Convert a token string to raw bytes using the ByteLevel reverse mapping.
/// Returns `None` if any character is not in the map (non-ByteLevel tokenizer).
fn token_str_to_bytes(token_str: &str, reverse_map: &HashMap<char, u8>) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    for ch in token_str.chars() {
        if let Some(&byte) = reverse_map.get(&ch) {
            bytes.push(byte);
        } else {
            return None;
        }
    }
    Some(bytes)
}

/// Extract valid UTF-8 prefixes from a byte buffer, leaving incomplete trailing bytes.
fn flush_valid_utf8(buf: &mut Vec<u8>) -> String {
    if buf.is_empty() {
        return String::new();
    }
    match std::str::from_utf8(buf) {
        Ok(_) => {
            let taken = std::mem::take(buf);
            String::from_utf8(taken).unwrap()
        }
        Err(e) => {
            let valid_up_to = e.valid_up_to();
            if valid_up_to == 0 {
                String::new()
            } else {
                let valid: Vec<u8> = buf.drain(..valid_up_to).collect();
                String::from_utf8(valid).unwrap()
            }
        }
    }
}

/// Detect whether this tokenizer uses ByteLevel encoding by checking if
/// decoding individual early-vocabulary tokens produces replacement characters.
/// ByteLevel tokenizers (GPT-2, Qwen) need per-token byte buffering;
/// SentencePiece tokenizers (Mistral, Gemma) need cumulative decoding.
fn needs_byte_buffering(tokenizer: &tokenizers::Tokenizer) -> bool {
    for tid in 0..256u32 {
        if let Ok(decoded) = tokenizer.decode(&[tid], true) {
            if decoded.contains('\u{FFFD}') {
                return true;
            }
        }
    }
    false
}

/// Decode a single token incrementally using ByteLevel byte buffering.
///
/// For GPT-2-style ByteLevel BPE tokenizers, single tokens can represent partial
/// multi-byte UTF-8 sequences. Decoding them in isolation produces replacement
/// characters (U+FFFD). This function accumulates raw bytes across calls and emits
/// only completed UTF-8 sequences.
fn incremental_decode_token(
    tokenizer: &tokenizers::Tokenizer,
    token_id: u32,
    reverse_map: &HashMap<char, u8>,
    byte_buf: &mut Vec<u8>,
) -> String {
    if let Some(token_str) = tokenizer.id_to_token(token_id) {
        if let Some(raw_bytes) = token_str_to_bytes(&token_str, reverse_map) {
            byte_buf.extend_from_slice(&raw_bytes);
            return flush_valid_utf8(byte_buf);
        }
        // Not a ByteLevel token — flush pending bytes, then use the tokenizer's
        // native decode to handle SentencePiece, WordPiece, etc.
        let mut text = flush_valid_utf8(byte_buf);
        if let Ok(decoded) = tokenizer.decode(&[token_id], true) {
            text.push_str(&decoded);
        } else {
            text.push_str(&token_str);
        }
        return text;
    }
    // Token not in vocabulary (shouldn't happen during normal generation).
    flush_valid_utf8(byte_buf)
}

/// Text generation pipeline.
pub struct GenerationPipeline<M> {
    model: M,
    tokenizer: Tokenizer,
    sampler: Sampler,
    strip_thinking: bool,
    stop_strings: Vec<String>,
    stop_signal: Option<Arc<AtomicBool>>,
}

impl<M: CausalLM> GenerationPipeline<M> {
    pub fn new(model: M, tokenizer: Tokenizer, sampler: Sampler) -> Self {
        Self {
            model,
            tokenizer,
            sampler,
            strip_thinking: false,
            stop_strings: Vec::new(),
            stop_signal: None,
        }
    }

    pub fn with_strip_thinking(mut self, strip: bool) -> Self {
        self.strip_thinking = strip;
        self
    }

    pub fn with_stop_strings(mut self, strings: Vec<String>) -> Self {
        self.stop_strings = strings;
        self
    }

    /// Attach a stop signal. When set to `true`, generation aborts at the
    /// next loop iteration and returns what has been generated so far with
    /// `stop_reason = "cancelled"`.
    pub fn with_stop_signal(mut self, signal: Arc<AtomicBool>) -> Self {
        self.stop_signal = Some(signal);
        self
    }

    fn is_stopped(&self) -> bool {
        self.stop_signal
            .as_ref()
            .is_some_and(|s| s.load(Ordering::Relaxed))
    }

    fn check_stop_strings(&self, output: &str) -> Option<usize> {
        self.stop_strings
            .iter()
            .filter_map(|s| output.find(s))
            .min()
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
        on_piece: F,
    ) -> Result<(String, GenerationMetrics)>
    where
        F: FnMut(u32, &str),
    {
        let tokenize_t0 = Instant::now();
        let input_ids = self.tokenizer.encode(prompt)?;
        let tokenize_s = tokenize_t0.elapsed().as_secs_f64();
        let prompt_i32: Vec<i32> = input_ids.iter().map(|&x| x as i32).collect();
        let input = Array::from_slice_i32(&prompt_i32)?
            .reshape(&[1, prompt_i32.len() as i32])?;
        let (output, mut metrics) =
            self.generate_from_ids_with_metrics(&input, None, max_tokens, on_piece)?;
        metrics.total_s += tokenize_s;
        if metrics.ttft_s > 0.0 {
            metrics.ttft_s += tokenize_s;
        }
        if let Some(p) = metrics.profile.as_mut() {
            p.tokenize_s = tokenize_s;
        }
        Ok((output, metrics))
    }

    /// Generate from pre-tokenized input IDs with optional pixel values for
    /// multimodal (vision-language) models.
    ///
    /// During prefill, uses multimodal forward if `pixel_values` is `Some`.
    /// During decode, always uses text-only forward.
    ///
    /// If `max_tokens` is `None`, generation is uncapped and stops only on EOS.
    pub fn generate_from_ids_with_metrics<F>(
        &mut self,
        input_ids: &Array,
        pixel_values: Option<&Array>,
        max_tokens: Option<usize>,
        mut on_piece: F,
    ) -> Result<(String, GenerationMetrics)>
    where
        F: FnMut(u32, &str),
    {
        let t0 = Instant::now();
        let mut ttft_s: Option<f64> = None;
        #[allow(unused_assignments)]
        let mut stop_reason: Option<&'static str> = None;
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

        // Repetition controls should apply only to generated tokens, not prompt tokens.
        let mut history_tokens: Vec<u32> = Vec::new();

        // Prefill: run the full prompt through the model, with optional pixel values
        let stage_t0 = Instant::now();
        let logits = if let Some(pv) = pixel_values {
            self.model
                .forward_last_token_logits_multimodal(input_ids, Some(pv))
                .map_err(|e| anyhow::anyhow!("forward failed: {e}"))?
        } else {
            self.model
                .forward_last_token_logits(input_ids)
                .map_err(|e| anyhow::anyhow!("forward failed: {e}"))?
        };
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
        let mut last_token_id: Option<u32> = None;
        let reverse_map = build_byte_reverse_map();
        let mut byte_buf: Vec<u8> = Vec::new();
        let needs_byte_buf = needs_byte_buffering(self.tokenizer.inner());
        if self.sampler.is_greedy() {
            let stage_t0 = Instant::now();
            let mut token_arr = self
                .sampler
                .sample_raw_last_token_logits_array_at_step(&logits, 0)?;
            async_eval(&[&token_arr])?;
            if let Some(p) = profile.as_mut() {
                p.first_sample_s += stage_t0.elapsed().as_secs_f64();
                p.sample_calls += 1;
            }

            let mut generated = 0usize;
            loop {
                if self.is_stopped() {
                    stop_reason = Some("cancelled");
                    break;
                }

                let can_schedule_next = max_tokens.is_none_or(|limit| generated + 1 < limit);
                let next_token_arr = if can_schedule_next {
                    let input = sampled_token_array_to_input(&token_arr)?;
                    let stage_t0 = Instant::now();
                    let logits = self
                        .model
                        .forward_last_token_logits(&input)
                        .map_err(|e| anyhow::anyhow!("forward failed: {e}"))?;
                    if let Some(p) = profile.as_mut() {
                        p.decode_forward_s += stage_t0.elapsed().as_secs_f64();
                    }

                    let stage_t0 = Instant::now();
                    let next = self
                        .sampler
                        .sample_raw_last_token_logits_array_at_step(&logits, generated + 1)?;
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
                    stop_reason = Some("stop");
                    break;
                }

                generated_tokens.push(token);
                if has_runaway_repeat(&generated_tokens) {
                    stop_reason = Some("repeat_guard");
                    break;
                }

                let stage_t0 = Instant::now();
                let decode_result = if needs_byte_buf {
                    Ok(incremental_decode_token(
                        self.tokenizer.inner(),
                        token,
                        &reverse_map,
                        &mut byte_buf,
                    ))
                } else {
                    let prev_len = output.len();
                    self.tokenizer
                        .decode(&generated_tokens)
                        .map(|full| full[prev_len..].to_string())
                };
                let piece = decode_result?;
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
                if let Some(idx) = self.check_stop_strings(&output) {
                    output.truncate(idx);
                    stop_reason = Some("stop");
                    break;
                }
                history_tokens.push(token);
                token_count += 1;
                if ttft_s.is_none() {
                    ttft_s = Some(t0.elapsed().as_secs_f64());
                }

                if let Some(limit) = max_tokens {
                    if generated >= limit {
                        stop_reason = Some("length");
                        break;
                    }
                }

                if generated.is_multiple_of(32) {
                    trace_memory("decode", generated);
                }

                token_arr = match next_token_arr {
                    Some(next) => next,
                    None => {
                        stop_reason = Some("length");
                        break;
                    }
                };
            }
            trace_memory("after_first_token", 1);
        } else {
            let stage_t0 = Instant::now();
            let mut token = self
                .sampler
                .sample_raw_last_token_logits(&logits, &history_tokens)?;
            last_token_id = Some(token);
            if let Some(p) = profile.as_mut() {
                p.first_sample_s += stage_t0.elapsed().as_secs_f64();
                p.sample_calls += 1;
                if self.sampler.uses_host_sampling() {
                    p.cpu_logits_extractions += 1;
                }
            }
            if !self.tokenizer.is_stop_token(token) {
                generated_tokens.push(token);
                let stage_t0 = Instant::now();
                let decode_result = if needs_byte_buf {
                    Ok(incremental_decode_token(
                        self.tokenizer.inner(),
                        token,
                        &reverse_map,
                        &mut byte_buf,
                    ))
                } else {
                    let prev_len = output.len();
                    self.tokenizer
                        .decode(&generated_tokens)
                        .map(|full| full[prev_len..].to_string())
                };
                let piece = decode_result?;
                if let Some(p) = profile.as_mut() {
                    p.first_decode_s += stage_t0.elapsed().as_secs_f64();
                    p.decoded_pieces += 1;
                }
                on_piece(token, &piece);
                output.push_str(&piece);
                history_tokens.push(token);
                token_count += 1;
                if ttft_s.is_none() {
                    ttft_s = Some(t0.elapsed().as_secs_f64());
                }
                trace_memory("after_first_token", 1);
            }

            // Autoregressive decode loop. Uncapped when max_tokens is None.
            let mut generated = 1usize;
            loop {
                if self.is_stopped() {
                    stop_reason = Some("cancelled");
                    break;
                }

                if let Some(limit) = max_tokens {
                    if generated >= limit {
                        stop_reason = Some("length");
                        break;
                    }
                }
                if self.tokenizer.is_stop_token(token) {
                    stop_reason = Some("stop");
                    break;
                }
                let input = Array::from_int(token as i32)?.reshape(&[1, 1])?;

                let stage_t0 = Instant::now();
                let logits = self
                    .model
                    .forward_last_token_logits(&input)
                    .map_err(|e| anyhow::anyhow!("forward failed: {e}"))?;
                if let Some(p) = profile.as_mut() {
                    p.decode_forward_s += stage_t0.elapsed().as_secs_f64();
                }

                let stage_t0 = Instant::now();
                token = self
                    .sampler
                    .sample_raw_last_token_logits(&logits, &history_tokens)?;
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
                    stop_reason = Some("stop");
                    break;
                }

                generated_tokens.push(token);
                if has_runaway_repeat(&generated_tokens) {
                    stop_reason = Some("repeat_guard");
                    break;
                }

                let stage_t0 = Instant::now();
                let decode_result = if needs_byte_buf {
                    Ok(incremental_decode_token(
                        self.tokenizer.inner(),
                        token,
                        &reverse_map,
                        &mut byte_buf,
                    ))
                } else {
                    let prev_len = output.len();
                    self.tokenizer
                        .decode(&generated_tokens)
                        .map(|full| full[prev_len..].to_string())
                };
                let piece = decode_result?;
                if let Some(p) = profile.as_mut() {
                    p.decode_text_s += stage_t0.elapsed().as_secs_f64();
                    p.decoded_pieces += 1;
                }
                on_piece(token, &piece);
                output.push_str(&piece);
                if let Some(idx) = self.check_stop_strings(&output) {
                    output.truncate(idx);
                    stop_reason = Some("stop");
                    break;
                }
                history_tokens.push(token);
                token_count += 1;
                if generated.is_multiple_of(32) {
                    trace_memory("decode", generated);
                }
            }
        }
        trace_memory("end", token_count);
        // Flush any remaining bytes in the buffer (incomplete UTF-8 sequences
        // from the final tokens, if any).
        let leftover = String::from_utf8_lossy(&byte_buf).into_owned();
        if !leftover.is_empty() {
            on_piece(last_token_id.unwrap_or(0), &leftover);
            output.push_str(&leftover);
        }
        if let Some(p) = profile.as_mut() {
            p.finish_kv_stats(kv_cache_stats());
            let moe = mlx_models::moe_profile_stats();
            p.moe_router_host_s = moe.router_host_s;
            p.moe_routing_build_s = moe.routing_build_s;
            p.moe_expert_forward_s = moe.expert_forward_s;
            p.moe_shared_expert_s = moe.shared_expert_s;
            p.moe_single_token_fast_path_hits = moe.single_token_fast_path_hits;
            p.moe_device_router_shadow_checks = moe.device_router_shadow_checks;
            p.moe_device_router_shadow_mismatches = moe.device_router_shadow_mismatches;
        }
        if self.strip_thinking {
            output = strip_thinking_blocks(&output);
        }
        let metrics = GenerationMetrics {
            ttft_s: ttft_s.unwrap_or(0.0),
            total_s: t0.elapsed().as_secs_f64(),
            tokens: token_count,
            stop_reason: stop_reason.unwrap_or("unknown"),
            last_token_id,
            generated_token_ids: generated_tokens.clone(),
            tail_token_ids: generated_tokens
                .iter()
                .rev()
                .take(8)
                .copied()
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect(),
            stop_token_ids: self.tokenizer.stop_token_ids().to_vec(),
            profile,
        };
        Ok((output, metrics))
    }
}
