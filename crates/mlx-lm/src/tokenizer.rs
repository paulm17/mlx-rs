use anyhow::Result;
use std::path::Path;

/// Tokenizer wrapper around HuggingFace tokenizers.
#[derive(Clone)]
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    eos_token_id: u32,
    stop_token_ids: Vec<u32>,
}

impl Tokenizer {
    /// Load a tokenizer from a file (tokenizer.json).
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        // Try to detect EOS token ID
        let eos_token_id = inner
            .token_to_id("</s>")
            .or_else(|| inner.token_to_id("<|endoftext|>"))
            .or_else(|| inner.token_to_id("<|im_end|>"))
            .unwrap_or(2);
        let stop_token_ids = vec![eos_token_id];

        Ok(Self {
            inner,
            eos_token_id,
            stop_token_ids,
        })
    }

    /// Set the EOS token ID explicitly.
    pub fn with_eos(mut self, eos: u32) -> Self {
        self.eos_token_id = eos;
        self.stop_token_ids = vec![eos];
        self
    }

    /// Set the full list of stop token IDs.
    pub fn with_stop_tokens(mut self, mut stop_ids: Vec<u32>) -> Self {
        stop_ids.sort_unstable();
        stop_ids.dedup();
        if stop_ids.is_empty() {
            stop_ids.push(self.eos_token_id);
        }
        self.eos_token_id = stop_ids[0];
        self.stop_token_ids = stop_ids;
        self
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("encoding failed: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs to text.
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.inner
            .decode(tokens, true)
            .map_err(|e| anyhow::anyhow!("decoding failed: {e}"))
    }

    /// Get the EOS token ID.
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Get the configured stop token IDs.
    pub fn stop_token_ids(&self) -> &[u32] {
        &self.stop_token_ids
    }

    /// Check if a token ID is a configured stop token.
    pub fn is_stop_token(&self, token_id: u32) -> bool {
        self.stop_token_ids.contains(&token_id)
    }

    /// Get a reference to the underlying tokenizer.
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }
}
