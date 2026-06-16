use anyhow::Result;

use crate::types::{
    ChatMessage, EmbeddingOutput, GenerateOutput, GenerationMetrics, GenerationOptions,
    LoadedModelInfo,
};

pub trait Backend: Send {
    fn model_info(&self) -> LoadedModelInfo;

    fn tokenize(&self, text: &str, add_bos: bool) -> Result<Vec<i32>>;

    fn detokenize(&self, tokens: &[i32]) -> Result<String>;

    fn detokenize_piece(&self, token_id: i32) -> Result<String>;

    fn is_eog(&self, token_id: i32) -> bool;

    fn token_eos(&self) -> i32;

    fn embeddings_enabled(&self) -> bool;

    fn supports_chat_template(&self) -> bool {
        true
    }

    fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String>;

    fn generate(&mut self, prompt: &str, options: &GenerationOptions) -> Result<GenerateOutput>;

    fn generate_stream(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
        on_token: Box<dyn FnMut(&str) -> bool + Send>,
    ) -> Result<GenerationMetrics>;

    fn generate_stream_output(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
        on_token: Box<dyn FnMut(&str) -> bool + Send>,
    ) -> Result<GenerateOutput>;

    fn embed(&mut self, text: &str) -> Result<EmbeddingOutput>;

    fn memory_info(&self) -> Option<String> {
        None
    }
}
