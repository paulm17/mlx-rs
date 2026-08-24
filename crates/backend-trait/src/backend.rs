use anyhow::Result;

use crate::types::{
    AppliedChatTemplate, ChatMessage, ChatTemplateOptions, EmbeddingOutput, GenerateOutput,
    GenerationMetrics, GenerationOptions, LoadedModelInfo,
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

    fn apply_chat_template_with_options(
        &self,
        messages: &[ChatMessage],
        _options: &ChatTemplateOptions,
    ) -> Result<AppliedChatTemplate> {
        Ok(AppliedChatTemplate {
            prompt: self.apply_chat_template(messages)?,
            additional_stops: Vec::new(),
            parser: None,
            generation_prompt: String::new(),
            chat_format: 0,
            parse_tool_calls: false,
        })
    }

    fn parse_chat_response(
        &self,
        _template: &AppliedChatTemplate,
        text: &str,
        _is_partial: bool,
    ) -> Result<String> {
        Ok(text.to_string())
    }

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

    fn generate_chat_stream_output(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
        _template: &AppliedChatTemplate,
        mut on_delta: Box<dyn FnMut(&str) -> bool + Send>,
    ) -> Result<GenerateOutput> {
        self.generate_stream_output(
            prompt,
            options,
            Box::new(move |piece| {
                let delta = serde_json::json!({"content": piece}).to_string();
                on_delta(&delta)
            }),
        )
    }

    fn embed(&mut self, text: &str) -> Result<EmbeddingOutput>;

    fn memory_info(&self) -> Option<String> {
        None
    }
}
