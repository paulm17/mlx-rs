use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationMetrics {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub total_tokens: usize,
    pub ttft_s: Option<f64>,
    pub total_s: Option<f64>,
    pub tokens_per_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateOutput {
    pub text: String,
    pub stop_reason: StopReason,
    pub metrics: GenerationMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StopReason {
    Eos,
    MaxTokens,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedModelInfo {
    pub model_path: String,
    pub context_length: Option<usize>,
    pub embedding_dimension: Option<usize>,
    pub vocab_size: Option<usize>,
}
