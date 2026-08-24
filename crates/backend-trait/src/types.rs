use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tool_calls: Vec<ChatToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ChatToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatToolFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Deserialize)]
struct ChatMessageWire {
    role: String,
    #[serde(default)]
    content: Value,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ChatToolCall>,
    #[serde(default)]
    tool_call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

impl<'de> serde::Deserialize<'de> for ChatMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ChatMessageWire::deserialize(deserializer)?;
        let content = match wire.content {
            Value::Null => String::new(),
            Value::String(value) => value,
            Value::Array(parts) => {
                let mut text = String::new();
                for (index, part) in parts.into_iter().enumerate() {
                    let object = part.as_object().ok_or_else(|| {
                        serde::de::Error::custom(format!(
                            "content[{index}] must be an object with type=text"
                        ))
                    })?;
                    if object.get("type").and_then(Value::as_str) != Some("text") {
                        return Err(serde::de::Error::custom(format!(
                            "content[{index}] has unsupported type; only text parts are supported"
                        )));
                    }
                    let value = object.get("text").and_then(Value::as_str).ok_or_else(|| {
                        serde::de::Error::custom(format!(
                            "content[{index}].text must be a string"
                        ))
                    })?;
                    text.push_str(value);
                }
                text
            }
            other => {
                return Err(serde::de::Error::custom(format!(
                    "content must be a string, null, or an array of text parts (got {})",
                    other
                )))
            }
        };

        Ok(Self {
            role: wire.role,
            content,
            reasoning_content: wire.reasoning_content,
            tool_calls: wire.tool_calls,
            tool_call_id: wire.tool_call_id,
            name: wire.name,
        })
    }
}

impl ChatMessage {
    pub fn system(content: &str) -> Self {
        Self {
            role: "system".to_string(),
            content: content.to_string(),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    pub fn user(content: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: content.to_string(),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    pub fn assistant(content: &str) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.to_string(),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTemplateOptions {
    pub enable_thinking: bool,
    pub reasoning_format: Option<String>,
    pub tools_json: Option<String>,
    pub tool_choice: Option<String>,
    pub parallel_tool_calls: bool,
    pub chat_template_kwargs: Option<String>,
    pub parse_tool_calls: bool,
}

impl Default for ChatTemplateOptions {
    fn default() -> Self {
        Self {
            enable_thinking: false,
            reasoning_format: None,
            tools_json: None,
            tool_choice: None,
            parallel_tool_calls: true,
            chat_template_kwargs: None,
            parse_tool_calls: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedChatTemplate {
    pub prompt: String,
    pub additional_stops: Vec<String>,
    pub parser: Option<String>,
    pub generation_prompt: String,
    pub chat_format: i32,
    pub parse_tool_calls: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationOptions {
    pub max_tokens: Option<usize>,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub stop: Option<Vec<String>>,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: None,
            temperature: 0.6,
            top_p: 0.9,
            top_k: 40,
            min_p: 0.0,
            stop: None,
        }
    }
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
pub struct EmbeddingData {
    pub object: String,
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingOutput {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedModelInfo {
    pub model_path: String,
    pub context_length: Option<usize>,
    pub embedding_dimension: Option<usize>,
    pub vocab_size: Option<usize>,
}
