//! Chat template support for LLM generation.
//!
//! Jinja-based chat template rendering compatible with HuggingFace's
//! `tokenizer.apply_chat_template()`, following candle-rs patterns.
//!
//! # Example
//!
//! ```no_run
//! use mlx_lm::chat_template::{ChatTemplate, Message};
//!
//! let template = ChatTemplate::from_tokenizer_config("path/to/tokenizer_config.json").unwrap();
//! let messages = vec![
//!     Message::system("You are helpful."),
//!     Message::user("Hello!"),
//! ];
//! let prompt = template.apply_for_generation(&messages).unwrap();
//! ```

use minijinja::{context, Environment};
use serde::{Deserialize, Serialize};
use std::path::Path;

// -----------------------------------------------------------------------
// Message
// -----------------------------------------------------------------------

/// A chat message with role and content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

// -----------------------------------------------------------------------
// Options
// -----------------------------------------------------------------------

/// Options for applying a chat template.
#[derive(Debug, Clone, Default)]
pub struct ChatTemplateOptions {
    /// Add tokens that prompt the model to generate an assistant response.
    pub add_generation_prompt: bool,
    /// Continue the final message instead of starting a new one.
    pub continue_final_message: bool,
    /// Enable thinking/reasoning mode.
    pub enable_thinking: bool,
}

impl ChatTemplateOptions {
    pub fn for_generation() -> Self {
        Self {
            add_generation_prompt: true,
            ..Default::default()
        }
    }

    pub fn for_training() -> Self {
        Self {
            add_generation_prompt: false,
            ..Default::default()
        }
    }

    pub fn with_thinking(mut self) -> Self {
        self.enable_thinking = true;
        self
    }
}

// -----------------------------------------------------------------------
// Config deserialization (tokenizer_config.json)
// -----------------------------------------------------------------------

#[derive(Debug, Clone, Default, Deserialize)]
struct TokenConfig {
    #[serde(default)]
    bos_token: Option<StringOrToken>,
    #[serde(default)]
    eos_token: Option<StringOrToken>,
    #[serde(default)]
    chat_template: Option<ChatTemplateConfig>,
}

/// Handle both `"<s>"` and `{"content": "<s>"}` forms.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum StringOrToken {
    String(String),
    Token { content: String },
}

impl StringOrToken {
    fn as_str(&self) -> &str {
        match self {
            Self::String(s) => s,
            Self::Token { content } => content,
        }
    }
}

/// Chat template can be a single string or multiple named templates.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum ChatTemplateConfig {
    Single(String),
    Multiple(Vec<NamedTemplate>),
}

#[derive(Debug, Clone, Deserialize)]
struct NamedTemplate {
    name: String,
    template: String,
}

// -----------------------------------------------------------------------
// ChatTemplate
// -----------------------------------------------------------------------

/// Chat template renderer using MiniJinja.
pub struct ChatTemplate {
    env: Environment<'static>,
    bos_token: String,
    eos_token: String,
}

impl ChatTemplate {
    /// Create from a Jinja template string.
    pub fn new(
        template: impl Into<String>,
        bos_token: impl Into<String>,
        eos_token: impl Into<String>,
    ) -> Result<Self, ChatTemplateError> {
        let mut env = Environment::new();
        // `raise_exception` is used by many HF templates
        env.add_function("raise_exception", |msg: String| -> Result<String, _> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        });

        let tmpl: String = template.into();
        env.add_template_owned("chat".to_string(), tmpl)
            .map_err(|e| ChatTemplateError::Template(e.to_string()))?;

        Ok(Self {
            env,
            bos_token: bos_token.into(),
            eos_token: eos_token.into(),
        })
    }

    /// Load from a `tokenizer_config.json` file.
    pub fn from_tokenizer_config(path: impl AsRef<Path>) -> Result<Self, ChatTemplateError> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| ChatTemplateError::Io(e.to_string()))?;
        Self::from_tokenizer_config_str(&content)
    }

    /// Load from a model directory.
    ///
    /// Resolution order:
    /// 1) `tokenizer_config.json` with inline `chat_template`
    /// 2) `chat_template.jinja` file in the same directory
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self, ChatTemplateError> {
        let model_dir = model_dir.as_ref();
        let config_path = model_dir.join("tokenizer_config.json");
        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)
                .map_err(|e| ChatTemplateError::Io(e.to_string()))?;
            if let Ok(cfg) = serde_json::from_str::<TokenConfig>(&content) {
                if let Some(template) = cfg.chat_template {
                    let template = match template {
                        ChatTemplateConfig::Single(t) => t,
                        ChatTemplateConfig::Multiple(templates) => templates
                            .iter()
                            .find(|t| t.name == "default")
                            .or_else(|| templates.first())
                            .map(|t| t.template.clone())
                            .ok_or(ChatTemplateError::NoTemplate)?,
                    };
                    let bos = cfg
                        .bos_token
                        .as_ref()
                        .map(|t| t.as_str().to_string())
                        .unwrap_or_default();
                    let eos = cfg
                        .eos_token
                        .as_ref()
                        .map(|t| t.as_str().to_string())
                        .unwrap_or_default();
                    return Self::new(template, bos, eos);
                }

                // Fallback for models that store template in chat_template.jinja.
                let jinja_path = model_dir.join("chat_template.jinja");
                if jinja_path.exists() {
                    let template = std::fs::read_to_string(jinja_path)
                        .map_err(|e| ChatTemplateError::Io(e.to_string()))?;
                    let bos = cfg
                        .bos_token
                        .as_ref()
                        .map(|t| t.as_str().to_string())
                        .unwrap_or_default();
                    let eos = cfg
                        .eos_token
                        .as_ref()
                        .map(|t| t.as_str().to_string())
                        .unwrap_or_default();
                    return Self::new(template, bos, eos);
                }
            }
        }

        Err(ChatTemplateError::NoTemplate)
    }

    /// Load from `tokenizer_config.json` content.
    pub fn from_tokenizer_config_str(json: &str) -> Result<Self, ChatTemplateError> {
        let config: TokenConfig =
            serde_json::from_str(json).map_err(|e| ChatTemplateError::Parse(e.to_string()))?;

        let template = match config.chat_template {
            Some(ChatTemplateConfig::Single(t)) => t,
            Some(ChatTemplateConfig::Multiple(templates)) => templates
                .iter()
                .find(|t| t.name == "default")
                .or_else(|| templates.first())
                .map(|t| t.template.clone())
                .ok_or(ChatTemplateError::NoTemplate)?,
            None => return Err(ChatTemplateError::NoTemplate),
        };

        let bos = config
            .bos_token
            .map(|t| t.as_str().to_string())
            .unwrap_or_default();
        let eos = config
            .eos_token
            .map(|t| t.as_str().to_string())
            .unwrap_or_default();

        Self::new(template, bos, eos)
    }

    // -------------------------------------------------------------------
    // Presets
    // -------------------------------------------------------------------

    /// ChatML template (Qwen, SmolLM, etc.).
    pub fn chatml() -> Self {
        let template = r#"
{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\n' + message.content | trim + '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{%- endif %}
"#;
        Self::new(template, "", "<|im_end|>").unwrap()
    }

    /// Qwen 3.5 text template fallback (for MiniJinja-incompatible HF templates).
    pub fn qwen35() -> Self {
        let template = r#"
{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\n' + message.content | trim + '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' }}
{%- endif %}
"#;
        Self::new(template, "", "<|im_end|>").unwrap()
    }

    /// Llama 3 / 3.1 / 3.2 chat template.
    pub fn llama3() -> Self {
        let template = r#"
{%- set loop_messages = messages %}
{%- for message in loop_messages %}
    {%- set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}
    {%- if loop.index0 == 0 %}
        {{- bos_token + content }}
    {%- else %}
        {{- content }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"#;
        Self::new(template, "<|begin_of_text|>", "<|eot_id|>").unwrap()
    }

    /// Gemma template.
    pub fn gemma() -> Self {
        let template = r#"
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- '<start_of_turn>user\n' + message['content'] + '<end_of_turn>\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- '<start_of_turn>model\n' + message['content'] + '<end_of_turn>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<start_of_turn>model\n' }}
{%- endif %}
"#;
        Self::new(template, "<bos>", "<eos>").unwrap()
    }

    /// Mistral Instruct template.
    pub fn mistral() -> Self {
        let template = r#"
{{- bos_token }}
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- '[INST] ' + message['content'] + ' [/INST]' }}
    {%- elif message['role'] == 'assistant' %}
        {{- ' ' + message['content'] + eos_token }}
    {%- endif %}
{%- endfor %}
"#;
        Self::new(template, "<s>", "</s>").unwrap()
    }

    // -------------------------------------------------------------------
    // Apply
    // -------------------------------------------------------------------

    /// Apply the chat template to messages.
    pub fn apply(
        &self,
        messages: &[Message],
        options: &ChatTemplateOptions,
    ) -> Result<String, ChatTemplateError> {
        let template = self
            .env
            .get_template("chat")
            .map_err(|e| ChatTemplateError::Template(e.to_string()))?;

        let result = template
            .render(context! {
                messages => messages,
                add_generation_prompt => options.add_generation_prompt,
                continue_final_message => options.continue_final_message,
                enable_thinking => options.enable_thinking,
                bos_token => &self.bos_token,
                eos_token => &self.eos_token,
            })
            .map_err(|e| ChatTemplateError::Render(e.to_string()))?;

        Ok(result.trim_start().to_string())
    }

    /// Convenience: apply with `add_generation_prompt = true`.
    pub fn apply_for_generation(
        &self,
        messages: &[Message],
    ) -> Result<String, ChatTemplateError> {
        self.apply(messages, &ChatTemplateOptions::for_generation())
    }
}

// -----------------------------------------------------------------------
// Conversation
// -----------------------------------------------------------------------

/// Multi-turn conversation manager.
pub struct Conversation {
    messages: Vec<Message>,
    template: ChatTemplate,
    options: ChatTemplateOptions,
}

impl Conversation {
    /// Create a new conversation with a system prompt.
    pub fn new(template: ChatTemplate, system_prompt: impl Into<String>) -> Self {
        Self {
            messages: vec![Message::system(system_prompt)],
            template,
            options: ChatTemplateOptions::for_generation(),
        }
    }

    /// Create without a system prompt.
    pub fn without_system(template: ChatTemplate) -> Self {
        Self {
            messages: Vec::new(),
            template,
            options: ChatTemplateOptions::for_generation(),
        }
    }

    /// Set options (e.g., enable thinking mode).
    pub fn with_options(mut self, options: ChatTemplateOptions) -> Self {
        self.options = options;
        self
    }

    /// Add a user message and return the formatted prompt for generation.
    pub fn user_turn(
        &mut self,
        content: impl Into<String>,
    ) -> Result<String, ChatTemplateError> {
        self.messages.push(Message::user(content));
        self.template.apply(&self.messages, &self.options)
    }

    /// Record the assistant's response after generation.
    pub fn assistant_response(&mut self, content: impl Into<String>) {
        self.messages.push(Message::assistant(content));
    }

    /// Get the conversation history.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Clear history (keeps system prompt if present).
    pub fn clear(&mut self) {
        if let Some(first) = self.messages.first() {
            if first.role == "system" {
                let system = self.messages.remove(0);
                self.messages.clear();
                self.messages.push(system);
                return;
            }
        }
        self.messages.clear();
    }
}

// -----------------------------------------------------------------------
// Error
// -----------------------------------------------------------------------

/// Errors that can occur with chat templates.
#[derive(Debug)]
pub enum ChatTemplateError {
    Io(String),
    Parse(String),
    Template(String),
    Render(String),
    NoTemplate,
}

impl std::fmt::Display for ChatTemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::Parse(e) => write!(f, "Parse error: {e}"),
            Self::Template(e) => write!(f, "Template error: {e}"),
            Self::Render(e) => write!(f, "Render error: {e}"),
            Self::NoTemplate => write!(f, "No chat_template found in config"),
        }
    }
}

impl std::error::Error for ChatTemplateError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chatml_basic() {
        let template = ChatTemplate::chatml();
        let messages = vec![Message::system("You are helpful."), Message::user("Hello")];

        let result = template.apply_for_generation(&messages).unwrap();

        assert!(result.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(result.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_llama3_format() {
        let template = ChatTemplate::llama3();
        let messages = vec![Message::system("You are helpful."), Message::user("Hello")];

        let result = template.apply_for_generation(&messages).unwrap();

        assert!(result.contains("<|begin_of_text|>"));
        assert!(result.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(result.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(result.contains("<|eot_id|>"));
    }

    #[test]
    fn test_multi_turn() {
        let mut conv = Conversation::new(ChatTemplate::chatml(), "You are helpful.");
        let prompt1 = conv.user_turn("Hi").unwrap();
        assert!(prompt1.contains("Hi"));

        conv.assistant_response("Hello!");
        let prompt2 = conv.user_turn("How are you?").unwrap();
        assert!(prompt2.contains("Hi"));
        assert!(prompt2.contains("Hello!"));
        assert!(prompt2.contains("How are you?"));
    }

    #[test]
    fn test_from_json_config() {
        let json = r#"{
            "bos_token": "<s>",
            "eos_token": "</s>",
            "chat_template": "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
        }"#;

        let template = ChatTemplate::from_tokenizer_config_str(json).unwrap();
        let messages = vec![Message::user("test")];
        let result = template.apply_for_generation(&messages).unwrap();
        assert!(result.contains("user: test"));
    }
}
