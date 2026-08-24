use std::path::Path;

use anyhow::{Context, Result};
use minijinja::{context, Environment};
use serde::Deserialize;

const DEFAULT_LLAMA3_TEMPLATE: &str = "{% for message in messages %}\n<|start_header_id|>{{ message.role }}<|end_header_id|>\n\n{{ message.content }}<|eot_id|>\n{% endfor %}\n<|start_header_id|>assistant<|end_header_id|>\n\n";

const GEMMA_TEMPLATE: &str = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% set loop_messages = messages[1:] %}{% else %}{% set system_message = '' %}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}<start_of_turn>user\n{% if loop.index0 == 0 and system_message %}{{ system_message }}\n\n{% endif %}{{ message.content }}<end_of_turn>\n{% elif message['role'] == 'assistant' %}<start_of_turn>model\n{{ message.content }}<end_of_turn>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}";

const QWEN_TEMPLATE: &str = "{% for message in messages %}\n<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}\n<|im_start|>assistant\n<think>\n\n</think>\n\n";


#[derive(Debug, Clone, Deserialize)]
struct TemplateEntry {
    name: String,
    template: String,
}

#[derive(Debug, Clone)]
pub struct ChatTemplate {
    template: String,
    bos_token: String,
    eos_token: String,
}

impl ChatTemplate {
    pub fn load(dir: &Path, architecture: Option<&str>) -> Result<Self> {
        let config_path = dir.join("tokenizer_config.json");
        if !config_path.exists() {
            return Ok(Self::default_for_architecture(architecture));
        }

        let data = std::fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?;

        let config: serde_json::Value = serde_json::from_str(&data)
            .with_context(|| format!("failed to parse {}", config_path.display()))?;

        let (default_bos, default_eos) = default_tokens_for_architecture(architecture);

        let bos_token = extract_token(&config, "bos_token").unwrap_or_else(|| default_bos.to_string());
        let eos_token = extract_token(&config, "eos_token").unwrap_or_else(|| default_eos.to_string());

        let template = extract_template(&config)
            .unwrap_or_else(|| default_template_for_architecture(architecture).to_string());

        Ok(Self {
            template,
            bos_token,
            eos_token,
        })
    }

    pub fn render(&self, messages: &[backend_trait::types::ChatMessage]) -> Result<String> {
        let mut env = Environment::new();
        env.add_template("chat", &self.template)
            .context("failed to compile chat template")?;

        let tmpl = env.get_template("chat").unwrap();

        let msg_values: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
            .collect();

        let rendered = tmpl.render(context! {
            messages => msg_values,
            bos_token => self.bos_token,
            eos_token => self.eos_token,
            add_generation_prompt => true,
        })
        .context("failed to render chat template")?;

        Ok(rendered)
    }

    pub fn bos_token(&self) -> &str {
        &self.bos_token
    }

    pub fn default_llama3() -> Self {
        Self {
            template: DEFAULT_LLAMA3_TEMPLATE.to_string(),
            bos_token: "<|begin_of_text|>".to_string(),
            eos_token: "<|end_of_text|>".to_string(),
        }
    }

    pub fn default_for_architecture(architecture: Option<&str>) -> Self {
        let (bos, eos) = default_tokens_for_architecture(architecture);
        Self {
            template: default_template_for_architecture(architecture).to_string(),
            bos_token: bos.to_string(),
            eos_token: eos.to_string(),
        }
    }
}

fn is_gemma_architecture(architecture: Option<&str>) -> bool {
    match architecture {
        Some(arch) => arch.starts_with("Gemma"),
        None => false,
    }
}


fn is_qwen_architecture(architecture: Option<&str>) -> bool {
    match architecture {
        Some(arch) => arch.starts_with("Qwen"),
        None => false,
    }
}
fn default_template_for_architecture(architecture: Option<&str>) -> &'static str {
    if is_gemma_architecture(architecture) {
        GEMMA_TEMPLATE
    } else if is_qwen_architecture(architecture) {
        QWEN_TEMPLATE
    } else {
        DEFAULT_LLAMA3_TEMPLATE
    }
}

fn default_tokens_for_architecture(architecture: Option<&str>) -> (&'static str, &'static str) {
    if is_gemma_architecture(architecture) {
        ("<bos>", "<eos>")
    } else if is_qwen_architecture(architecture) {
        ("", "</think>")
    } else {
        ("<|begin_of_text|>", "<|end_of_text|>")
    }
}

fn extract_token(config: &serde_json::Value, key: &str) -> Option<String> {
    match config.get(key)? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(obj) => obj.get("content").and_then(|v| v.as_str()).map(|s| s.to_string()),
        _ => None,
    }
}

fn extract_template(config: &serde_json::Value) -> Option<String> {
    match config.get("chat_template")? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(arr) => {
            for entry in arr {
                if let Ok(e) = serde_json::from_value::<TemplateEntry>(entry.clone()) {
                    if e.name == "default" {
                        return Some(e.template);
                    }
                }
            }
            arr.first()
                .and_then(|v| serde_json::from_value::<TemplateEntry>(v.clone()).ok())
                .map(|e| e.template)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend_trait::types::ChatMessage;
    use std::fs;

    #[test]
    fn test_default_template() {
        let tmpl = ChatTemplate::default_llama3();
        let messages = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there"),
            ChatMessage::user("How are you?"),
        ];
        let rendered = tmpl.render(&messages).unwrap();
        assert!(rendered.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(rendered.contains("Hello"));
        assert!(rendered.contains("<|start_header_id|>assistant<|end_header_id|>"));
    }

    #[test]
    fn test_default_gemma_template() {
        let tmpl = ChatTemplate::default_for_architecture(Some("Gemma4ForCausalLM"));
        let messages = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there"),
        ];
        let rendered = tmpl.render(&messages).unwrap();
        assert!(rendered.contains("<start_of_turn>user"));
        assert!(rendered.contains("Hello"));
        assert!(rendered.contains("<start_of_turn>model"));
        assert!(rendered.contains("<end_of_turn>"));
    }

    #[test]
    fn test_gemma_template_with_system() {
        let tmpl = ChatTemplate::default_for_architecture(Some("Gemma3ForCausalLM"));
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are helpful.".to_string(),
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                name: None,
            },
            ChatMessage::user("Hello"),
        ];
        let rendered = tmpl.render(&messages).unwrap();
        assert!(rendered.contains("<start_of_turn>user"));
        assert!(rendered.contains("You are helpful."));
        assert!(rendered.contains("<start_of_turn>model"));
    }

    #[test]
    fn test_load_from_tokenizer_config() {
        let dir = tempfile::tempdir().unwrap();
        let config = serde_json::json!({
            "chat_template": "{% for message in messages %}[{{ message.role }}] {{ message.content }}\n{% endfor %}[assistant]\n",
            "bos_token": "<s>",
            "eos_token": "</s>"
        });
        fs::write(dir.path().join("tokenizer_config.json"), config.to_string()).unwrap();

        let tmpl = ChatTemplate::load(dir.path(), None).unwrap();
        let messages = vec![ChatMessage::user("Hi")];
        let rendered = tmpl.render(&messages).unwrap();
        assert!(rendered.contains("[user] Hi"));
        assert!(rendered.contains("[assistant]"));
    }

    #[test]
    fn test_load_missing_config() {
        let dir = tempfile::tempdir().unwrap();
        let tmpl = ChatTemplate::load(dir.path(), None).unwrap();
        let messages = vec![ChatMessage::user("test")];
        let rendered = tmpl.render(&messages).unwrap();
        assert!(rendered.contains("<|start_header_id|>user<|end_header_id|>"));
    }

    #[test]
    fn test_load_missing_config_gemma_architecture() {
        let dir = tempfile::tempdir().unwrap();
        let tmpl = ChatTemplate::load(dir.path(), Some("Gemma4ForCausalLM")).unwrap();
        let messages = vec![ChatMessage::user("test")];
        let rendered = tmpl.render(&messages).unwrap();
        assert!(rendered.contains("<start_of_turn>user"));
        assert!(rendered.contains("<start_of_turn>model"));
    }

    #[test]
    fn test_load_null_template_gemma_architecture() {
        let dir = tempfile::tempdir().unwrap();
        let config = serde_json::json!({
            "chat_template": null,
            "bos_token": "<bos>",
            "eos_token": "<eos>"
        });
        fs::write(dir.path().join("tokenizer_config.json"), config.to_string()).unwrap();

        let tmpl = ChatTemplate::load(dir.path(), Some("Gemma4ForCausalLM")).unwrap();
        assert_eq!(tmpl.bos_token(), "<bos>");
        assert_eq!(tmpl.eos_token, "<eos>");
        let messages = vec![ChatMessage::user("Hello")];
        let rendered = tmpl.render(&messages).unwrap();
        assert!(rendered.contains("<start_of_turn>user"));
    }

    #[test]
    fn test_template_array_format() {
        let dir = tempfile::tempdir().unwrap();
        let config = serde_json::json!({
            "chat_template": [
                {"name": "default", "template": "{% for message in messages %}{{ message.content }}{% endfor %}"},
                {"name": "tool_use", "template": "tools mode"}
            ]
        });
        fs::write(dir.path().join("tokenizer_config.json"), config.to_string()).unwrap();

        let tmpl = ChatTemplate::load(dir.path(), None).unwrap();
        let messages = vec![ChatMessage::user("hello")];
        let rendered = tmpl.render(&messages).unwrap();
        assert_eq!(rendered, "hello");
    }

    #[test]
    fn test_token_as_object() {
        let dir = tempfile::tempdir().unwrap();
        let config = serde_json::json!({
            "bos_token": {"content": "<s>", "lstrip": false},
            "eos_token": {"content": "</s>", "rstrip": false}
        });
        fs::write(dir.path().join("tokenizer_config.json"), config.to_string()).unwrap();

        let tmpl = ChatTemplate::load(dir.path(), None).unwrap();
        assert_eq!(tmpl.bos_token(), "<s>");
        assert_eq!(tmpl.eos_token, "</s>");
    }
}
