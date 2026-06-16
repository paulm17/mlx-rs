use std::collections::HashMap;

use crate::array::Array;
use crate::llama::{LlamaConfig, LlamaModel};
use crate::model::Model;
use crate::qwen3::{Qwen3Config, Qwen3Model};
use crate::gemma3::{Gemma3Config, Gemma3Model};

pub fn detect_architecture(config: &serde_json::Value) -> String {
    config
        .get("architectures")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|v| v.as_str())
        .unwrap_or("LlamaForCausalLM")
        .to_string()
}

pub fn create_model(
    architecture: &str,
    tensors: HashMap<String, Array>,
    config: &serde_json::Value,
) -> anyhow::Result<Box<dyn Model>> {
    match architecture {
        "LlamaForCausalLM" | "LlamaForSequenceClassification" => {
            let cfg = LlamaConfig::from_json(config)?;
            let model = LlamaModel::load_from_tensors(tensors, cfg)?;
            Ok(Box::new(model))
        }
        "Qwen3ForCausalLM" => {
            let cfg = Qwen3Config::from_json(config)?;
            let model = Qwen3Model::load_from_tensors(tensors, cfg)?;
            Ok(Box::new(model))
        }
        "Gemma3ForCausalLM" | "Gemma3ForConditionalGeneration" => {
            let cfg = Gemma3Config::from_json(config)?;
            let model = Gemma3Model::load_from_tensors(tensors, cfg)?;
            Ok(Box::new(model))
        }
        _ => anyhow::bail!("unsupported architecture: {architecture}"),
    }
}

pub fn supported_architectures() -> Vec<&'static str> {
    vec![
        "LlamaForCausalLM",
        "LlamaForSequenceClassification",
        "Qwen3ForCausalLM",
        "Gemma3ForCausalLM",
        "Gemma3ForConditionalGeneration",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_llama_architecture() {
        let config = serde_json::json!({
            "architectures": ["LlamaForCausalLM"]
        });
        assert_eq!(detect_architecture(&config), "LlamaForCausalLM");
    }

    #[test]
    fn test_detect_qwen3_architecture() {
        let config = serde_json::json!({
            "architectures": ["Qwen3ForCausalLM"]
        });
        assert_eq!(detect_architecture(&config), "Qwen3ForCausalLM");
    }

    #[test]
    fn test_detect_gemma3_architecture() {
        let config = serde_json::json!({
            "architectures": ["Gemma3ForCausalLM"]
        });
        assert_eq!(detect_architecture(&config), "Gemma3ForCausalLM");
    }

    #[test]
    fn test_detect_missing_architecture_defaults_to_llama() {
        let config = serde_json::json!({});
        assert_eq!(detect_architecture(&config), "LlamaForCausalLM");
    }

    #[test]
    fn test_unsupported_architecture() {
        let tensors = HashMap::new();
        let config = serde_json::json!({});
        let result = create_model("UnsupportedModel", tensors, &config);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("unsupported architecture"));
    }

    #[test]
    fn test_supported_architectures_list() {
        let archs = supported_architectures();
        assert!(archs.contains(&"LlamaForCausalLM"));
        assert!(archs.contains(&"Qwen3ForCausalLM"));
        assert!(archs.contains(&"Gemma3ForCausalLM"));
    }
}
