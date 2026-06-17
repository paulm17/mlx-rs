use std::collections::HashMap;

use crate::array::Array;
use crate::bert::{BertConfig, BertModel};
use crate::llama::{LlamaConfig, LlamaModel};
use crate::model::{EncoderModel, Model};
use crate::qwen3::{Qwen3Config, Qwen3Model};
use crate::qwen3_5::{Qwen3_5Config, Qwen3_5Model};
use crate::gemma3::{Gemma3Config, Gemma3Model};
use crate::gemma4::{Gemma4Config, Gemma4Model};

pub fn detect_architecture(config: &serde_json::Value) -> String {
    config
        .get("architectures")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|v| v.as_str())
        .unwrap_or("LlamaForCausalLM")
        .to_string()
}

pub fn is_encoder_architecture(architecture: &str) -> bool {
    matches!(architecture, "BertModel" | "BertForMaskedLM" | "BertForSequenceClassification")
}

pub fn create_model(
    architecture: &str,
    tensors: HashMap<String, Array>,
    config: &serde_json::Value,
) -> anyhow::Result<Box<dyn Model>> {
    match architecture {
        "LlamaForCausalLM" | "LlamaForSequenceClassification" | "Qwen2ForCausalLM" => {
            let cfg = LlamaConfig::from_json(config)?;
            let model = LlamaModel::load_from_tensors(tensors, cfg)?;
            Ok(Box::new(model))
        }
        "Qwen3ForCausalLM" => {
            let cfg = Qwen3Config::from_json(config)?;
            let model = Qwen3Model::load_from_tensors(tensors, cfg)?;
            Ok(Box::new(model))
        }
        "Qwen3_5ForCausalLM" | "Qwen3_5ForConditionalGeneration"
        | "Qwen3NextForCausalLM" | "Qwen3NextForConditionalGeneration"
        | "Qwen3_5MoeForConditionalGeneration" | "Qwen3_5MoeForCausalLM"
        | "Qwen3NextMoeForConditionalGeneration" | "Qwen3NextMoeForCausalLM" => {
            let cfg = Qwen3_5Config::from_json(config)?;
            let model = Qwen3_5Model::load_from_tensors(tensors, cfg)?;
            Ok(Box::new(model))
        }
        "Gemma3ForCausalLM" | "Gemma3ForConditionalGeneration" => {
            let cfg = Gemma3Config::from_json(config)?;
            let model = Gemma3Model::load_from_tensors(tensors, cfg)?;
            Ok(Box::new(model))
        }
        "Gemma4ForCausalLM" | "Gemma4ForConditionalGeneration" => {
            let cfg = Gemma4Config::from_json(config)?;
            let model = Gemma4Model::load_from_tensors(tensors, cfg)?;
            Ok(Box::new(model))
        }
        "BertModel" | "BertForMaskedLM" | "BertForSequenceClassification" => {
            let cfg = BertConfig::from_json(config)?;
            let model = BertModel::load_from_tensors(tensors, cfg)?;
            Ok(Box::new(model))
        }
        _ => anyhow::bail!("unsupported architecture: {architecture}"),
    }
}

pub fn create_encoder_model(
    architecture: &str,
    tensors: HashMap<String, Array>,
    config: &serde_json::Value,
) -> anyhow::Result<Box<dyn EncoderModel>> {
    match architecture {
        "BertModel" | "BertForMaskedLM" | "BertForSequenceClassification" => {
            let cfg = BertConfig::from_json(config)?;
            let model = BertModel::load_from_tensors(tensors, cfg)?;
            Ok(Box::new(model))
        }
        _ => anyhow::bail!("architecture {architecture} is not an encoder model"),
    }
}

pub fn supported_architectures() -> Vec<&'static str> {
    vec![
        "LlamaForCausalLM",
        "LlamaForSequenceClassification",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3_5ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3NextForCausalLM",
        "Qwen3NextForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
        "Qwen3NextMoeForConditionalGeneration",
        "Qwen3NextMoeForCausalLM",
        "Gemma3ForCausalLM",
        "Gemma3ForConditionalGeneration",
        "Gemma4ForCausalLM",
        "Gemma4ForConditionalGeneration",
        "BertModel",
        "BertForMaskedLM",
        "BertForSequenceClassification",
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
