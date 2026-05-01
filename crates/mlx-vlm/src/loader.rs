use anyhow::Result;
use mlx_models::{sanitize_weights, Gemma4, Gemma4Config};
use mlx_nn::VarBuilder;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

use crate::processing::Gemma4ImageProcessor;

/// Components loaded for a VLM.
pub struct VlmComponents {
    pub model: Gemma4,
    pub tokenizer: Tokenizer,
    pub config: Gemma4Config,
    pub processor: Gemma4ImageProcessor,
    pub eos_token_id: u32,
}

/// Load a Gemma4 VLM from a local directory.
///
/// Expects `config.json`, `tokenizer.json`, and safetensors weight files
/// in `model_dir`.
pub fn load_gemma4_vlm(model_dir: &Path) -> Result<VlmComponents> {
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config: Gemma4Config = serde_json::from_str(&config_str)?;

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    // Determine EOS token ID from config and tokenizer_config
    let eos_token_id = load_eos_token_id(model_dir, &config_str)?;

    // Load raw weights, sanitize, then build VarBuilder
    let shards = VarBuilder::discover_shards(model_dir)?;
    let mut all_weights = HashMap::new();
    for shard in &shards {
        let weights = mlx_core::safetensors::load(shard)?;
        all_weights.extend(weights);
    }
    let sanitized = sanitize_weights(all_weights, &config);
    let vb = VarBuilder::from_weights(sanitized, mlx_core::DType::Float16);
    let model = Gemma4::new(&vb, &config)?;

    let processor = Gemma4ImageProcessor::new(
        config.vision_config.patch_size,
        config.vision_soft_tokens_per_image,
        config.vision_config.pooling_kernel_size,
    );

    Ok(VlmComponents {
        model,
        tokenizer,
        config,
        processor,
        eos_token_id,
    })
}

fn load_eos_token_id(model_dir: &Path, config_str: &str) -> Result<u32> {
    let config: Value = serde_json::from_str(config_str)?;

    // 1. Check config.json top-level eos_token_id
    if let Some(id) = extract_eos_id(&config.get("eos_token_id")) {
        return Ok(id);
    }
    // 2. Check config.json text_config.eos_token_id
    if let Some(tc) = config.get("text_config") {
        if let Some(id) = extract_eos_id(&tc.get("eos_token_id")) {
            return Ok(id);
        }
    }
    // 3. Check tokenizer_config.json
    let tk_path = model_dir.join("tokenizer_config.json");
    if let Ok(raw) = std::fs::read_to_string(&tk_path) {
        if let Ok(tk_cfg) = serde_json::from_str::<Value>(&raw) {
            if let Some(id) = extract_eos_id(&tk_cfg.get("eos_token_id")) {
                return Ok(id);
            }
            // Also check eos_token string -> token ID via tokenizer
            if let Some(eos_str) = tk_cfg.get("eos_token").and_then(|v| v.as_str()) {
                let tokenizer = Tokenizer::from_file(&model_dir.join("tokenizer.json"))
                    .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
                if let Some(id) = tokenizer.token_to_id(eos_str) {
                    return Ok(id);
                }
            }
        }
    }
    // 4. Fallback: try common EOS token strings
    let tokenizer = Tokenizer::from_file(&model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
    for tok in &["</s>", "<eos>", "<|endoftext|>", "<|im_end|>"] {
        if let Some(id) = tokenizer.token_to_id(tok) {
            return Ok(id);
        }
    }
    Ok(2)
}

/// Extract a single eos_token_id from a JSON value that could be an integer or array.
fn extract_eos_id(value: &Option<&Value>) -> Option<u32> {
    match value {
        Some(Value::Number(n)) => n.as_u64().map(|v| v as u32),
        Some(Value::Array(arr)) => arr.first().and_then(|v| v.as_u64()).map(|v| v as u32),
        _ => None,
    }
}
