use anyhow::Result;
use mlx_models::{Gemma4, Gemma4Config};
use mlx_nn::VarBuilder;
use std::path::Path;
use tokenizers::Tokenizer;

/// Components loaded for a VLM.
pub struct VlmComponents {
    pub model: Gemma4,
    pub tokenizer: Tokenizer,
    pub config: Gemma4Config,
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

    let vb = VarBuilder::from_dir(model_dir, mlx_core::DType::Float16)?;
    let model = Gemma4::new(&vb, &config)?;

    Ok(VlmComponents {
        model,
        tokenizer,
        config,
    })
}
