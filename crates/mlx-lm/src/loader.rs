use anyhow::Result;
use mlx_core::DType;
use mlx_nn::VarBuilder;
use serde_json::Value;
use std::path::Path;

use crate::generate::CausalLM;

fn parse_env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.trim().parse::<usize>().ok())
}

fn configure_mlx_cache_policy() {
    // Priority:
    // 1) MLX_CACHE_LIMIT_BYTES
    // 2) MLX_CACHE_LIMIT_MB
    // 3) auto cap based on active memory
    let explicit_bytes = parse_env_usize("MLX_CACHE_LIMIT_BYTES");
    let explicit_mb = parse_env_usize("MLX_CACHE_LIMIT_MB");
    let target = if let Some(v) = explicit_bytes {
        v
    } else if let Some(v) = explicit_mb {
        v.saturating_mul(1024 * 1024)
    } else {
        let active = mlx_core::metal::memory_info().active_memory;
        // Conservative default for production behavior:
        // keep allocator cache small so RSS tracks active model memory.
        // Auto target is active/8, clamped to [256 MiB, 512 MiB].
        let auto = active / 8;
        let min_b = 256usize * 1024 * 1024;
        let max_b = 512usize * 1024 * 1024;
        auto.clamp(min_b, max_b)
    };
    mlx_core::metal::set_cache_limit(target);
    eprintln!(
        "Configured MLX cache limit: {:.2} GiB",
        target as f64 / 1024.0 / 1024.0 / 1024.0
    );
}

/// Supported model architectures.
#[derive(Debug, Clone)]
pub enum ModelArch {
    Llama,
    Qwen3,
    Qwen35,
    QwenMoe,
    QwenMoePythonPort,
    Lfm2Moe,
    Lfm2MoePythonPort,
}

/// Detect model architecture from config.json.
pub fn detect_architecture(config: &serde_json::Value) -> Result<ModelArch> {
    // Check model_type field
    if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
        match model_type.to_lowercase().as_str() {
            "llama" => return Ok(ModelArch::Llama),
            "qwen3_5" | "qwen3.5" => return Ok(ModelArch::Qwen35),
            "qwen3" => {
                if config.get("num_experts").is_some() || config.get("num_experts_per_tok").is_some() {
                    return Err(anyhow::anyhow!(
                        "qwen3_moe is not supported in the current runtime; use dense qwen3"
                    ));
                }
                return Ok(ModelArch::Qwen3);
            }
            "qwen2_moe" | "qwen1.5_moe" => return Ok(ModelArch::QwenMoe),
            "qwen2_moe_python_port" | "qwen1.5_moe_python_port" | "qwen_moe_python_port" => {
                return Ok(ModelArch::QwenMoePythonPort);
            }
            "lfm2_moe" | "lfm2" => return Ok(ModelArch::Lfm2Moe),
            "lfm2_moe_python_port" | "lfm2_python_port" => {
                return Ok(ModelArch::Lfm2MoePythonPort);
            }
            _ => {}
        }
    }

    // Check architectures field
    if let Some(archs) = config.get("architectures").and_then(|v| v.as_array()) {
        for arch in archs {
            if let Some(arch_str) = arch.as_str() {
                let lower = arch_str.to_lowercase();
                if lower.contains("qwen3_5") || lower.contains("qwen3.5") {
                    return Ok(ModelArch::Qwen35);
                }
                if lower.contains("llama") {
                    return Ok(ModelArch::Llama);
                }
                if lower.contains("qwen3") {
                    if config.get("num_experts").is_some() || config.get("num_experts_per_tok").is_some() {
                        return Err(anyhow::anyhow!(
                            "qwen3_moe is not supported in the current runtime; use dense qwen3"
                        ));
                    }
                    return Ok(ModelArch::Qwen3);
                }
                if lower.contains("qwen2moe") || lower.contains("qwen2_moe") {
                    return Ok(ModelArch::QwenMoe);
                }
                if lower.contains("qwen2moepythonport")
                    || lower.contains("qwen2_moe_python_port")
                    || lower.contains("qwen1.5_moe_python_port")
                    || lower.contains("qwen_moe_python_port")
                {
                    return Ok(ModelArch::QwenMoePythonPort);
                }
                if lower.contains("lfm2moepythonport")
                    || lower.contains("lfm2_moe_python_port")
                    || lower.contains("lfm2_python_port")
                {
                    return Ok(ModelArch::Lfm2MoePythonPort);
                }
                if lower.contains("lfm2moe") || lower.contains("lfm2_moe") || lower.contains("lfm2") {
                    return Ok(ModelArch::Lfm2Moe);
                }
            }
        }
    }

    Err(anyhow::anyhow!("could not detect model architecture"))
}

fn push_token_id(tokenizer: &crate::tokenizer::Tokenizer, stops: &mut Vec<u32>, token: &str) {
    if let Some(id) = tokenizer.inner().token_to_id(token) {
        stops.push(id);
    }
}

fn extend_stop_tokens_from_value(
    tokenizer: &crate::tokenizer::Tokenizer,
    stops: &mut Vec<u32>,
    value: &Value,
) {
    match value {
        Value::Number(n) => {
            if let Some(id) = n.as_u64().map(|v| v as u32) {
                stops.push(id);
            }
        }
        Value::String(s) => push_token_id(tokenizer, stops, s),
        Value::Object(map) => {
            if let Some(Value::String(s)) = map.get("content") {
                push_token_id(tokenizer, stops, s);
            }
        }
        Value::Array(arr) => {
            for item in arr {
                extend_stop_tokens_from_value(tokenizer, stops, item);
            }
        }
        _ => {}
    }
}

fn load_stop_tokens(
    model_dir: &Path,
    tokenizer: crate::tokenizer::Tokenizer,
    config: &Value,
) -> crate::tokenizer::Tokenizer {
    let mut stops: Vec<u32> = Vec::new();

    if let Some(eos_id) = config
        .get("eos_token_id")
        .or_else(|| config.get("text_config").and_then(|tc| tc.get("eos_token_id")))
    {
        extend_stop_tokens_from_value(&tokenizer, &mut stops, eos_id);
    }

    for filename in ["tokenizer_config.json", "special_tokens_map.json"] {
        let path = model_dir.join(filename);
        if let Ok(raw) = std::fs::read_to_string(&path) {
            if let Ok(obj) = serde_json::from_str::<Value>(&raw) {
                if let Some(v) = obj.get("eos_token") {
                    extend_stop_tokens_from_value(&tokenizer, &mut stops, v);
                }
                if let Some(v) = obj.get("eos_token_id") {
                    extend_stop_tokens_from_value(&tokenizer, &mut stops, v);
                }
                if let Some(v) = obj.get("additional_special_tokens") {
                    extend_stop_tokens_from_value(&tokenizer, &mut stops, v);
                }
            }
        }
    }

    for tok in ["<|im_end|>", "<|eot_id|>", "<|im_start|>"] {
        push_token_id(&tokenizer, &mut stops, tok);
    }

    tokenizer.with_stop_tokens(stops)
}

/// Load a model from a directory containing config.json and .safetensors files.
pub fn load_model(model_dir: &Path) -> Result<(Box<dyn CausalLM>, crate::tokenizer::Tokenizer)> {
    // Read config.json
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("failed to read config.json: {e}"))?;
    let config: serde_json::Value = serde_json::from_str(&config_str)?;

    // Detect architecture
    let arch = detect_architecture(&config)?;
    eprintln!("Detected architecture: {:?}", arch);
    // Load weights
    let vb = VarBuilder::from_dir(model_dir, DType::Float16)?;
    eprintln!("Loaded {} tensors", vb.data().len());

    // Build model
    let model: Box<dyn CausalLM> = match arch {
        ModelArch::Llama => {
            let cfg: mlx_models::LlamaConfig = serde_json::from_value(config.clone())?;
            Box::new(mlx_models::Llama::new(&cfg, &vb)?)
        }
        ModelArch::Qwen3 => {
            let cfg: mlx_models::Qwen3Config = serde_json::from_value(config.clone())?;
            Box::new(mlx_models::Qwen3::new(&cfg, &vb)?)
        }
        ModelArch::Qwen35 => {
            let cfg: mlx_models::Qwen35Config = serde_json::from_value(config.clone())?;
            Box::new(mlx_models::Qwen35::new(&cfg, &vb)?)
        }
        ModelArch::QwenMoe => {
            let cfg: mlx_models::Qwen3MoeConfig = serde_json::from_value(config.clone())?;
            Box::new(mlx_models::Qwen3Moe::new(&cfg, &vb)?)
        }
        ModelArch::QwenMoePythonPort => {
            let cfg: mlx_models::Qwen3MoePythonPortConfig = serde_json::from_value(config.clone())?;
            Box::new(mlx_models::Qwen3MoePythonPort::new(&cfg, &vb)?)
        }
        ModelArch::Lfm2Moe => {
            let cfg: mlx_models::Lfm2MoeConfig = serde_json::from_value(config.clone())?;
            Box::new(mlx_models::Lfm2Moe::new(&cfg, &vb)?)
        }
        ModelArch::Lfm2MoePythonPort => {
            let cfg: mlx_models::Lfm2MoePythonPortConfig = serde_json::from_value(config.clone())?;
            Box::new(mlx_models::Lfm2MoePythonPort::new(&cfg, &vb)?)
        }
    };

    configure_mlx_cache_policy();

    // Load tokenizer
    let tokenizer_path = model_dir.join("tokenizer.json");
    let mut tokenizer = crate::tokenizer::Tokenizer::from_file(&tokenizer_path)?;

    tokenizer = load_stop_tokens(model_dir, tokenizer, &config);

    Ok((model, tokenizer))
}
