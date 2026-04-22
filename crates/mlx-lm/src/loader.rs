use anyhow::Result;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::Cache;
use mlx_core::DType;
use mlx_nn::VarBuilder;
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::generate::ModelRuntime;

#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct HuggingFaceOptions {
    pub hf_token: Option<String>,
}

fn looks_like_hf_repo_id(model: &str) -> bool {
    !model.is_empty()
        && !model.starts_with('.')
        && !model.starts_with('~')
        && !Path::new(model).is_absolute()
        && model.matches('/').count() == 1
}

fn cached_hf_snapshot_dir(model: &str) -> Option<PathBuf> {
    if let Some(snapshot_dir) = Cache::from_env()
        .model(model.to_string())
        .get("config.json")
        .and_then(|path| path.parent().map(Path::to_path_buf))
    {
        return Some(snapshot_dir);
    }

    let cache_root = Cache::from_env().path().clone();
    let repo_dir = cache_root.join(format!("models--{}", model.replace('/', "--")));
    let snapshots_dir = repo_dir.join("snapshots");
    if !snapshots_dir.is_dir() {
        return None;
    }
    let mut snapshots = std::fs::read_dir(&snapshots_dir)
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_dir() && path.join("config.json").is_file())
        .collect::<Vec<_>>();

    snapshots.sort_by_key(|path| {
        std::fs::metadata(path)
            .and_then(|metadata| metadata.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH)
    });
    snapshots.pop()
}

fn resolved_model_map_path() -> PathBuf {
    Cache::from_env().path().join("mlx-rs-resolved-models.json")
}

fn read_resolved_model_map() -> HashMap<String, String> {
    let path = resolved_model_map_path();
    std::fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str::<HashMap<String, String>>(&raw).ok())
        .unwrap_or_default()
}

fn read_persisted_snapshot_dir(model: &str) -> Option<PathBuf> {
    let path = PathBuf::from(read_resolved_model_map().get(model)?.clone());
    if path.join("config.json").is_file() {
        Some(path)
    } else {
        None
    }
}

fn persist_snapshot_dir(model: &str, snapshot_dir: &Path) {
    let mut resolved = read_resolved_model_map();
    resolved.insert(model.to_string(), snapshot_dir.display().to_string());
    let path = resolved_model_map_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(raw) = serde_json::to_vec_pretty(&resolved) {
        let _ = std::fs::write(path, raw);
    }
}

pub fn resolve_model_dir(model: &str, hf: Option<&HuggingFaceOptions>) -> Result<PathBuf> {
    let path = PathBuf::from(model);
    if path.exists() {
        return Ok(path);
    }

    if !looks_like_hf_repo_id(model) {
        anyhow::bail!("model path does not exist: {model}");
    }

    if let Some(snapshot_dir) = read_persisted_snapshot_dir(model) {
        eprintln!(
            "Using persisted Hugging Face snapshot for {} at {:?}",
            model, snapshot_dir
        );
        return Ok(snapshot_dir);
    }

    if let Some(snapshot_dir) = cached_hf_snapshot_dir(model) {
        eprintln!(
            "Using cached Hugging Face snapshot for {} at {:?}",
            model, snapshot_dir
        );
        persist_snapshot_dir(model, &snapshot_dir);
        return Ok(snapshot_dir);
    }
    eprintln!(
        "No cached Hugging Face snapshot found for {} in {:?}",
        model,
        Cache::from_env().path()
    );

    let token = hf
        .and_then(|cfg| cfg.hf_token.as_ref())
        .map(|token| token.trim())
        .filter(|token| !token.is_empty())
        .map(ToOwned::to_owned);

    eprintln!("Resolving model from Hugging Face repo {model} ...");
    let api = ApiBuilder::from_env()
        .with_token(token)
        .build()
        .map_err(|e| anyhow::anyhow!("failed to initialize Hugging Face client: {e}"))?;
    let repo = api.model(model.to_string());
    let info = repo
        .info()
        .map_err(|e| anyhow::anyhow!("failed to query Hugging Face repo {model}: {e}"))?;

    for sibling in &info.siblings {
        let filename = sibling.rfilename.as_str();
        if filename.ends_with('/') || filename == ".gitattributes" {
            continue;
        }
        repo.download(filename).map_err(|e| {
            anyhow::anyhow!("failed to download {filename} from Hugging Face repo {model}: {e}")
        })?;
    }

    let snapshot_dir = repo
        .get("config.json")
        .map_err(|e| anyhow::anyhow!("failed to locate config.json for {model}: {e}"))?
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| anyhow::anyhow!("invalid cached snapshot layout for {model}"))?;

    eprintln!(
        "Resolved Hugging Face repo {} to local snapshot {:?}",
        model, snapshot_dir
    );
    persist_snapshot_dir(model, &snapshot_dir);
    Ok(snapshot_dir)
}

fn parse_env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
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
    Bert,
    Gemma3,
    Gemma4,
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
            "bert" => return Ok(ModelArch::Bert),
            "gemma3" => return Ok(ModelArch::Gemma3),
            "gemma4" => return Ok(ModelArch::Gemma4),
            "llama" => return Ok(ModelArch::Llama),
            "qwen3_5" | "qwen3.5" => return Ok(ModelArch::Qwen35),
            "qwen3" => {
                if config.get("num_experts").is_some()
                    || config.get("num_experts_per_tok").is_some()
                {
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

    if let Some(model_type) = config
        .get("text_config")
        .and_then(|v| v.get("model_type"))
        .and_then(|v| v.as_str())
    {
        if model_type.eq_ignore_ascii_case("gemma3_text") {
            return Ok(ModelArch::Gemma3);
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
                if lower.contains("bertmodel") || lower == "bert" {
                    return Ok(ModelArch::Bert);
                }
                if lower.contains("gemma4") {
                    return Ok(ModelArch::Gemma4);
                }
                if lower.contains("gemma3") {
                    return Ok(ModelArch::Gemma3);
                }
                if lower.contains("llama") {
                    return Ok(ModelArch::Llama);
                }
                if lower.contains("qwen3") {
                    if config.get("num_experts").is_some()
                        || config.get("num_experts_per_tok").is_some()
                    {
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
                if lower.contains("lfm2moe") || lower.contains("lfm2_moe") || lower.contains("lfm2")
                {
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

fn gemma3_text_config(config: &Value) -> Result<Value> {
    let mut text = config
        .get("text_config")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("gemma3 config missing text_config"))?;

    let text_obj = text
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("gemma3 text_config must be an object"))?;

    for key in [
        "quantization",
        "quantization_config",
        "eos_token_id",
        "bos_token_id",
        "tie_word_embeddings",
    ] {
        if let Some(value) = config.get(key) {
            text_obj
                .entry(key.to_string())
                .or_insert_with(|| value.clone());
        }
    }

    Ok(text)
}

fn load_stop_tokens(
    model_dir: &Path,
    tokenizer: crate::tokenizer::Tokenizer,
    config: &Value,
) -> crate::tokenizer::Tokenizer {
    let mut stops: Vec<u32> = Vec::new();

    if let Some(eos_id) = config.get("eos_token_id").or_else(|| {
        config
            .get("text_config")
            .and_then(|tc| tc.get("eos_token_id"))
    }) {
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
pub fn load_model(
    model_dir: &Path,
) -> Result<(Box<dyn ModelRuntime>, crate::tokenizer::Tokenizer)> {
    // Read config.json
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("failed to read config.json: {e}"))?;
    let config: serde_json::Value = serde_json::from_str(&config_str)?;

    // Detect architecture
    let arch = detect_architecture(&config)?;
    eprintln!("Detected architecture: {:?}", arch);
    // Load weights
    let vb = VarBuilder::from_dir(model_dir, DType::BFloat16)?;
    eprintln!("Loaded {} tensors", vb.data().len());

    // Build model
    let model: Box<dyn ModelRuntime> = match arch {
        ModelArch::Bert => {
            let cfg: mlx_models::BertConfig = serde_json::from_value(config.clone())?;
            Box::new(mlx_models::Bert::new(&cfg, &vb)?)
        }
        ModelArch::Gemma3 => {
            let cfg: mlx_models::Gemma3Config =
                serde_json::from_value(gemma3_text_config(&config)?)?;
            Box::new(mlx_models::Gemma3::new(&cfg, &vb)?)
        }
        ModelArch::Gemma4 => {
            let cfg: mlx_models::Gemma4Config = serde_json::from_value(config.clone())?;
            Box::new(mlx_models::Gemma4::new(&vb, &cfg)?)
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock")
    }

    #[test]
    fn detects_bert_from_model_type() {
        let config = json!({"model_type": "bert"});
        assert!(matches!(
            detect_architecture(&config).unwrap(),
            ModelArch::Bert
        ));
    }

    #[test]
    fn detects_bert_from_architecture() {
        let config = json!({"architectures": ["BertModel"]});
        assert!(matches!(
            detect_architecture(&config).unwrap(),
            ModelArch::Bert
        ));
    }

    #[test]
    fn detects_gemma3_from_model_type() {
        let config = json!({"model_type": "gemma3"});
        assert!(matches!(
            detect_architecture(&config).unwrap(),
            ModelArch::Gemma3
        ));
    }

    #[test]
    fn detects_gemma3_from_nested_text_model_type() {
        let config = json!({"text_config": {"model_type": "gemma3_text"}});
        assert!(matches!(
            detect_architecture(&config).unwrap(),
            ModelArch::Gemma3
        ));
    }

    #[test]
    fn detects_gemma3_from_architecture() {
        let config = json!({"architectures": ["Gemma3ForConditionalGeneration"]});
        assert!(matches!(
            detect_architecture(&config).unwrap(),
            ModelArch::Gemma3
        ));
    }

    #[test]
    fn detects_gemma4_from_model_type() {
        let config = json!({"model_type": "gemma4"});
        assert!(matches!(
            detect_architecture(&config).unwrap(),
            ModelArch::Gemma4
        ));
    }

    #[test]
    fn detects_gemma4_from_architecture() {
        let config = json!({"architectures": ["Gemma4ForConditionalGeneration"]});
        assert!(matches!(
            detect_architecture(&config).unwrap(),
            ModelArch::Gemma4
        ));
    }

    #[test]
    fn extracts_gemma3_nested_config_with_top_level_quantization() {
        let config = json!({
            "model_type": "gemma3",
            "quantization": {"group_size": 64, "bits": 4},
            "text_config": {
                "hidden_size": 3840,
                "intermediate_size": 15360,
                "vocab_size": 262208,
                "num_hidden_layers": 48,
                "num_attention_heads": 16,
                "num_key_value_heads": 8
            }
        });
        let nested = gemma3_text_config(&config).unwrap();
        let cfg: mlx_models::Gemma3Config = serde_json::from_value(nested).unwrap();
        assert_eq!(cfg.head_dim(), 240);
        assert_eq!(cfg.sliding_window_pattern, 6);
        assert_eq!(cfg.quantization.unwrap().bits, 4);
    }

    #[test]
    fn finds_cached_hf_snapshot_dir() {
        let _guard = env_lock();
        let unique = format!(
            "mlx-rs-loader-cache-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        let snapshot_dir = root
            .join("hub")
            .join("models--mlx-community--demo-model")
            .join("snapshots")
            .join("snapshot-a");
        fs::create_dir_all(&snapshot_dir).expect("create snapshot dir");
        fs::write(snapshot_dir.join("config.json"), "{}").expect("write config");
        std::env::set_var("HF_HOME", &root);

        let resolved = cached_hf_snapshot_dir("mlx-community/demo-model");

        std::env::remove_var("HF_HOME");
        let _ = fs::remove_dir_all(&root);

        assert_eq!(resolved, Some(snapshot_dir));
    }

    #[test]
    fn finds_persisted_snapshot_dir() {
        let _guard = env_lock();
        let unique = format!(
            "mlx-rs-loader-persisted-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        let snapshot_dir = root
            .join("hub")
            .join("models--mlx-community--demo-model")
            .join("snapshots")
            .join("snapshot-a");
        fs::create_dir_all(&snapshot_dir).expect("create snapshot dir");
        fs::write(snapshot_dir.join("config.json"), "{}").expect("write config");
        std::env::set_var("HF_HOME", &root);

        persist_snapshot_dir("mlx-community/demo-model", &snapshot_dir);
        let resolved = read_persisted_snapshot_dir("mlx-community/demo-model");

        std::env::remove_var("HF_HOME");
        let _ = fs::remove_dir_all(&root);

        assert_eq!(resolved, Some(snapshot_dir));
    }

    #[test]
    #[ignore = "requires local MLX/Metal runtime and the Gemma snapshot to be available"]
    fn loads_local_gemma3_snapshot_and_runs_forward() {
        if !matches!(
            std::env::var("MLX_RS_RUN_LOCAL_MODEL_TESTS").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE")
        ) {
            return;
        }
        let model_dir = PathBuf::from("/Volumes/Data/Users/paul/.cache/huggingface/hub/models--mlx-community--gemma-3-text-12b-it-4bit/snapshots/83eda451975103fff6f11e97bf75cd0f1c61af2c");
        if !model_dir.join("config.json").is_file() {
            return;
        }

        let (mut model, _tokenizer) = load_model(&model_dir).expect("load gemma3 snapshot");
        let prompt = mlx_core::Array::from_slice_i32(&[2, 42, 43])
            .expect("prompt")
            .reshape(&[1, 3])
            .expect("prompt reshape");
        let logits = model
            .forward_last_token_logits(&prompt)
            .expect("prefill forward");
        assert_eq!(logits.ndim(), 3);

        let decode = mlx_core::Array::from_slice_i32(&[44])
            .expect("decode")
            .reshape(&[1, 1])
            .expect("decode reshape");
        let next_logits = model
            .forward_last_token_logits(&decode)
            .expect("decode forward");
        assert_eq!(next_logits.ndim(), 3);
    }
}
