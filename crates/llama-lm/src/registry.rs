use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use anyhow::Result;

use crate::backend::Backend;
use crate::config::LlamaCppConfig;
use crate::llamacpp::LlamaCppBackend;
use crate::loader::{resolve_model_path, resolve_hf_safetensors_dir, looks_like_hf_repo_id, looks_like_hf_gguf_ref};
use crate::subprocess::RunnerSubprocess;
use crate::types::LoadedModelInfo;

pub enum ModelFormat {
    Gguf,
    Safetensors,
}

type SafetensorsFactory = Box<dyn Fn(&std::path::Path) -> Result<Box<dyn Backend>> + Send + Sync>;

fn safetensors_registry() -> &'static Mutex<Vec<SafetensorsFactory>> {
    static REGISTRY: OnceLock<Mutex<Vec<SafetensorsFactory>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Vec::new()))
}

pub fn register_safetensors_backend<F>(factory: F)
where
    F: Fn(&std::path::Path) -> Result<Box<dyn Backend>> + Send + Sync + 'static,
{
    let mut reg = safetensors_registry().lock().unwrap();
    reg.push(Box::new(factory));
}

pub fn detect_format(path: &str) -> Result<ModelFormat> {
    let raw = PathBuf::from(path);

    if raw.extension().map_or(false, |e| e == "gguf") {
        let _resolved = resolve_model_path(path)?;
        return Ok(ModelFormat::Gguf);
    }

    if raw.is_dir() {
        let has_safetensors = raw
            .read_dir()
            .ok()
            .and_then(|mut entries| {
                entries.find(|e| {
                    e.as_ref()
                        .ok()
                        .and_then(|e| e.path().extension().map(|ext| ext == "safetensors"))
                        .unwrap_or(false)
                })
            })
            .is_some();

        if has_safetensors || raw.join("config.json").is_file() {
            return Ok(ModelFormat::Safetensors);
        }
    }

    if looks_like_hf_repo_id(path) {
        // For HF repo IDs, detect format without downloading.
        // GGUF references (owner/repo/file.gguf) are handled by resolve_model_path.
        // Everything else (owner/repo or owner/repo/subdir) is considered safetensors.
        if looks_like_hf_gguf_ref(path) {
            return Ok(ModelFormat::Gguf);
        }
        return Ok(ModelFormat::Safetensors);
    }

    let resolved: PathBuf = resolve_model_path(path)?;

    if resolved.extension().map_or(false, |e| e == "gguf") {
        return Ok(ModelFormat::Gguf);
    }

    anyhow::bail!(
        "Unable to detect model format for: {}",
        path
    )
}

pub fn create_backend(path: &str, config: LlamaCppConfig) -> Result<Box<dyn Backend>> {
    let format = detect_format(path)?;

    match format {
        ModelFormat::Gguf => {
            let resolved = resolve_model_path(path)?;
            let backend = LlamaCppBackend::new(resolved.to_str().unwrap(), config)?;
            Ok(Box::new(backend))
        }
        ModelFormat::Safetensors => {
            let resolved = if PathBuf::from(path).is_dir() {
                PathBuf::from(path)
            } else {
                resolve_hf_safetensors_dir(path)?
            };

            let reg = safetensors_registry().lock().unwrap();
            if let Some(factory) = reg.first() {
                return factory(&resolved);
            }

            let subprocess = RunnerSubprocess::spawn(&resolved)?;
            let model_info = LoadedModelInfo {
                model_path: resolved.display().to_string(),
                context_length: None,
                embedding_dimension: None,
                vocab_size: None,
            };
            let client = subprocess.client(model_info);
            Ok(Box::new(client))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_detect_format_gguf_file() {
        let dir = tempfile::tempdir().unwrap();
        let gguf = dir.path().join("model.gguf");
        fs::write(&gguf, b"fake").unwrap();
        let result = detect_format(gguf.to_str().unwrap()).unwrap();
        assert!(matches!(result, ModelFormat::Gguf));
    }

    #[test]
    fn test_detect_format_safetensors_dir() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model.safetensors"), b"fake").unwrap();
        fs::write(dir.path().join("config.json"), b"{}").unwrap();
        let result = detect_format(dir.path().to_str().unwrap()).unwrap();
        assert!(matches!(result, ModelFormat::Safetensors));
    }

    #[test]
    fn test_detect_format_empty_dir_fails() {
        let dir = tempfile::tempdir().unwrap();
        let result = detect_format(dir.path().to_str().unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_format_nonexistent_fails() {
        let result = detect_format("/nonexistent/model.gguf");
        assert!(result.is_err());
    }

    #[test]
    fn test_create_backend_safetensors_spawns_subprocess() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model.safetensors"), b"fake").unwrap();
        fs::write(dir.path().join("config.json"), b"{}").unwrap();
        std::env::set_var("MLX_RS_RUNNER_TIMEOUT", "5");
        let result = create_backend(dir.path().to_str().unwrap(), LlamaCppConfig::default());
        std::env::remove_var("MLX_RS_RUNNER_TIMEOUT");
        match result {
            Ok(_) => panic!("safetensors subprocess spawning should fail without runner binary"),
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("failed to spawn")
                        || msg.contains("not yet supported")
                        || msg.contains("did not become ready"),
                    "unexpected error: {}",
                    msg
                );
            }
        }
    }

    #[test]
    fn test_create_backend_gguf_env_gated() {
        let model_path = match std::env::var("MLX_RS_TEST_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("Skipping backend creation test: MLX_RS_TEST_GGUF not set");
                return;
            }
        };

        let backend = create_backend(
            &model_path,
            LlamaCppConfig {
                n_ctx: Some(512),
                n_gpu_layers: Some(0),
                ..Default::default()
            },
        )
        .expect("Failed to create backend");

        let info = backend.model_info();
        assert!(info.context_length.is_some());
        assert!(info.embedding_dimension.is_some());
        assert!(backend.supports_chat_template());
        assert!(backend.memory_info().is_some());
    }
}
