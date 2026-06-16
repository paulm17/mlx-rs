use std::path::Path;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::config::LlamaCppConfig;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ServerConfig {
    pub bind: Option<String>,
    pub port: Option<u16>,
    pub model_path: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub rate_limit_rpm: Option<u32>,
    pub thinking: Option<bool>,
    pub embeddings_batch_size: Option<usize>,
    // llama.cpp runtime config
    pub n_ctx: Option<u32>,
    pub n_batch: Option<u32>,
    pub n_ubatch: Option<u32>,
    pub n_gpu_layers: Option<u32>,
    pub n_threads: Option<u32>,
    pub n_threads_batch: Option<u32>,
    pub embedding: Option<bool>,
    pub pooling: Option<String>,
    pub use_mmap: Option<bool>,
    pub use_mlock: Option<bool>,
    pub flash_attn: Option<bool>,
}

impl ServerConfig {
    pub fn from_toml_path(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(path)?;
        Self::from_toml_str(&content)
    }

    pub fn from_toml_str(content: &str) -> Result<Self> {
        let parsed: serde_json::Value = basic_toml_to_json(content)
            .map_err(|e| anyhow::anyhow!("Failed to parse config: {e}"))?;
        let server = parsed
            .get("server")
            .cloned()
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
        let cfg: Self = serde_json::from_value(server)
            .map_err(|e| anyhow::anyhow!("Failed to parse server config: {e}"))?;
        Ok(cfg)
    }

    pub fn to_llamacpp_config(&self) -> LlamaCppConfig {
        LlamaCppConfig {
            n_ctx: self.n_ctx,
            n_batch: self.n_batch,
            n_ubatch: self.n_ubatch,
            n_gpu_layers: self.n_gpu_layers,
            n_threads: self.n_threads,
            n_threads_batch: self.n_threads_batch,
            embedding: self.embedding,
            pooling: self.pooling.clone(),
            use_mmap: self.use_mmap,
            use_mlock: self.use_mlock,
            flash_attn: self.flash_attn,
        }
    }
}

fn basic_toml_to_json(input: &str) -> Result<serde_json::Value> {
    let mut root = serde_json::Map::new();
    let mut current_section: Option<String> = None;

    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            let name = line[1..line.len() - 1].trim().to_string();
            current_section = Some(name);
            continue;
        }
        if let Some(eq_pos) = line.find('=') {
            let key = line[..eq_pos].trim();
            let value_str = line[eq_pos + 1..].trim();
            let value = parse_toml_value(value_str);

            let section = current_section.as_deref().unwrap_or("");
            if section.is_empty() {
                root.insert(key.to_string(), value);
            } else {
                let section_map = root
                    .entry(section.to_string())
                    .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
                if let serde_json::Value::Object(ref mut m) = section_map {
                    m.insert(key.to_string(), value);
                }
            }
        }
    }

    Ok(serde_json::Value::Object(root))
}

fn parse_toml_value(s: &str) -> serde_json::Value {
    let s = s.trim();
    if s == "true" {
        serde_json::Value::Bool(true)
    } else if s == "false" {
        serde_json::Value::Bool(false)
    } else if let Ok(n) = s.parse::<u64>() {
        serde_json::Value::Number(n.into())
    } else if let Ok(n) = s.parse::<i64>() {
        serde_json::Value::Number(n.into())
    } else if let Some(inner) = s.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
        serde_json::Value::String(inner.to_string())
    } else {
        serde_json::Value::String(s.to_string())
    }
}

pub fn run_server(_config: ServerConfig) -> Result<()> {
    bail!("Server not yet implemented; llama.cpp backend coming in milestone 1.7+")
}

pub fn run_server_from_toml_path(path: impl AsRef<Path>) -> Result<()> {
    let cfg = ServerConfig::from_toml_path(path.as_ref())?;
    run_server(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_toml(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[test]
    fn test_parse_empty_config() {
        let cfg = ServerConfig::from_toml_path(Path::new("/nonexistent")).unwrap();
        assert!(cfg.bind.is_none());
        assert!(cfg.port.is_none());
        assert!(cfg.model_path.is_none());
    }

    #[test]
    fn test_parse_basic_server_config() {
        let toml = r#"
[server]
bind = "0.0.0.0:3000"
port = 8080
model_path = "/models/llama.gguf"
api_key = "secret"
rate_limit_rpm = 120
thinking = true
embeddings_batch_size = 32
"#;
        let f = write_temp_toml(toml);
        let cfg = ServerConfig::from_toml_path(f.path()).unwrap();
        assert_eq!(cfg.bind.as_deref(), Some("0.0.0.0:3000"));
        assert_eq!(cfg.port, Some(8080));
        assert_eq!(cfg.model_path.as_deref(), Some("/models/llama.gguf"));
        assert_eq!(cfg.api_key.as_deref(), Some("secret"));
        assert_eq!(cfg.rate_limit_rpm, Some(120));
        assert_eq!(cfg.thinking, Some(true));
        assert_eq!(cfg.embeddings_batch_size, Some(32));
    }

    #[test]
    fn test_parse_llamacpp_config_keys() {
        let toml = r#"
[server]
n_ctx = 4096
n_batch = 512
n_ubatch = 256
n_gpu_layers = 99
n_threads = 8
n_threads_batch = 8
embedding = true
pooling = "mean"
use_mmap = true
use_mlock = false
flash_attn = true
"#;
        let f = write_temp_toml(toml);
        let cfg = ServerConfig::from_toml_path(f.path()).unwrap();
        assert_eq!(cfg.n_ctx, Some(4096));
        assert_eq!(cfg.n_batch, Some(512));
        assert_eq!(cfg.n_ubatch, Some(256));
        assert_eq!(cfg.n_gpu_layers, Some(99));
        assert_eq!(cfg.n_threads, Some(8));
        assert_eq!(cfg.n_threads_batch, Some(8));
        assert_eq!(cfg.embedding, Some(true));
        assert_eq!(cfg.pooling.as_deref(), Some("mean"));
        assert_eq!(cfg.use_mmap, Some(true));
        assert_eq!(cfg.use_mlock, Some(false));
        assert_eq!(cfg.flash_attn, Some(true));
    }

    #[test]
    fn test_parse_comments_and_blanks() {
        let toml = r#"
# This is a comment

[server]
bind = "127.0.0.1:3000"
# model_path = "/ignored"
port = 3001
"#;
        let f = write_temp_toml(toml);
        let cfg = ServerConfig::from_toml_path(f.path()).unwrap();
        assert_eq!(cfg.bind.as_deref(), Some("127.0.0.1:3000"));
        assert_eq!(cfg.port, Some(3001));
        assert!(cfg.model_path.is_none());
    }

    #[test]
    fn test_parse_partial_config() {
        let toml = r#"
[server]
model = "llama-3.2-1b"
n_gpu_layers = 99
"#;
        let f = write_temp_toml(toml);
        let cfg = ServerConfig::from_toml_path(f.path()).unwrap();
        assert_eq!(cfg.model.as_deref(), Some("llama-3.2-1b"));
        assert_eq!(cfg.n_gpu_layers, Some(99));
        assert!(cfg.bind.is_none());
        assert!(cfg.n_ctx.is_none());
    }

    #[test]
    fn test_to_llamacpp_config() {
        let cfg = ServerConfig {
            n_ctx: Some(2048),
            n_gpu_layers: Some(32),
            flash_attn: Some(true),
            ..Default::default()
        };
        let llamacpp = cfg.to_llamacpp_config();
        assert_eq!(llamacpp.n_ctx, Some(2048));
        assert_eq!(llamacpp.n_gpu_layers, Some(32));
        assert_eq!(llamacpp.flash_attn, Some(true));
        assert!(llamacpp.n_batch.is_none());
    }

    #[test]
    fn test_toml_value_types() {
        assert_eq!(parse_toml_value("true"), serde_json::Value::Bool(true));
        assert_eq!(parse_toml_value("false"), serde_json::Value::Bool(false));
        assert_eq!(
            parse_toml_value("42"),
            serde_json::Value::Number(42u64.into())
        );
        assert_eq!(
            parse_toml_value("\"hello\""),
            serde_json::Value::String("hello".to_string())
        );
        // unquoted string fallback
        assert_eq!(
            parse_toml_value("mean"),
            serde_json::Value::String("mean".to_string())
        );
    }
}
