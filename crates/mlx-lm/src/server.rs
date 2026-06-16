use std::path::Path;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

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
}

impl ServerConfig {
    pub fn from_toml_path(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(path)?;
        let parsed: serde_json::Value = basic_toml_to_json(&content)
            .map_err(|e| anyhow::anyhow!("Failed to parse config: {e}"))?;
        let server = parsed.get("server").cloned().unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
        let cfg: Self = serde_json::from_value(server)
            .map_err(|e| anyhow::anyhow!("Failed to parse server config: {e}"))?;
        Ok(cfg)
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
