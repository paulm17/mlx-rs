use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

/// Resolve a model input path to a single GGUF file.
///
/// Accepted inputs:
/// - A direct `.gguf` file path
/// - A directory containing exactly one `.gguf` file
/// - A directory containing split GGUF shards (e.g., `model-00001-of-00003.gguf`)
///
/// Rejected inputs:
/// - Missing paths
/// - Directories with only safetensors/config.json (MLX models)
/// - Directories with multiple unrelated `.gguf` files
pub fn resolve_model_path(input: &str) -> Result<PathBuf> {
    let path = Path::new(input);

    if !path.exists() {
        bail!("Model path does not exist: {}", input);
    }

    if path.is_file() {
        if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            return Ok(path.to_path_buf());
        }
        bail!(
            "Model file is not a GGUF file: {}",
            path.display()
        );
    }

    if !path.is_dir() {
        bail!("Model path is not a file or directory: {}", input);
    }

    resolve_from_dir(path)
}

fn resolve_from_dir(dir: &Path) -> Result<PathBuf> {
    let entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("Failed to read directory: {}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();

    let gguf_files: Vec<&PathBuf> = entries
        .iter()
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("gguf"))
        .collect();

    if gguf_files.is_empty() {
        // Check if this looks like a safetensors/MLX directory
        let has_safetensors = entries.iter().any(|p| {
            p.extension().and_then(|e| e.to_str()) == Some("safetensors")
        });
        let has_config_json = entries.iter().any(|p| {
            p.file_name().and_then(|n| n.to_str()) == Some("config.json")
        });

        if has_safetensors || has_config_json {
            bail!(
                "safetensors/MLX model directories are not supported by the llama.cpp runtime; \
                 provide a GGUF model file"
            );
        }

        bail!("No GGUF files found in directory: {}", dir.display());
    }

    if gguf_files.len() == 1 {
        return Ok(gguf_files[0].clone());
    }

    // Multiple GGUF files — check if they are split shards
    if let Some(first_shard) = try_resolve_split_shards(&gguf_files) {
        return Ok(first_shard);
    }

    bail!(
        "Multiple unrelated GGUF files found in {}; \
         provide a direct path to a single GGUF file",
        dir.display()
    )
}

/// Try to identify split GGUF shards and return the first one.
///
/// Split GGUF naming convention: `name-NNNNN-of-NNNNN.gguf`
fn try_resolve_split_shards(files: &[&PathBuf]) -> Option<PathBuf> {
    let mut shard_groups: std::collections::HashMap<String, Vec<&PathBuf>> =
        std::collections::HashMap::new();

    for file in files {
        let name = file.file_stem()?.to_str()?;
        if let Some((base, _shard_info)) = parse_shard_name(name) {
            shard_groups
                .entry(base.to_string())
                .or_default()
                .push(file);
        }
    }

    // Only accept if exactly one shard group exists
    if shard_groups.len() != 1 {
        return None;
    }

    let (_base, mut shards) = shard_groups.into_iter().next()?;
    if shards.len() < 2 {
        return None;
    }

    // Sort by shard number to get the first shard
    shards.sort_by_key(|p| {
        p.file_stem()
            .and_then(|n| n.to_str())
            .and_then(parse_shard_name)
            .map(|(_, (idx, _))| idx)
            .unwrap_or(0)
    });

    Some(shards[0].clone())
}

/// Parse a shard filename like `model-00001-of-00003` into (base, (index, total)).
/// Returns None if not a shard pattern.
fn parse_shard_name(name: &str) -> Option<(&str, (u32, u32))> {
    // Pattern: <base>-<5digits>-of-<5digits>
    let of_pos = name.rfind("-of-")?;
    let total_str = &name[of_pos + 4..];
    let total: u32 = total_str.parse().ok()?;

    let before_of = &name[..of_pos];
    let last_dash = before_of.rfind('-')?;
    let index_str = &before_of[last_dash + 1..];
    let index: u32 = index_str.parse().ok()?;

    let base = &before_of[..last_dash];
    Some((base, (index, total)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn create_temp_dir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn test_direct_gguf_file() {
        let dir = create_temp_dir();
        let gguf = dir.path().join("model.gguf");
        fs::write(&gguf, b"fake").unwrap();
        let result = resolve_model_path(gguf.to_str().unwrap()).unwrap();
        assert_eq!(result, gguf);
    }

    #[test]
    fn test_nonexistent_path() {
        let result = resolve_model_path("/nonexistent/path/model.gguf");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    #[test]
    fn test_non_gguf_file() {
        let dir = create_temp_dir();
        let file = dir.path().join("model.bin");
        fs::write(&file, b"fake").unwrap();
        let result = resolve_model_path(file.to_str().unwrap());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not a GGUF"));
    }

    #[test]
    fn test_dir_with_single_gguf() {
        let dir = create_temp_dir();
        fs::write(dir.path().join("model.gguf"), b"fake").unwrap();
        fs::write(dir.path().join("README.md"), b"info").unwrap();
        let result = resolve_model_path(dir.path().to_str().unwrap()).unwrap();
        assert_eq!(result, dir.path().join("model.gguf"));
    }

    #[test]
    fn test_dir_with_split_shards() {
        let dir = create_temp_dir();
        fs::write(dir.path().join("llama-00001-of-00003.gguf"), b"fake").unwrap();
        fs::write(dir.path().join("llama-00002-of-00003.gguf"), b"fake").unwrap();
        fs::write(dir.path().join("llama-00003-of-00003.gguf"), b"fake").unwrap();
        let result = resolve_model_path(dir.path().to_str().unwrap()).unwrap();
        assert_eq!(result, dir.path().join("llama-00001-of-00003.gguf"));
    }

    #[test]
    fn test_dir_with_multiple_unrelated_gguf() {
        let dir = create_temp_dir();
        fs::write(dir.path().join("model_a.gguf"), b"fake").unwrap();
        fs::write(dir.path().join("model_b.gguf"), b"fake").unwrap();
        let result = resolve_model_path(dir.path().to_str().unwrap());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Multiple unrelated"));
    }

    #[test]
    fn test_dir_with_only_safetensors() {
        let dir = create_temp_dir();
        fs::write(dir.path().join("model.safetensors"), b"fake").unwrap();
        fs::write(dir.path().join("config.json"), b"{}").unwrap();
        let result = resolve_model_path(dir.path().to_str().unwrap());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("safetensors/MLX"));
        assert!(err.contains("not supported"));
    }

    #[test]
    fn test_dir_with_config_json_only() {
        let dir = create_temp_dir();
        fs::write(dir.path().join("config.json"), b"{}").unwrap();
        let result = resolve_model_path(dir.path().to_str().unwrap());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("safetensors/MLX"));
    }

    #[test]
    fn test_empty_dir() {
        let dir = create_temp_dir();
        let result = resolve_model_path(dir.path().to_str().unwrap());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No GGUF files"));
    }

    #[test]
    fn test_parse_shard_name() {
        let (base, (idx, total)) = parse_shard_name("llama-00001-of-00003").unwrap();
        assert_eq!(base, "llama");
        assert_eq!(idx, 1);
        assert_eq!(total, 3);
    }

    #[test]
    fn test_parse_shard_name_not_shard() {
        assert!(parse_shard_name("model").is_none());
        assert!(parse_shard_name("model.gguf").is_none());
        assert!(parse_shard_name("model-abc-of-00003").is_none());
    }

    #[test]
    fn test_dir_split_shards_unsorted() {
        let dir = create_temp_dir();
        // Write out of order to verify sorting
        fs::write(dir.path().join("m-00003-of-00003.gguf"), b"fake").unwrap();
        fs::write(dir.path().join("m-00001-of-00003.gguf"), b"fake").unwrap();
        fs::write(dir.path().join("m-00002-of-00003.gguf"), b"fake").unwrap();
        let result = resolve_model_path(dir.path().to_str().unwrap()).unwrap();
        assert_eq!(result, dir.path().join("m-00001-of-00003.gguf"));
    }
}
