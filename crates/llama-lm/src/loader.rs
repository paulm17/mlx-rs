use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};

use hf_hub::api::sync::ApiBuilder;

/// Resolve a model input path to a single GGUF file.
///
/// Accepted inputs:
/// - A direct `.gguf` file path
/// - A directory containing exactly one `.gguf` file
/// - A directory containing split GGUF shards (e.g., `model-00001-of-00003.gguf`)
/// - A Hugging Face reference like `owner/repo/model.gguf`
///
/// Rejected inputs:
/// - Missing paths
/// - Directories with only safetensors/config.json (MLX models)
/// - Directories with multiple unrelated `.gguf` files
pub fn resolve_model_path(input: &str) -> Result<PathBuf> {
    let path = Path::new(input);

    if !path.exists() {
        if let Some(reference) = parse_hf_gguf_reference(input) {
            return resolve_hf_gguf_reference(&reference);
        }
        bail!("Model path does not exist: {}", input);
    }

    if path.is_file() {
        if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            return Ok(path.to_path_buf());
        }
        bail!("Model file is not a GGUF file: {}", path.display());
    }

    if !path.is_dir() {
        bail!("Model path is not a file or directory: {}", input);
    }

    resolve_from_dir(path)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HfGgufReference {
    repo_id: String,
    filename: String,
}

pub(crate) fn looks_like_hf_gguf_ref(input: &str) -> bool {
    parse_hf_gguf_reference(input).is_some()
}

fn parse_hf_gguf_reference(input: &str) -> Option<HfGgufReference> {
    if input.starts_with('/')
        || input.starts_with("./")
        || input.starts_with("../")
        || input.contains("://")
    {
        return None;
    }

    let parts = input.split('/').collect::<Vec<_>>();
    if parts.len() < 3 || parts.iter().any(|part| part.is_empty()) {
        return None;
    }

    let filename = parts[2..].join("/");
    if Path::new(&filename).extension().and_then(|e| e.to_str()) != Some("gguf") {
        return None;
    }

    Some(HfGgufReference {
        repo_id: format!("{}/{}", parts[0], parts[1]),
        filename,
    })
}

fn resolve_hf_gguf_reference(reference: &HfGgufReference) -> Result<PathBuf> {
    let path = hf_cache_path(reference)?;
    if path.exists() {
        return Ok(path);
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create model cache dir: {}", parent.display()))?;
    }

    download_hf_gguf(reference, &path)?;

    if path.exists() {
        Ok(path)
    } else {
        bail!(
            "Download completed but model file was not found at {}",
            path.display()
        )
    }
}

fn hf_cache_root() -> Result<PathBuf> {
    let cache_root = if let Some(path) = std::env::var_os("LLAMA_RS_MODEL_CACHE") {
        PathBuf::from(path)
    } else if let Some(path) = std::env::var_os("HF_HOME") {
        PathBuf::from(path).join("llama-rs").join("models")
    } else {
        home_dir()
            .context("Cannot determine model cache directory; set LLAMA_RS_MODEL_CACHE")?
            .join(".cache")
            .join("llama-rs")
            .join("models")
    };

    Ok(cache_root)
}

fn hf_cache_repo_dir(reference: &HfGgufReference) -> Result<PathBuf> {
    Ok(hf_cache_root()?.join(reference.repo_id.replace('/', "--")))
}

fn hf_cache_path(reference: &HfGgufReference) -> Result<PathBuf> {
    Ok(hf_cache_repo_dir(reference)?.join(&reference.filename))
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

fn download_hf_gguf(reference: &HfGgufReference, target: &Path) -> Result<()> {
    let local_dir = hf_cache_repo_dir(reference)?;

    let hf_status = Command::new("hf")
        .arg("download")
        .arg(&reference.repo_id)
        .arg(&reference.filename)
        .arg("--local-dir")
        .arg(&local_dir)
        .status();

    if matches!(hf_status, Ok(status) if status.success()) {
        return Ok(());
    }

    let hf_cli_status = Command::new("huggingface-cli")
        .arg("download")
        .arg(&reference.repo_id)
        .arg(&reference.filename)
        .arg("--local-dir")
        .arg(&local_dir)
        .status();

    if matches!(hf_cli_status, Ok(status) if status.success()) {
        return Ok(());
    }

    let url = hf_resolve_url(reference);
    let mut curl = Command::new("curl");
    curl.arg("--fail")
        .arg("--location")
        .arg("--create-dirs")
        .arg("--output")
        .arg(target);

    if let Some(token) = std::env::var_os("HF_TOKEN") {
        curl.arg("--header")
            .arg(format!("Authorization: Bearer {}", token.to_string_lossy()));
    }

    let curl_status = curl.arg(&url).status();
    if matches!(curl_status, Ok(status) if status.success()) {
        return Ok(());
    }

    bail!(
        "Failed to download Hugging Face model {}/{}; install `hf`, or ensure `curl` can reach {}",
        reference.repo_id,
        reference.filename,
        url
    )
}

fn hf_resolve_url(reference: &HfGgufReference) -> String {
    format!(
        "https://huggingface.co/{}/resolve/main/{}",
        reference.repo_id,
        url_path_encode(&reference.filename)
    )
}

fn url_path_encode(path: &str) -> String {
    path.split('/')
        .map(url_component_encode)
        .collect::<Vec<_>>()
        .join("/")
}

fn url_component_encode(component: &str) -> String {
    let mut encoded = String::new();
    for byte in component.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            _ => encoded.push_str(&format!("%{byte:02X}")),
        }
    }
    encoded
}

pub(crate) fn looks_like_hf_repo_id(model: &str) -> bool {
    !model.is_empty()
        && !model.starts_with('.')
        && !model.starts_with('~')
        && !Path::new(model).is_absolute()
        && model.matches('/').count() == 1
}

pub fn resolve_hf_safetensors_dir(model: &str) -> Result<PathBuf> {
    if let Some(dir) = cached_hf_safetensors_dir(model) {
        return Ok(dir);
    }
    download_hf_safetensors_repo(model)
}

fn cached_hf_safetensors_dir(model: &str) -> Option<PathBuf> {
    if let Some(snapshot_dir) = hf_hub::Cache::from_env()
        .model(model.to_string())
        .get("config.json")
        .and_then(|path| path.parent().map(Path::to_path_buf))
    {
        if snapshot_dir.join("config.json").is_file() {
            return Some(snapshot_dir);
        }
    }

    let cache_root = hf_hub::Cache::from_env().path().clone();
    let repo_dir = cache_root.join(format!("models--{}", model.replace('/', "--")));
    let snapshots_dir = repo_dir.join("snapshots");
    if !snapshots_dir.is_dir() {
        return None;
    }
    let mut snapshots: Vec<PathBuf> = std::fs::read_dir(&snapshots_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.join("config.json").is_file())
        .collect();
    snapshots.sort_by_key(|p| {
        std::fs::metadata(p)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH)
    });
    snapshots.pop()
}

fn download_hf_safetensors_repo(model: &str) -> Result<PathBuf> {
    let token = std::env::var("HF_TOKEN")
        .ok()
        .map(|t| t.trim().to_owned())
        .filter(|t| !t.is_empty());

    eprintln!("Resolving safetensors model from Hugging Face repo {model} ...");

    let hf_status = Command::new("hf")
        .arg("download")
        .arg(model)
        .status();

    if let Ok(status) = hf_status {
        if status.success() {
            let cache_dir = hf_hub::Cache::from_env().path().clone();
            let repo_dir = cache_dir.join(format!("models--{}", model.replace('/', "--")));
            let snapshots_dir = repo_dir.join("snapshots");
            if snapshots_dir.is_dir() {
                return find_latest_snapshot(&snapshots_dir, model);
            }
        }
    }

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

    eprintln!("Resolved Hugging Face repo {model} to {snapshot_dir:?}");
    Ok(snapshot_dir)
}

fn find_latest_snapshot(snapshots_dir: &Path, model: &str) -> Result<PathBuf> {
    let mut snapshots: Vec<PathBuf> = std::fs::read_dir(snapshots_dir)
        .with_context(|| format!("failed to read snapshots dir: {}", snapshots_dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.join("config.json").is_file())
        .collect();
    snapshots.sort_by_key(|p| {
        std::fs::metadata(p)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH)
    });
    snapshots
        .pop()
        .ok_or_else(|| anyhow::anyhow!("no snapshot with config.json found for {}", model))
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
        let has_safetensors = entries
            .iter()
            .any(|p| p.extension().and_then(|e| e.to_str()) == Some("safetensors"));
        let has_config_json = entries
            .iter()
            .any(|p| p.file_name().and_then(|n| n.to_str()) == Some("config.json"));

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

    // Multiple GGUF files - accept only one complete split shard set.
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
            shard_groups.entry(base.to_string()).or_default().push(file);
        } else {
            return None;
        }
    }

    // Only accept if exactly one shard group exists.
    if shard_groups.len() != 1 {
        return None;
    }

    let (_base, mut shards) = shard_groups.into_iter().next()?;
    if shards.len() < 2 {
        return None;
    }

    let mut indexes = Vec::with_capacity(shards.len());
    let mut expected_total = None;

    for shard in &shards {
        let (_base, (idx, total)) = shard
            .file_stem()
            .and_then(|n| n.to_str())
            .and_then(parse_shard_name)?;

        if idx == 0 || idx > total {
            return None;
        }

        if let Some(previous_total) = expected_total {
            if previous_total != total {
                return None;
            }
        } else {
            expected_total = Some(total);
        }

        indexes.push(idx);
    }

    let total = expected_total?;
    if shards.len() != total as usize {
        return None;
    }

    indexes.sort_unstable();
    indexes.dedup();
    if indexes.len() != total as usize || indexes != (1..=total).collect::<Vec<_>>() {
        return None;
    }

    // Sort by shard number to get the first shard.
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
    fn test_parse_hf_gguf_reference() {
        let reference =
            parse_hf_gguf_reference("unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf")
                .unwrap();

        assert_eq!(reference.repo_id, "unsloth/gemma-4-E2B-it-GGUF");
        assert_eq!(reference.filename, "gemma-4-E2B-it-Q4_K_M.gguf");
    }

    #[test]
    fn test_parse_hf_gguf_reference_nested_file() {
        let reference = parse_hf_gguf_reference("owner/repo/sub/dir/model.gguf").unwrap();

        assert_eq!(reference.repo_id, "owner/repo");
        assert_eq!(reference.filename, "sub/dir/model.gguf");
    }

    #[test]
    fn test_parse_hf_gguf_reference_rejects_local_or_invalid_inputs() {
        assert!(parse_hf_gguf_reference("./owner/repo/model.gguf").is_none());
        assert!(parse_hf_gguf_reference("../owner/repo/model.gguf").is_none());
        assert!(parse_hf_gguf_reference("/owner/repo/model.gguf").is_none());
        assert!(parse_hf_gguf_reference("https://huggingface.co/owner/repo/model.gguf").is_none());
        assert!(parse_hf_gguf_reference("owner/repo/model.bin").is_none());
        assert!(parse_hf_gguf_reference("owner/repo").is_none());
    }

    #[test]
    fn test_hf_resolve_url_encodes_filename_path() {
        let reference = HfGgufReference {
            repo_id: "owner/repo".to_string(),
            filename: "sub dir/model file-Q4_K_M.gguf".to_string(),
        };

        assert_eq!(
            hf_resolve_url(&reference),
            "https://huggingface.co/owner/repo/resolve/main/sub%20dir/model%20file-Q4_K_M.gguf"
        );
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
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Multiple unrelated"));
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

    #[test]
    fn test_dir_split_shards_with_unrelated_gguf_rejected() {
        let dir = create_temp_dir();
        fs::write(dir.path().join("m-00001-of-00002.gguf"), b"fake").unwrap();
        fs::write(dir.path().join("m-00002-of-00002.gguf"), b"fake").unwrap();
        fs::write(dir.path().join("other.gguf"), b"fake").unwrap();
        let result = resolve_model_path(dir.path().to_str().unwrap());
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Multiple unrelated"));
    }

    #[test]
    fn test_dir_incomplete_split_shards_rejected() {
        let dir = create_temp_dir();
        fs::write(dir.path().join("m-00001-of-00003.gguf"), b"fake").unwrap();
        fs::write(dir.path().join("m-00002-of-00003.gguf"), b"fake").unwrap();
        let result = resolve_model_path(dir.path().to_str().unwrap());
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Multiple unrelated"));
    }

    #[test]
    fn test_dir_duplicate_split_shards_rejected() {
        let dir = create_temp_dir();
        fs::write(dir.path().join("m-00001-of-00002.gguf"), b"fake").unwrap();
        fs::write(dir.path().join("m-copy-00001-of-00002.gguf"), b"fake").unwrap();
        let result = resolve_model_path(dir.path().to_str().unwrap());
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Multiple unrelated"));
    }
}
