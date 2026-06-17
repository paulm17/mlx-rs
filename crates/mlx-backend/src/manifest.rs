use std::path::{Path, PathBuf};

use anyhow::Context;
use serde_json::Value;
use crate::tensors::SafetensorsFile;

#[derive(Debug)]
pub struct ModelManifest {
    dir: PathBuf,
    config: Value,
    safetensors_files: Vec<PathBuf>,
}

pub struct TensorEntry {
    pub file_index: usize,
    pub dtype: String,
    pub shape: Vec<usize>,
}

impl ModelManifest {
    pub fn open(dir: &Path) -> anyhow::Result<Self> {
        if !dir.is_dir() {
            anyhow::bail!("not a directory: {}", dir.display());
        }

        let config_path = dir.join("config.json");
        if !config_path.exists() {
            anyhow::bail!(
                "config.json not found in model directory: {}",
                dir.display()
            );
        }

        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?;
        let config: Value =
            serde_json::from_str(&config_str).context("failed to parse config.json")?;

        let mut safetensors_files = Vec::new();

        let index_path = dir.join("model.safetensors.index.json");
        if index_path.exists() {
            let index_str = std::fs::read_to_string(&index_path)
                .with_context(|| format!("failed to read {}", index_path.display()))?;
            let index: Value =
                serde_json::from_str(&index_str).context("failed to parse model index")?;

            if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
                let mut file_set = std::collections::HashSet::new();
                for file in weight_map.values() {
                    if let Some(f) = file.as_str() {
                        file_set.insert(f.to_string());
                    }
                }
                let mut files: Vec<String> = file_set.into_iter().collect();
                files.sort();
                for f in files {
                    safetensors_files.push(dir.join(f));
                }
            }
        }

        if safetensors_files.is_empty() {
            let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
                .with_context(|| format!("failed to read directory: {}", dir.display()))?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.extension()
                        .map_or(false, |ext| ext == "safetensors")
                })
                .collect();
            entries.sort();
            safetensors_files = entries;
        }

        if safetensors_files.is_empty() {
            anyhow::bail!(
                "no .safetensors files found in: {}",
                dir.display()
            );
        }

        Ok(Self {
            dir: dir.to_path_buf(),
            config,
            safetensors_files,
        })
    }

    pub fn config(&self) -> &Value {
        &self.config
    }

    pub fn model_type(&self) -> Option<&str> {
        self.config
            .get("model_type")
            .and_then(|v| v.as_str())
            .or_else(|| {
                self.config
                    .get("architectures")
                    .and_then(|v| v.as_array())
                    .and_then(|a| a.first())
                    .and_then(|v| v.as_str())
            })
    }

    pub fn num_hidden_layers(&self) -> Option<usize> {
        self.config
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    }

    pub fn hidden_size(&self) -> Option<usize> {
        self.config
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    }

    pub fn vocab_size(&self) -> Option<usize> {
        self.config
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    }

    pub fn safetensors_files(&self) -> &[PathBuf] {
        &self.safetensors_files
    }

    pub fn safetensors_dir(&self) -> &Path {
        &self.dir
    }

    pub fn load_all_tensors(&self) -> anyhow::Result<std::collections::HashMap<String, crate::array::Array>> {
        let mut all = std::collections::HashMap::new();
        for path in &self.safetensors_files {
            let st = SafetensorsFile::load(path)
                .with_context(|| format!("failed to load {}", path.display()))?;
            let tensors = st.load_all()?;
            all.extend(tensors);
        }

        // Remap quantized tensor names following Ollama's convention:
        // "foo.weight" + "foo.scales" -> "foo.weight" (packed) + "foo_weight_scale"
        // "foo.biases" when foo.scales exists -> "foo_weight_qbias"
        // Only plural ".scales"/".biases" are quantization metadata.
        // Singular ".scale"/".bias" are NOT quantization tensors.
        let keys: Vec<String> = all.keys().cloned().collect();
        let scale_bases: std::collections::HashSet<String> = keys.iter()
            .filter(|k| k.ends_with(".scales"))
            .filter_map(|k| {
                k.strip_suffix(".scales").map(|s| s.to_string())
            })
            .collect();

        if !scale_bases.is_empty() {
            let mut renames: Vec<(String, String)> = Vec::new();
            for key in &keys {
                if let Some(base) = key.strip_suffix(".scales") {
                    if scale_bases.contains(base) {
                        renames.push((key.clone(), format!("{base}_scale")));
                    }
                } else if let Some(base) = key.strip_suffix(".biases") {
                    if scale_bases.contains(base) {
                        renames.push((key.clone(), format!("{base}_qbias")));
                    }
                }
            }
            for (old, new) in renames {
                if let Some(val) = all.remove(&old) {
                    all.insert(new, val);
                }
            }
        }

        Ok(all)
    }

    pub fn tensor_names(&self) -> anyhow::Result<Vec<String>> {
        let mut all = Vec::new();
        for path in &self.safetensors_files {
            let st = SafetensorsFile::load(path)
                .with_context(|| format!("failed to load {}", path.display()))?;
            all.extend(st.tensor_names()?);
        }
        all.sort();
        all.dedup();
        Ok(all)
    }

    pub fn total_tensor_bytes(&self) -> anyhow::Result<usize> {
        let mut total = 0;
        for path in &self.safetensors_files {
            let st = SafetensorsFile::load(path)
                .with_context(|| format!("failed to load {}", path.display()))?;
            for name in st.tensor_names()? {
                // We don't have direct byte count, estimate from tensor data
                // This is only used for display, so approximate is fine
                let _ = &name;
            }
        }
        // Fallback: use file size as estimate
        for path in &self.safetensors_files {
            let meta = std::fs::metadata(path)
                .with_context(|| format!("failed to stat {}", path.display()))?;
            total += meta.len() as usize;
        }
        Ok(total)
    }

    pub fn metadata(&self) -> anyhow::Result<std::collections::HashMap<String, String>> {
        Ok(std::collections::HashMap::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn make_model_dir(name: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let config = serde_json::json!({
            "model_type": "llama",
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "vocab_size": 100
        });
        fs::write(
            dir.path().join("config.json"),
            serde_json::to_string_pretty(&config).unwrap(),
        )
        .unwrap();

        if name == "multi" {
            let index = serde_json::json!({
                "weight_map": {
                    "model.embed.weight": "model-00001-of-00002.safetensors",
                    "model.head.weight": "model-00002-of-00002.safetensors"
                }
            });
            fs::write(
                dir.path().join("model.safetensors.index.json"),
                serde_json::to_string_pretty(&index).unwrap(),
            )
            .unwrap();
            create_dummy_safetensors(
                &dir.path().join("model-00001-of-00002.safetensors"),
                &[("model.embed.weight", vec![1.0f32; 64], vec![1, 64])],
            );
            create_dummy_safetensors(
                &dir.path().join("model-00002-of-00002.safetensors"),
                &[("model.head.weight", vec![2.0f32; 64], vec![1, 64])],
            );
        } else {
            create_dummy_safetensors(
                &dir.path().join("model.safetensors"),
                &[(name, vec![1.0f32; 4], vec![2, 2])],
            );
        }

        dir
    }

    fn create_dummy_safetensors(path: &Path, tensors: &[(&str, Vec<f32>, Vec<usize>)]) {
        let mut all_data = Vec::new();
        let mut entries = Vec::new();

        for (name, data, shape) in tensors {
            let start = all_data.len();
            let raw: &[u8] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
            };
            all_data.extend_from_slice(raw);
            let end = all_data.len();
            entries.push((name.to_string(), shape.clone(), start, end));
        }

        let mut header_map = std::collections::BTreeMap::new();
        for (name, shape, start, end) in &entries {
            header_map.insert(
                name.clone(),
                safetensors::tensor::TensorView::new(
                    safetensors::tensor::Dtype::F32,
                    shape.clone(),
                    &all_data[*start..*end],
                )
                .unwrap(),
            );
        }

        safetensors::serialize_to_file(header_map, &None, path).unwrap();
    }

    #[test]
    fn test_open_model_dir() {
        let dir = make_model_dir("weight");
        let manifest = ModelManifest::open(dir.path()).unwrap();
        assert_eq!(manifest.model_type(), Some("llama"));
        assert_eq!(manifest.num_hidden_layers(), Some(2));
        assert_eq!(manifest.hidden_size(), Some(64));
        assert_eq!(manifest.vocab_size(), Some(100));
        assert_eq!(manifest.safetensors_files().len(), 1);
    }

    #[test]
    fn test_tensor_names() {
        let dir = make_model_dir("weight");
        let manifest = ModelManifest::open(dir.path()).unwrap();
        let names = manifest.tensor_names().unwrap();
        assert_eq!(names, vec!["weight"]);
    }

    #[test]
    fn test_open_nonexistent_dir() {
        let result = ModelManifest::open(Path::new("/nonexistent"));
        assert!(result.is_err());
    }

    #[test]
    fn test_open_dir_without_config() {
        let dir = tempfile::tempdir().unwrap();
        let result = ModelManifest::open(dir.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("config.json"));
    }

    #[test]
    fn test_open_dir_without_safetensors() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("config.json"), "{}").unwrap();
        let result = ModelManifest::open(dir.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no .safetensors"));
    }

    #[test]
    fn test_multi_file_model() {
        let dir = make_model_dir("multi");
        let manifest = ModelManifest::open(dir.path()).unwrap();
        assert_eq!(manifest.safetensors_files().len(), 2);
        let names = manifest.tensor_names().unwrap();
        assert!(names.contains(&"model.embed.weight".to_string()));
        assert!(names.contains(&"model.head.weight".to_string()));
    }

    #[test]
    fn test_total_tensor_bytes() {
        let dir = make_model_dir("weight");
        let manifest = ModelManifest::open(dir.path()).unwrap();
        let bytes = manifest.total_tensor_bytes().unwrap();
        assert_eq!(bytes, 16); // 4 floats * 4 bytes
    }

    #[test]
    fn test_load_all_tensors() {
        if crate::loader::check_init().is_err() {
            return;
        }
        let dir = make_model_dir("weight");
        let manifest = ModelManifest::open(dir.path()).unwrap();
        let tensors = manifest.load_all_tensors().unwrap();
        assert_eq!(tensors.len(), 1);
        let arr = tensors.get("weight").unwrap();
        assert_eq!(arr.shape(), vec![2, 2]);
    }
}
