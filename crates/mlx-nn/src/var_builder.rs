use mlx_core::Array;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Weight loading abstraction inspired by candle's VarBuilder.
///
/// Provides scoped, prefix-based access to named tensors loaded from
/// safetensors files. The key innovation is the `pp()` method which
/// creates a new VarBuilder with a pushed prefix, enabling clean
/// hierarchical weight access.
///
/// # Example
/// ```no_run
/// use mlx_nn::VarBuilder;
/// use mlx_core::DType;
///
/// let vb = VarBuilder::from_safetensors(&["model.safetensors"], DType::Float16).unwrap();
/// let layer_vb = vb.pp("model.layers.0.self_attn");
/// let q_weight = layer_vb.get("q_proj.weight").unwrap();
/// // resolves to "model.layers.0.self_attn.q_proj.weight"
/// ```
#[derive(Clone)]
pub struct VarBuilder {
    data: std::sync::Arc<HashMap<String, Array>>,
    prefix: Vec<String>,
    dtype: mlx_core::DType,
}

impl VarBuilder {
    /// Create a VarBuilder from safetensors shard files.
    pub fn from_safetensors<P: AsRef<Path>>(
        paths: &[P],
        dtype: mlx_core::DType,
    ) -> anyhow::Result<Self> {
        let mut all_tensors = HashMap::new();
        for path in paths {
            let tensors = mlx_core::safetensors::load(path)
                .map_err(|e| anyhow::anyhow!("failed loading {}: {e}", path.as_ref().display()))?;
            all_tensors.extend(tensors);
        }
        Ok(Self {
            data: std::sync::Arc::new(all_tensors),
            prefix: Vec::new(),
            dtype,
        })
    }

    /// Create a VarBuilder from a pre-loaded weight map.
    pub fn from_weights(weights: HashMap<String, Array>, dtype: mlx_core::DType) -> Self {
        Self {
            data: std::sync::Arc::new(weights),
            prefix: Vec::new(),
            dtype,
        }
    }

    /// Push a prefix scope ("push prefix").
    ///
    /// Returns a new VarBuilder with the given segment appended to the prefix.
    /// This is the key method for hierarchical weight access.
    pub fn pp(&self, segment: impl ToString) -> Self {
        let mut new_prefix = self.prefix.clone();
        new_prefix.push(segment.to_string());
        Self {
            data: self.data.clone(),
            prefix: new_prefix,
            dtype: self.dtype,
        }
    }

    /// Get the current prefix as a dot-separated string.
    pub fn prefix(&self) -> String {
        self.prefix.join(".")
    }

    /// Resolve a tensor name to its full path.
    fn full_path(&self, name: &str) -> String {
        if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.prefix.join("."), name)
        }
    }

    /// Get a tensor by name at the current prefix.
    pub fn get(&self, name: &str) -> anyhow::Result<Array> {
        let path = self.full_path(name);
        self.data
            .get(&path).cloned()
            .ok_or_else(|| anyhow::anyhow!("tensor not found: {path}"))
    }

    /// Check if a tensor exists at the current prefix.
    pub fn contains(&self, name: &str) -> bool {
        let path = self.full_path(name);
        self.data.contains_key(&path)
    }

    /// Get the dtype.
    pub fn dtype(&self) -> mlx_core::DType {
        self.dtype
    }

    /// Get a reference to the underlying data map.
    pub fn data(&self) -> &HashMap<String, Array> {
        &self.data
    }

    /// Return all tensor names.
    pub fn tensor_names(&self) -> Vec<&String> {
        self.data.keys().collect()
    }

    /// Discover all safetensors shards in a directory.
    pub fn discover_shards(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
        let entries = std::fs::read_dir(dir)
            .map_err(|e| anyhow::anyhow!("failed reading directory {}: {e}", dir.display()))?;

        let mut shards: Vec<PathBuf> = entries
            .filter_map(std::result::Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
            .collect();
        shards.sort();

        if shards.is_empty() {
            return Err(anyhow::anyhow!(
                "no .safetensors files found in {}",
                dir.display()
            ));
        }
        Ok(shards)
    }

    /// Create a VarBuilder by loading all safetensors files from a directory.
    pub fn from_dir(dir: &Path, dtype: mlx_core::DType) -> anyhow::Result<Self> {
        let shards = Self::discover_shards(dir)?;
        let shard_refs: Vec<&Path> = shards.iter().map(|p| p.as_path()).collect();
        Self::from_safetensors(&shard_refs, dtype)
    }
}

impl std::fmt::Debug for VarBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VarBuilder")
            .field("prefix", &self.prefix())
            .field("dtype", &self.dtype)
            .field("num_tensors", &self.data.len())
            .finish()
    }
}
