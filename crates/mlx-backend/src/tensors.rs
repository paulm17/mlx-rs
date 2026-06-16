use std::collections::HashMap;
use std::path::Path;

use anyhow::Context;
use safetensors::tensor::Dtype;

use crate::array::Array;
use crate::ffi::MlxDtype;

pub struct SafetensorsFile {
    tensors: HashMap<String, TensorInfo>,
    metadata: HashMap<String, String>,
}

struct TensorInfo {
    dtype: MlxDtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

fn st_dtype_to_mlx(dtype: Dtype) -> anyhow::Result<MlxDtype> {
    match dtype {
        Dtype::BOOL => Ok(MlxDtype::Bool),
        Dtype::U8 => Ok(MlxDtype::Uint8),
        Dtype::I8 => Ok(MlxDtype::Int8),
        Dtype::I16 => Ok(MlxDtype::Int16),
        Dtype::U16 => Ok(MlxDtype::Uint16),
        Dtype::I32 => Ok(MlxDtype::Int32),
        Dtype::U32 => Ok(MlxDtype::Uint32),
        Dtype::I64 => Ok(MlxDtype::Int64),
        Dtype::U64 => Ok(MlxDtype::Uint64),
        Dtype::F16 => Ok(MlxDtype::Float16),
        Dtype::BF16 => Ok(MlxDtype::Bfloat16),
        Dtype::F32 => Ok(MlxDtype::Float32),
        Dtype::F64 => Ok(MlxDtype::Float64),
        _ => Err(anyhow::anyhow!("unsupported safetensors dtype: {:?}", dtype)),
    }
}

impl SafetensorsFile {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read(path)
            .with_context(|| format!("failed to read safetensors file: {}", path.display()))?;
        Self::from_bytes(&data)
    }

    pub fn from_bytes(data: &[u8]) -> anyhow::Result<Self> {
        let st = safetensors::SafeTensors::deserialize(data)
            .context("failed to deserialize safetensors")?;

        let mut tensors = HashMap::new();
        for (name, tensor) in st.tensors() {
            let dtype = st_dtype_to_mlx(tensor.dtype())?;
            let shape: Vec<usize> = tensor.shape().to_vec();
            let data = tensor.data().to_vec();
            tensors.insert(name, TensorInfo { dtype, shape, data });
        }

        let metadata = Self::parse_metadata(data).unwrap_or_default();

        Ok(Self { tensors, metadata })
    }

    fn parse_metadata(data: &[u8]) -> anyhow::Result<HashMap<String, String>> {
        use std::io::Read;
        let mut cursor = std::io::Cursor::new(data);
        let mut size_buf = [0u8; 8];
        cursor.read_exact(&mut size_buf)?;
        let header_size = u64::from_le_bytes(size_buf) as usize;
        if header_size > 100 * 1024 * 1024 {
            anyhow::bail!("header too large");
        }
        let mut header_buf = vec![0u8; header_size];
        cursor.read_exact(&mut header_buf)?;
        let header: serde_json::Value = serde_json::from_slice(&header_buf)?;
        if let Some(meta) = header.get("__metadata__").and_then(|v| v.as_object()) {
            Ok(meta
                .iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect())
        } else {
            Ok(HashMap::new())
        }
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tensors.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    pub fn tensor_info(&self, name: &str) -> Option<(&[usize], MlxDtype)> {
        self.tensors
            .get(name)
            .map(|info| (info.shape.as_slice(), info.dtype))
    }

    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    pub fn get_tensor(&self, name: &str) -> anyhow::Result<Array> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("tensor not found: {name}"))?;
        Array::from_data(&info.data, &info.shape, info.dtype)
    }

    pub fn load_all(&self) -> anyhow::Result<HashMap<String, Array>> {
        let mut result = HashMap::new();
        for (name, info) in &self.tensors {
            let arr = Array::from_data(&info.data, &info.shape, info.dtype)?;
            result.insert(name.clone(), arr);
        }
        Ok(result)
    }

    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }

    pub fn total_bytes(&self) -> usize {
        self.tensors.values().map(|t| t.data.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_safetensors_bytes(
        tensors: &[(&str, Vec<f32>, Vec<usize>)],
    ) -> Vec<u8> {
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

        safetensors::serialize(&header_map, &None).unwrap()
    }

    #[test]
    fn test_load_safetensors_from_bytes() {
        let bytes = make_safetensors_bytes(&[
            ("weight", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            ("bias", vec![0.5, -0.5], vec![2]),
        ]);

        let st = SafetensorsFile::from_bytes(&bytes).unwrap();
        assert_eq!(st.num_tensors(), 2);
        assert_eq!(st.tensor_names(), vec!["bias", "weight"]);
    }

    #[test]
    fn test_tensor_info() {
        let bytes = make_safetensors_bytes(&[
            ("layer.weight", vec![1.0; 6], vec![2, 3]),
        ]);

        let st = SafetensorsFile::from_bytes(&bytes).unwrap();
        let (shape, dtype) = st.tensor_info("layer.weight").unwrap();
        assert_eq!(shape, &[2, 3]);
        assert_eq!(dtype, MlxDtype::Float32);
    }

    #[test]
    fn test_get_tensor_creates_mlx_array() {
        if crate::loader::check_init().is_err() {
            return;
        }

        let bytes = make_safetensors_bytes(&[
            ("w", vec![1.0, 2.0, 3.0], vec![3]),
        ]);

        let st = SafetensorsFile::from_bytes(&bytes).unwrap();
        let arr = st.get_tensor("w").unwrap();
        assert_eq!(arr.shape(), vec![3]);
        assert_eq!(arr.dtype().unwrap(), MlxDtype::Float32);
        arr.eval().unwrap();
        let data = arr.data_f32().unwrap();
        assert_eq!(data, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_load_all() {
        if crate::loader::check_init().is_err() {
            return;
        }

        let bytes = make_safetensors_bytes(&[
            ("a", vec![1.0], vec![1]),
            ("b", vec![2.0, 3.0], vec![2]),
        ]);

        let st = SafetensorsFile::from_bytes(&bytes).unwrap();
        let all = st.load_all().unwrap();
        assert_eq!(all.len(), 2);
        assert!(all.contains_key("a"));
        assert!(all.contains_key("b"));
    }

    #[test]
    fn test_total_bytes() {
        let bytes = make_safetensors_bytes(&[
            ("x", vec![1.0, 2.0], vec![2]),
            ("y", vec![3.0, 4.0, 5.0], vec![3]),
        ]);

        let st = SafetensorsFile::from_bytes(&bytes).unwrap();
        assert_eq!(st.total_bytes(), 20); // (2+3) * 4 bytes each
    }

    #[test]
    fn test_nonexistent_tensor() {
        let bytes = make_safetensors_bytes(&[]);
        let st = SafetensorsFile::from_bytes(&bytes).unwrap();
        let result = st.get_tensor("nonexistent");
        assert!(result.is_err());
    }
}
