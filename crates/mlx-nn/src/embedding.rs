use crate::linear::QuantConfig;
use crate::var_builder::VarBuilder;
use mlx_core::{Array, Module, Result};

/// Embedding layer supporting both full-precision and quantized weights.
///
/// Maps integer token IDs to dense vectors. Implements `Module` where
/// `forward` takes a tensor of token IDs and returns embeddings.
pub struct Embedding {
    weight: Array,
    // Quantization parameters
    scales: Option<Array>,
    biases: Option<Array>,
    group_size: i32,
    bits: i32,
}

impl Embedding {
    /// Load an embedding layer from the VarBuilder.
    pub fn new(vb: &VarBuilder, config: &QuantConfig) -> anyhow::Result<Self> {
        let weight = vb.get("weight")?;
        let scales = if vb.contains("scales") {
            Some(vb.get("scales")?)
        } else {
            None
        };
        let biases = if scales.is_some() && vb.contains("biases") {
            Some(vb.get("biases")?)
        } else {
            None
        };

        let (group_size, bits) = if let Some(ref s) = scales {
            let ws = weight.shape_raw();
            let ss = s.shape_raw();
            let packed = ws.last().copied().unwrap_or(0) as i64;
            let n_groups = ss.last().copied().unwrap_or(0) as i64;
            let gs = config.group_size as i64;
            if packed > 0 && n_groups > 0 && gs > 0 {
                let unpacked = n_groups * gs;
                let bits_val = if unpacked > 0 && (packed * 32) % unpacked == 0 {
                    let b = (packed * 32 / unpacked) as i32;
                    if matches!(b, 2 | 4 | 8) {
                        b
                    } else {
                        config.bits
                    }
                } else {
                    config.bits
                };
                (config.group_size, bits_val)
            } else {
                (config.group_size, config.bits)
            }
        } else {
            (0, 0)
        };

        Ok(Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
        })
    }

    /// Create from a raw weight tensor (for tied embeddings).
    pub fn from_weight(weight: Array) -> Self {
        Self {
            weight,
            scales: None,
            biases: None,
            group_size: 0,
            bits: 0,
        }
    }

    /// Get a reference to the weight tensor.
    pub fn weight(&self) -> &Array {
        &self.weight
    }

    /// Create a Linear layer that shares this embedding's weight (and quantization params).
    ///
    /// Used for tied word embeddings (lm_head = embed_tokens).
    pub fn as_linear(&self) -> crate::Linear {
        crate::Linear::from_weights_quantized(
            self.weight.clone(),
            None,
            self.scales.clone(),
            self.biases.clone(),
            self.group_size,
            self.bits,
        )
    }
}

impl Module for Embedding {
    fn forward(&self, token_ids: &Array) -> Result<Array> {
        if let Some(ref scales) = self.scales {
            // Quantized embedding lookup: gather then dequantize
            let q_rows = self.weight.take(token_ids, 0)?;
            let s_rows = scales.take(token_ids, 0)?;
            let b_rows = self
                .biases
                .as_ref()
                .map(|b| b.take(token_ids, 0))
                .transpose()?;
            q_rows.dequantize(&s_rows, b_rows.as_ref(), self.group_size, self.bits)
        } else {
            // Full-precision: simple gather
            self.weight.take(token_ids, 0)
        }
    }
}
