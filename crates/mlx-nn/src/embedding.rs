use crate::linear::{infer_quant_params, QuantConfig};
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
            infer_quant_params(&weight.shape_raw(), &s.shape_raw(), config)
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
            // Quantized embedding lookup: gather then dequantize.
            // MLX dequantize requires a 2D matrix, so we must ensure the
            // gathered weight has at least 2 dims. Flatten/squeeze indices
            // as needed, then reshape back.
            let orig_ndim = token_ids.ndim();
            let flat_ids = if orig_ndim == 0 {
                token_ids.reshape(&[1])?
            } else if orig_ndim > 1 {
                token_ids.reshape(&[-1])?
            } else {
                token_ids.clone()
            };
            let q_rows = self.weight.take(&flat_ids, 0)?;
            let s_rows = scales.take(&flat_ids, 0)?;
            let b_rows = self
                .biases
                .as_ref()
                .map(|b| b.take(&flat_ids, 0))
                .transpose()?;
            // MLX dequantize requires at least 2D; ensure q_rows is 2D by
            // adding a leading dim when flat_ids has only one element.
            let (q_rows, s_rows, b_rows) = if q_rows.ndim() == 1 {
                let packed = q_rows.shape_raw().last().copied().unwrap_or(0) as i32;
                (
                    q_rows.reshape(&[1, packed])?,
                    s_rows.reshape(&[1, s_rows.shape_raw().last().copied().unwrap_or(0)])?,
                    b_rows.map(|b| b.reshape(&[1, b.shape_raw().last().copied().unwrap_or(0)])).transpose()?,
                )
            } else {
                (q_rows, s_rows, b_rows)
            };
            let deq = q_rows.dequantize(&s_rows, b_rows.as_ref(), self.group_size, self.bits)?;
            if orig_ndim == 0 {
                // Return [hidden] (squeeze the batch dim)
                let packed = q_rows.shape_raw().last().copied().unwrap_or(0) as i32;
                let hidden = packed * 32 / self.bits.max(1);
                deq.reshape(&[hidden])
            } else if orig_ndim > 1 {
                let mut out_shape = token_ids.shape_raw();
                let packed = q_rows.shape_raw().last().copied().unwrap_or(0) as i32;
                let hidden = packed * 32 / self.bits.max(1);
                out_shape.push(hidden);
                deq.reshape(&out_shape)
            } else {
                Ok(deq)
            }
        } else {
            // Full-precision: simple gather (MLX take handles multi-dim indices natively)
            self.weight.take(token_ids, 0)
        }
    }
}
