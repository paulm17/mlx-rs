use crate::VarBuilder;
use mlx_core::{Array, Module, Result};

/// RMS Normalization layer.
///
/// Uses MLX's hardware-accelerated `fast_rms_norm` kernel.
pub struct RmsNorm {
    weight: Array,
    eps: f32,
}

impl RmsNorm {
    /// Load from a VarBuilder.
    pub fn new(eps: f32, vb: &VarBuilder) -> anyhow::Result<Self> {
        let weight = vb.get("weight")?;
        Ok(Self { weight, eps })
    }

    /// Create from a raw weight tensor.
    pub fn from_weight(weight: Array, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// Get a reference to the weight tensor.
    pub fn weight(&self) -> &Array {
        &self.weight
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Array) -> Result<Array> {
        x.fast_rms_norm(&self.weight, self.eps)
    }
}
