use crate::VarBuilder;
use mlx_core::{Array, DType, Module, Result};

/// Standard LayerNorm over the final dimension.
pub struct LayerNorm {
    weight: Array,
    bias: Array,
    eps: f32,
}

impl LayerNorm {
    pub fn new(eps: f32, vb: &VarBuilder) -> anyhow::Result<Self> {
        Ok(Self {
            weight: vb.get("weight")?,
            bias: vb.get("bias")?,
            eps,
        })
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Array) -> Result<Array> {
        let x_dtype = x.dtype();
        let x = x.as_type(DType::Float32)?;
        let mean = x.mean_axis(-1, true)?;
        let centered = x.subtract(&mean)?;
        let variance = centered.square()?.mean_axis(-1, true)?;
        let eps = Array::from_float(self.eps)?;
        let inv_std = variance.add(&eps)?.sqrt()?;
        let normalized = centered.divide(&inv_std)?;
        let weight = self.weight.as_type(DType::Float32)?;
        let bias = self.bias.as_type(DType::Float32)?;
        normalized.multiply(&weight)?.add(&bias)?.as_type(x_dtype)
    }
}
