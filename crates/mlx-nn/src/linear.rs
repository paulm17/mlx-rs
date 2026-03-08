use crate::VarBuilder;
use mlx_core::{Array, Module, Result};

/// A linear layer supporting both full-precision and quantized weights.
///
/// For full-precision: stores the pre-transposed weight to avoid
/// transposing on every forward call.
/// For quantized: uses `quantized_matmul` which has a built-in transpose flag.
pub struct Linear {
    weight: Array,
    /// Pre-transposed weight for full-precision layers (None for quantized).
    weight_t: Option<Array>,
    bias: Option<Array>,
    // Quantization parameters (None if full-precision)
    scales: Option<Array>,
    biases: Option<Array>, // quantization zero-point biases
    group_size: i32,
    bits: i32,
}

impl Linear {
    /// Load a linear layer from the VarBuilder at the current prefix.
    ///
    /// Automatically detects quantized weights if `scales` and `biases` are present.
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
        let bias = if vb.contains("bias") {
            Some(vb.get("bias")?)
        } else {
            None
        };

        let (group_size, bits) = if let Some(ref s) = scales {
            (
                config.group_size,
                infer_bits(
                    &weight.shape_raw(),
                    &s.shape_raw(),
                    config.group_size,
                    config.bits,
                ),
            )
        } else {
            (0, 0)
        };

        // Pre-transpose for full-precision layers to avoid per-call transpose
        let weight_t = if scales.is_none() {
            Some(weight.transpose()?)
        } else {
            None
        };

        Ok(Self {
            weight,
            weight_t,
            scales,
            biases,
            bias,
            group_size,
            bits,
        })
    }

    /// Create from raw tensors (for testing or tied weights).
    pub fn from_weights(weight: Array, bias: Option<Array>) -> Self {
        let weight_t = Some(weight.transpose().expect("transpose failed"));
        Self {
            weight,
            weight_t,
            bias,
            scales: None,
            biases: None,
            group_size: 0,
            bits: 0,
        }
    }

    /// Create from raw tensors with quantization parameters (for tied quantized weights).
    pub fn from_weights_quantized(
        weight: Array,
        bias: Option<Array>,
        scales: Option<Array>,
        biases: Option<Array>,
        group_size: i32,
        bits: i32,
    ) -> Self {
        let weight_t = if scales.is_none() {
            Some(weight.transpose().expect("transpose failed"))
        } else {
            None
        };
        Self {
            weight,
            weight_t,
            bias,
            scales,
            biases,
            group_size,
            bits,
        }
    }

    /// Get a reference to the weight tensor.
    pub fn weight(&self) -> &Array {
        &self.weight
    }

    /// Check if this linear layer exists at the given VarBuilder prefix.
    pub fn has_weight(vb: &VarBuilder) -> bool {
        vb.contains("weight")
    }
}

impl Module for Linear {
    fn forward(&self, x: &Array) -> Result<Array> {
        let out = if let Some(ref scales) = self.scales {
            // Quantized: uses built-in transpose flag
            x.quantized_matmul(
                &self.weight,
                scales,
                self.biases.as_ref(),
                true,
                self.group_size,
                self.bits,
            )?
        } else {
            // Full-precision: use pre-transposed weight
            x.matmul(self.weight_t.as_ref().unwrap())?
        };
        if let Some(ref bias) = self.bias {
            out.add(bias)
        } else {
            Ok(out)
        }
    }
}

/// Quantization configuration.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    pub group_size: i32,
    pub bits: i32,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            group_size: 64,
            bits: 4,
        }
    }
}

/// Infer the number of quantization bits from weight and scales shapes.
fn infer_bits(weight_shape: &[i32], scales_shape: &[i32], group_size: i32, fallback: i32) -> i32 {
    let packed = weight_shape.last().copied().unwrap_or(0) as i64;
    let n_groups = scales_shape.last().copied().unwrap_or(0) as i64;
    if packed <= 0 || n_groups <= 0 || group_size <= 0 {
        return fallback;
    }
    let unpacked = n_groups * group_size as i64;
    if unpacked <= 0 {
        return fallback;
    }
    let num = packed * 32;
    if num % unpacked != 0 {
        return fallback;
    }
    let bits = (num / unpacked) as i32;
    match bits {
        2 | 4 | 8 => bits,
        _ => fallback,
    }
}
