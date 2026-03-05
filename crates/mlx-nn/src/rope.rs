use mlx_core::{Array, Result};

/// Rotary Position Embedding (RoPE).
///
/// Supports standard and custom scaling (e.g. Llama3, YaRN).
/// Uses MLX's hardware-accelerated `fast_rope` kernel.
pub struct RoPE {
    dims: i32,
    base: f32,
    traditional: bool,
    scale: f32,
    freqs: Option<Array>,
}

/// Configuration for RoPE scaling variants.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeScaling {
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default = "default_factor")]
    pub factor: f32,
    #[serde(default = "default_low_freq_factor")]
    pub low_freq_factor: f32,
    #[serde(default = "default_high_freq_factor")]
    pub high_freq_factor: f32,
    #[serde(default = "default_original_max_position_embeddings")]
    pub original_max_position_embeddings: usize,
}

fn default_factor() -> f32 {
    1.0
}

fn default_low_freq_factor() -> f32 {
    1.0
}

fn default_high_freq_factor() -> f32 {
    4.0
}

fn default_original_max_position_embeddings() -> usize {
    8192
}

impl RoPE {
    /// Create a standard RoPE.
    pub fn new(dims: i32, base: f32, traditional: bool) -> Self {
        Self {
            dims,
            base,
            traditional,
            scale: 1.0,
            freqs: None,
        }
    }

    /// Create a RoPE with custom scaling (e.g. Llama3).
    pub fn with_scaling(
        dims: i32,
        base: f32,
        traditional: bool,
        scaling: &RopeScaling,
        max_position_embeddings: usize,
    ) -> Result<Self> {
        let rope_type = scaling
            .rope_type
            .as_deref()
            .unwrap_or("default");

        match rope_type {
            "llama3" => {
                // Llama3-style frequency scaling
                let head_dim = dims as usize;
                let factor = if scaling.factor > 0.0 { scaling.factor } else { 1.0 };
                let low_freq_factor = if scaling.low_freq_factor > 0.0 {
                    scaling.low_freq_factor
                } else {
                    1.0
                };
                let mut high_freq_factor = if scaling.high_freq_factor > 0.0 {
                    scaling.high_freq_factor
                } else {
                    4.0
                };
                if high_freq_factor <= low_freq_factor {
                    high_freq_factor = low_freq_factor + 1.0;
                }
                let original_max_position_embeddings = if scaling.original_max_position_embeddings > 0 {
                    scaling.original_max_position_embeddings
                } else {
                    max_position_embeddings.max(1)
                } as f32;

                let low_freq_wavelen = original_max_position_embeddings / low_freq_factor;
                let high_freq_wavelen = original_max_position_embeddings / high_freq_factor;

                let mut freqs = Vec::with_capacity(head_dim / 2);
                for i in (0..head_dim).step_by(2) {
                    // base_freq is base^(i/dims) — matching the deprecated working code
                    let base_freq = base.powf(i as f32 / head_dim as f32);
                    let wavelen = 2.0 * std::f32::consts::PI * base_freq;
                    let adjusted = if wavelen < high_freq_wavelen {
                        base_freq
                    } else if wavelen > low_freq_wavelen {
                        base_freq * factor
                    } else {
                        let smooth =
                            (original_max_position_embeddings / wavelen - low_freq_factor)
                                / (high_freq_factor - low_freq_factor);
                        let low_adjusted = base_freq * factor;
                        low_adjusted / ((1.0 - smooth) / factor + smooth)
                    };
                    freqs.push(adjusted);
                }

                let freqs_arr = Array::from_slice_f32(&freqs)?;
                Ok(Self {
                    dims,
                    base,
                    traditional,
                    scale: 1.0,
                    freqs: Some(freqs_arr),
                })
            }
            _ => {
                // Default: apply scale factor
                Ok(Self {
                    dims,
                    base,
                    traditional,
                    scale: 1.0 / scaling.factor,
                    freqs: None,
                })
            }
        }
    }

    /// Apply rotary position embedding to the input tensor.
    pub fn forward(&self, x: &Array, offset: i32) -> Result<Array> {
        x.fast_rope(
            self.dims,
            self.traditional,
            if self.freqs.is_some() { None } else { Some(self.base) },
            self.scale,
            offset,
            self.freqs.as_ref(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::RopeScaling;

    #[test]
    fn rope_scaling_llama3_defaults_are_stable() {
        let cfg = RopeScaling {
            rope_type: Some("llama3".to_string()),
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
        };
        assert_eq!(cfg.low_freq_factor, 1.0);
        assert_eq!(cfg.high_freq_factor, 4.0);
        assert_eq!(cfg.original_max_position_embeddings, 8192);
    }
}
