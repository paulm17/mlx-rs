//! Gemma 4 multimodal model implementation (2B variant only).
//!
//! Supports vision-language tasks with per-layer gating, KV cache sharing,
//! and ClippableLinear vision components.

use mlx_core::{Array, DType, Module, Result};
use mlx_nn::{repeat_kv, Embedding, KvCache, Linear, QuantConfig, VarBuilder};
use std::collections::HashMap;

// ------------------------------------------------------------------
// Config structs
// ------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RopeParams {
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
}

fn default_partial_rotary_factor() -> f32 {
    0.25
}

impl<'de> serde::Deserialize<'de> for RopeParams {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct RopeParamsVisitor;

        impl<'de> serde::de::Visitor<'de> for RopeParamsVisitor {
            type Value = RopeParams;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a RopeParams object or a bare float (rope_theta)")
            }

            fn visit_f64<E>(self, v: f64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(RopeParams {
                    rope_theta: v as f32,
                    partial_rotary_factor: default_partial_rotary_factor(),
                })
            }

            fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(RopeParams {
                    rope_theta: v as f32,
                    partial_rotary_factor: default_partial_rotary_factor(),
                })
            }

            fn visit_u64<E>(self, v: u64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(RopeParams {
                    rope_theta: v as f32,
                    partial_rotary_factor: default_partial_rotary_factor(),
                })
            }

            fn visit_str<E>(self, v: &str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                if v == "default" {
                    Ok(RopeParams {
                        rope_theta: 10000.0,
                        partial_rotary_factor: default_partial_rotary_factor(),
                    })
                } else {
                    Err(serde::de::Error::custom(format!(
                        "unexpected rope_parameters string: {}",
                        v
                    )))
                }
            }

            fn visit_map<M>(self, mut map: M) -> std::result::Result<Self::Value, M::Error>
            where
                M: serde::de::MapAccess<'de>,
            {
                let mut rope_theta = None;
                let mut partial_rotary_factor = default_partial_rotary_factor();

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "rope_theta" => rope_theta = Some(map.next_value::<f32>()?),
                        "partial_rotary_factor" => {
                            partial_rotary_factor = map.next_value::<f32>()?
                        }
                        _ => {
                            let _ = map.next_value::<serde_json::Value>()?;
                        }
                    }
                }

                Ok(RopeParams {
                    rope_theta: rope_theta
                        .ok_or_else(|| serde::de::Error::missing_field("rope_theta"))?,
                    partial_rotary_factor,
                })
            }
        }

        deserializer.deserialize_any(RopeParamsVisitor)
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Gemma4TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub num_global_key_value_heads: Option<usize>,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub sliding_window: usize,
    #[serde(default = "default_sliding_window_pattern")]
    pub sliding_window_pattern: usize,
    pub layer_types: Vec<String>,
    pub rope_parameters: HashMap<String, RopeParams>,
    pub final_logit_softcapping: Option<f32>,
    pub tie_word_embeddings: bool,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    pub hidden_activation: String,
    pub enable_moe_block: bool,
    pub num_experts: Option<usize>,
    pub top_k_experts: Option<usize>,
    pub moe_intermediate_size: Option<usize>,
    pub attention_k_eq_v: bool,
    pub num_kv_shared_layers: usize,
    pub hidden_size_per_layer_input: usize,
    pub use_double_wide_mlp: bool,
    #[serde(default = "default_vocab_size_per_layer_input")]
    pub vocab_size_per_layer_input: usize,
}

fn default_vocab_size_per_layer_input() -> usize {
    262144
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_sliding_window_pattern() -> usize {
    6
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Gemma4VisionConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub position_embedding_size: usize,
    pub pooling_kernel_size: usize,
    pub default_output_length: usize,
    #[serde(default)]
    pub max_patches: usize,
    #[serde(default)]
    pub standardize: bool,
    #[serde(default)]
    pub rope_parameters: HashMap<String, RopeParams>,
    #[serde(default)]
    pub use_clipped_linears: bool,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Gemma4AudioConfig {
    // Placeholder; not used for 2B
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Gemma4Config {
    pub text_config: Gemma4TextConfig,
    pub vision_config: Gemma4VisionConfig,
    pub audio_config: Option<Gemma4AudioConfig>,
    pub model_type: String,
    pub image_token_id: u32,
    pub audio_token_id: Option<u32>,
    pub boi_token_id: u32,
    pub eoi_token_id: u32,
    pub boa_token_id: Option<u32>,
    pub eoa_token_id: Option<u32>,
    pub vision_soft_tokens_per_image: usize,
}

// ------------------------------------------------------------------
// RMS Norm variants
// ------------------------------------------------------------------

struct RmsNormNoScale {
    eps: f32,
}

impl RmsNormNoScale {
    fn new(_vb: &VarBuilder, eps: f32) -> anyhow::Result<Self> {
        Ok(Self { eps })
    }
}

impl Module for RmsNormNoScale {
    fn forward(&self, x: &Array) -> Result<Array> {
        let shape = x.shape_raw();
        let last_dim = shape.last().copied().unwrap_or(1);
        let ones = Array::ones(&[last_dim], x.dtype())?;
        x.fast_rms_norm(&ones, self.eps)
    }
}

pub struct RmsNormZeroShift {
    weight: Array,
    eps: f32,
}

impl RmsNormZeroShift {
    fn new(vb: &VarBuilder, eps: f32) -> anyhow::Result<Self> {
        let weight = vb.get("weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNormZeroShift {
    fn forward(&self, x: &Array) -> Result<Array> {
        x.fast_rms_norm(&self.weight, self.eps)
    }
}

struct VisionRmsNorm {
    weight: Array,
    eps: f32,
}

impl VisionRmsNorm {
    fn new(vb: &VarBuilder, eps: f32) -> anyhow::Result<Self> {
        let weight = vb.get("weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for VisionRmsNorm {
    fn forward(&self, x: &Array) -> Result<Array> {
        let orig_dtype = x.dtype();
        let x_f32 = x.as_type(DType::Float32)?;
        let weight_f32 = self.weight.as_type(DType::Float32)?;
        let normed = x_f32.fast_rms_norm(&weight_f32, self.eps)?;
        normed.as_type(orig_dtype)
    }
}

struct VisionRmsNormNoScale {
    eps: f32,
}

impl VisionRmsNormNoScale {
    fn new(_vb: &VarBuilder, eps: f32) -> anyhow::Result<Self> {
        Ok(Self { eps })
    }
}

impl Module for VisionRmsNormNoScale {
    fn forward(&self, x: &Array) -> Result<Array> {
        let orig_dtype = x.dtype();
        let x_f32 = x.as_type(DType::Float32)?;
        let shape = x.shape_raw();
        let last_dim = shape.last().copied().unwrap_or(1);
        let ones = Array::ones(&[last_dim], DType::Float32)?;
        let normed = x_f32.fast_rms_norm(&ones, self.eps)?;
        normed.as_type(orig_dtype)
    }
}

// ------------------------------------------------------------------
// Utility functions
// ------------------------------------------------------------------

fn clamp_array(x: &Array, min: Option<&Array>, max: Option<&Array>) -> Result<Array> {
    let mut result = x.clone();
    if let Some(min_val) = min {
        let mask = result.greater(min_val)?;
        result = mask.where_cond(&result, min_val)?;
    }
    if let Some(max_val) = max {
        let mask = result.less(max_val)?;
        result = mask.where_cond(&result, max_val)?;
    }
    Ok(result)
}

fn gelu_approx(x: &Array) -> Result<Array> {
    let half = Array::from_float(0.5)?;
    let one = Array::from_float(1.0)?;
    let sqrt_2_over_pi = Array::from_float((2.0f32 / std::f32::consts::PI).sqrt())?;
    let coeff = Array::from_float(0.044715f32)?;
    let three = Array::from_float(3.0)?;

    let x3 = x.power(&three)?;
    let inner = x.add(&x3.multiply(&coeff)?)?;
    let inner = inner.multiply(&sqrt_2_over_pi)?;
    let tanh_val = inner.tanh()?;
    let gate = one.add(&tanh_val)?;
    let gated = x.multiply(&gate)?;
    half.multiply(&gated)
}

/// Build the layer-index → cache-index mapping used for KV sharing.
///
/// Non-shared layers each own a cache. Shared layers reuse the cache of the
/// last concrete layer with the same attention type (full or sliding).
fn build_layer_idx_to_cache_idx(
    num_hidden_layers: usize,
    num_kv_shared_layers: usize,
    layer_types: &[String],
) -> Vec<usize> {
    let first_kv_shared_layer_idx = if num_kv_shared_layers > 0 {
        num_hidden_layers - num_kv_shared_layers
    } else {
        num_hidden_layers
    };
    let num_caches = first_kv_shared_layer_idx;
    let mut mapping: Vec<usize> = (0..num_caches).collect();

    if first_kv_shared_layer_idx < num_hidden_layers {
        let concrete_layers = &layer_types[..first_kv_shared_layer_idx];
        let shared_full_idx = concrete_layers.len()
            - 1
            - concrete_layers
                .iter()
                .rev()
                .position(|t| t == "full_attention")
                .unwrap_or(0);
        let shared_sliding_idx = concrete_layers.len()
            - 1
            - concrete_layers
                .iter()
                .rev()
                .position(|t| t == "sliding_attention")
                .unwrap_or(0);
        for i in first_kv_shared_layer_idx..num_hidden_layers {
            if layer_types[i] == "full_attention" {
                mapping.push(shared_full_idx);
            } else {
                mapping.push(shared_sliding_idx);
            }
        }
    }

    mapping
}

fn sliding_window_mask(
    batch: usize,
    q_len: usize,
    offset: usize,
    sliding_window: usize,
    dtype: DType,
) -> Result<Array> {
    let kv_len = offset + q_len;
    let mut mask = vec![0f32; batch * q_len * kv_len];
    for b in 0..batch {
        let batch_offset = b * q_len * kv_len;
        for i in 0..q_len {
            let q_pos = offset + i;
            let row_offset = batch_offset + i * kv_len;
            for j in 0..kv_len {
                if j > q_pos || q_pos.saturating_sub(j) >= sliding_window {
                    mask[row_offset + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
    Array::from_slice_f32(&mask)?
        .reshape(&[batch as i32, 1, q_len as i32, kv_len as i32])?
        .as_type(dtype)
}

// ------------------------------------------------------------------
// ClippableLinear (2B vision)
// ------------------------------------------------------------------

struct ClippableLinear {
    linear: Linear,
    input_min: Option<Array>,
    input_max: Option<Array>,
    output_min: Option<Array>,
    output_max: Option<Array>,
}

impl ClippableLinear {
    fn new(vb: &VarBuilder, config: &QuantConfig) -> anyhow::Result<Self> {
        let linear = if vb.contains("linear.weight") {
            Linear::new(&vb.pp("linear"), config)?
        } else {
            Linear::new(vb, config)?
        };
        let input_min = if vb.contains("input_min") {
            Some(vb.get("input_min")?)
        } else {
            None
        };
        let input_max = if vb.contains("input_max") {
            Some(vb.get("input_max")?)
        } else {
            None
        };
        let output_min = if vb.contains("output_min") {
            Some(vb.get("output_min")?)
        } else {
            None
        };
        let output_max = if vb.contains("output_max") {
            Some(vb.get("output_max")?)
        } else {
            None
        };
        Ok(Self {
            linear,
            input_min,
            input_max,
            output_min,
            output_max,
        })
    }
}

impl Module for ClippableLinear {
    fn forward(&self, x: &Array) -> Result<Array> {
        let mut h = clamp_array(x, self.input_min.as_ref(), self.input_max.as_ref())?;
        h = self.linear.forward(&h)?;
        clamp_array(&h, self.output_min.as_ref(), self.output_max.as_ref())
    }
}

// ------------------------------------------------------------------
// RoPE helpers
// ------------------------------------------------------------------

fn apply_rope(x: &Array, dims: i32, theta: f32, offset: usize) -> Result<Array> {
    x.fast_rope(dims, false, Some(theta), 1.0, offset as i32, None)
}

fn apply_proportional_rope(
    x: &Array,
    dims: i32,
    rotated_dims: i32,
    theta: f32,
    offset: usize,
) -> Result<Array> {
    let half = dims / 2;
    let rot_half = rotated_dims / 2;
    let last_axis = (x.ndim() - 1) as i32;

    // left = x[..., :half]
    let left_start = vec![0i32; x.ndim()];
    let mut left_stop = x.shape_raw();
    left_stop[x.ndim() - 1] = half;
    let left = x.slice(&left_start, &left_stop)?;

    // right = x[..., half:dims]
    let mut right_start = vec![0i32; x.ndim()];
    right_start[x.ndim() - 1] = half;
    let mut right_stop = x.shape_raw();
    right_stop[x.ndim() - 1] = dims;
    let right = x.slice(&right_start, &right_stop)?;

    // rotated = concatenate([left[..., :rot_half], right[..., :rot_half]], axis=-1)
    let lr_start = vec![0i32; left.ndim()];
    let mut lr_stop = left.shape_raw();
    lr_stop[left.ndim() - 1] = rot_half;
    let left_rot = left.slice(&lr_start, &lr_stop)?;

    let rr_start = vec![0i32; right.ndim()];
    let mut rr_stop = right.shape_raw();
    rr_stop[right.ndim() - 1] = rot_half;
    let right_rot = right.slice(&rr_start, &rr_stop)?;

    let rotated = Array::concatenate(&[&left_rot, &right_rot], last_axis)?;

    // Apply standard rope to the interleaved rotated portion
    let rotated = rotated.fast_rope(rotated_dims, false, Some(theta), 1.0, offset as i32, None)?;

    // Reconstruct left = [rotated[..., :rot_half], left[..., rot_half:]]
    let r1_start = vec![0i32; rotated.ndim()];
    let mut r1_stop = rotated.shape_raw();
    r1_stop[rotated.ndim() - 1] = rot_half;
    let r1 = rotated.slice(&r1_start, &r1_stop)?;

    let mut l2_start = vec![0i32; left.ndim()];
    l2_start[left.ndim() - 1] = rot_half;
    let l2_stop = left.shape_raw();
    let l2 = left.slice(&l2_start, &l2_stop)?;

    let left_new = Array::concatenate(&[&r1, &l2], last_axis)?;

    // Reconstruct right = [rotated[..., rot_half:], right[..., rot_half:]]
    let mut r2_start = vec![0i32; rotated.ndim()];
    r2_start[rotated.ndim() - 1] = rot_half;
    let r2_stop = rotated.shape_raw();
    let r2 = rotated.slice(&r2_start, &r2_stop)?;

    let mut rr2_start = vec![0i32; right.ndim()];
    rr2_start[right.ndim() - 1] = rot_half;
    let rr2_stop = right.shape_raw();
    let rr2 = right.slice(&rr2_start, &rr2_stop)?;

    let right_new = Array::concatenate(&[&r2, &rr2], last_axis)?;

    // head = concatenate([left_new, right_new], axis=-1)
    Array::concatenate(&[&left_new, &right_new], last_axis)
}

// ------------------------------------------------------------------
// 2D RoPE helpers for vision
// ------------------------------------------------------------------

fn repeat_interleave(arr: &Array, repeats: usize, axis: i32) -> Result<Array> {
    let shape = arr.shape_raw();
    let ndim = shape.len() as i32;
    let axis_norm = if axis < 0 { ndim + axis } else { axis };
    let dim = shape[axis_norm as usize];
    let mut indices = Vec::with_capacity(dim as usize * repeats);
    for i in 0..dim {
        for _ in 0..repeats {
            indices.push(i);
        }
    }
    let idx = Array::from_slice_i32(&indices)?;
    arr.take(&idx, axis)
}

fn rotate_half(x: &Array) -> Result<Array> {
    let shape = x.shape_raw();
    let ndim = shape.len();
    let head_dim = shape[ndim - 1];

    let mut new_shape = shape[..ndim - 1].to_vec();
    new_shape.push(head_dim / 2);
    new_shape.push(2);
    let x_reshaped = x.reshape(&new_shape)?;

    let mut start = vec![0i32; ndim + 1];
    let mut stop = new_shape.clone();
    stop[ndim] = 1;
    let x0 = x_reshaped.slice(&start, &stop)?;

    start[ndim] = 1;
    stop[ndim] = 2;
    let x1 = x_reshaped.slice(&start, &stop)?;

    let neg_x1 = x1.negative()?;
    let rotated = Array::concatenate(&[&neg_x1, &x0], ndim as i32)?;

    let mut final_shape = shape[..ndim - 1].to_vec();
    final_shape.push(head_dim);
    rotated.reshape(&final_shape)
}

fn apply_1d_rope_with_positions(
    x: &Array,
    positions: &Array,
    theta: f32,
) -> Result<Array> {
    let shape = x.shape_raw();
    let seq_len = shape[2] as usize;
    let head_dim = shape[3] as usize;
    let half_dim = head_dim / 2;

    let mut freqs = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        let freq = theta.powf(-(i as f32) / (half_dim as f32));
        freqs.push(freq);
    }
    let freqs_arr = Array::from_slice_f32(&freqs)?;

    let pos_rs = positions.reshape(&[seq_len as i32, 1])?;
    let freqs_rs = freqs_arr.reshape(&[1, half_dim as i32])?;
    let angles = pos_rs.multiply(&freqs_rs)?;

    let cos = angles.cos()?;
    let sin = angles.sin()?;

    let cos_exp = cos.expand_dims(0)?.expand_dims(0)?;
    let sin_exp = sin.expand_dims(0)?.expand_dims(0)?;

    let cos_rep = repeat_interleave(&cos_exp, 2, -1)?;
    let sin_rep = repeat_interleave(&sin_exp, 2, -1)?;

    let x_rotated = rotate_half(x)?;

    let term1 = x.multiply(&cos_rep)?;
    let term2 = x_rotated.multiply(&sin_rep)?;

    term1.add(&term2)
}

fn apply_multidimensional_rope(
    x: &Array,
    positions_h: &Array,
    positions_w: &Array,
    theta: f32,
) -> Result<Array> {
    let shape = x.shape_raw();
    let head_dim = shape[3] as usize;
    let half_dim = head_dim / 2;
    let ndim = shape.len();

    let mut start = vec![0i32; ndim];
    let mut stop = shape.clone();
    stop[ndim - 1] = half_dim as i32;
    let x_h = x.slice(&start, &stop)?;

    start[ndim - 1] = half_dim as i32;
    stop[ndim - 1] = head_dim as i32;
    let x_w = x.slice(&start, &stop)?;

    let x_h_rot = apply_1d_rope_with_positions(&x_h, positions_h, theta)?;
    let x_w_rot = apply_1d_rope_with_positions(&x_w, positions_w, theta)?;

    Array::concatenate(&[&x_h_rot, &x_w_rot], -1)
}

// ------------------------------------------------------------------
// Attention (text)
// ------------------------------------------------------------------

struct Gemma4Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Option<Linear>,
    o_proj: Linear,
    q_norm: RmsNormZeroShift,
    k_norm: RmsNormZeroShift,
    v_norm: RmsNormNoScale,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    is_sliding: bool,
    is_kv_shared_layer: bool,
    use_proportional_rope: bool,
    rope_theta: f32,
    partial_rotary_factor: f32,
    sliding_window: usize,
    k_eq_v: bool,
}

impl Gemma4Attention {
    fn load(
        vb: &VarBuilder,
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        head_dim: usize,
        is_sliding: bool,
    ) -> anyhow::Result<Self> {
        let qc = QuantConfig::default();
        let vb_attn = vb.pp("self_attn");

        let q_proj = Linear::new(&vb_attn.pp("q_proj"), &qc)?;
        let k_proj = Linear::new(&vb_attn.pp("k_proj"), &qc)?;

        let k_eq_v = cfg.attention_k_eq_v && !is_sliding;
        let v_proj = if k_eq_v {
            None
        } else {
            Some(Linear::new(&vb_attn.pp("v_proj"), &qc)?)
        };

        let o_proj = Linear::new(&vb_attn.pp("o_proj"), &qc)?;

        let q_norm = RmsNormZeroShift::new(&vb_attn.pp("q_norm"), cfg.rms_norm_eps as f32)?;
        let k_norm = RmsNormZeroShift::new(&vb_attn.pp("k_norm"), cfg.rms_norm_eps as f32)?;
        let v_norm = RmsNormNoScale::new(&vb_attn.pp("v_norm"), cfg.rms_norm_eps as f32)?;

        let n_kv_heads = if is_sliding {
            cfg.num_key_value_heads
        } else {
            cfg.num_global_key_value_heads.unwrap_or(cfg.num_key_value_heads)
        };
        let scale = 1.0;

        let first_kv_shared_layer_idx = cfg.num_hidden_layers.saturating_sub(cfg.num_kv_shared_layers);
        let is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx && first_kv_shared_layer_idx > 0;

        let layer_type = cfg
            .layer_types
            .get(layer_idx)
            .cloned()
            .unwrap_or_else(|| "sliding_attention".to_string());
        let use_proportional_rope = layer_type == "full_attention";

        let rope_params = if use_proportional_rope {
            cfg.rope_parameters
                .get("full_attention")
                .cloned()
                .unwrap_or(RopeParams {
                    rope_theta: 1_000_000.0,
                    partial_rotary_factor: 0.25,
                })
        } else {
            cfg.rope_parameters
                .get("sliding_attention")
                .cloned()
                .unwrap_or(RopeParams {
                    rope_theta: 10_000.0,
                    partial_rotary_factor: 1.0,
                })
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm,
            n_heads: cfg.num_attention_heads,
            n_kv_heads,
            head_dim,
            scale,
            is_sliding,
            is_kv_shared_layer,
            use_proportional_rope,
            rope_theta: rope_params.rope_theta,
            partial_rotary_factor: rope_params.partial_rotary_factor,
            sliding_window: cfg.sliding_window,
            k_eq_v,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        _mask: Option<&Array>,
        cache: &mut KvCache,
    ) -> Result<Array> {
        let shape = x.shape_raw();
        let (b, seq_len, _) = (shape[0], shape[1], shape[2]);
        let offset = cache.offset();

        let q = self.q_proj.forward(x)?;

        let q = q
            .reshape(&[b, seq_len, self.n_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let q = self.q_norm.forward(&q)?;

        let (k, v) = if self.is_kv_shared_layer {
            // For KV-shared layers, reuse keys/values from the shared cache

            cache.fetch()?
        } else {
            let k = self.k_proj.forward(x)?;
            let v = if self.k_eq_v {
                self.k_proj.forward(x)?
            } else {
                self.v_proj.as_ref().unwrap().forward(x)?
            };

            let k = k
                .reshape(&[b, seq_len, self.n_kv_heads as i32, self.head_dim as i32])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let v = v
                .reshape(&[b, seq_len, self.n_kv_heads as i32, self.head_dim as i32])?
                .transpose_axes(&[0, 2, 1, 3])?;

            let k = self.k_norm.forward(&k)?;
            let v = self.v_norm.forward(&v)?;

            let k = if self.use_proportional_rope {
                let rotated_dims = (self.partial_rotary_factor * self.head_dim as f32) as i32;
                apply_proportional_rope(&k, self.head_dim as i32, rotated_dims, self.rope_theta, offset)?
            } else {
                apply_rope(&k, self.head_dim as i32, self.rope_theta, offset)?
            };

            let (k_updated, v_updated) = cache.update(&k, &v)?;
            (k_updated, v_updated)
        };

        let q = if self.use_proportional_rope {
            let rotated_dims = (self.partial_rotary_factor * self.head_dim as f32) as i32;
            apply_proportional_rope(&q, self.head_dim as i32, rotated_dims, self.rope_theta, offset)?
        } else {
            apply_rope(&q, self.head_dim as i32, self.rope_theta, offset)?
        };
        let n_rep = self.n_heads / self.n_kv_heads;
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;


        let attn = if self.is_sliding {
            let window_mask = if self.is_kv_shared_layer {
                // For KV-shared layers the cache already contains all KV tokens,
                // so kv_len == cache size and queries map to the last seq_len positions.
                let kv_len = offset;
                let q_start = offset.saturating_sub(seq_len as usize);
                let batch = b as usize;
                let q_len = seq_len as usize;
                let mut mask = vec![0f32; batch * q_len * kv_len];
                for bi in 0..batch {
                    let batch_off = bi * q_len * kv_len;
                    for i in 0..q_len {
                        let q_pos = q_start + i;
                        let row_off = batch_off + i * kv_len;
                        for j in 0..kv_len {
                            if j > q_pos || q_pos.saturating_sub(j) >= self.sliding_window {
                                mask[row_off + j] = f32::NEG_INFINITY;
                            }
                        }
                    }
                }
                Array::from_slice_f32(&mask)?
                    .reshape(&[b, 1, seq_len, kv_len as i32])?
                    .as_type(q.dtype())?
            } else {
                sliding_window_mask(
                    b as usize,
                    seq_len as usize,
                    offset,
                    self.sliding_window,
                    q.dtype(),
                )?
            };

            let mut scores = q.matmul(&k.transpose_axes(&[0, 1, 3, 2])?)?;
            scores = scores.multiply(&Array::from_float(self.scale)?)?;
            let probs = scores.add(&window_mask)?.softmax(-1)?;
            probs.matmul(&v)?
        } else {
            let mask_mode = if seq_len > 1 { "causal" } else { "" };
            q.fast_scaled_dot_product_attention(&k, &v, self.scale, mask_mode, None)?
        };

        let attn = attn.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            b,
            seq_len,
            (self.n_heads * self.head_dim) as i32,
        ])?;
        self.o_proj.forward(&attn)
    }
}

// ------------------------------------------------------------------
// MLP (GeGLU)
// ------------------------------------------------------------------

struct Gemma4Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Gemma4Mlp {
    fn load(vb: &VarBuilder) -> anyhow::Result<Self> {
        let qc = QuantConfig::default();
        let vb_mlp = vb.pp("mlp");
        Ok(Self {
            gate_proj: Linear::new(&vb_mlp.pp("gate_proj"), &qc)?,
            up_proj: Linear::new(&vb_mlp.pp("up_proj"), &qc)?,
            down_proj: Linear::new(&vb_mlp.pp("down_proj"), &qc)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let gate = gelu_approx(&gate)?;
        self.down_proj.forward(&gate.multiply(&up)?)
    }
}

// ------------------------------------------------------------------
// MoE components
// ------------------------------------------------------------------

struct Router {
    proj: Linear,
    scale: Array,
    per_expert_scale: Array,
    top_k: usize,
    root_size: f32,
}

impl Router {
    fn load(vb: &VarBuilder, cfg: &Gemma4TextConfig) -> anyhow::Result<Self> {
        let qc = QuantConfig::default();
        let proj = Linear::new(&vb.pp("proj"), &qc)?;
        let scale = vb.get("scale")?;
        let per_expert_scale = vb.get("per_expert_scale")?;
        Ok(Self {
            proj,
            scale,
            per_expert_scale,
            top_k: cfg.top_k_experts.unwrap_or(1),
            root_size: (cfg.hidden_size as f32).powf(-0.5),
        })
    }

    fn forward(&self, x: &Array) -> Result<(Array, Array)> {
        let shape = x.shape_raw();
        let hidden = shape[shape.len() - 1];
        let flat = x.reshape(&[-1, hidden])?;

        // norm(x) * root_size * scale
        let mut h = flat.fast_rms_norm(
            &Array::ones(&[hidden], flat.dtype())?,
            1e-6,
        )?;
        h = h.multiply(&Array::from_float(self.root_size)?)?;
        h = h.multiply(&self.scale)?;

        let logits = self.proj.forward(&h)?;
        let probs = logits.softmax(-1)?;

        let kth = (self.top_k - 1) as i32;
        let neg_logits = logits.negative()?;
        let partition = neg_logits.argpartition(kth, -1)?;
        let top_k_indices = partition.slice(&[0, 0], &[partition.shape_raw()[0], self.top_k as i32])?;

        let mut top_k_weights = probs.take_along_axis(&top_k_indices, -1)?;
        let denom = top_k_weights.sum_axis(-1, true)?;
        top_k_weights = top_k_weights.divide(&denom)?;

        let expert_scale = self.per_expert_scale.reshape(&[1, self.per_expert_scale.shape_raw()[0]])?
            .take_along_axis(&top_k_indices, -1)?;
        top_k_weights = top_k_weights.multiply(&expert_scale)?;

        Ok((top_k_indices, top_k_weights))
    }
}

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
    match (num / unpacked) as i32 {
        2 | 4 | 8 => (num / unpacked) as i32,
        _ => fallback,
    }
}

struct SwitchLinear {
    weight: Array,
    scales: Option<Array>,
    biases: Option<Array>,
    bias: Option<Array>,
    group_size: i32,
    bits: i32,
}

impl SwitchLinear {
    fn load(vb: &VarBuilder, _cfg: &Gemma4TextConfig) -> anyhow::Result<Self> {
        let weight = vb.get("weight")?;
        let scales = if vb.contains("scales") {
            Some(vb.get("scales")?)
        } else {
            None
        };
        let biases = if vb.contains("biases") {
            Some(vb.get("biases")?)
        } else {
            None
        };
        let bias = if vb.contains("bias") {
            Some(vb.get("bias")?)
        } else {
            None
        };

        let group_size = 64;
        let bits = if let Some(ref s) = scales {
            infer_bits(&weight.shape_raw(), &s.shape_raw(), group_size, 4)
        } else {
            0
        };

        Ok(Self {
            weight,
            scales,
            biases,
            bias,
            group_size,
            bits,
        })
    }

    fn forward(&self, x: &Array, indices: &Array) -> Result<Array> {
        let sorted_indices = false;
        let mut out = if let Some(scales) = &self.scales {
            x.gather_qmm(
                &self.weight,
                scales,
                self.biases.as_ref(),
                None,
                Some(indices),
                true,
                self.group_size,
                self.bits,
                sorted_indices,
            )?
        } else {
            let wt = self.weight.transpose_axes(&[0, 2, 1])?;
            x.gather_mm(&wt, None, Some(indices), sorted_indices)?
        };
        if let Some(bias) = &self.bias {
            let gathered = bias.take(indices, 0)?.expand_dims(-2)?;
            out = out.add(&gathered)?;
        }
        Ok(out)
    }
}

struct SwitchGlu {
    gate_proj: SwitchLinear,
    up_proj: SwitchLinear,
    down_proj: SwitchLinear,
}

impl SwitchGlu {
    fn load(vb: &VarBuilder, cfg: &Gemma4TextConfig) -> anyhow::Result<Self> {
        Ok(Self {
            gate_proj: SwitchLinear::load(&vb.pp("gate_proj"), cfg)?,
            up_proj: SwitchLinear::load(&vb.pp("up_proj"), cfg)?,
            down_proj: SwitchLinear::load(&vb.pp("down_proj"), cfg)?,
        })
    }

    fn forward(&self, x: &Array, indices: &Array) -> Result<Array> {
        let x = x.expand_dims(-2)?.expand_dims(-2)?;
        let x_up = self.up_proj.forward(&x, indices)?;
        let x_gate = self.gate_proj.forward(&x, indices)?;
        let hidden = gelu_approx(&x_gate)?.multiply(&x_up)?;
        let out = self.down_proj.forward(&hidden, indices)?;
        out.squeeze(2)
    }
}

struct SparseMoeBlock {
    router: Router,
    switch_glu: SwitchGlu,
}

impl SparseMoeBlock {
    fn load(vb: &VarBuilder, cfg: &Gemma4TextConfig) -> anyhow::Result<Self> {
        Ok(Self {
            router: Router::load(&vb.pp("router"), cfg)?,
            switch_glu: SwitchGlu::load(&vb.pp("experts.switch_glu"), cfg)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let shape = x.shape_raw();
        let hidden = shape[shape.len() - 1];
        let orig_shape = shape.clone();
        let flat = x.reshape(&[-1, hidden])?;
        let num_tokens = flat.shape_raw()[0] as usize;

        let (top_k_indices, top_k_weights) = self.router.forward(x)?;

        let indices_flat = top_k_indices.reshape(&[-1, self.router.top_k as i32])?;
        let expert_out = self.switch_glu.forward(&flat, &indices_flat)?;

        let weights = top_k_weights.reshape(&[num_tokens as i32, self.router.top_k as i32, 1])?
            .as_type(expert_out.dtype())?;
        let out = expert_out.multiply(&weights)?.sum_axis(1, false)?;
        out.reshape(&orig_shape)
    }
}

// ------------------------------------------------------------------
// DecoderLayer
// ------------------------------------------------------------------

pub struct Gemma4DecoderLayer {
    input_layernorm: RmsNormZeroShift,
    post_attn_layernorm: RmsNormZeroShift,
    pre_ffw_layernorm: RmsNormZeroShift,
    post_ffw_layernorm: RmsNormZeroShift,
    self_attn: Gemma4Attention,
    mlp: Gemma4Mlp,
    moe: Option<SparseMoeBlock>,
    post_ffw_layernorm_1: Option<RmsNormZeroShift>,
    post_ffw_layernorm_2: Option<RmsNormZeroShift>,
    pre_ffw_layernorm_2: Option<RmsNormZeroShift>,
    layer_scalar: Option<Array>,
    is_sliding: bool,
    per_layer_input_gate: Option<Linear>,
    per_layer_projection: Option<Linear>,
    post_per_layer_input_norm: Option<RmsNormZeroShift>,
}

impl Gemma4DecoderLayer {
    fn load(
        vb: &VarBuilder,
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
    ) -> anyhow::Result<Self> {
        let layer_type = cfg
            .layer_types
            .get(layer_idx)
            .cloned()
            .unwrap_or_else(|| "sliding_attention".to_string());
        let is_sliding = layer_type == "sliding_attention";

        let head_dim = if is_sliding {
            cfg.head_dim
        } else {
            cfg.global_head_dim
        };

        let layer_scalar = if vb.contains("layer_scalar") {
            Some(vb.get("layer_scalar")?)
        } else {
            None
        };

        let per_layer_input_gate = if cfg.hidden_size_per_layer_input > 0
            && vb.pp("per_layer_input_gate").contains("weight")
        {
            Some(Linear::new(
                &vb.pp("per_layer_input_gate"),
                &QuantConfig::default(),
            )?)
        } else {
            None
        };

        let per_layer_projection = if cfg.hidden_size_per_layer_input > 0
            && vb.pp("per_layer_projection").contains("weight")
        {
            Some(Linear::new(
                &vb.pp("per_layer_projection"),
                &QuantConfig::default(),
            )?)
        } else {
            None
        };

        let post_per_layer_input_norm = if cfg.hidden_size_per_layer_input > 0
            && vb.contains("post_per_layer_input_norm.weight")
        {
            Some(RmsNormZeroShift::new(
                &vb.pp("post_per_layer_input_norm"),
                cfg.rms_norm_eps as f32,
            )?)
        } else {
            None
        };

        let (moe, post_ffw_layernorm_1, post_ffw_layernorm_2, pre_ffw_layernorm_2) =
            if cfg.enable_moe_block {
                (
                    Some(SparseMoeBlock::load(vb, cfg)?),
                    Some(RmsNormZeroShift::new(
                        &vb.pp("post_feedforward_layernorm_1"),
                        cfg.rms_norm_eps as f32,
                    )?),
                    Some(RmsNormZeroShift::new(
                        &vb.pp("post_feedforward_layernorm_2"),
                        cfg.rms_norm_eps as f32,
                    )?),
                    Some(RmsNormZeroShift::new(
                        &vb.pp("pre_feedforward_layernorm_2"),
                        cfg.rms_norm_eps as f32,
                    )?),
                )
            } else {
                (None, None, None, None)
            };

        Ok(Self {
            input_layernorm: RmsNormZeroShift::new(
                &vb.pp("input_layernorm"),
                cfg.rms_norm_eps as f32,
            )?,
            post_attn_layernorm: RmsNormZeroShift::new(
                &vb.pp("post_attention_layernorm"),
                cfg.rms_norm_eps as f32,
            )?,
            pre_ffw_layernorm: RmsNormZeroShift::new(
                &vb.pp("pre_feedforward_layernorm"),
                cfg.rms_norm_eps as f32,
            )?,
            post_ffw_layernorm: RmsNormZeroShift::new(
                &vb.pp("post_feedforward_layernorm"),
                cfg.rms_norm_eps as f32,
            )?,
            self_attn: Gemma4Attention::load(vb, cfg, layer_idx, head_dim, is_sliding)?,
            mlp: Gemma4Mlp::load(vb)?,
            moe,
            post_ffw_layernorm_1,
            post_ffw_layernorm_2,
            pre_ffw_layernorm_2,
            layer_scalar,
            is_sliding,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
        })
    }

    /// Apply per-layer input gating when all gating components are present.
    ///
    /// Computes: `residual + post_norm(proj(gelu_approx(gate_proj(h)) * layer_emb))`
    fn apply_per_layer_gate(&self, h: &Array, layer_emb: &Array) -> Result<Array> {
        if let (Some(ref gate_proj), Some(ref proj_proj), Some(ref post_norm)) =
            (&self.per_layer_input_gate, &self.per_layer_projection, &self.post_per_layer_input_norm)
        {
            let residual = h.clone();
            let gate = gate_proj.forward(h)?;
            let gate = gelu_approx(&gate)?;
            let gate = gate.multiply(layer_emb)?;
            let gate = proj_proj.forward(&gate)?;
            let gate = post_norm.forward(&gate)?;
            residual.add(&gate)
        } else {
            Ok(h.clone())
        }
    }

    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut KvCache,
        layer_emb: Option<&Array>,
    ) -> Result<Array> {
        // Attention sublayer
        let residual = x.clone();
        let h_norm = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&h_norm, mask, cache)?;
        let h = self.post_attn_layernorm.forward(&attn_out)?;
        let h = residual.add(&h)?;

        // MLP / MoE sublayer
        let residual = h.clone();
        let mut h = if let Some(ref moe) = self.moe {
            let h1 = self.pre_ffw_layernorm.forward(&h)?;
            let h1 = self.mlp.forward(&h1)?;
            let h1 = self.post_ffw_layernorm_1.as_ref().unwrap().forward(&h1)?;

            let h2 = self.pre_ffw_layernorm_2.as_ref().unwrap().forward(&h)?;
            let h2 = moe.forward(&h2)?;
            let h2 = self.post_ffw_layernorm_2.as_ref().unwrap().forward(&h2)?;

            let mut h = h1.add(&h2)?;
            h = self.post_ffw_layernorm.forward(&h)?;
            residual.add(&h)?
        } else {
            let h_norm = self.pre_ffw_layernorm.forward(&h)?;
            let mlp_out = self.mlp.forward(&h_norm)?;
            let h = self.post_ffw_layernorm.forward(&mlp_out)?;
            residual.add(&h)?
        };

        // Per-layer input gating (after MLP, matching Python)
        if let Some(layer_emb) = layer_emb {
            h = self.apply_per_layer_gate(&h, layer_emb)?;
        }

        // Layer scalar
        if let Some(ref scalar) = self.layer_scalar {
            h = h.multiply(scalar)?;
        }

        Ok(h)
    }
}

// ------------------------------------------------------------------
// TextModel
// ------------------------------------------------------------------

pub struct Gemma4TextModel {
    pub embed_tokens: Embedding,
    pub layers: Vec<Gemma4DecoderLayer>,
    pub norm: RmsNormZeroShift,
    pub embed_scale: f32,
    pub embed_tokens_per_layer: Option<Embedding>,
    pub embed_tokens_per_layer_scale: f32,
    pub per_layer_model_projection: Option<Linear>,
    pub per_layer_projection_norm: Option<RmsNormZeroShift>,
    pub per_layer_input_scale: f32,
    pub num_hidden_layers: usize,
    pub hidden_size_per_layer_input: usize,
    pub sliding_window: usize,
    pub caches: Vec<KvCache>,
    pub layer_idx_to_cache_idx: Vec<usize>,
}

impl Gemma4TextModel {
    fn new(vb: &VarBuilder, cfg: &Gemma4TextConfig) -> anyhow::Result<Self> {
        let qc = QuantConfig::default();
        let embed_tokens = Embedding::new(&vb.pp("embed_tokens"), &qc)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Gemma4DecoderLayer::load(
                &vb.pp(format!("layers.{i}")),
                cfg,
                i,
            )?);
        }

        let norm = RmsNormZeroShift::new(&vb.pp("norm"), cfg.rms_norm_eps as f32)?;

        let (embed_tokens_per_layer, embed_tokens_per_layer_scale) =
            if cfg.hidden_size_per_layer_input > 0 {
                (
                    Some(Embedding::new(
                        &vb.pp("embed_tokens_per_layer"),
                        &qc,
                    )?),
                    (cfg.hidden_size_per_layer_input as f32).sqrt(),
                )
            } else {
                (None, 0.0)
            };

        let per_layer_model_projection = if cfg.hidden_size_per_layer_input > 0
            && vb.pp("per_layer_model_projection").contains("weight")
        {
            Some(Linear::new(
                &vb.pp("per_layer_model_projection"),
                &qc,
            )?)
        } else {
            None
        };

        let per_layer_projection_norm = if cfg.hidden_size_per_layer_input > 0
            && vb.contains("per_layer_projection_norm.weight")
        {
            Some(RmsNormZeroShift::new(
                &vb.pp("per_layer_projection_norm"),
                cfg.rms_norm_eps as f32,
            )?)
        } else {
            None
        };

        let layer_idx_to_cache_idx = build_layer_idx_to_cache_idx(
            cfg.num_hidden_layers,
            cfg.num_kv_shared_layers,
            &cfg.layer_types,
        );
        let num_caches = layer_idx_to_cache_idx.iter().copied().max().map_or(0, |m| m + 1);
        let caches = (0..num_caches).map(|_| KvCache::new()).collect();

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            embed_scale: (cfg.hidden_size as f32).sqrt(),
            embed_tokens_per_layer,
            embed_tokens_per_layer_scale,
            per_layer_model_projection,
            per_layer_projection_norm,
            per_layer_input_scale: 2.0f32.powf(-0.5),
            num_hidden_layers: cfg.num_hidden_layers,
            hidden_size_per_layer_input: cfg.hidden_size_per_layer_input,
            sliding_window: cfg.sliding_window,
            caches,
            layer_idx_to_cache_idx,
        })
    }

    /// Compute per-layer inputs from token IDs and embeddings.
    /// Returns an array of shape [B, S, num_hidden_layers, hidden_size_per_layer_input].
    fn compute_per_layer_inputs(
        &mut self,
        input_ids: &Array,
        embeddings: &Array,
    ) -> Result<Array> {
        let embed_table = self.embed_tokens_per_layer.as_ref().unwrap();
        let proj = self.per_layer_model_projection.as_ref().unwrap();
        let norm = self.per_layer_projection_norm.as_ref().unwrap();

        // Token-based per-layer inputs
        let mut per_layer_inputs = embed_table.forward(input_ids)?;
        per_layer_inputs = per_layer_inputs.multiply(&Array::from_float(self.embed_tokens_per_layer_scale)?)?;
        let mut shape = per_layer_inputs.shape_raw();
        shape.pop(); // Remove last dim (num_layers * hidden_per_layer)
        shape.push(self.num_hidden_layers as i32);
        shape.push(self.hidden_size_per_layer_input as i32);
        per_layer_inputs = per_layer_inputs.reshape(&shape)?;

        // Embedding-based projection
        let mut per_layer_projection = proj.forward(embeddings)?;
        // Apply scale: ScaledLinear in Python multiplies by hidden_size**-0.5
        let hidden = embeddings.shape_raw().last().copied().unwrap_or(0) as f32;
        per_layer_projection = per_layer_projection.multiply(&Array::from_float(hidden.powf(-0.5))?)?;

        let mut proj_shape = per_layer_projection.shape_raw();
        proj_shape.pop(); // Remove last dim (num_layers * hidden_per_layer)
        proj_shape.push(self.num_hidden_layers as i32);
        proj_shape.push(self.hidden_size_per_layer_input as i32);
        per_layer_projection = per_layer_projection.reshape(&proj_shape)?;
        per_layer_projection = norm.forward(&per_layer_projection)?;

        // Combine
        let combined = per_layer_projection.add(&per_layer_inputs)?;
        let result = combined.multiply(&Array::from_float(self.per_layer_input_scale)?)?;
        Ok(result)
    }

    fn forward_embeddings(
        layers: &mut [Gemma4DecoderLayer],
        layer_idx_to_cache_idx: &[usize],
        norm: &RmsNormZeroShift,
        embeddings: &Array,
        per_layer_inputs: Option<&Array>,
        caches: &mut [KvCache],
        sliding_window: usize,
    ) -> Result<Array> {
        let mut h = embeddings.clone();
        let shape = h.shape_raw();
        let seq_len = shape[1] as usize;
        let batch = shape[0] as usize;

        for (i, layer) in layers.iter_mut().enumerate() {
            let cache_idx = layer_idx_to_cache_idx[i];

            let mask = if layer.is_sliding {
                let offset = caches[cache_idx].offset();
                Some(sliding_window_mask(
                    batch,
                    seq_len,
                    offset,
                    sliding_window,
                    h.dtype(),
                )?)
            } else {
                None
            };

            let layer_emb = per_layer_inputs.map(|pli| {
                // Extract per_layer_inputs[:, :, i, :]
                let start = vec![0i32, 0, i as i32, 0];
                let mut stop = pli.shape_raw();
                stop[2] = (i + 1) as i32;
                pli.slice(&start, &stop)?.squeeze(2)
            }).transpose()?;

            h = layer.forward(&h, mask.as_ref(), &mut caches[cache_idx], layer_emb.as_ref())?;
        }

        let out = norm.forward(&h)?;
        Ok(out)
    }

    fn forward(&mut self, input_ids: &Array) -> Result<Array> {
        let mut h = self.embed_tokens.forward(input_ids)?;
        h = h.multiply(&Array::from_float(self.embed_scale)?)?;

        let per_layer_inputs = if self.embed_tokens_per_layer.is_some() {
            Some(self.compute_per_layer_inputs(input_ids, &h)?)
        } else {
            None
        };

        Self::forward_embeddings(
            &mut self.layers,
            &self.layer_idx_to_cache_idx,
            &self.norm,
            &h,
            per_layer_inputs.as_ref(),
            &mut self.caches,
            self.sliding_window,
        )
    }

    fn clear_cache(&mut self) {
        for c in &mut self.caches {
            c.reset();
        }
    }
}

// ------------------------------------------------------------------
// LanguageModel wrapper
// ------------------------------------------------------------------

pub struct LanguageModel {
    pub model: Gemma4TextModel,
    pub lm_head: Linear,
    pub final_logit_softcapping: Option<f32>,
}

impl LanguageModel {
    fn new(vb: &VarBuilder, cfg: &Gemma4TextConfig) -> anyhow::Result<Self> {
        let model_vb = vb.pp("model");
        let model = Gemma4TextModel::new(&model_vb, cfg)?;

        let lm_head = if cfg.tie_word_embeddings && !vb.pp("lm_head").contains("weight") {
            model.embed_tokens.as_linear()
        } else {
            Linear::new(&vb.pp("lm_head"), &QuantConfig::default())?
        };

        Ok(Self {
            model,
            lm_head,
            final_logit_softcapping: cfg.final_logit_softcapping,
        })
    }

    fn forward_hidden_states(&mut self, input_ids: &Array) -> Result<Array> {
        self.model.forward(input_ids)
    }
}

// ------------------------------------------------------------------
// Vision components
// ------------------------------------------------------------------

struct VisionPatchEmbedder {
    input_proj: ClippableLinear,
    position_embedding_table: Array,
    patch_size: usize,
    position_embedding_size: usize,
}

impl VisionPatchEmbedder {
    fn new(vb: &VarBuilder, cfg: &Gemma4VisionConfig) -> anyhow::Result<Self> {
        let qc = QuantConfig::default();
        let pos_emb_weight = vb.get("position_embedding_table")?;
        Ok(Self {
            input_proj: ClippableLinear::new(&vb.pp("input_proj"), &qc)?,
            position_embedding_table: pos_emb_weight,
            patch_size: cfg.patch_size,
            position_embedding_size: cfg.position_embedding_size,
        })
    }

    fn forward(&self, images: &Array) -> Result<Array> {
        let shape = images.shape_raw();
        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let patch_size = self.patch_size as i32;
        let num_patches_h = h / patch_size;
        let num_patches_w = w / patch_size;
        let num_patches = num_patches_h * num_patches_w;

        // Reshape to patches: [B, C, Hh, Ph, Ww, Pw] -> [B, Hh, Ww, C, Ph, Pw] -> [B, num_patches, C*Ph*Pw]
        let patches = images
            .reshape(&[b, c, num_patches_h, patch_size, num_patches_w, patch_size])?
            .transpose_axes(&[0, 2, 4, 1, 3, 5])?
            .reshape(&[b, num_patches, c * patch_size * patch_size])?;

        // Normalize patches to match Python: 2 * (patches - 0.5)
        let half = Array::from_float(0.5f32)?;
        let two = Array::from_float(2.0f32)?;
        let patches = patches.subtract(&half)?.multiply(&two)?;

        let features = self.input_proj.forward(&patches)?;

        // Position embeddings: table has shape [2, position_embedding_size, hidden_size]
        // We take embeddings for x (table[0]) and y (table[1]) separately and sum them.
        let mut x_positions = Vec::with_capacity(num_patches as usize);
        let mut y_positions = Vec::with_capacity(num_patches as usize);
        for row in 0..num_patches_h {
            for col in 0..num_patches_w {
                x_positions.push(col as i32);
                y_positions.push(row as i32);
            }
        }
        let x_pos_arr = Array::from_slice_i32(&x_positions)?;
        let y_pos_arr = Array::from_slice_i32(&y_positions)?;

        let pos_size = self.position_embedding_size as i32;
        let hidden_size = self.position_embedding_table.dim(2) as i32;
        let table_0 = self
            .position_embedding_table
            .slice(&[0, 0, 0], &[1, pos_size, hidden_size])?
            .squeeze(0)?;
        let table_1 = self
            .position_embedding_table
            .slice(&[1, 0, 0], &[2, pos_size, hidden_size])?
            .squeeze(0)?;

        let x_emb = table_0.take(&x_pos_arr, 0)?;
        let y_emb = table_1.take(&y_pos_arr, 0)?;
        let pos_emb = x_emb.add(&y_emb)?;

        // Broadcast to batch dimension
        let pos_emb = pos_emb.expand_dims(0)?.broadcast_to(&features.shape_raw())?;

        features.add(&pos_emb)
    }
}

struct VisionPooler {
    pooling_kernel_size: usize,
    default_output_length: usize,
}

impl VisionPooler {
    fn new(_vb: &VarBuilder, cfg: &Gemma4VisionConfig) -> anyhow::Result<Self> {
        Ok(Self {
            pooling_kernel_size: cfg.pooling_kernel_size,
            default_output_length: cfg.default_output_length,
        })
    }

    fn forward(
        &self,
        features: &Array,
        patch_positions: &Array,
        padding_positions: &Array,
    ) -> Result<(Array, Array)> {
        let shape = features.shape_raw();
        let b = shape[0];
        let seq_len = shape[1] as usize;
        let hidden_size = shape[2] as usize;
        let length = self.default_output_length;

        // Zero out padding tokens before pooling (matches HF masked_fill)
        let expanded_padding = padding_positions.expand_dims(-1)?.broadcast_to(&shape)?;
        let zero = Array::zeros(&shape, features.dtype())?;
        let features = expanded_padding.where_cond(&zero, features)?;

        if seq_len == length {
            let scaled = features.multiply(&Array::from_float((hidden_size as f32).sqrt())?)?;
            let mask = padding_positions.logical_not()?;
            return Ok((scaled, mask));
        }

        let k = ((seq_len / length) as f32).sqrt() as i32;
        let k_sq = (k * k) as f32;

        // Clamp patch positions to >= 0 (padding positions are -1)
        let zero_arr = Array::from_int(0)?;
        let clamped = patch_positions.maximum(&zero_arr)?;

        // max_x = max(clamped[..., 0]) + 1
        let x_coords = clamped.slice(&[0, 0, 0], &[b, seq_len as i32, 1])?.squeeze(2)?;
        let max_x = x_coords.max(Some(1), true)?.add(&Array::from_int(1)?)?;

        // kernel_idxs = floor(clamped / k)
        let k_f32 = Array::from_float(k as f32)?;
        let kernel_idxs = clamped
            .as_type(DType::Float32)?
            .divide(&k_f32)?
            .floor()?
            .as_type(DType::Int32)?;

        // kernel_idx = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        let idx_0 = kernel_idxs
            .slice(&[0, 0, 0], &[b, seq_len as i32, 1])?
            .squeeze(2)?;
        let idx_1 = kernel_idxs
            .slice(&[0, 0, 1], &[b, seq_len as i32, 2])?
            .squeeze(2)?;
        let max_x_div_k = max_x
            .as_type(DType::Float32)?
            .divide(&k_f32)?
            .floor()?
            .as_type(DType::Int32)?;
        let kernel_idx = idx_0.add(&idx_1.multiply(&max_x_div_k)?)?;

        // one_hot(kernel_idx, length) using equal + broadcast
        let kernel_idx_3d = kernel_idx.expand_dims(-1)?; // [B, seq_len, 1]
        let arange = Array::arange_int(length as i32, DType::Int32)?
            .expand_dims(0)?
            .expand_dims(0)?; // [1, 1, length]
        let one_hot = kernel_idx_3d.equal(&arange)?; // [B, seq_len, length]

        // weights = one_hot.astype(float32) / k_squared
        let weights = one_hot
            .as_type(DType::Float32)?
            .divide(&Array::from_float(k_sq)?)?;

        // output = einsum("bLl,bLd->bld", weights, features)
        let w_exp = weights.expand_dims(-1)?; // [B, seq_len, length, 1]
        let f_exp = features.expand_dims(2)?; // [B, seq_len, 1, hidden_size]
        let prod = w_exp.multiply(&f_exp)?; // [B, seq_len, length, hidden_size]
        let output = prod.sum_axis(1, false)?; // [B, length, hidden_size]
        let output = output.as_type(features.dtype())?;

        // mask = any(one_hot, axis=1) -> True if any patch maps to this output cell
        let one_hot_f32 = one_hot.as_type(DType::Float32)?;
        let has_any = one_hot_f32.max(Some(1), true)?; // [B, 1, length]
        let mask = has_any
            .greater(&Array::from_float(0.0)?)?
            .squeeze(1)?; // [B, length]

        // Scale by sqrt(hidden_size)
        let scaled = output.multiply(&Array::from_float((hidden_size as f32).sqrt())?)?;

        Ok((scaled, mask))
    }
}

struct VisionAttention {
    q_proj: ClippableLinear,
    k_proj: ClippableLinear,
    v_proj: ClippableLinear,
    q_norm: VisionRmsNorm,
    k_norm: VisionRmsNorm,
    v_norm: VisionRmsNormNoScale,
    out_proj: ClippableLinear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    rope_theta: f32,
}

impl VisionAttention {
    fn new(vb: &VarBuilder, cfg: &Gemma4VisionConfig) -> anyhow::Result<Self> {
        let qc = QuantConfig::default();
        let vb_attn = vb.pp("self_attn");

        let q_proj = ClippableLinear::new(&vb_attn.pp("q_proj"), &qc)?;
        let k_proj = ClippableLinear::new(&vb_attn.pp("k_proj"), &qc)?;
        let v_proj = ClippableLinear::new(&vb_attn.pp("v_proj"), &qc)?;
        let out_proj = ClippableLinear::new(&vb_attn.pp("o_proj"), &qc)?;

        let q_norm = VisionRmsNorm::new(&vb_attn.pp("q_norm"), 1e-6)?;
        let k_norm = VisionRmsNorm::new(&vb_attn.pp("k_norm"), 1e-6)?;
        let v_norm = VisionRmsNormNoScale::new(&vb_attn.pp("v_norm"), 1e-6)?;

        let rope_params = cfg
            .rope_parameters
            .get("default")
            .cloned()
            .unwrap_or(RopeParams {
                rope_theta: 10_000.0,
                partial_rotary_factor: 1.0,
            });

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            q_norm,
            k_norm,
            v_norm,
            out_proj,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            scale: 1.0 / (cfg.head_dim as f32).sqrt(),
            rope_theta: rope_params.rope_theta,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        positions_h: &Array,
        positions_w: &Array,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let shape = x.shape_raw();
        let (b, seq_len, _) = (shape[0], shape[1], shape[2]);

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape(&[b, seq_len, self.n_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[b, seq_len, self.n_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[b, seq_len, self.n_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let v = self.v_norm.forward(&v)?;

        let q = apply_multidimensional_rope(&q, positions_h, positions_w, self.rope_theta)?;
        let k = apply_multidimensional_rope(&k, positions_h, positions_w, self.rope_theta)?;

        let n_rep = self.n_heads / self.n_kv_heads;
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;

        let attn = if let Some(m) = mask {
            q.fast_scaled_dot_product_attention(&k, &v, self.scale, "array", Some(m))?
        } else {
            q.fast_scaled_dot_product_attention(&k, &v, self.scale, "", None)?
        };

        let attn = attn.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            b,
            seq_len,
            (self.n_heads * self.head_dim) as i32,
        ])?;
        self.out_proj.forward(&attn)
    }
}

struct VisionMlp {
    gate_proj: ClippableLinear,
    up_proj: ClippableLinear,
    down_proj: ClippableLinear,
}

impl VisionMlp {
    fn new(vb: &VarBuilder, _cfg: &Gemma4VisionConfig) -> anyhow::Result<Self> {
        let qc = QuantConfig::default();
        let vb_mlp = vb.pp("mlp");
        Ok(Self {
            gate_proj: ClippableLinear::new(&vb_mlp.pp("gate_proj"), &qc)?,
            up_proj: ClippableLinear::new(&vb_mlp.pp("up_proj"), &qc)?,
            down_proj: ClippableLinear::new(&vb_mlp.pp("down_proj"), &qc)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let gate = gelu_approx(&gate)?;
        self.down_proj.forward(&gate.multiply(&up)?)
    }
}

struct VisionEncoderLayer {
    input_layernorm: VisionRmsNorm,
    post_attention_layernorm: VisionRmsNorm,
    pre_feedforward_layernorm: VisionRmsNorm,
    post_feedforward_layernorm: VisionRmsNorm,
    self_attn: VisionAttention,
    mlp: VisionMlp,
}

impl VisionEncoderLayer {
    fn new(vb: &VarBuilder, cfg: &Gemma4VisionConfig) -> anyhow::Result<Self> {
        Ok(Self {
            input_layernorm: VisionRmsNorm::new(&vb.pp("input_layernorm"), 1e-6)?,
            post_attention_layernorm: VisionRmsNorm::new(
                &vb.pp("post_attention_layernorm"),
                1e-6,
            )?,
            pre_feedforward_layernorm: VisionRmsNorm::new(
                &vb.pp("pre_feedforward_layernorm"),
                1e-6,
            )?,
            post_feedforward_layernorm: VisionRmsNorm::new(
                &vb.pp("post_feedforward_layernorm"),
                1e-6,
            )?,
            self_attn: VisionAttention::new(vb, cfg)?,
            mlp: VisionMlp::new(vb, cfg)?,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        positions_h: &Array,
        positions_w: &Array,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h, positions_h, positions_w, mask)?;
        let h = self.post_attention_layernorm.forward(&h)?;
        let h = residual.add(&h)?;

        let residual = h.clone();
        let h = self.pre_feedforward_layernorm.forward(&h)?;
        let h = self.mlp.forward(&h)?;
        let h = self.post_feedforward_layernorm.forward(&h)?;
        residual.add(&h)
    }
}

pub fn build_vision_attention_mask(
    num_real: usize,
    max_patches: usize,
    dtype: DType,
) -> Result<Array> {
    let mut mask_data = vec![0.0f32; max_patches * max_patches];
    for i in 0..max_patches {
        for j in 0..max_patches {
            if i >= num_real || j >= num_real {
                mask_data[i * max_patches + j] = f32::NEG_INFINITY;
            }
        }
    }
    Array::from_slice_f32(&mask_data)?
        .reshape(&[1, 1, max_patches as i32, max_patches as i32])?
        .as_type(dtype)
}

pub struct VisionModel {
    patch_embedder: VisionPatchEmbedder,
    pooler: VisionPooler,
    layers: Vec<VisionEncoderLayer>,
    norm: Option<VisionRmsNorm>,
    std_bias: Option<Array>,
    std_scale: Option<Array>,
}

impl VisionModel {
    fn new(vb: &VarBuilder, cfg: &Gemma4VisionConfig) -> anyhow::Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(VisionEncoderLayer::new(
                &vb.pp(format!("encoder.layers.{i}")),
                cfg,
            )?);
        }

        let norm = if vb.contains("norm") {
            Some(VisionRmsNorm::new(&vb.pp("norm"), 1e-6)?)
        } else {
            None
        };

        let std_bias = if vb.contains("std_bias") {
            Some(vb.get("std_bias")?)
        } else {
            None
        };

        let std_scale = if vb.contains("std_scale") {
            Some(vb.get("std_scale")?)
        } else {
            None
        };

        Ok(Self {
            patch_embedder: VisionPatchEmbedder::new(&vb.pp("patch_embedder"), cfg)?,
            pooler: VisionPooler::new(&vb.pp("pooler"), cfg)?,
            layers,
            norm,
            std_bias,
            std_scale,
        })
    }

    pub fn forward(&mut self, images: &Array) -> Result<Array> {
        let shape = images.shape_raw();
        let (b, _c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let patch_size = self.patch_embedder.patch_size as i32;
        let num_patches_h = h / patch_size;
        let num_patches_w = w / patch_size;
        let num_patches = (num_patches_h * num_patches_w) as usize;

        let mut features = self.patch_embedder.forward(images)?;
        let feature_shape = features.shape_raw();
        let hidden_size = feature_shape[2] as usize;

        let max_patches = self.pooler.default_output_length
            * self.pooler.pooling_kernel_size
            * self.pooler.pooling_kernel_size;
        let num_real = num_patches.min(max_patches);

        // Truncate if too many patches
        if num_patches > max_patches {
            let start = vec![0i32, 0i32, 0i32];
            let mut stop = feature_shape.clone();
            stop[1] = max_patches as i32;
            features = features.slice(&start, &stop)?;
        }

        // Build patch_positions [B, max_patches, 2] (x, y); padding = (-1, -1)
        let mut patch_positions_data = Vec::with_capacity(b as usize * max_patches * 2);
        for _bi in 0..b as usize {
            for row in 0..num_patches_h {
                for col in 0..num_patches_w {
                    if patch_positions_data.len() >= (_bi + 1) * max_patches * 2 {
                        break;
                    }
                    patch_positions_data.push(col as i32); // x
                    patch_positions_data.push(row as i32); // y
                }
            }
            while patch_positions_data.len() < (_bi + 1) * max_patches * 2 {
                patch_positions_data.push(-1i32);
                patch_positions_data.push(-1i32);
            }
        }
        let patch_positions = Array::from_slice_i32(&patch_positions_data)?
            .reshape(&[b, max_patches as i32, 2])?;

        // Build padding_positions [B, max_patches] (bool)
        let mut padding_data = Vec::with_capacity(b as usize * max_patches);
        for _bi in 0..b as usize {
            for _ in 0..num_real {
                padding_data.push(0i32);
            }
            for _ in num_real..max_patches {
                padding_data.push(1i32);
            }
        }
        let padding_positions = Array::from_slice_i32(&padding_data)?
            .reshape(&[b, max_patches as i32])?
            .as_type(DType::Bool)?;

        // Pad embeddings to max_patches with zeros
        let num_padding = max_patches.saturating_sub(num_real);
        if num_padding > 0 {
            let pad_data = vec![0.0f32; (b as usize) * num_padding * hidden_size];
            let pad = Array::from_slice_f32(&pad_data)?
                .reshape(&[b, num_padding as i32, hidden_size as i32])?;
            features = Array::concatenate(&[&features, &pad], 1)?;
        }

        // Build position arrays for max_patches (for RoPE)
        let mut pos_h = Vec::with_capacity(max_patches);
        let mut pos_w = Vec::with_capacity(max_patches);
        for row in 0..num_patches_h {
            for col in 0..num_patches_w {
                if pos_h.len() >= num_real {
                    break;
                }
                pos_h.push(row as i32);
                pos_w.push(col as i32);
            }
            if pos_h.len() >= num_real {
                break;
            }
        }
        while pos_h.len() < max_patches {
            pos_h.push(0i32);
            pos_w.push(0i32);
        }
        let positions_h = Array::from_slice_i32(&pos_h)?;
        let positions_w = Array::from_slice_i32(&pos_w)?;

        let mask = build_vision_attention_mask(num_real, max_patches, features.dtype())?;

        for layer in &mut self.layers {
            features = layer.forward(&features, &positions_h, &positions_w, Some(&mask))?;
        }

        if let Some(ref norm) = self.norm {
            features = norm.forward(&features)?;
        }

        // Pool with position-based averaging
        let (pooled, pool_mask) =
            self.pooler
                .forward(&features, &patch_positions, &padding_positions)?;

        // Strip padding: count valid tokens per batch and slice
        let valid_counts = pool_mask.as_type(DType::Int32)?.sum_axis(1, false)?; // [B]
        let valid_counts_vec: Vec<i32> = valid_counts.to_vec_i32()?;

        let stripped = if b == 1 {
            let n_valid = valid_counts_vec[0] as usize;
            let start = vec![0i32, 0i32, 0i32];
            let mut stop = pooled.shape_raw();
            stop[1] = n_valid as i32;
            pooled.slice(&start, &stop)?
        } else {
            let mut all_real = Vec::with_capacity(b as usize);
            for bi in 0..b as usize {
                let n_valid = valid_counts_vec[bi] as usize;
                let start = vec![bi as i32, 0i32, 0i32];
                let mut stop = pooled.shape_raw();
                stop[0] = bi as i32 + 1;
                stop[1] = n_valid as i32;
                let sliced = pooled.slice(&start, &stop)?;
                all_real.push(sliced);
            }
            let concatenated = if all_real.len() == 1 {
                all_real[0].clone()
            } else {
                Array::concatenate(&all_real.iter().collect::<Vec<_>>(), 0)?
            };
            concatenated.reshape(&[1, concatenated.dim(0), concatenated.dim(1)])?
        };

        // Apply standardize if configured (matches Python: after pooling)
        if let (Some(ref bias), Some(ref scale)) = (&self.std_bias, &self.std_scale) {
            let h = stripped.subtract(bias)?.multiply(scale)?;
            Ok(h)
        } else {
            Ok(stripped)
        }
    }
}

// ------------------------------------------------------------------
// Multimodal embedder
// ------------------------------------------------------------------

pub struct MultimodalEmbedder {
    embedding_pre_projection_norm: RmsNormNoScale,
    embedding_projection: Linear,
}

impl MultimodalEmbedder {
    fn new(vb: &VarBuilder, _text_hidden: usize, _vision_hidden: usize) -> anyhow::Result<Self> {
        let qc = QuantConfig::default();
        Ok(Self {
            embedding_pre_projection_norm: RmsNormNoScale::new(
                &vb.pp("embedding_pre_projection_norm"),
                1e-6,
            )?,
            embedding_projection: Linear::new(&vb.pp("embedding_projection"), &qc)?,
        })
    }

    pub fn forward(&self, features: &Array) -> Result<Array> {
        let h = self.embedding_pre_projection_norm.forward(features)?;
        self.embedding_projection.forward(&h)
    }
}

// ------------------------------------------------------------------
// Top-level Gemma4 model
// ------------------------------------------------------------------

pub struct Gemma4 {
    pub language_model: LanguageModel,
    pub vision_tower: VisionModel,
    pub embed_vision: MultimodalEmbedder,
    config: Gemma4Config,
}

/// Sanitize raw safetensors weights before model construction.
///
/// Applies the same transformations as Python's `Gemma4.sanitize()`:
/// 1. Strip clipping params when `use_clipped_linears=false` (vision)
/// 2. Strip `rotary_emb` params
/// 3. Strip audio weights when no audio tower
/// 4. Remap keys (`model.` prefix, `language_model.` → `language_model.model.`)
/// 5. Transpose conv weights (PyTorch → MLX layout)
/// 6. Split MoE `experts.gate_up_proj` into separate gate/up weights
pub fn sanitize_weights(
    weights: HashMap<String, Array>,
    config: &Gemma4Config,
) -> HashMap<String, Array> {
    let mut sanitized = HashMap::with_capacity(weights.len());
    let use_clipped = config.vision_config.use_clipped_linears;
    let has_audio = config.audio_config.is_some();

    for (key, mut value) in weights {
        // Skip clipping parameters when not used
        if key.contains("input_max")
            || key.contains("input_min")
            || key.contains("output_max")
            || key.contains("output_min")
        {
            if key.contains("vision_tower") && !use_clipped {
                continue;
            }
            if !key.contains("vision_tower") && !key.contains("audio_tower") {
                continue;
            }
        }

        // Skip rotary embedding params
        if key.contains("rotary_emb.inv_freq") || key.contains("rotary_emb") {
            continue;
        }

        // Skip audio weights when no audio tower
        if !has_audio && (key.contains("audio_tower") || key.contains("embed_audio")) {
            continue;
        }

        // Key remapping
        let mut new_key = if key.starts_with("model.") {
            key["model.".len()..].to_string()
        } else {
            key.clone()
        };

        if new_key.starts_with("language_model.") && !new_key.starts_with("language_model.model.") {
            new_key = format!("language_model.model.{}", &new_key["language_model.".len()..]);
        }

        // Conv2d: PyTorch [out, in, kH, kW] → MLX [out, kH, kW, in]
        if new_key.contains("subsample_conv_projection")
            && new_key.contains("conv.weight")
            && value.ndim() == 4
        {
            value = value.transpose_axes(&[0, 2, 3, 1]).unwrap_or(value);
        }

        // Conv1d: PyTorch [out, in, kW] → MLX [out, kW, in]
        if new_key.contains("depthwise_conv1d.weight") && value.ndim() == 3 {
            value = value.transpose_axes(&[0, 2, 1]).unwrap_or(value);
        }

        // MoE: experts.down_proj → experts.switch_glu.down_proj.weight
        if new_key.ends_with(".experts.down_proj") {
            new_key = new_key.replace(".experts.down_proj", ".experts.switch_glu.down_proj.weight");
        }

        // MoE: experts.gate_up_proj → split into gate_proj + up_proj
        if new_key.ends_with(".experts.gate_up_proj") {
            let gate_key = new_key.replace(".experts.gate_up_proj", ".experts.switch_glu.gate_proj.weight");
            let up_key = new_key.replace(".experts.gate_up_proj", ".experts.switch_glu.up_proj.weight");

            // swapaxes(-1, -2): transpose last two dims
            let ndim = value.ndim() as i32;
            let mut perm: Vec<i32> = (0..ndim).collect();
            perm.swap((ndim - 1) as usize, (ndim - 2) as usize);
            let value = value.transpose_axes(&perm).unwrap_or(value);

            // Split last dim in half
            let shape = value.shape_raw();
            let last_dim = shape.last().copied().unwrap_or(0) as usize;
            let mid = last_dim / 2;

            // Slice first half for gate
            let start_gate = vec![0i32; shape.len()];
            let mut stop_gate = shape.clone();
            stop_gate[shape.len() - 1] = mid as i32;
            let gate_val = value.slice(&start_gate, &stop_gate).unwrap_or(value.clone());

            // Swap back
            let gate_val = gate_val.transpose_axes(&perm).unwrap_or(gate_val);
            sanitized.insert(gate_key, gate_val);

            // Slice second half for up
            let mut start_up = vec![0i32; shape.len()];
            start_up[shape.len() - 1] = mid as i32;
            let mut stop_up = shape.clone();
            stop_up[shape.len() - 1] = last_dim as i32;
            let up_val = value.slice(&start_up, &stop_up).unwrap_or(value.clone());

            // Swap back
            let up_val = up_val.transpose_axes(&perm).unwrap_or(up_val);
            sanitized.insert(up_key, up_val);

            continue;
        }

        sanitized.insert(new_key, value);
    }

    sanitized
}

impl Gemma4 {
    pub fn new(vb: &VarBuilder, config: &Gemma4Config) -> anyhow::Result<Self> {
        let language_model = LanguageModel::new(
            &vb.pp("language_model"),
            &config.text_config,
        )?;
        let vision_tower = VisionModel::new(&vb.pp("vision_tower"), &config.vision_config)?;
        let embed_vision = MultimodalEmbedder::new(
            &vb.pp("embed_vision"),
            config.text_config.hidden_size,
            config.vision_config.hidden_size,
        )?;

        Ok(Self {
            language_model,
            vision_tower,
            embed_vision,
            config: config.clone(),
        })
    }

    pub fn config(&self) -> &Gemma4Config {
        &self.config
    }

    pub fn encode_image(&mut self, pixel_values: &Array) -> Result<Array> {
        let vision_features = self.vision_tower.forward(pixel_values)?;
        self.embed_vision.forward(&vision_features)
    }

    pub fn masked_scatter(input_tensor: &Array, mask: &Array, source: &Array) -> Result<Array> {
        let mask_flat = mask.flatten(0, -1)?.as_type(DType::Int32)?;
        let indices = mask_flat.cumsum(0, false, true)?.subtract(&Array::from_int(1)?)?;

        let source_flat = source.flatten(0, -1)?;
        let source_size = source_flat.elem_count() as i32;
        let mod_indices = if source_size > 0 {
            indices.remainder(&Array::from_int(source_size)?)?.as_type(DType::Int32)?
        } else {
            indices
        };
        let aligned = source_flat.take(&mod_indices, 0)?;

        let input_flat = input_tensor.flatten(0, -1)?;
        let result = mask_flat.where_cond(&aligned, &input_flat)?;
        result.reshape(&input_tensor.shape_raw())
    }

    fn compute_embeddings(
        &mut self,
        input_ids: &Array,
        pixel_values: Option<&Array>,
    ) -> Result<(Array, Option<Array>)> {
        let mut h = self.language_model.model.embed_tokens.forward(input_ids)?;
        h = h.multiply(&Array::from_float(self.language_model.model.embed_scale)?)?;

        if let Some(pixel_values) = pixel_values {
            let vision_features = self.vision_tower.forward(pixel_values)?;
            let vision_emb = self.embed_vision.forward(&vision_features)?;

            let image_token_id_arr = Array::from_int(self.config.image_token_id as i32)?;
            let image_mask = input_ids.equal(&image_token_id_arr)?;
            let image_mask_expanded = image_mask.expand_dims(-1)?.broadcast_to(&h.shape_raw())?;
            h = Self::masked_scatter(&h, &image_mask_expanded, &vision_emb)?;
        }

        let per_layer_inputs = if self.language_model.model.embed_tokens_per_layer.is_some() {
            let pli = self.language_model.model.compute_per_layer_inputs(input_ids, &h)?;
            Some(pli)
        } else {
            None
        };

        Ok((h, per_layer_inputs))
    }

    pub fn forward_logits(
        &mut self,
        input_ids: &Array,
        pixel_values: Option<&Array>,
    ) -> Result<Array> {
        let (h, per_layer_inputs) = self.compute_embeddings(input_ids, pixel_values)?;
        let h = Gemma4TextModel::forward_embeddings(
            &mut self.language_model.model.layers,
            &self.language_model.model.layer_idx_to_cache_idx,
            &self.language_model.model.norm,
            &h,
            per_layer_inputs.as_ref(),
            &mut self.language_model.model.caches,
            self.language_model.model.sliding_window,
        )?;
        let mut logits = self.language_model.lm_head.forward(&h)?;
        if let Some(softcap) = self.language_model.final_logit_softcapping {
            logits = logits
                .divide(&Array::from_float(softcap)?)?
                .tanh()?
                .multiply(&Array::from_float(softcap)?)?;
        }
        Ok(logits)
    }

    pub fn forward_logits_with_cache(
        &mut self,
        input_ids: &Array,
        pixel_values: Option<&Array>,
        cache: &mut [KvCache],
    ) -> Result<Array> {
        let (h, per_layer_inputs) = self.compute_embeddings(input_ids, pixel_values)?;
        let h = Gemma4TextModel::forward_embeddings(
            &mut self.language_model.model.layers,
            &self.language_model.model.layer_idx_to_cache_idx,
            &self.language_model.model.norm,
            &h,
            per_layer_inputs.as_ref(),
            cache,
            self.language_model.model.sliding_window,
        )?;
        let mut logits = self.language_model.lm_head.forward(&h)?;
        if let Some(softcap) = self.language_model.final_logit_softcapping {
            logits = logits
                .divide(&Array::from_float(softcap)?)?
                .tanh()?
                .multiply(&Array::from_float(softcap)?)?;
        }
        Ok(logits)
    }

    pub fn forward_last_token_logits(
        &mut self,
        input_ids: &Array,
        pixel_values: Option<&Array>,
    ) -> Result<Array> {
        let seq_len = input_ids.shape_raw()[input_ids.ndim() - 1];
        let mut logits = self.forward_logits(input_ids, pixel_values)?;
        if seq_len > 1 {
            let logits_shape = logits.shape_raw();
            let mut start = vec![0i32; logits_shape.len()];
            let mut stop = logits_shape.clone();
            start[logits_shape.len() - 2] = seq_len - 1;
            stop[logits_shape.len() - 2] = seq_len;
            logits = logits.slice(&start, &stop)?;
        }
        Ok(logits)
    }

    pub fn forward_last_token_logits_with_cache(
        &mut self,
        input_ids: &Array,
        pixel_values: Option<&Array>,
        cache: &mut [KvCache],
    ) -> Result<Array> {
        let seq_len = input_ids.shape_raw()[input_ids.ndim() - 1];
        let mut logits = self.forward_logits_with_cache(input_ids, pixel_values, cache)?;
        if seq_len > 1 {
            let logits_shape = logits.shape_raw();
            let mut start = vec![0i32; logits_shape.len()];
            let mut stop = logits_shape.clone();
            start[logits_shape.len() - 2] = seq_len - 1;
            stop[logits_shape.len() - 2] = seq_len;
            logits = logits.slice(&start, &stop)?;
        }
        Ok(logits)
    }

    pub fn forward_hidden_states(&mut self, input_ids: &Array) -> Result<Array> {
        self.language_model.forward_hidden_states(input_ids)
    }

    pub fn clear_cache(&mut self) {
        self.language_model.model.clear_cache();
    }
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_are_stable() {
        let text_cfg: Gemma4TextConfig = serde_json::from_str(
            r#"{
                "hidden_size": 1536,
                "intermediate_size": 6144,
                "vocab_size": 262208,
                "num_hidden_layers": 35,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "head_dim": 256,
                "global_head_dim": 512,
                "sliding_window": 512,
                "sliding_window_pattern": 6,
                "layer_types": ["sliding_attention"],
                "rope_parameters": {},
                "hidden_activation": "gelu",
                "tie_word_embeddings": false,
                "attention_k_eq_v": false,
                "num_kv_shared_layers": 20,
                "hidden_size_per_layer_input": 256,
                "use_double_wide_mlp": true,
                "enable_moe_block": false
            }"#,
        )
        .unwrap();

        assert_eq!(text_cfg.head_dim, 256);
        assert_eq!(text_cfg.sliding_window, 512);
        assert_eq!(text_cfg.num_key_value_heads, 1);
        assert!(text_cfg.use_double_wide_mlp);
    }

    #[test]
    fn kv_sharing_maps_to_last_concrete_layer() {
        // Simulate a 35-layer 2B model with sliding_window_pattern=5:
        // 4 sliding, 1 full, repeating.
        let mut layer_types: Vec<String> = Vec::new();
        for i in 0..35 {
            if (i + 1) % 5 == 0 {
                layer_types.push("full_attention".to_string());
            } else {
                layer_types.push("sliding_attention".to_string());
            }
        }

        let mapping = build_layer_idx_to_cache_idx(35, 20, &layer_types);

        // 15 non-shared layers each own a cache
        assert_eq!(mapping[0], 0);
        assert_eq!(mapping[14], 14);

        // Concrete layers 0-14:
        //   full at indices: 4, 9, 14   => last = 14
        //   sliding at indices: 0,1,2,3,5,6,7,8,10,11,12,13 => last = 13
        // Shared layers 15-34 map to last concrete layer of same type
        for i in 15..35 {
            let expected = if layer_types[i] == "full_attention" { 14 } else { 13 };
            assert_eq!(mapping[i], expected, "layer {i} ({}) should map to cache {expected}", layer_types[i]);
        }
    }

    #[test]
    fn kv_sharing_no_sharing() {
        let layer_types: Vec<String> = (0..10).map(|i| {
            if i % 2 == 0 { "full_attention".to_string() } else { "sliding_attention".to_string() }
        }).collect();

        let mapping = build_layer_idx_to_cache_idx(10, 0, &layer_types);

        // No sharing: each layer has its own cache
        assert_eq!(mapping, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn kv_sharing_all_shared() {
        let layer_types: Vec<String> = (0..6).map(|_| "full_attention".to_string()).collect();

        // 2 non-shared, 4 shared
        let mapping = build_layer_idx_to_cache_idx(6, 4, &layer_types);

        assert_eq!(mapping[0], 0);
        assert_eq!(mapping[1], 1);
        // All shared layers map to last concrete full_attention layer = 1
        assert_eq!(mapping[2], 1);
        assert_eq!(mapping[3], 1);
        assert_eq!(mapping[4], 1);
        assert_eq!(mapping[5], 1);
    }

    #[test]
    fn rope_params_deserializes_bare_float() {
        let params: RopeParams = serde_json::from_str("100.0").unwrap();
        assert!((params.rope_theta - 100.0).abs() < 1e-6);
        assert!((params.partial_rotary_factor - 0.25).abs() < 1e-6);
    }

    #[test]
    fn rope_params_deserializes_object() {
        let params: RopeParams = serde_json::from_str(
            r#"{"rope_theta": 10000.0, "rope_type": "default", "partial_rotary_factor": 1.0}"#,
        )
        .unwrap();
        assert!((params.rope_theta - 10000.0).abs() < 1e-6);
        assert!((params.partial_rotary_factor - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rope_params_deserializes_object_with_defaults() {
        let params: RopeParams = serde_json::from_str(
            r#"{"rope_theta": 10000.0, "rope_type": "default"}"#,
        )
        .unwrap();
        assert!((params.rope_theta - 10000.0).abs() < 1e-6);
        assert!((params.partial_rotary_factor - 0.25).abs() < 1e-6);
    }

    #[test]
    fn vision_pooler_shortcut_zeros_padding() {
        // seq_len == default_output_length => shortcut path
        let cfg = Gemma4VisionConfig {
            hidden_size: 2,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 2,
            global_head_dim: 2,
            intermediate_size: 4,
            patch_size: 2,
            position_embedding_size: 2,
            pooling_kernel_size: 2,
            default_output_length: 2,
            max_patches: 4,
            standardize: false,
            rope_parameters: HashMap::new(),
            use_clipped_linears: false,
        };
        let vb = VarBuilder::from_weights(HashMap::new(), mlx_core::DType::Float32);
        let pooler = VisionPooler::new(&vb, &cfg).unwrap();

        // features: [1, 2, 3, 4] -> [1, 2, 2]
        let features = Array::from_slice_f32(&[1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .reshape(&[1, 2, 2])
            .unwrap();
        // patch positions don't matter on shortcut path
        let patch_positions = Array::from_slice_i32(&[0, 0, 0, 0])
            .unwrap()
            .reshape(&[1, 2, 2])
            .unwrap();
        // padding_positions = [false, true] => second token is padding
        let padding_positions = Array::from_slice_f32(&[0.0, 1.0])
            .unwrap()
            .reshape(&[1, 2])
            .unwrap()
            .greater(&Array::from_float(0.5).unwrap())
            .unwrap();

        let (pooled, mask) = pooler.forward(&features, &patch_positions, &padding_positions).unwrap();

        // After zeroing padding and scaling by sqrt(2):
        // [[1*sqrt(2), 2*sqrt(2)], [0, 0]]
        let sqrt2 = 2.0f32.sqrt();
        let pooled_data: Vec<f32> = pooled.to_vec_f32().unwrap();
        assert_eq!(pooled_data.len(), 4);
        assert!((pooled_data[0] - 1.0 * sqrt2).abs() < 1e-4);
        assert!((pooled_data[1] - 2.0 * sqrt2).abs() < 1e-4);
        assert!((pooled_data[2] - 0.0).abs() < 1e-4);
        assert!((pooled_data[3] - 0.0).abs() < 1e-4);

        // mask should be [true, false]
        let mask_data: Vec<i32> = mask.as_type(mlx_core::DType::Int32).unwrap().to_vec_i32().unwrap();
        assert_eq!(mask_data, vec![1, 0]);
    }

    #[test]
    fn vision_pooler_position_based_pooling() {
        // 2x2 grid of patches -> single pooled token (k=2)
        let cfg = Gemma4VisionConfig {
            hidden_size: 1,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 1,
            global_head_dim: 1,
            intermediate_size: 4,
            patch_size: 2,
            position_embedding_size: 2,
            pooling_kernel_size: 2,
            default_output_length: 1,
            max_patches: 4,
            standardize: false,
            rope_parameters: HashMap::new(),
            use_clipped_linears: false,
        };
        let vb = VarBuilder::from_weights(HashMap::new(), mlx_core::DType::Float32);
        let pooler = VisionPooler::new(&vb, &cfg).unwrap();

        // features: [1, 2, 3, 4] -> [1, 4, 1]
        let features = Array::from_slice_f32(&[1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .reshape(&[1, 4, 1])
            .unwrap();
        // patch positions: (0,0), (0,1), (1,0), (1,1)
        let patch_positions = Array::from_slice_i32(&[0, 0, 0, 1, 1, 0, 1, 1])
            .unwrap()
            .reshape(&[1, 4, 2])
            .unwrap();
        // no padding
        let padding_positions = Array::from_slice_f32(&[0.0, 0.0, 0.0, 0.0])
            .unwrap()
            .reshape(&[1, 4])
            .unwrap()
            .greater(&Array::from_float(0.5).unwrap())
            .unwrap();

        let (pooled, mask) = pooler.forward(&features, &patch_positions, &padding_positions).unwrap();

        // All 4 patches map to kernel cell 0; average = (1+2+3+4)/4 = 2.5
        // scaled by sqrt(1) = 1.0
        let pooled_data: Vec<f32> = pooled.to_vec_f32().unwrap();
        assert_eq!(pooled_data.len(), 1);
        assert!((pooled_data[0] - 2.5).abs() < 1e-4);

        let mask_data: Vec<i32> = mask.as_type(mlx_core::DType::Int32).unwrap().to_vec_i32().unwrap();
        assert_eq!(mask_data, vec![1]);
    }

    #[test]
    fn vision_pooler_zeros_padding_before_pooling() {
        // Same as above but one patch is padding -> should be zeroed before pooling
        let cfg = Gemma4VisionConfig {
            hidden_size: 1,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 1,
            global_head_dim: 1,
            intermediate_size: 4,
            patch_size: 2,
            position_embedding_size: 2,
            pooling_kernel_size: 2,
            default_output_length: 1,
            max_patches: 4,
            standardize: false,
            rope_parameters: HashMap::new(),
            use_clipped_linears: false,
        };
        let vb = VarBuilder::from_weights(HashMap::new(), mlx_core::DType::Float32);
        let pooler = VisionPooler::new(&vb, &cfg).unwrap();

        let features = Array::from_slice_f32(&[1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .reshape(&[1, 4, 1])
            .unwrap();
        let patch_positions = Array::from_slice_i32(&[0, 0, 0, 1, 1, 0, 1, 1])
            .unwrap()
            .reshape(&[1, 4, 2])
            .unwrap();
        // third patch is padding
        let padding_positions = Array::from_slice_f32(&[0.0, 0.0, 1.0, 0.0])
            .unwrap()
            .reshape(&[1, 4])
            .unwrap()
            .greater(&Array::from_float(0.5).unwrap())
            .unwrap();

        let (pooled, _mask) = pooler.forward(&features, &patch_positions, &padding_positions).unwrap();

        // After zeroing: [1, 2, 0, 4]; average = (1+2+0+4)/4 = 1.75
        let pooled_data: Vec<f32> = pooled.to_vec_f32().unwrap();
        assert_eq!(pooled_data.len(), 1);
        assert!((pooled_data[0] - 1.75).abs() < 1e-4);
    }

    #[test]
    fn per_layer_gate_computation() {
        // Verify the per-layer gating math: residual + post_norm(proj(gelu_approx(gate_proj(h)) * layer_emb))
        let hidden_size = 4;
        let hidden_per_layer = 2;

        // gate_proj: [hidden_per_layer, hidden_size] — extracts first 2 elements
        let gate_proj_w = Array::from_slice_f32(&[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ])
        .unwrap()
        .reshape(&[hidden_per_layer as i32, hidden_size as i32])
        .unwrap();

        // proj_proj: [hidden_size, hidden_per_layer] — pads back to 4
        let proj_proj_w = Array::from_slice_f32(&[
            1.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
            0.0, 0.0,
        ])
        .unwrap()
        .reshape(&[hidden_size as i32, hidden_per_layer as i32])
        .unwrap();

        let layer = Gemma4DecoderLayer {
            input_layernorm: RmsNormZeroShift {
                weight: Array::from_slice_f32(&[1.0, 1.0, 1.0, 1.0]).unwrap(),
                eps: 1e-6,
            },
            post_attn_layernorm: RmsNormZeroShift {
                weight: Array::from_slice_f32(&[1.0, 1.0, 1.0, 1.0]).unwrap(),
                eps: 1e-6,
            },
            pre_ffw_layernorm: RmsNormZeroShift {
                weight: Array::from_slice_f32(&[1.0, 1.0, 1.0, 1.0]).unwrap(),
                eps: 1e-6,
            },
            post_ffw_layernorm: RmsNormZeroShift {
                weight: Array::from_slice_f32(&[1.0, 1.0, 1.0, 1.0]).unwrap(),
                eps: 1e-6,
            },
            self_attn: Gemma4Attention {
                q_proj: Linear::from_weights(
                    Array::zeros(&[4, 4], DType::Float32).unwrap(),
                    None,
                ),
                k_proj: Linear::from_weights(
                    Array::zeros(&[4, 4], DType::Float32).unwrap(),
                    None,
                ),
                v_proj: Some(Linear::from_weights(
                    Array::zeros(&[4, 4], DType::Float32).unwrap(),
                    None,
                )),
                o_proj: Linear::from_weights(
                    Array::zeros(&[4, 4], DType::Float32).unwrap(),
                    None,
                ),
                q_norm: RmsNormZeroShift {
                    weight: Array::from_slice_f32(&[1.0, 1.0, 1.0, 1.0]).unwrap(),
                    eps: 1e-6,
                },
                k_norm: RmsNormZeroShift {
                    weight: Array::from_slice_f32(&[1.0, 1.0, 1.0, 1.0]).unwrap(),
                    eps: 1e-6,
                },
                v_norm: RmsNormNoScale { eps: 1e-6 },
                n_heads: 1,
                n_kv_heads: 1,
                head_dim: 4,
                scale: 1.0,
                is_sliding: false,
                is_kv_shared_layer: false,
                use_proportional_rope: false,
                rope_theta: 10000.0,
                partial_rotary_factor: 0.25,
                sliding_window: 512,
                k_eq_v: false,
            },
            mlp: Gemma4Mlp {
                gate_proj: Linear::from_weights(
                    Array::zeros(&[4, 4], DType::Float32).unwrap(),
                    None,
                ),
                up_proj: Linear::from_weights(
                    Array::zeros(&[4, 4], DType::Float32).unwrap(),
                    None,
                ),
                down_proj: Linear::from_weights(
                    Array::zeros(&[4, 4], DType::Float32).unwrap(),
                    None,
                ),
            },
            moe: None,
            post_ffw_layernorm_1: None,
            post_ffw_layernorm_2: None,
            pre_ffw_layernorm_2: None,
            layer_scalar: None,
            is_sliding: false,
            per_layer_input_gate: Some(Linear::from_weights(gate_proj_w, None)),
            per_layer_projection: Some(Linear::from_weights(proj_proj_w, None)),
            post_per_layer_input_norm: Some(RmsNormZeroShift {
                weight: Array::from_slice_f32(&[1.0, 1.0, 1.0, 1.0]).unwrap(),
                eps: 1e-6,
            }),
        };

        // h = [1, 2, 3, 4]
        let h = Array::from_slice_f32(&[1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .reshape(&[1, 1, hidden_size as i32])
            .unwrap();

        // layer_emb = [0.5, 0.5]
        let layer_emb = Array::from_slice_f32(&[0.5, 0.5])
            .unwrap()
            .reshape(&[1, 1, hidden_per_layer as i32])
            .unwrap();

        let result = layer.apply_per_layer_gate(&h, &layer_emb).unwrap();

        // Verify output shape
        assert_eq!(result.shape_raw(), vec![1, 1, hidden_size as i32]);

        // gate_proj(h) extracts [1, 2]
        // gelu_approx([1, 2]) ≈ [0.841, 1.935]
        // multiply by layer_emb [0.5, 0.5] ≈ [0.4205, 0.9675]
        // proj_proj pads back to [0.4205, 0.9675, 0, 0]
        // post_norm with weight=1 is approximately identity (with slight RMS scaling)
        // residual + gate ≈ [1.4205, 2.9675, 3, 4]
        let data: Vec<f32> = result.to_vec_f32().unwrap();
        assert!(data[0] > 1.0); // 1 + positive gate
        assert!(data[1] > 2.0); // 2 + positive gate
        assert!(data[2] == 3.0); // unchanged (proj pads with zeros)
        assert!(data[3] == 4.0); // unchanged
    }

    fn make_test_config(use_clipped: bool) -> Gemma4Config {
        Gemma4Config {
            model_type: "gemma4".to_string(),
            image_token_id: 258880,
            audio_token_id: None,
            boi_token_id: 255999,
            eoi_token_id: 258882,
            boa_token_id: None,
            eoa_token_id: None,
            vision_soft_tokens_per_image: 280,
            text_config: Gemma4TextConfig {
                hidden_size: 4,
                num_hidden_layers: 2,
                num_attention_heads: 1,
                num_key_value_heads: 1,
                head_dim: 4,
                global_head_dim: 4,
                num_global_key_value_heads: None,
                intermediate_size: 8,
                vocab_size: 100,
                sliding_window: 512,
                sliding_window_pattern: 5,
                layer_types: vec!["sliding_attention".to_string(), "full_attention".to_string()],
                rope_parameters: HashMap::new(),
                final_logit_softcapping: None,
                tie_word_embeddings: false,
                rms_norm_eps: 1e-6,
                hidden_activation: "gelu".to_string(),
                enable_moe_block: false,
                num_experts: None,
                top_k_experts: None,
                moe_intermediate_size: None,
                attention_k_eq_v: false,
                num_kv_shared_layers: 0,
                hidden_size_per_layer_input: 0,
                use_double_wide_mlp: false,
                vocab_size_per_layer_input: 262144,
            },
            vision_config: Gemma4VisionConfig {
                hidden_size: 4,
                num_hidden_layers: 1,
                num_attention_heads: 1,
                num_key_value_heads: 1,
                head_dim: 4,
                global_head_dim: 4,
                intermediate_size: 8,
                patch_size: 2,
                position_embedding_size: 2,
                pooling_kernel_size: 2,
                default_output_length: 1,
                max_patches: 4,
                standardize: false,
                rope_parameters: HashMap::new(),
                use_clipped_linears: use_clipped,
            },
            audio_config: None,
        }
    }

    #[test]
    fn sanitize_strips_clipping_params_when_not_used() {
        let mut weights = HashMap::new();
        weights.insert(
            "vision_tower.some_layer.input_max".to_string(),
            Array::from_slice_f32(&[1.0]).unwrap(),
        );
        weights.insert(
            "vision_tower.some_layer.weight".to_string(),
            Array::from_slice_f32(&[2.0]).unwrap(),
        );
        weights.insert(
            "language_model.some_layer.output_min".to_string(),
            Array::from_slice_f32(&[3.0]).unwrap(),
        );

        let config = make_test_config(false);
        let sanitized = sanitize_weights(weights, &config);

        // vision clipping params stripped when use_clipped_linears=false
        assert!(!sanitized.contains_key("vision_tower.some_layer.input_max"));
        // non-vision clipping params always stripped
        assert!(!sanitized.contains_key("language_model.some_layer.output_min"));
        // normal params kept
        assert!(sanitized.contains_key("vision_tower.some_layer.weight"));
    }

    #[test]
    fn sanitize_keeps_clipping_params_when_used() {
        let mut weights = HashMap::new();
        weights.insert(
            "vision_tower.some_layer.input_max".to_string(),
            Array::from_slice_f32(&[1.0]).unwrap(),
        );
        weights.insert(
            "vision_tower.some_layer.weight".to_string(),
            Array::from_slice_f32(&[2.0]).unwrap(),
        );

        let config = make_test_config(true);
        let sanitized = sanitize_weights(weights, &config);

        // vision clipping params kept when use_clipped_linears=true
        assert!(sanitized.contains_key("vision_tower.some_layer.input_max"));
        assert!(sanitized.contains_key("vision_tower.some_layer.weight"));
    }

    #[test]
    fn sanitize_remaps_keys() {
        let mut weights = HashMap::new();
        weights.insert(
            "model.language_model.layers.0.weight".to_string(),
            Array::from_slice_f32(&[1.0]).unwrap(),
        );
        weights.insert(
            "model.vision_tower.patch_embed.weight".to_string(),
            Array::from_slice_f32(&[2.0]).unwrap(),
        );

        let config = make_test_config(false);
        let sanitized = sanitize_weights(weights, &config);

        // model. prefix stripped
        assert!(!sanitized.contains_key("model.language_model.layers.0.weight"));
        assert!(!sanitized.contains_key("model.vision_tower.patch_embed.weight"));

        // language_model. -> language_model.model.
        assert!(sanitized.contains_key("language_model.model.layers.0.weight"));
        // vision_tower unchanged
        assert!(sanitized.contains_key("vision_tower.patch_embed.weight"));
    }

    #[test]
    fn sanitize_strips_rotary_emb() {
        let mut weights = HashMap::new();
        weights.insert(
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq".to_string(),
            Array::from_slice_f32(&[1.0]).unwrap(),
        );
        weights.insert(
            "language_model.model.layers.0.self_attn.q_proj.weight".to_string(),
            Array::from_slice_f32(&[2.0]).unwrap(),
        );

        let config = make_test_config(false);
        let sanitized = sanitize_weights(weights, &config);

        assert!(!sanitized.contains_key("language_model.model.layers.0.self_attn.rotary_emb.inv_freq"));
        assert!(sanitized.contains_key("language_model.model.layers.0.self_attn.q_proj.weight"));
    }

    #[test]
    fn sanitize_strips_audio_when_no_audio_tower() {
        let mut weights = HashMap::new();
        weights.insert(
            "audio_tower.encoder.weight".to_string(),
            Array::from_slice_f32(&[1.0]).unwrap(),
        );
        weights.insert(
            "embed_audio.projection.weight".to_string(),
            Array::from_slice_f32(&[2.0]).unwrap(),
        );
        weights.insert(
            "vision_tower.weight".to_string(),
            Array::from_slice_f32(&[3.0]).unwrap(),
        );

        let config = make_test_config(false);
        let sanitized = sanitize_weights(weights, &config);

        assert!(!sanitized.contains_key("audio_tower.encoder.weight"));
        assert!(!sanitized.contains_key("embed_audio.projection.weight"));
        assert!(sanitized.contains_key("vision_tower.weight"));
    }

    #[test]
    fn sanitize_splits_moe_gate_up_proj() {
        // In PyTorch checkpoint, gate_up_proj is [experts, 2*intermediate, hidden].
        // [2 experts, 2*intermediate=4, hidden=6] => intermediate=2, hidden=6.
        // After swapaxes(-1,-2): [2, 6, 4]
        // Split on last dim (4 → 2+2): gate [2, 6, 2], up [2, 6, 2]
        // After swapaxes back: gate [2, 2, 6], up [2, 2, 6]
        let data: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let gate_up = Array::from_slice_f32(&data)
            .unwrap()
            .reshape(&[2, 4, 6])
            .unwrap();

        let mut weights = HashMap::new();
        weights.insert(
            "language_model.model.layers.0.moe.experts.gate_up_proj".to_string(),
            gate_up,
        );

        let config = make_test_config(false);
        let sanitized = sanitize_weights(weights, &config);

        assert!(!sanitized.contains_key("language_model.model.layers.0.moe.experts.gate_up_proj"));
        assert!(sanitized.contains_key("language_model.model.layers.0.moe.experts.switch_glu.gate_proj.weight"));
        assert!(sanitized.contains_key("language_model.model.layers.0.moe.experts.switch_glu.up_proj.weight"));

        let gate = sanitized.get("language_model.model.layers.0.moe.experts.switch_glu.gate_proj.weight").unwrap();
        let up = sanitized.get("language_model.model.layers.0.moe.experts.switch_glu.up_proj.weight").unwrap();

        assert_eq!(gate.shape_raw(), vec![2, 2, 6]);
        assert_eq!(up.shape_raw(), vec![2, 2, 6]);
    }

    #[test]
    fn sanitize_transposes_conv_weights() {
        // Use vision_tower keys so they are not stripped (no audio tower in config)
        // Conv2d: PyTorch [out=2, in=3, kH=4, kW=5] → MLX [2, 4, 5, 3]
        let conv2d = Array::from_slice_f32(&vec![0.0f32; 120])
            .unwrap()
            .reshape(&[2, 3, 4, 5])
            .unwrap();
        // Conv1d: PyTorch [out=2, in=3, kW=4] → MLX [2, 4, 3]
        let conv1d = Array::from_slice_f32(&vec![0.0f32; 24])
            .unwrap()
            .reshape(&[2, 3, 4])
            .unwrap();

        let mut weights = HashMap::new();
        weights.insert(
            "vision_tower.subsample_conv_projection.conv.weight".to_string(),
            conv2d,
        );
        weights.insert(
            "vision_tower.depthwise_conv1d.weight".to_string(),
            conv1d,
        );

        let config = make_test_config(false);
        let sanitized = sanitize_weights(weights, &config);

        let t2d = sanitized.get("vision_tower.subsample_conv_projection.conv.weight").unwrap();
        let t1d = sanitized.get("vision_tower.depthwise_conv1d.weight").unwrap();

        assert_eq!(t2d.shape_raw(), vec![2, 4, 5, 3]);
        assert_eq!(t1d.shape_raw(), vec![2, 4, 3]);
    }
}
