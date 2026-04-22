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

struct RmsNormZeroShift {
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

#[allow(dead_code)]
fn cache_index(layer_idx: usize, num_non_shared: usize) -> usize {
    if num_non_shared > 0 {
        layer_idx % num_non_shared
    } else {
        layer_idx
    }
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
        let linear = Linear::new(&vb.pp("linear"), config)?;
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
    let dim = shape[axis as usize];
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
// DecoderLayer
// ------------------------------------------------------------------

struct Gemma4DecoderLayer {
    input_layernorm: RmsNormZeroShift,
    post_attn_layernorm: RmsNormZeroShift,
    pre_ffw_layernorm: RmsNormZeroShift,
    post_ffw_layernorm: RmsNormZeroShift,
    self_attn: Gemma4Attention,
    mlp: Gemma4Mlp,
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
            layer_scalar,
            is_sliding,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
        })
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

        // MLP sublayer
        let residual = h.clone();
        let h_norm = self.pre_ffw_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&h_norm)?;
        let h = self.post_ffw_layernorm.forward(&mlp_out)?;
        let mut h = residual.add(&h)?;

        // Per-layer input gating (after MLP, matching Python)
        if let (Some(ref gate_proj), Some(ref proj_proj), Some(ref post_norm), Some(layer_emb)) =
            (&self.per_layer_input_gate, &self.per_layer_projection, &self.post_per_layer_input_norm, layer_emb)
        {
            let residual = h.clone();
            let gate = gate_proj.forward(&h)?;
            let gate = gelu_approx(&gate)?;
            let gate = gate.multiply(layer_emb)?;
            let gate = proj_proj.forward(&gate)?;
            let gate = post_norm.forward(&gate)?;
            h = residual.add(&gate)?;
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

struct Gemma4TextModel {
    embed_tokens: Embedding,
    layers: Vec<Gemma4DecoderLayer>,
    norm: RmsNormZeroShift,
    embed_scale: f32,
    embed_tokens_per_layer: Option<Embedding>,
    embed_tokens_per_layer_scale: f32,
    per_layer_model_projection: Option<Linear>,
    per_layer_projection_norm: Option<RmsNormZeroShift>,
    per_layer_input_scale: f32,
    num_hidden_layers: usize,
    hidden_size_per_layer_input: usize,
    sliding_window: usize,
    caches: Vec<KvCache>,
    layer_idx_to_cache_idx: Vec<usize>,
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

        let first_kv_shared_layer_idx = if cfg.num_kv_shared_layers > 0 {
            cfg.num_hidden_layers - cfg.num_kv_shared_layers
        } else {
            cfg.num_hidden_layers
        };
        let num_caches = first_kv_shared_layer_idx;
        let caches = (0..num_caches).map(|_| KvCache::new()).collect();

        // Build layer_idx_to_cache_idx matching Python logic
        let mut layer_idx_to_cache_idx: Vec<usize> = (0..num_caches).collect();
        if first_kv_shared_layer_idx < cfg.num_hidden_layers {
            let concrete_layers = &cfg.layer_types[..first_kv_shared_layer_idx];
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
            for i in first_kv_shared_layer_idx..cfg.num_hidden_layers {
                if cfg.layer_types[i] == "full_attention" {
                    layer_idx_to_cache_idx.push(shared_full_idx);
                } else {
                    layer_idx_to_cache_idx.push(shared_sliding_idx);
                }
            }
        }

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
        &mut self,
        embeddings: &Array,
        per_layer_inputs: Option<&Array>,
    ) -> Result<Array> {
        let mut h = embeddings.clone();
        let shape = h.shape_raw();
        let seq_len = shape[1] as usize;
        let batch = shape[0] as usize;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let cache_idx = self.layer_idx_to_cache_idx[i];

            let mask = if layer.is_sliding {
                let offset = self.caches[cache_idx].offset();
                Some(sliding_window_mask(
                    batch,
                    seq_len,
                    offset,
                    self.sliding_window,
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

            h = layer.forward(&h, mask.as_ref(), &mut self.caches[cache_idx], layer_emb.as_ref())?;
        }

        let out = self.norm.forward(&h)?;
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

        self.forward_embeddings(&h, per_layer_inputs.as_ref())
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

struct LanguageModel {
    model: Gemma4TextModel,
    lm_head: Linear,
    final_logit_softcapping: Option<f32>,
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
    input_proj: Linear,
    position_embedding_table: Embedding,
    patch_size: usize,
}

impl VisionPatchEmbedder {
    fn new(vb: &VarBuilder, cfg: &Gemma4VisionConfig) -> anyhow::Result<Self> {
        let qc = QuantConfig::default();
        let pos_emb_weight = vb.get("position_embedding_table")?;
        Ok(Self {
            input_proj: Linear::new(&vb.pp("input_proj"), &qc)?,
            position_embedding_table: Embedding::from_weight(pos_emb_weight),
            patch_size: cfg.patch_size,
        })
    }

    fn forward(&self, images: &Array) -> Result<(Array, Array)> {
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

        let features = self.input_proj.forward(&patches)?;

        // Position embeddings
        let mut pos_indices = Vec::with_capacity(num_patches as usize);
        for row in 0..num_patches_h {
            for col in 0..num_patches_w {
                pos_indices.push((row * num_patches_w + col) as i32);
            }
        }
        let pos_indices_arr = Array::from_slice_i32(&pos_indices)?;
        let pos_emb = self.position_embedding_table.forward(&pos_indices_arr)?;

        Ok((features, pos_emb))
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

    fn forward(&self, features: &Array) -> Result<Array> {
        let shape = features.shape_raw();
        let batch = shape[0];
        let seq_len = shape[1];
        let hidden_size = shape[2];

        let kernel = self.pooling_kernel_size as i32;
        let num_pools = seq_len / kernel;

        if num_pools == 0 {
            // Not enough tokens, return as-is truncated
            let start = vec![0i32; 3];
            let mut stop = shape.clone();
            stop[1] = self.default_output_length as i32;
            return features.slice(&start, &stop);
        }

        let pooled = features
            .reshape(&[batch, num_pools, kernel, hidden_size])?
            .mean_axis(2, true)?;

        pooled.squeeze(2)
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

    fn forward(&mut self, x: &Array, positions_h: &Array, positions_w: &Array) -> Result<Array> {
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

        let attn = q.fast_scaled_dot_product_attention(&k, &v, self.scale, "", None)?;

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

    fn forward(&mut self, x: &Array, positions_h: &Array, positions_w: &Array) -> Result<Array> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h, positions_h, positions_w)?;
        let h = self.post_attention_layernorm.forward(&h)?;
        let h = residual.add(&h)?;

        let residual = h.clone();
        let h = self.pre_feedforward_layernorm.forward(&h)?;
        let h = self.mlp.forward(&h)?;
        let h = self.post_feedforward_layernorm.forward(&h)?;
        residual.add(&h)
    }
}

struct VisionModel {
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

    fn forward(&mut self, images: &Array) -> Result<Array> {
        let shape = images.shape_raw();
        let (_b, _c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let patch_size = self.patch_embedder.patch_size as i32;
        let num_patches_h = h / patch_size;
        let num_patches_w = w / patch_size;
        let num_patches = num_patches_h * num_patches_w;

        let (mut features, pos_emb) = self.patch_embedder.forward(images)?;

        if let (Some(ref bias), Some(ref scale)) = (&self.std_bias, &self.std_scale) {
            features = features.subtract(bias)?;
            features = features.divide(scale)?;
        }

        features = features.add(&pos_emb)?;

        let mut pos_h = Vec::with_capacity(num_patches as usize);
        let mut pos_w = Vec::with_capacity(num_patches as usize);
        for row in 0..num_patches_h {
            for col in 0..num_patches_w {
                pos_h.push(row as i32);
                pos_w.push(col as i32);
            }
        }
        let positions_h = Array::from_slice_i32(&pos_h)?;
        let positions_w = Array::from_slice_i32(&pos_w)?;

        for layer in &mut self.layers {
            features = layer.forward(&features, &positions_h, &positions_w)?;
        }

        if let Some(ref norm) = self.norm {
            features = norm.forward(&features)?;
        }
        self.pooler.forward(&features)
    }
}

// ------------------------------------------------------------------
// Multimodal embedder
// ------------------------------------------------------------------

struct MultimodalEmbedder {
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

    fn forward(&self, features: &Array) -> Result<Array> {
        let h = self.embedding_pre_projection_norm.forward(features)?;
        self.embedding_projection.forward(&h)
    }
}

// ------------------------------------------------------------------
// Top-level Gemma4 model
// ------------------------------------------------------------------

pub struct Gemma4 {
    language_model: LanguageModel,
    vision_tower: VisionModel,
    embed_vision: MultimodalEmbedder,
    config: Gemma4Config,
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

    pub fn forward_logits(
        &mut self,
        input_ids: &Array,
        pixel_values: Option<&Array>,
    ) -> Result<Array> {
        let mut h = self.language_model.model.embed_tokens.forward(input_ids)?;
        h = h.multiply(&Array::from_float(self.language_model.model.embed_scale)?)?;

        if let Some(pixel_values) = pixel_values {
            let vision_features = self.vision_tower.forward(pixel_values)?;
            let vision_emb = self.embed_vision.forward(&vision_features)?;

            let num_vision_tokens = vision_emb.shape_raw()[1];
            let start = vec![0i32; 3];
            let mut stop = h.shape_raw();
            stop[1] = num_vision_tokens;
            let strides = vec![1i32; 3];
            h = h.slice_update(&vision_emb, &start, &stop, &strides)?;
        }

        let per_layer_inputs = if self.language_model.model.embed_tokens_per_layer.is_some() {
            let pli = self.language_model.model.compute_per_layer_inputs(input_ids, &h)?;
            Some(pli)
        } else {
            None
        };

        let h = self.language_model.model.forward_embeddings(&h, per_layer_inputs.as_ref())?;
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
    fn cache_index_works() {
        // 35 layers, 15 non-shared, 20 shared
        assert_eq!(cache_index(0, 15), 0);
        assert_eq!(cache_index(14, 15), 14);
        assert_eq!(cache_index(15, 15), 0);
        assert_eq!(cache_index(29, 15), 14);
        assert_eq!(cache_index(30, 15), 0);
        assert_eq!(cache_index(34, 15), 4);
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
}
