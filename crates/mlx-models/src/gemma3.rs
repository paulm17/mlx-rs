//! Gemma 3 text model implementation.

use mlx_core::{Array, DType, Module, Result};
use mlx_nn::{repeat_kv, Embedding, KvCache, Linear, QuantConfig, RoPE, RopeScaling, VarBuilder};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Gemma3Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_rope_local_base_freq")]
    pub rope_local_base_freq: f32,
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    #[serde(default = "default_sliding_window_pattern")]
    pub sliding_window_pattern: usize,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_head_dim")]
    pub head_dim: Option<usize>,
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,
    #[serde(default = "default_hidden_activation")]
    pub hidden_activation: String,
    #[serde(default)]
    pub hidden_act: Option<String>,
    #[serde(default)]
    pub query_pre_attn_scalar: Option<usize>,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub attn_logit_softcapping: Option<f32>,
    pub quantization: Option<super::llama::QuantizationConfig>,
}

fn default_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    1_000_000.0
}
fn default_rope_local_base_freq() -> f32 {
    10_000.0
}
fn default_sliding_window() -> usize {
    1024
}
fn default_sliding_window_pattern() -> usize {
    6
}
fn default_max_pos() -> usize {
    131072
}
fn default_head_dim() -> Option<usize> {
    None
}
fn default_attention_bias() -> bool {
    false
}
fn default_hidden_activation() -> String {
    "silu".to_string()
}

impl Gemma3Config {
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn quant_config(&self) -> QuantConfig {
        match &self.quantization {
            Some(q) => QuantConfig {
                group_size: q.group_size,
                bits: q.bits,
            },
            None => QuantConfig::default(),
        }
    }

    fn rope_scaling_for_window(&self, sliding_window: bool) -> Option<RopeScaling> {
        if !sliding_window {
            return self.rope_scaling.clone();
        }
        let mut scaling = self.rope_scaling.clone().unwrap_or(RopeScaling {
            rope_type: Some("linear".to_string()),
            factor: 1.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: self.max_position_embeddings,
        });
        if scaling.rope_type.is_none() {
            scaling.rope_type = Some("linear".to_string());
        }
        Some(scaling)
    }

    fn mlp_uses_silu(&self) -> bool {
        matches!(
            self.hidden_act
                .as_deref()
                .or(Some(self.hidden_activation.as_str())),
            Some("silu") | Some("SiLU")
        )
    }
}

struct GemmaRmsNorm {
    weight_plus_one: Array,
    eps: f32,
}

impl GemmaRmsNorm {
    fn new(eps: f32, vb: &VarBuilder) -> anyhow::Result<Self> {
        let weight = vb.get("weight")?;
        let weight_plus_one = weight.add(&Array::from_float(1.0)?)?;
        Ok(Self {
            weight_plus_one,
            eps,
        })
    }
}

impl Module for GemmaRmsNorm {
    fn forward(&self, x: &Array) -> Result<Array> {
        x.fast_rms_norm(&self.weight_plus_one, self.eps)
    }
}

struct GemmaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: GemmaRmsNorm,
    k_norm: GemmaRmsNorm,
    rope: RoPE,
    kv_cache: KvCache,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    sliding_window: Option<usize>,
    attn_logit_softcapping: Option<f32>,
}

impl GemmaAttention {
    fn load(
        vb: &VarBuilder,
        cfg: &Gemma3Config,
        sliding_window: Option<usize>,
    ) -> anyhow::Result<Self> {
        let qc = cfg.quant_config();
        let q_proj = Linear::new(&vb.pp("q_proj"), &qc)?;
        let k_proj = Linear::new(&vb.pp("k_proj"), &qc)?;
        let v_proj = Linear::new(&vb.pp("v_proj"), &qc)?;
        let o_proj = Linear::new(&vb.pp("o_proj"), &qc)?;
        let q_norm = GemmaRmsNorm::new(cfg.rms_norm_eps, &vb.pp("q_norm"))?;
        let k_norm = GemmaRmsNorm::new(cfg.rms_norm_eps, &vb.pp("k_norm"))?;

        let head_dim = if let Some(hd) = cfg.head_dim {
            hd
        } else {
            // Infer from q_proj weight shape when config omits head_dim.
            // For quantized weights the packed tensor is [out_features, ...];
            // for full-precision it's [in_features, out_features].
            let is_quantized = vb.pp("q_proj").contains("scales");
            let shape = q_proj.weight().shape_raw();
            let out_features = if is_quantized {
                shape[0] as usize
            } else {
                shape[1] as usize
            };
            out_features / cfg.num_attention_heads
        };
        let rope_theta = if sliding_window.is_some() {
            cfg.rope_local_base_freq
        } else {
            cfg.rope_theta
        };
        let rope = match cfg.rope_scaling_for_window(sliding_window.is_some()) {
            Some(scaling) => RoPE::with_scaling(
                head_dim as i32,
                rope_theta,
                false,
                &scaling,
                cfg.max_position_embeddings,
            )?,
            None => RoPE::new(head_dim as i32, rope_theta, false),
        };

        let mut scale = 1.0 / (head_dim as f32).sqrt();
        if let Some(pre_attn) = cfg.query_pre_attn_scalar {
            if pre_attn > 0 {
                scale = 1.0 / (pre_attn as f32).sqrt();
            }
        }

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rope,
            kv_cache: KvCache::new(),
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_kv_heads(),
            head_dim,
            scale,
            sliding_window,
            attn_logit_softcapping: cfg.attn_logit_softcapping,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        let shape = x.shape_raw();
        let (b, seq_len, _) = (shape[0], shape[1], shape[2]);
        let offset = self.kv_cache.offset();

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape(&[b, seq_len, self.num_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[b, seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[b, seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let q = self.rope.forward(&q, offset as i32)?;
        let k = self.rope.forward(&k, offset as i32)?;

        let (k, v) = self.kv_cache.update(&k, &v)?;
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;

        let attn = if let Some(window) = self.sliding_window {
            let mask =
                sliding_window_mask(b as usize, seq_len as usize, offset, window, q.dtype())?;
            let mut scores = q.matmul(&k.transpose_axes(&[0, 1, 3, 2])?)?;
            scores = scores.multiply(&Array::from_float(self.scale)?)?;
            if let Some(softcap) = self.attn_logit_softcapping {
                scores = scores
                    .divide(&Array::from_float(softcap)?)?
                    .tanh()?
                    .multiply(&Array::from_float(softcap)?)?;
            }
            let probs = scores.add(&mask)?.softmax(-1)?;
            probs.matmul(&v)?
        } else {
            let mask_mode = if seq_len > 1 { "causal" } else { "" };
            let mut attn =
                q.fast_scaled_dot_product_attention(&k, &v, self.scale, mask_mode, None)?;
            if let Some(softcap) = self.attn_logit_softcapping {
                attn = attn
                    .divide(&Array::from_float(softcap)?)?
                    .tanh()?
                    .multiply(&Array::from_float(softcap)?)?;
            }
            attn
        };

        let attn = attn.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            b,
            seq_len,
            (self.num_heads * self.head_dim) as i32,
        ])?;
        self.o_proj.forward(&attn)
    }

    fn clear_cache(&mut self) {
        self.kv_cache.reset();
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

#[derive(Clone, Copy)]
enum Activation {
    Silu,
    Gelu,
}

struct GemmaMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    activation: Activation,
}

impl GemmaMlp {
    fn load(vb: &VarBuilder, cfg: &Gemma3Config) -> anyhow::Result<Self> {
        let qc = cfg.quant_config();
        let activation = cfg.mlp_uses_silu().then_some(Activation::Silu).unwrap_or(Activation::Gelu);
        Ok(Self {
            gate_proj: Linear::new(&vb.pp("gate_proj"), &qc)?,
            up_proj: Linear::new(&vb.pp("up_proj"), &qc)?,
            down_proj: Linear::new(&vb.pp("down_proj"), &qc)?,
            activation,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let gate = match self.activation {
            Activation::Silu => gate.multiply(&gate.sigmoid()?)?,
            Activation::Gelu => gelu_approx(&gate)?,
        };
        self.down_proj.forward(&gate.multiply(&up)?)
    }
}

struct GemmaBlock {
    attn: GemmaAttention,
    mlp: GemmaMlp,
    input_layernorm: GemmaRmsNorm,
    post_attention_layernorm: GemmaRmsNorm,
    pre_feedforward_layernorm: GemmaRmsNorm,
    post_feedforward_layernorm: GemmaRmsNorm,
}

impl GemmaBlock {
    fn load(
        vb: &VarBuilder,
        cfg: &Gemma3Config,
        sliding_window: Option<usize>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            attn: GemmaAttention::load(&vb.pp("self_attn"), cfg, sliding_window)?,
            mlp: GemmaMlp::load(&vb.pp("mlp"), cfg)?,
            input_layernorm: GemmaRmsNorm::new(cfg.rms_norm_eps, &vb.pp("input_layernorm"))?,
            post_attention_layernorm: GemmaRmsNorm::new(
                cfg.rms_norm_eps,
                &vb.pp("post_attention_layernorm"),
            )?,
            pre_feedforward_layernorm: GemmaRmsNorm::new(
                cfg.rms_norm_eps,
                &vb.pp("pre_feedforward_layernorm"),
            )?,
            post_feedforward_layernorm: GemmaRmsNorm::new(
                cfg.rms_norm_eps,
                &vb.pp("post_feedforward_layernorm"),
            )?,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let h = self.attn.forward(&h)?;
        let h = self.post_attention_layernorm.forward(&h)?;
        let x = residual.add(&h)?;

        let residual = x.clone();
        let h = self.pre_feedforward_layernorm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        let h = self.post_feedforward_layernorm.forward(&h)?;
        residual.add(&h)
    }

    fn clear_cache(&mut self) {
        self.attn.clear_cache();
    }
}

pub struct Gemma3 {
    embed_tokens: Embedding,
    layers: Vec<GemmaBlock>,
    norm: GemmaRmsNorm,
    lm_head: Linear,
    embed_scale: Array,
    final_logit_softcapping: Option<f32>,
}

impl Gemma3 {
    pub fn new(cfg: &Gemma3Config, vb: &VarBuilder) -> anyhow::Result<Self> {
        let qc = cfg.quant_config();

        // Text-only Gemma3 (e.g. gemma-3-1b-it) stores weights at model.* / lm_head.*
        // Multimodal Gemma3 (e.g. gemma-3-4b-it) nests them under language_model.*
        let has_language_model = vb.pp("language_model").pp("model").pp("embed_tokens").contains("weight");
        let (model_vb, lm_head_vb) = if has_language_model {
            (vb.pp("language_model").pp("model"), vb.pp("language_model").pp("lm_head"))
        } else {
            (vb.pp("model"), vb.pp("lm_head"))
        };

        let embed_tokens = Embedding::new(&model_vb.pp("embed_tokens"), &qc)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let sliding_window = if (i + 1) % cfg.sliding_window_pattern > 0 {
                Some(cfg.sliding_window)
            } else {
                None
            };
            layers.push(GemmaBlock::load(
                &model_vb.pp(format!("layers.{i}")),
                cfg,
                sliding_window,
            )?);
        }

        let norm = GemmaRmsNorm::new(cfg.rms_norm_eps, &model_vb.pp("norm"))?;
        let lm_head = if !lm_head_vb.contains("weight") {
            embed_tokens.as_linear()
        } else {
            Linear::new(&lm_head_vb, &qc)?
        };
        let embed_scale = Array::from_float((cfg.hidden_size as f32).sqrt())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            embed_scale,
            final_logit_softcapping: cfg.final_logit_softcapping,
        })
    }

    pub fn forward_logits(&mut self, input_ids: &Array) -> Result<Array> {
        let mut h = self.embed_tokens.forward(input_ids)?;
        h = h.multiply(&self.embed_scale)?;
        for layer in &mut self.layers {
            h = layer.forward(&h)?;
        }
        h = self.norm.forward(&h)?;
        let mut logits = self.lm_head.forward(&h)?;
        if let Some(softcap) = self.final_logit_softcapping {
            logits = logits
                .divide(&Array::from_float(softcap)?)?
                .tanh()?
                .multiply(&Array::from_float(softcap)?)?;
        }
        Ok(logits)
    }

    pub fn forward_hidden_states(&mut self, input_ids: &Array) -> Result<Array> {
        let mut h = self.embed_tokens.forward(input_ids)?;
        h = h.multiply(&self.embed_scale)?;
        for layer in &mut self.layers {
            h = layer.forward(&h)?;
        }
        self.norm.forward(&h)
    }

    pub fn forward_last_token_logits(&mut self, input_ids: &Array) -> Result<Array> {
        let seq_len = input_ids.shape_raw()[input_ids.ndim() - 1];
        let mut logits = self.forward_logits(input_ids)?;
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

    pub fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Gemma3Config;

    #[test]
    fn config_defaults_are_stable() {
        let cfg: Gemma3Config = serde_json::from_str(
            r#"{
                "hidden_size": 3840,
                "intermediate_size": 15360,
                "vocab_size": 262208,
                "num_hidden_layers": 48,
                "num_attention_heads": 16,
                "num_key_value_heads": 8
            }"#,
        )
        .unwrap();

        assert_eq!(cfg.head_dim(), 240);
        assert_eq!(cfg.sliding_window_pattern, 6);
        assert_eq!(cfg.sliding_window, 1024);
        assert_eq!(cfg.rms_norm_eps, 1e-6);
        assert_eq!(cfg.rope_theta, 1_000_000.0);
    }
}
