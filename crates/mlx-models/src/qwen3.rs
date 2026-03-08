//! Qwen3 model implementation.
//!
//! Follows the same pattern as Llama but with Qwen-specific config and
//! differences in attention (e.g. qk_norm, bias options).

use mlx_core::{Array, Module, Result};
use mlx_nn::{Embedding, KvCache, Linear, QuantConfig, RmsNorm, RoPE, RopeScaling, VarBuilder};

// ------------------------------------------------------------------
// Config
// ------------------------------------------------------------------

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen3Config {
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
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_head_dim")]
    pub head_dim: Option<usize>,
    pub quantization: Option<super::llama::QuantizationConfig>,
}

fn default_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    1_000_000.0
}
fn default_max_pos() -> usize {
    32768
}
fn default_head_dim() -> Option<usize> {
    None
}

impl Qwen3Config {
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
}

// ------------------------------------------------------------------
// Attention (Qwen3 has optional QK norm)
// ------------------------------------------------------------------

struct Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    rope: RoPE,
    kv_cache: KvCache,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl Qwen3Attention {
    fn load(vb: &VarBuilder, cfg: &Qwen3Config) -> anyhow::Result<Self> {
        let qc = cfg.quant_config();
        let q_proj = Linear::new(&vb.pp("q_proj"), &qc)?;
        let k_proj = Linear::new(&vb.pp("k_proj"), &qc)?;
        let v_proj = Linear::new(&vb.pp("v_proj"), &qc)?;
        let o_proj = Linear::new(&vb.pp("o_proj"), &qc)?;

        let q_norm = if vb.pp("q_norm").contains("weight") {
            Some(RmsNorm::new(cfg.rms_norm_eps, &vb.pp("q_norm"))?)
        } else {
            None
        };
        let k_norm = if vb.pp("k_norm").contains("weight") {
            Some(RmsNorm::new(cfg.rms_norm_eps, &vb.pp("k_norm"))?)
        } else {
            None
        };

        let head_dim = cfg.head_dim();
        let rope = match &cfg.rope_scaling {
            Some(scaling) => RoPE::with_scaling(
                head_dim as i32,
                cfg.rope_theta,
                false,
                scaling,
                cfg.max_position_embeddings,
            )?,
            None => RoPE::new(head_dim as i32, cfg.rope_theta, false),
        };

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
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        let shape = x.shape_raw();
        let (b, seq_len, _) = (shape[0], shape[1], shape[2]);
        let offset = self.kv_cache.offset() as i32;

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

        // Optional QK norm
        let q = match &self.q_norm {
            Some(norm) => norm.forward(&q)?,
            None => q,
        };
        let k = match &self.k_norm {
            Some(norm) => norm.forward(&k)?,
            None => k,
        };

        let q = self.rope.forward(&q, offset)?;
        let k = self.rope.forward(&k, offset)?;

        let (k, v) = self.kv_cache.update(&k, &v)?;

        // GQA: MLX's SDPA handles grouped query attention natively.
        let mask_mode = if seq_len > 1 { "causal" } else { "" };
        let attn = q.fast_scaled_dot_product_attention(&k, &v, self.scale, mask_mode, None)?;

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

// ------------------------------------------------------------------
// MLP
// ------------------------------------------------------------------

struct Qwen3Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen3Mlp {
    fn load(vb: &VarBuilder, cfg: &Qwen3Config) -> anyhow::Result<Self> {
        let qc = cfg.quant_config();
        Ok(Self {
            gate_proj: Linear::new(&vb.pp("gate_proj"), &qc)?,
            up_proj: Linear::new(&vb.pp("up_proj"), &qc)?,
            down_proj: Linear::new(&vb.pp("down_proj"), &qc)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let gate_silu = gate.multiply(&gate.sigmoid()?)?;
        self.down_proj.forward(&gate_silu.multiply(&up)?)
    }
}

// ------------------------------------------------------------------
// Block
// ------------------------------------------------------------------

struct Qwen3Block {
    attn: Qwen3Attention,
    mlp: Qwen3Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Qwen3Block {
    fn load(vb: &VarBuilder, cfg: &Qwen3Config) -> anyhow::Result<Self> {
        Ok(Self {
            attn: Qwen3Attention::load(&vb.pp("self_attn"), cfg)?,
            mlp: Qwen3Mlp::load(&vb.pp("mlp"), cfg)?,
            input_layernorm: RmsNorm::new(cfg.rms_norm_eps, &vb.pp("input_layernorm"))?,
            post_attention_layernorm: RmsNorm::new(
                cfg.rms_norm_eps,
                &vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let h = self.attn.forward(&h)?;
        let x = residual.add(&h)?;

        let residual = x.clone();
        let h = self.post_attention_layernorm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        residual.add(&h)
    }

    fn clear_cache(&mut self) {
        self.attn.clear_cache();
    }
}

// ------------------------------------------------------------------
// Qwen3 Model
// ------------------------------------------------------------------

/// Qwen3 language model.
pub struct Qwen3 {
    embed_tokens: Embedding,
    layers: Vec<Qwen3Block>,
    norm: RmsNorm,
    lm_head: Linear,
}

impl Qwen3 {
    pub fn new(cfg: &Qwen3Config, vb: &VarBuilder) -> anyhow::Result<Self> {
        let qc = cfg.quant_config();
        let model_vb = vb.pp("model");

        let embed_tokens = Embedding::new(&model_vb.pp("embed_tokens"), &qc)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Qwen3Block::load(&model_vb.pp(format!("layers.{i}")), cfg)?);
        }

        let norm = RmsNorm::new(cfg.rms_norm_eps, &model_vb.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            embed_tokens.as_linear()
        } else {
            Linear::new(&vb.pp("lm_head"), &qc)?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    pub fn forward(&mut self, input_ids: &Array) -> Result<Array> {
        let shape = input_ids.shape_raw();
        let seq_len = shape[shape.len() - 1];

        let mut h = self.forward_hidden_states(input_ids)?;

        if seq_len > 1 {
            let h_shape = h.shape_raw();
            let mut start = vec![0i32; h_shape.len()];
            let mut stop = h_shape.clone();
            start[h_shape.len() - 2] = seq_len - 1;
            stop[h_shape.len() - 2] = seq_len;
            h = h.slice(&start, &stop)?;
        }

        self.lm_head.forward(&h)
    }

    pub fn forward_hidden_states(&mut self, input_ids: &Array) -> Result<Array> {
        let mut h = self.embed_tokens.forward(input_ids)?;
        for layer in &mut self.layers {
            h = layer.forward(&h)?;
        }
        self.norm.forward(&h)
    }

    pub fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }
}
