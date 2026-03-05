//! Llama model implementation.
//!
//! Follows candle's llama.rs pattern: Config → Attention → MLP → Block → Llama.

use mlx_core::{Array, Module, Result};
use mlx_nn::{
    Embedding, KvCache, Linear, QuantConfig, RmsNorm, RoPE, RopeScaling, VarBuilder,
};

// ------------------------------------------------------------------
// Config
// ------------------------------------------------------------------

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlamaConfig {
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
    // Quantization (from config.json)
    pub quantization: Option<QuantizationConfig>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct QuantizationConfig {
    #[serde(default = "default_group_size")]
    pub group_size: i32,
    #[serde(default = "default_bits")]
    pub bits: i32,
}

fn default_eps() -> f32 { 1e-5 }
fn default_rope_theta() -> f32 { 10_000.0 }
fn default_max_pos() -> usize { 4096 }
fn default_group_size() -> i32 { 64 }
fn default_bits() -> i32 { 4 }

impl LlamaConfig {
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
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
// Attention
// ------------------------------------------------------------------

struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rope: RoPE,
    kv_cache: KvCache,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl CausalSelfAttention {
    fn load(vb: &VarBuilder, cfg: &LlamaConfig) -> anyhow::Result<Self> {
        let qc = cfg.quant_config();
        let q_proj = Linear::new(&vb.pp("q_proj"), &qc)?;
        let k_proj = Linear::new(&vb.pp("k_proj"), &qc)?;
        let v_proj = Linear::new(&vb.pp("v_proj"), &qc)?;
        let o_proj_vb = if vb.pp("o_proj").contains("weight") {
            vb.pp("o_proj")
        } else {
            vb.pp("out_proj")
        };
        let o_proj = Linear::new(&o_proj_vb, &qc)?;

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

        // Reshape to [B, seq, heads, head_dim] then transpose to [B, heads, seq, head_dim]
        let q = q
            .reshape(&[b, seq_len, self.num_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[b, seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[b, seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE
        let q = self.rope.forward(&q, offset)?;
        let k = self.rope.forward(&k, offset)?;

        // Update KV cache
        let (k, v) = self.kv_cache.update(&k, &v)?;

        // GQA: MLX's SDPA handles grouped query attention natively
        let mask_mode = if seq_len > 1 { "causal" } else { "" };
        let attn = q.fast_scaled_dot_product_attention(&k, &v, self.scale, mask_mode, None)?;

        // Reshape back to [B, seq, hidden]
        let attn = attn
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, seq_len, (self.num_heads * self.head_dim) as i32])?;

        self.o_proj.forward(&attn)
    }

    fn clear_cache(&mut self) {
        self.kv_cache.reset();
    }
}

// ------------------------------------------------------------------
// MLP
// ------------------------------------------------------------------

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(vb: &VarBuilder, cfg: &LlamaConfig) -> anyhow::Result<Self> {
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
        // SiLU(gate) = gate * sigmoid(gate)
        let gate_silu = gate.multiply(&gate.sigmoid()?)?;
        let hidden = gate_silu.multiply(&up)?;
        self.down_proj.forward(&hidden)
    }
}

// ------------------------------------------------------------------
// Decoder Block
// ------------------------------------------------------------------

struct Block {
    attn: CausalSelfAttention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Block {
    fn load(vb: &VarBuilder, cfg: &LlamaConfig) -> anyhow::Result<Self> {
        Ok(Self {
            attn: CausalSelfAttention::load(&vb.pp("self_attn"), cfg)?,
            mlp: Mlp::load(&vb.pp("mlp"), cfg)?,
            input_layernorm: RmsNorm::new(cfg.rms_norm_eps, &vb.pp("input_layernorm"))?,
            post_attention_layernorm: RmsNorm::new(
                cfg.rms_norm_eps,
                &vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        // Pre-norm attention
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let h = self.attn.forward(&h)?;
        let x = residual.add(&h)?;

        // Pre-norm MLP
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
// Llama Model
// ------------------------------------------------------------------

/// Llama language model.
pub struct Llama {
    embed_tokens: Embedding,
    layers: Vec<Block>,
    norm: RmsNorm,
    lm_head: Linear,
}

impl Llama {
    /// Load a Llama model from a VarBuilder.
    pub fn new(cfg: &LlamaConfig, vb: &VarBuilder) -> anyhow::Result<Self> {
        let qc = cfg.quant_config();
        let model_vb = vb.pp("model");

        let embed_tokens = Embedding::new(&model_vb.pp("embed_tokens"), &qc)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Block::load(&model_vb.pp(format!("layers.{i}")), cfg)?);
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

    /// Forward pass. Returns logits for all input tokens.
    pub fn forward_logits(&mut self, input_ids: &Array) -> Result<Array> {
        let mut h = self.embed_tokens.forward(input_ids)?;

        for layer in &mut self.layers {
            h = layer.forward(&h)?;
            h.eval()?;
        }

        h = self.norm.forward(&h)?;
        self.lm_head.forward(&h)
    }

    /// Forward pass. Returns logits for the last token.
    pub fn forward_last_token_logits(&mut self, input_ids: &Array) -> Result<Array> {
        let shape = input_ids.shape_raw();
        let seq_len = shape[shape.len() - 1];

        let mut logits = self.forward_logits(input_ids)?;

        // Take last-token logits to match cached decode semantics.
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

    /// Backward-compatible forward alias returning last-token logits.
    pub fn forward(&mut self, input_ids: &Array) -> Result<Array> {
        self.forward_last_token_logits(input_ids)
    }

    /// Clear all KV caches.
    pub fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }
}
