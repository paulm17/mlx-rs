use mlx_core::{Array, Module, Result};
use mlx_nn::{repeat_kv, Embedding, KvCache, Linear, QuantConfig, RmsNorm, RoPE, VarBuilder};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen35Config {
    pub model_type: String,
    pub text_config: Qwen35TextConfig,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub quantization: Option<super::llama::QuantizationConfig>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen35TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub layer_types: Vec<String>,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f32,
    pub head_dim: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_linear_num_key_heads")]
    pub linear_num_key_heads: usize,
    #[serde(default = "default_linear_num_value_heads")]
    pub linear_num_value_heads: usize,
    #[serde(default = "default_linear_key_head_dim")]
    pub linear_key_head_dim: usize,
    #[serde(default = "default_linear_value_head_dim")]
    pub linear_value_head_dim: usize,
    #[serde(default = "default_linear_conv_kernel_dim")]
    pub linear_conv_kernel_dim: usize,
    pub rope_parameters: Option<RopeParameters>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub partial_rotary_factor: Option<f32>,
}

fn default_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    10_000_000.0
}
fn default_linear_num_key_heads() -> usize {
    16
}
fn default_linear_num_value_heads() -> usize {
    16
}
fn default_linear_key_head_dim() -> usize {
    128
}
fn default_linear_value_head_dim() -> usize {
    128
}
fn default_linear_conv_kernel_dim() -> usize {
    4
}

impl Qwen35Config {
    fn quant_config(&self) -> QuantConfig {
        match &self.quantization {
            Some(q) => QuantConfig {
                group_size: q.group_size,
                bits: q.bits,
            },
            None => QuantConfig::default(),
        }
    }

    fn rope_theta(&self) -> f32 {
        self.text_config
            .rope_parameters
            .as_ref()
            .map(|r| r.rope_theta)
            .unwrap_or(default_rope_theta())
    }

    fn partial_rotary_factor(&self) -> f32 {
        self.text_config
            .rope_parameters
            .as_ref()
            .and_then(|r| r.partial_rotary_factor)
            .unwrap_or(1.0)
    }

    fn tie_word_embeddings_effective(&self) -> bool {
        self.tie_word_embeddings || self.text_config.tie_word_embeddings
    }
}

struct FullAttention {
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

impl FullAttention {
    fn load(vb: &VarBuilder, cfg: &Qwen35Config) -> anyhow::Result<Self> {
        let qc = cfg.quant_config();
        let text = &cfg.text_config;
        let q_proj = Linear::new(&vb.pp("q_proj"), &qc)?;
        let k_proj = Linear::new(&vb.pp("k_proj"), &qc)?;
        let v_proj = Linear::new(&vb.pp("v_proj"), &qc)?;
        let o_proj = Linear::new(&vb.pp("o_proj"), &qc)?;

        let q_norm = if vb.pp("q_norm").contains("weight") {
            Some(RmsNorm::new(text.rms_norm_eps, &vb.pp("q_norm"))?)
        } else {
            None
        };
        let k_norm = if vb.pp("k_norm").contains("weight") {
            Some(RmsNorm::new(text.rms_norm_eps, &vb.pp("k_norm"))?)
        } else {
            None
        };

        let head_dim = text.head_dim;
        let rope_dim = ((head_dim as f32) * cfg.partial_rotary_factor()) as i32;
        let rope_dim = rope_dim.clamp(1, head_dim as i32);
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rope: RoPE::new(rope_dim, cfg.rope_theta(), false),
            kv_cache: KvCache::new(),
            num_heads: text.num_attention_heads,
            num_kv_heads: text.num_key_value_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        let shape = x.shape_raw();
        let (b, seq_len) = (shape[0], shape[1]);
        let offset = self.kv_cache.offset() as i32;

        let q_proj = self.q_proj.forward(x)?;
        let k_proj = self.k_proj.forward(x)?;
        let v_proj = self.v_proj.forward(x)?;

        let q_dim = (self.num_heads * self.head_dim) as i32;
        let kv_dim = (self.num_kv_heads * self.head_dim) as i32;
        let q_proj = q_proj.slice(&[0, 0, 0], &[b, seq_len, 2 * q_dim])?;
        let k_proj = k_proj.slice(&[0, 0, 0], &[b, seq_len, kv_dim])?;
        let v_proj = v_proj.slice(&[0, 0, 0], &[b, seq_len, kv_dim])?;

        // Qwen3.5 packs q_proj per-head as [q_head, gate_head] (not [all_q, all_gate]).
        let qg = q_proj.reshape(&[
            b,
            seq_len,
            self.num_heads as i32,
            (2 * self.head_dim) as i32,
        ])?;
        let q = qg.slice(
            &[0, 0, 0, 0],
            &[b, seq_len, self.num_heads as i32, self.head_dim as i32],
        )?;
        let gate = qg
            .slice(
                &[0, 0, 0, self.head_dim as i32],
                &[
                    b,
                    seq_len,
                    self.num_heads as i32,
                    (2 * self.head_dim) as i32,
                ],
            )?
            .reshape(&[b, seq_len, q_dim])?;
        let q = q.reshape(&[b, seq_len, self.num_heads as i32, self.head_dim as i32])?;
        let q = match &self.q_norm {
            Some(n) => n.forward(&q)?,
            None => q,
        }
        .transpose_axes(&[0, 2, 1, 3])?;

        let k = k_proj.reshape(&[b, seq_len, self.num_kv_heads as i32, self.head_dim as i32])?;
        let k = match &self.k_norm {
            Some(n) => n.forward(&k)?,
            None => k,
        }
        .transpose_axes(&[0, 2, 1, 3])?;

        let v = v_proj
            .reshape(&[b, seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let q = self.rope.forward(&q, offset)?;
        let k = self.rope.forward(&k, offset)?;
        let (k, v) = self.kv_cache.update(&k, &v)?;

        let n_rep = (self.num_heads / self.num_kv_heads).max(1);
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;

        let mask_mode = if seq_len > 1 { "causal" } else { "" };
        let attn = q.fast_scaled_dot_product_attention(&k, &v, self.scale, mask_mode, None)?;

        let attn = attn.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            b,
            seq_len,
            (self.num_heads * self.head_dim) as i32,
        ])?;
        self.o_proj.forward(&attn.multiply(&gate.sigmoid()?)?)
    }

    fn clear_cache(&mut self) {
        self.kv_cache.reset();
    }
}

struct LinearAttention {
    in_proj_qkv: Linear,
    in_proj_a: Linear,
    in_proj_b: Linear,
    in_proj_z: Linear,
    out_proj: Linear,
    norm: RmsNorm,
    conv_weight: Array,
    conv_state: Option<Array>,
    ssm_state: Option<Array>,
    a_log: Array,
    dt_bias: Array,
    conv_kernel_size: usize,
    k_heads: usize,
    k_head_dim: usize,
    v_heads: usize,
    v_head_dim: usize,
}

impl LinearAttention {
    fn load(vb: &VarBuilder, cfg: &Qwen35Config) -> anyhow::Result<Self> {
        let qc = cfg.quant_config();
        let text = &cfg.text_config;
        let k_heads = text.linear_num_key_heads;
        let k_head_dim = text.linear_key_head_dim;
        let v_heads = text.linear_num_value_heads;
        let v_head_dim = text.linear_value_head_dim;
        let mut conv_weight = vb.pp("conv1d").get("weight")?;
        let wshape = conv_weight.shape_raw();
        if wshape.len() == 3 && wshape[2] > wshape[1] {
            conv_weight = conv_weight.transpose_axes(&[0, 2, 1])?;
        }
        Ok(Self {
            in_proj_qkv: Linear::new(&vb.pp("in_proj_qkv"), &qc)?,
            in_proj_a: Linear::new(&vb.pp("in_proj_a"), &qc)?,
            in_proj_b: Linear::new(&vb.pp("in_proj_b"), &qc)?,
            in_proj_z: Linear::new(&vb.pp("in_proj_z"), &qc)?,
            out_proj: Linear::new(&vb.pp("out_proj"), &qc)?,
            norm: RmsNorm::new(text.rms_norm_eps, &vb.pp("norm"))?,
            conv_weight,
            conv_state: None,
            ssm_state: None,
            a_log: vb.get("A_log")?,
            dt_bias: vb.get("dt_bias")?,
            conv_kernel_size: text.linear_conv_kernel_dim.max(1),
            k_heads,
            k_head_dim,
            v_heads,
            v_head_dim,
        })
    }

    fn softplus(x: &Array) -> Result<Array> {
        let one = Array::from_float(1.0)?;
        x.exp()?.add(&one)?.log()
    }

    fn repeat_heads(x: &Array, n_rep: usize) -> Result<Array> {
        if n_rep <= 1 {
            return Ok(x.clone());
        }
        // x: [B, T, H, D] -> repeat H
        let x_t = x.transpose_axes(&[0, 2, 1, 3])?;
        let y_t = repeat_kv(&x_t, n_rep)?;
        y_t.transpose_axes(&[0, 2, 1, 3])
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        let shape = x.shape_raw();
        let (b, seq_len) = (shape[0], shape[1]);
        let key_dim = (self.k_heads * self.k_head_dim) as i32;
        let value_dim = (self.v_heads * self.v_head_dim) as i32;
        let conv_dim = key_dim * 2 + value_dim;

        // qkv + z + b + a projections (qwen3_5 GatedDeltaNet layout)
        let mixed_qkv = self.in_proj_qkv.forward(x)?;
        let z = self.in_proj_z.forward(x)?;
        let b_proj = self.in_proj_b.forward(x)?;
        let a_proj = self.in_proj_a.forward(x)?;
        let mut mixed = mixed_qkv;
        let n_keep = self.conv_kernel_size.saturating_sub(1) as i32;
        if let Some(prev) = &self.conv_state {
            mixed = Array::concatenate(&[prev, &mixed], 1)?;
        } else if n_keep > 0 {
            let zeros = Array::zeros(&[b, n_keep, conv_dim], mixed.dtype())?;
            mixed = Array::concatenate(&[&zeros, &mixed], 1)?;
        }
        if n_keep > 0 {
            let total = mixed.shape_raw()[1];
            let next_conv_state = mixed.slice(&[0, total - n_keep, 0], &[b, total, conv_dim])?;
            next_conv_state.eval()?;
            self.conv_state = Some(next_conv_state);
        }
        let conv = mixed.conv1d(&self.conv_weight, 1, 0, 1, conv_dim)?;
        let conv = conv.multiply(&conv.sigmoid()?)?;

        let q = conv.slice(&[0, 0, 0], &[b, seq_len, key_dim])?.reshape(&[
            b,
            seq_len,
            self.k_heads as i32,
            self.k_head_dim as i32,
        ])?;
        let k = conv
            .slice(&[0, 0, key_dim], &[b, seq_len, 2 * key_dim])?
            .reshape(&[b, seq_len, self.k_heads as i32, self.k_head_dim as i32])?;
        let v = conv
            .slice(&[0, 0, 2 * key_dim], &[b, seq_len, 2 * key_dim + value_dim])?
            .reshape(&[b, seq_len, self.v_heads as i32, self.v_head_dim as i32])?;

        // q/k normalization and scaling
        let inv_scale = (self.k_head_dim as f32).powf(-0.5);
        let q = q
            .fast_rms_norm(&Array::ones(&[self.k_head_dim as i32], q.dtype())?, 1e-6)?
            .multiply(&Array::from_float(inv_scale * inv_scale)?)?;
        let k = k
            .fast_rms_norm(&Array::ones(&[self.k_head_dim as i32], k.dtype())?, 1e-6)?
            .multiply(&Array::from_float(inv_scale)?)?;

        // Compute beta and decay g; match mlx_lm.models.gated_delta.compute_g:
        // g = exp(-exp(A_log.float32) * softplus(a + dt_bias)).astype(A_log.dtype)
        let beta = b_proj.sigmoid()?; // [B,S,Hv]
        let a = a_proj.as_type(mlx_core::DType::Float32)?; // [B,S,Hv]
        let dt_bias = self.dt_bias.as_type(mlx_core::DType::Float32)?.reshape(&[
            1,
            1,
            self.v_heads as i32,
        ])?;
        let a_log_exp = self.a_log.as_type(mlx_core::DType::Float32)?.exp()?;
        let g_pre = a.add(&dt_bias)?;
        let g_sp = Self::softplus(&g_pre)?;
        let g = g_sp
            .multiply(&a_log_exp.reshape(&[1, 1, self.v_heads as i32])?)?
            .multiply(&Array::from_float(-1.0)?)?
            .exp()?
            .as_type(x.dtype())?;

        // state: [B, Hv, Dv, Dk]
        let mut state = if let Some(s) = &self.ssm_state {
            s.clone()
        } else {
            Array::zeros(
                &[
                    b,
                    self.v_heads as i32,
                    self.v_head_dim as i32,
                    self.k_head_dim as i32,
                ],
                x.dtype(),
            )?
        };

        // Repeat q/k heads to Hv if needed.
        let rep = self.v_heads / self.k_heads;
        let q = Self::repeat_heads(&q, rep)?;
        let k = Self::repeat_heads(&k, rep)?;

        let mut ys = Vec::with_capacity(seq_len as usize);
        for t in 0..(seq_len as usize) {
            let t0 = t as i32;
            let t1 = (t + 1) as i32;
            let q_t = q
                .slice(
                    &[0, t0, 0, 0],
                    &[b, t1, self.v_heads as i32, self.k_head_dim as i32],
                )?
                .squeeze(1)?;
            let k_t = k
                .slice(
                    &[0, t0, 0, 0],
                    &[b, t1, self.v_heads as i32, self.k_head_dim as i32],
                )?
                .squeeze(1)?;
            let v_t = v
                .slice(
                    &[0, t0, 0, 0],
                    &[b, t1, self.v_heads as i32, self.v_head_dim as i32],
                )?
                .squeeze(1)?;
            let g_t = g
                .slice(&[0, t0, 0], &[b, t1, self.v_heads as i32])?
                .squeeze(1)?
                .reshape(&[b, self.v_heads as i32, 1, 1])?;
            let beta_t = beta
                .slice(&[0, t0, 0], &[b, t1, self.v_heads as i32])?
                .squeeze(1)?
                .reshape(&[b, self.v_heads as i32, 1])?;

            state = state.multiply(&g_t)?;
            let kv_mem = state
                .multiply(&k_t.reshape(&[b, self.v_heads as i32, 1, self.k_head_dim as i32])?)?
                .sum_axis(-1, false)?;
            let delta = v_t
                .add(&kv_mem.multiply(&Array::from_float(-1.0)?)?)?
                .multiply(&beta_t)?;
            state = state.add(
                &k_t.reshape(&[b, self.v_heads as i32, 1, self.k_head_dim as i32])?
                    .multiply(&delta.reshape(&[
                        b,
                        self.v_heads as i32,
                        self.v_head_dim as i32,
                        1,
                    ])?)?,
            )?;
            let y_t = state
                .multiply(&q_t.reshape(&[b, self.v_heads as i32, 1, self.k_head_dim as i32])?)?
                .sum_axis(-1, false)?;
            ys.push(y_t.reshape(&[b, 1, self.v_heads as i32, self.v_head_dim as i32])?);
        }
        let y_refs: Vec<&Array> = ys.iter().collect();
        let y = Array::concatenate(&y_refs, 1)?; // [B,S,Hv,Dv]
        state.eval()?;
        self.ssm_state = Some(state);

        let z = z.reshape(&[b, seq_len, self.v_heads as i32, self.v_head_dim as i32])?;
        let mut y = self.norm.forward(&y)?;
        // swiglu(gate=z, x=y): silu(z) * y
        y = y.multiply(&z.multiply(&z.sigmoid()?)?)?;
        self.out_proj.forward(&y.reshape(&[b, seq_len, value_dim])?)
    }

    fn clear_cache(&mut self) {
        self.conv_state = None;
        self.ssm_state = None;
    }
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(vb: &VarBuilder, cfg: &Qwen35Config) -> anyhow::Result<Self> {
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

enum LayerOp {
    Full(FullAttention),
    Linear(LinearAttention),
}

impl LayerOp {
    fn forward(&mut self, x: &Array) -> Result<Array> {
        match self {
            Self::Full(a) => a.forward(x),
            Self::Linear(a) => a.forward(x),
        }
    }

    fn clear_cache(&mut self) {
        match self {
            Self::Full(a) => a.clear_cache(),
            Self::Linear(a) => a.clear_cache(),
        }
    }
}

struct Qwen35Layer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    op: LayerOp,
    mlp: Mlp,
}

impl Qwen35Layer {
    fn load(idx: usize, vb: &VarBuilder, cfg: &Qwen35Config) -> anyhow::Result<Self> {
        let layer_type = cfg
            .text_config
            .layer_types
            .get(idx)
            .map(|s| s.as_str())
            .unwrap_or("linear_attention");
        let op = if layer_type == "full_attention" {
            LayerOp::Full(FullAttention::load(&vb.pp("self_attn"), cfg)?)
        } else {
            LayerOp::Linear(LinearAttention::load(&vb.pp("linear_attn"), cfg)?)
        };
        Ok(Self {
            input_layernorm: RmsNorm::new(cfg.text_config.rms_norm_eps, &vb.pp("input_layernorm"))?,
            post_attention_layernorm: RmsNorm::new(
                cfg.text_config.rms_norm_eps,
                &vb.pp("post_attention_layernorm"),
            )?,
            op,
            mlp: Mlp::load(&vb.pp("mlp"), cfg)?,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let h = self.op.forward(&h)?;
        let x = residual.add(&h)?;

        let residual = x.clone();
        let h = self.post_attention_layernorm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        residual.add(&h)
    }

    fn clear_cache(&mut self) {
        self.op.clear_cache();
    }
}

pub struct Qwen35 {
    embed_tokens: Embedding,
    layers: Vec<Qwen35Layer>,
    norm: RmsNorm,
    lm_head: Linear,
}

impl Qwen35 {
    pub fn new(cfg: &Qwen35Config, vb: &VarBuilder) -> anyhow::Result<Self> {
        let qc = cfg.quant_config();
        let language_vb = vb.pp("language_model");
        let model_vb = language_vb.pp("model");

        let embed_tokens = Embedding::new(&model_vb.pp("embed_tokens"), &qc)?;
        let mut layers = Vec::with_capacity(cfg.text_config.num_hidden_layers);
        for i in 0..cfg.text_config.num_hidden_layers {
            layers.push(Qwen35Layer::load(
                i,
                &model_vb.pp(format!("layers.{i}")),
                cfg,
            )?);
        }

        let norm = RmsNorm::new(cfg.text_config.rms_norm_eps, &model_vb.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings_effective() {
            embed_tokens.as_linear()
        } else {
            Linear::new(&language_vb.pp("lm_head"), &qc)?
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
