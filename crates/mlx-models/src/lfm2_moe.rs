use mlx_core::{Array, Module, Result};
use mlx_nn::{
    Embedding, KvCache, Linear, QuantConfig, RmsNorm, RoPE, VarBuilder, repeat_kv,
};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Lfm2MoeConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: Option<usize>,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    pub layer_types: Option<Vec<String>>,
    #[serde(default = "default_conv_l_cache", rename = "conv_L_cache")]
    pub conv_l_cache: usize,
    #[serde(default)]
    pub conv_bias: bool,
    pub num_dense_layers: Option<usize>,
    pub num_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub norm_topk_prob: bool,
    pub quantization: Option<super::llama::QuantizationConfig>,
}

fn default_norm_eps() -> f32 { 1e-5 }
fn default_rope_theta() -> f32 { 10_000.0 }
fn default_max_pos() -> usize { 131072 }
fn default_conv_l_cache() -> usize { 3 }

impl Lfm2MoeConfig {
    fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    fn quant_config(&self) -> QuantConfig {
        match &self.quantization {
            Some(q) => QuantConfig {
                group_size: q.group_size,
                bits: q.bits,
            },
            None => QuantConfig::default(),
        }
    }

    fn num_experts(&self) -> usize {
        self.num_experts.unwrap_or(8)
    }

    fn num_experts_per_tok(&self) -> usize {
        self.num_experts_per_tok.unwrap_or(2)
    }

    fn layer_is_attention(&self, idx: usize) -> bool {
        match &self.layer_types {
            Some(v) => v
                .get(idx)
                .map(|s| s.eq_ignore_ascii_case("full_attention"))
                .unwrap_or(false),
            None => false,
        }
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
    fn load(vb: &VarBuilder, cfg: &Lfm2MoeConfig) -> anyhow::Result<Self> {
        let qc = cfg.quant_config();
        let q_proj = Linear::new(&vb.pp("q_proj"), &qc)?;
        let k_proj = Linear::new(&vb.pp("k_proj"), &qc)?;
        let v_proj = Linear::new(&vb.pp("v_proj"), &qc)?;
        let o_proj = if vb.pp("out_proj").contains("weight") {
            Linear::new(&vb.pp("out_proj"), &qc)?
        } else {
            Linear::new(&vb.pp("o_proj"), &qc)?
        };

        let q_norm = if vb.pp("q_layernorm").contains("weight") {
            Some(RmsNorm::new(cfg.norm_eps, &vb.pp("q_layernorm"))?)
        } else {
            None
        };
        let k_norm = if vb.pp("k_layernorm").contains("weight") {
            Some(RmsNorm::new(cfg.norm_eps, &vb.pp("k_layernorm"))?)
        } else {
            None
        };

        let head_dim = cfg.head_dim();
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rope: RoPE::new(head_dim as i32, cfg.rope_theta, false),
            kv_cache: KvCache::new(),
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_kv_heads(),
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        let shape = x.shape_raw();
        let (b, seq_len) = (shape[0], shape[1]);
        let offset = self.kv_cache.offset() as i32;

        let q = self
            .q_proj
            .forward(x)?
            .reshape(&[b, seq_len, self.num_heads as i32, self.head_dim as i32])?;
        let q = match &self.q_norm {
            Some(n) => n.forward(&q)?,
            None => q,
        }
        .transpose_axes(&[0, 2, 1, 3])?;

        let k = self
            .k_proj
            .forward(x)?
            .reshape(&[b, seq_len, self.num_kv_heads as i32, self.head_dim as i32])?;
        let k = match &self.k_norm {
            Some(n) => n.forward(&k)?,
            None => k,
        }
        .transpose_axes(&[0, 2, 1, 3])?;

        let v = self
            .v_proj
            .forward(x)?
            .reshape(&[b, seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let q = self.rope.forward(&q, offset)?;
        let k = self.rope.forward(&k, offset)?;
        let (k, v) = self.kv_cache.update(&k, &v)?;

        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;

        let mask_mode = if seq_len > 1 { "causal" } else { "" };
        let attn = q.fast_scaled_dot_product_attention(&k, &v, self.scale, mask_mode, None)?;

        let attn = attn
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, seq_len, (self.num_heads * self.head_dim) as i32])?;
        self.o_proj.forward(&attn)
    }

    fn clear_cache(&mut self) {
        self.kv_cache.reset();
    }
}

struct ShortConv {
    in_proj: Linear,
    out_proj: Linear,
    conv_weight: Array,
    conv_bias: Option<Array>,
    hidden_size: usize,
    l_cache: usize,
    state: Option<Array>,
}

impl ShortConv {
    fn load(vb: &VarBuilder, cfg: &Lfm2MoeConfig) -> anyhow::Result<Self> {
        let qc = cfg.quant_config();
        let mut conv_weight = vb.pp("conv").get("weight")?;
        let wshape = conv_weight.shape_raw();
        if wshape.len() == 3 && wshape[2] > wshape[1] {
            conv_weight = conv_weight.transpose_axes(&[0, 2, 1])?;
        }
        let conv_bias = if cfg.conv_bias && vb.pp("conv").contains("bias") {
            Some(vb.pp("conv").get("bias")?)
        } else {
            None
        };
        Ok(Self {
            in_proj: Linear::new(&vb.pp("in_proj"), &qc)?,
            out_proj: Linear::new(&vb.pp("out_proj"), &qc)?,
            conv_weight,
            conv_bias,
            hidden_size: cfg.hidden_size,
            l_cache: cfg.conv_l_cache.max(1),
            state: None,
        })
    }

    fn split_last_three(&self, x: &Array) -> Result<(Array, Array, Array)> {
        let shape = x.shape_raw();
        let (b, l, h) = (shape[0], shape[1], self.hidden_size as i32);
        let a = x.slice(&[0, 0, 0], &[b, l, h])?;
        let b_part = x.slice(&[0, 0, h], &[b, l, 2 * h])?;
        let c = x.slice(&[0, 0, 2 * h], &[b, l, 3 * h])?;
        Ok((a, b_part, c))
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        let shape = x.shape_raw();
        let bsz = shape[0];
        let h = self.hidden_size as i32;
        let n_keep = self.l_cache.saturating_sub(1) as i32;

        let bcx = self.in_proj.forward(x)?;
        let (b_proj, c_proj, x_proj) = self.split_last_three(&bcx)?;
        let mut bx = b_proj.multiply(&x_proj)?;

        if let Some(prev) = &self.state {
            bx = Array::concatenate(&[prev, &bx], 1)?;
        } else if n_keep > 0 {
            let zeros = Array::zeros(&[bsz, n_keep, h], bx.dtype())?;
            bx = Array::concatenate(&[&zeros, &bx], 1)?;
        }

        if n_keep > 0 {
            let total = bx.shape_raw()[1];
            let start = (total - n_keep).max(0);
            self.state = Some(bx.slice(&[0, start, 0], &[bsz, total, h])?);
        }

        let mut conv_out = bx.conv1d(&self.conv_weight, 1, 0, 1, self.hidden_size as i32)?;
        if let Some(bias) = &self.conv_bias {
            conv_out = conv_out.add(bias)?;
        }
        let y = c_proj.multiply(&conv_out)?;
        self.out_proj.forward(&y)
    }

    fn clear_cache(&mut self) {
        self.state = None;
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

struct ExpertLinear {
    weight: Array,
    scales: Option<Array>,
    biases: Option<Array>,
    group_size: i32,
    bits: i32,
}

impl ExpertLinear {
    fn load(vb: &VarBuilder, cfg: &Lfm2MoeConfig) -> anyhow::Result<Self> {
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
        let qc = cfg.quant_config();
        let bits = if let Some(ref s) = scales {
            infer_bits(&weight.shape_raw(), &s.shape_raw(), qc.group_size, qc.bits)
        } else {
            0
        };
        Ok(Self {
            weight,
            scales,
            biases,
            group_size: qc.group_size,
            bits,
        })
    }

    fn forward_selected(&self, x: &Array, expert_indices: &Array) -> Result<Array> {
        if let Some(scales) = &self.scales {
            let x3 = x.reshape(&[x.shape_raw()[0], 1, x.shape_raw()[1]])?;
            let out = x3.gather_qmm(
                &self.weight,
                scales,
                self.biases.as_ref(),
                None,
                Some(expert_indices),
                true,
                self.group_size,
                self.bits,
                false,
            )?;
            return out.squeeze(1);
        }

        let n = x.shape_raw()[0] as usize;
        let ids = expert_indices.to_vec_i32()?;
        let mut rows = Vec::with_capacity(n);
        for (row, &eid) in ids.iter().enumerate().take(n) {
            let idx = Array::from_slice_i32(&[eid])?;
            let w = self.weight.take(&idx, 0)?.squeeze(0)?;
            let w_t = w.transpose()?;
            let x_row = x
                .slice(&[row as i32, 0], &[(row + 1) as i32, x.shape_raw()[1]])?;
            rows.push(x_row.matmul(&w_t)?);
        }
        let refs: Vec<&Array> = rows.iter().collect();
        Array::concatenate(&refs, 0)
    }
}

struct MoeFeedForward {
    router: Linear,
    gate: ExpertLinear,
    up: ExpertLinear,
    down: ExpertLinear,
    expert_bias: Option<Array>,
    num_experts: usize,
    top_k: usize,
    norm_topk_prob: bool,
}

impl MoeFeedForward {
    fn load(vb: &VarBuilder, cfg: &Lfm2MoeConfig) -> anyhow::Result<Self> {
        let qc = cfg.quant_config();
        Ok(Self {
            router: Linear::new(&vb.pp("gate"), &qc)?,
            gate: ExpertLinear::load(&vb.pp("switch_mlp").pp("gate_proj"), cfg)?,
            up: ExpertLinear::load(&vb.pp("switch_mlp").pp("up_proj"), cfg)?,
            down: ExpertLinear::load(&vb.pp("switch_mlp").pp("down_proj"), cfg)?,
            expert_bias: if vb.contains("expert_bias") {
                Some(vb.get("expert_bias")?)
            } else {
                None
            },
            num_experts: cfg.num_experts(),
            top_k: cfg.num_experts_per_tok(),
            norm_topk_prob: cfg.norm_topk_prob,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let shape = x.shape_raw();
        let hidden = shape[shape.len() - 1];
        let orig_shape = shape.clone();
        let flat = x.reshape(&[-1, hidden])?;

        let mut router_logits = self.router.forward(&flat)?;
        if let Some(expert_bias) = &self.expert_bias {
            router_logits = router_logits.add(expert_bias)?;
        }

        let router_probs = router_logits.softmax(-1)?.to_vec_f32()?;
        let num_tokens = flat.shape_raw()[0] as usize;
        let num_experts = self.num_experts;
        let k = self.top_k.min(num_experts).max(1);

        let mut routed_by_expert: Vec<Vec<(usize, f32)>> =
            (0..num_experts).map(|_| Vec::new()).collect();
        let mut expert_ids: Vec<usize> = (0..num_experts).collect();

        for tok in 0..num_tokens {
            let row = &router_probs[tok * num_experts..(tok + 1) * num_experts];
            expert_ids.sort_unstable_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
            let top = &expert_ids[..k];
            let norm = if self.norm_topk_prob {
                top.iter().map(|&idx| row[idx]).sum::<f32>()
            } else {
                1.0
            };
            let denom = if norm > 0.0 { norm } else { 1.0 };
            for &expert_idx in top {
                let prob = row[expert_idx];
                let weight = if self.norm_topk_prob {
                    prob / denom
                } else {
                    prob
                };
                routed_by_expert[expert_idx].push((tok, weight));
            }
        }

        let mut token_accum: Vec<Option<Array>> = (0..num_tokens).map(|_| None).collect();
        for (expert_idx, assignments) in routed_by_expert.iter().enumerate() {
            if assignments.is_empty() {
                continue;
            }
            let token_ids: Vec<i32> = assignments.iter().map(|(tok, _)| *tok as i32).collect();
            let token_idx = Array::from_slice_i32(&token_ids)?;
            let tok_x = flat.take(&token_idx, 0)?;
            let eidx = Array::from_slice_i32(&vec![expert_idx as i32; assignments.len()])?;

            let gate = self.gate.forward_selected(&tok_x, &eidx)?;
            let up = self.up.forward_selected(&tok_x, &eidx)?;
            let hidden = gate.multiply(&gate.sigmoid()?)?.multiply(&up)?;
            let expert_out = self.down.forward_selected(&hidden, &eidx)?;

            let out_hidden = expert_out.shape_raw()[1];
            let weights: Vec<f32> = assignments.iter().map(|(_, w)| *w).collect();
            let weight_arr = Array::from_slice_f32(&weights)?
                .reshape(&[assignments.len() as i32, 1])?
                .as_type(expert_out.dtype())?;
            let weighted = expert_out.multiply(&weight_arr)?;

            for (i, (tok, _)) in assignments.iter().enumerate() {
                let contrib = weighted
                    .slice(&[i as i32, 0], &[(i + 1) as i32, out_hidden])?
                    .reshape(&[1, out_hidden])?;
                if let Some(prev) = token_accum[*tok].as_ref() {
                    token_accum[*tok] = Some(prev.add(&contrib)?);
                } else {
                    token_accum[*tok] = Some(contrib);
                }
            }
        }

        let mut rows = Vec::with_capacity(num_tokens);
        for token_contrib in token_accum {
            match token_contrib {
                Some(v) => rows.push(v),
                None => {
                    return Err(mlx_core::Error::Message(
                        "MoE router produced no assignments".into(),
                    ));
                }
            }
        }
        let refs: Vec<&Array> = rows.iter().collect();
        let out = Array::concatenate(&refs, 0)?;
        out.reshape(&orig_shape)
    }
}

struct DenseMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl DenseMlp {
    fn load(vb: &VarBuilder, cfg: &Lfm2MoeConfig) -> anyhow::Result<Self> {
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

enum LayerOperator {
    Attention(FullAttention),
    ShortConv(ShortConv),
}

impl LayerOperator {
    fn forward(&mut self, x: &Array) -> Result<Array> {
        match self {
            Self::Attention(a) => a.forward(x),
            Self::ShortConv(c) => c.forward(x),
        }
    }

    fn clear_cache(&mut self) {
        match self {
            Self::Attention(a) => a.clear_cache(),
            Self::ShortConv(c) => c.clear_cache(),
        }
    }
}

enum LayerFfn {
    Dense(DenseMlp),
    Moe(MoeFeedForward),
}

impl LayerFfn {
    fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Dense(m) => m.forward(x),
            Self::Moe(m) => m.forward(x),
        }
    }
}

struct Lfm2Layer {
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
    operator: LayerOperator,
    ffn: LayerFfn,
}

impl Lfm2Layer {
    fn load(idx: usize, vb: &VarBuilder, cfg: &Lfm2MoeConfig) -> anyhow::Result<Self> {
        let operator = if cfg.layer_is_attention(idx) {
            LayerOperator::Attention(FullAttention::load(&vb.pp("self_attn"), cfg)?)
        } else {
            LayerOperator::ShortConv(ShortConv::load(&vb.pp("conv"), cfg)?)
        };

        let is_moe = vb.pp("feed_forward").pp("switch_mlp").pp("gate_proj").contains("weight");
        let ffn = if is_moe {
            LayerFfn::Moe(MoeFeedForward::load(&vb.pp("feed_forward"), cfg)?)
        } else {
            LayerFfn::Dense(DenseMlp::load(&vb.pp("feed_forward"), cfg)?)
        };

        Ok(Self {
            operator_norm: RmsNorm::new(cfg.norm_eps, &vb.pp("operator_norm"))?,
            ffn_norm: RmsNorm::new(cfg.norm_eps, &vb.pp("ffn_norm"))?,
            operator,
            ffn,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        let residual = x.clone();
        let h = self.operator_norm.forward(x)?;
        let h = self.operator.forward(&h)?;
        let x = residual.add(&h)?;

        let residual = x.clone();
        let h = self.ffn_norm.forward(&x)?;
        let h = self.ffn.forward(&h)?;
        residual.add(&h)
    }

    fn clear_cache(&mut self) {
        self.operator.clear_cache();
    }
}

pub struct Lfm2Moe {
    embed_tokens: Embedding,
    layers: Vec<Lfm2Layer>,
    norm: RmsNorm,
    lm_head: Linear,
}

impl Lfm2Moe {
    pub fn new(cfg: &Lfm2MoeConfig, vb: &VarBuilder) -> anyhow::Result<Self> {
        let qc = cfg.quant_config();
        let model_vb = vb.pp("model");

        let embed_tokens = Embedding::new(&model_vb.pp("embed_tokens"), &qc)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Lfm2Layer::load(i, &model_vb.pp(format!("layers.{i}")), cfg)?);
        }

        let norm = if model_vb.pp("embedding_norm").contains("weight") {
            RmsNorm::new(cfg.norm_eps, &model_vb.pp("embedding_norm"))?
        } else {
            RmsNorm::new(cfg.norm_eps, &model_vb.pp("norm"))?
        };

        let lm_head = if vb.pp("lm_head").contains("weight") {
            Linear::new(&vb.pp("lm_head"), &qc)?
        } else {
            embed_tokens.as_linear()
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

        let mut h = self.embed_tokens.forward(input_ids)?;
        for layer in &mut self.layers {
            h = layer.forward(&h)?;
            h.eval()?;
        }
        h = self.norm.forward(&h)?;

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

    pub fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }
}
