//! Qwen3-MoE (Mixture of Experts) model implementation.

use mlx_core::{Array, Module, Result};
use mlx_nn::{
    Embedding, KvCache, Linear, QuantConfig, RmsNorm, RoPE, RopeScaling, VarBuilder,
};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

// ------------------------------------------------------------------
// Config
// ------------------------------------------------------------------

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen3MoeConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: Option<usize>,
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
    pub head_dim: Option<usize>,
    // MoE-specific
    pub num_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub norm_topk_prob: bool,
    // Which layers are MoE (if not all)
    pub decoder_sparse_step: Option<usize>,
    // Shared expert
    pub shared_expert_intermediate_size: Option<usize>,
    pub quantization: Option<super::llama::QuantizationConfig>,
}

fn default_eps() -> f32 { 1e-6 }
fn default_rope_theta() -> f32 { 1_000_000.0 }
fn default_max_pos() -> usize { 32768 }

#[derive(Debug, Clone, Default)]
pub struct MoeProfileStats {
    pub router_host_s: f64,
    pub routing_build_s: f64,
    pub expert_forward_s: f64,
    pub shared_expert_s: f64,
    pub single_token_fast_path_hits: usize,
}

static MOE_PROFILE_STATS: OnceLock<Mutex<MoeProfileStats>> = OnceLock::new();

fn moe_profile_stats_store() -> &'static Mutex<MoeProfileStats> {
    MOE_PROFILE_STATS.get_or_init(|| Mutex::new(MoeProfileStats::default()))
}

fn trace_generation_enabled() -> bool {
    matches!(
        std::env::var("MLX_TRACE_GENERATION").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn with_moe_profile_stats_mut<F: FnOnce(&mut MoeProfileStats)>(f: F) {
    if !trace_generation_enabled() {
        return;
    }
    if let Ok(mut stats) = moe_profile_stats_store().lock() {
        f(&mut stats);
    }
}

pub fn reset_moe_profile_stats() {
    if let Ok(mut stats) = moe_profile_stats_store().lock() {
        *stats = MoeProfileStats::default();
    }
}

pub fn moe_profile_stats() -> MoeProfileStats {
    moe_profile_stats_store()
        .lock()
        .map(|stats| stats.clone())
        .unwrap_or_default()
}

impl Qwen3MoeConfig {
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn quant_config(&self) -> QuantConfig {
        match &self.quantization {
            Some(q) => QuantConfig { group_size: q.group_size, bits: q.bits },
            None => QuantConfig::default(),
        }
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts.unwrap_or(8)
    }

    pub fn num_experts_per_tok(&self) -> usize {
        self.num_experts_per_tok.unwrap_or(2)
    }

    pub fn moe_intermediate_size(&self) -> usize {
        self.moe_intermediate_size.unwrap_or(self.intermediate_size)
    }

    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        match self.decoder_sparse_step {
            Some(step) if step > 0 => (layer_idx + 1) % step == 0,
            _ => true, // Default: all layers are MoE
        }
    }
}

// ------------------------------------------------------------------
// MoE Components
// ------------------------------------------------------------------

struct MoeGate {
    gate: Linear,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
}

impl MoeGate {
    fn load(vb: &VarBuilder, cfg: &Qwen3MoeConfig) -> anyhow::Result<Self> {
        let qc = cfg.quant_config();
        Ok(Self {
            gate: Linear::new(vb, &qc)?,
            num_experts_per_tok: cfg.num_experts_per_tok(),
            norm_topk_prob: cfg.norm_topk_prob,
        })
    }
}

struct MoeExpert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MoeExpert {
    fn load(vb: &VarBuilder, cfg: &Qwen3MoeConfig) -> anyhow::Result<Self> {
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

struct SparseMoeBlock {
    gate: MoeGate,
    experts: Vec<MoeExpert>,
    shared_expert: Option<SharedExpert>,
    shared_expert_gate: Option<Linear>,
}

struct SharedExpert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl SharedExpert {
    fn load(vb: &VarBuilder, cfg: &Qwen3MoeConfig) -> anyhow::Result<Self> {
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

impl SparseMoeBlock {
    fn load(vb: &VarBuilder, cfg: &Qwen3MoeConfig) -> anyhow::Result<Self> {
        let gate = MoeGate::load(&vb.pp("gate"), cfg)?;

        let mut experts = Vec::with_capacity(cfg.num_experts());
        for i in 0..cfg.num_experts() {
            experts.push(MoeExpert::load(&vb.pp(format!("experts.{i}")), cfg)?);
        }
        let shared_expert = if cfg.shared_expert_intermediate_size.is_some() {
            Some(SharedExpert::load(&vb.pp("shared_expert"), cfg)?)
        } else {
            None
        };

        let shared_expert_gate = if vb.pp("shared_expert_gate").contains("weight") {
            Some(Linear::new(&vb.pp("shared_expert_gate"), &cfg.quant_config())?)
        } else {
            None
        };

        Ok(Self {
            gate,
            experts,
            shared_expert,
            shared_expert_gate,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let shape = x.shape_raw();
        let hidden = shape[shape.len() - 1];

        // Flatten to [tokens, hidden]
        let orig_shape = shape.clone();
        let flat = x.reshape(&[-1, hidden])?;

        let num_tokens = flat.shape_raw()[0] as usize;
        let num_experts = self.experts.len();
        let k = self.gate.num_experts_per_tok.min(num_experts).max(1);

        if num_tokens == 1 {
            with_moe_profile_stats_mut(|stats| {
                stats.single_token_fast_path_hits += 1;
            });
            let stage_t0 = Instant::now();
            let row_logits = self.gate.gate.forward(&flat)?.to_vec_f32()?;
            with_moe_profile_stats_mut(|stats| {
                stats.router_host_s += stage_t0.elapsed().as_secs_f64();
            });
            let row_logits = &row_logits[..num_experts];
            let row_probs = softmax_row(row_logits);
            let top = select_top_k(row_logits, k);
            let norm = if self.gate.norm_topk_prob {
                top.iter().map(|&(idx, _)| row_probs[idx]).sum::<f32>()
            } else {
                1.0
            };
            let denom = if norm > 0.0 { norm } else { 1.0 };
            let mut out: Option<Array> = None;
            for (expert_idx, _) in top {
                let prob = row_probs[expert_idx];
                let weight = if self.gate.norm_topk_prob { prob / denom } else { prob };
                let stage_t0 = Instant::now();
                let expert_out = self.experts[expert_idx].forward(&flat)?;
                with_moe_profile_stats_mut(|stats| {
                    stats.expert_forward_s += stage_t0.elapsed().as_secs_f64();
                });
                let weight_arr = Array::from_float(weight)?.as_type(expert_out.dtype())?;
                let contrib = expert_out.multiply(&weight_arr)?;
                out = Some(match out {
                    Some(prev) => prev.add(&contrib)?,
                    None => contrib,
                });
            }

            let mut out = out.ok_or_else(|| {
                mlx_core::Error::Message("MoE router produced no assignments".into())
            })?;

            if let Some(ref shared) = self.shared_expert {
                let stage_t0 = Instant::now();
                let mut shared_out = shared.forward(&flat)?;
                if let Some(ref shared_gate) = self.shared_expert_gate {
                    let gate = shared_gate.forward(&flat)?.sigmoid()?;
                    shared_out = shared_out.multiply(&gate)?;
                }
                with_moe_profile_stats_mut(|stats| {
                    stats.shared_expert_s += stage_t0.elapsed().as_secs_f64();
                });
                out = out.add(&shared_out)?;
            }

            return out.reshape(&orig_shape);
        }

        let stage_t0 = Instant::now();
        let router_logits = self.gate.gate.forward(&flat)?.to_vec_f32()?;
        with_moe_profile_stats_mut(|stats| {
            stats.router_host_s += stage_t0.elapsed().as_secs_f64();
        });
        let mut routed_by_expert: Vec<Vec<(usize, f32)>> =
            (0..num_experts).map(|_| Vec::new()).collect();

        for tok in 0..num_tokens {
            let row_logits = &router_logits[tok * num_experts..(tok + 1) * num_experts];
            let row_probs = softmax_row(row_logits);
            let top = select_top_k(row_logits, k);
            let norm = if self.gate.norm_topk_prob {
                top.iter().map(|&(idx, _)| row_probs[idx]).sum::<f32>()
            } else {
                1.0
            };
            let denom = if norm > 0.0 { norm } else { 1.0 };
            for &(expert_idx, _) in &top {
                let prob = row_probs[expert_idx];
                let weight = if self.gate.norm_topk_prob { prob / denom } else { prob };
                routed_by_expert[expert_idx].push((tok, weight));
            }
        }
        with_moe_profile_stats_mut(|stats| {
            stats.routing_build_s += stage_t0.elapsed().as_secs_f64();
        });

        let mut all_token_ids: Vec<i32> = Vec::with_capacity(num_tokens * k);
        let mut weighted_parts: Vec<Array> = Vec::new();
        for (expert_idx, assignments) in routed_by_expert.iter().enumerate() {
            if assignments.is_empty() {
                continue;
            }
            let token_ids: Vec<i32> = assignments.iter().map(|(tok, _)| *tok as i32).collect();
            let token_idx = Array::from_slice_i32(&token_ids)?;
            let tok_x = flat.take(&token_idx, 0)?;
            let stage_t0 = Instant::now();
            let expert_out = self.experts[expert_idx].forward(&tok_x)?;
            with_moe_profile_stats_mut(|stats| {
                stats.expert_forward_s += stage_t0.elapsed().as_secs_f64();
            });
            let expert_out = expert_out.reshape(&[assignments.len() as i32, hidden])?;

            let weights: Vec<f32> = assignments.iter().map(|(_, w)| *w).collect();
            let weight_arr = Array::from_slice_f32(&weights)?
                .reshape(&[assignments.len() as i32, 1])?
                .as_type(expert_out.dtype())?;
            let weighted = expert_out.multiply(&weight_arr)?;
            all_token_ids.extend(token_ids);
            weighted_parts.push(weighted);
        }

        if all_token_ids.len() != num_tokens * k {
            return Err(mlx_core::Error::Message(
                format!(
                    "MoE router assignment count mismatch: expected {}, got {}",
                    num_tokens * k,
                    all_token_ids.len()
                ),
            ));
        }
        let weighted_refs: Vec<&Array> = weighted_parts.iter().collect();
        let all_weighted = Array::concatenate(&weighted_refs, 0)?;

        let mut order: Vec<i32> = (0..all_token_ids.len() as i32).collect();
        order.sort_unstable_by_key(|&i| all_token_ids[i as usize]);
        let order_idx = Array::from_slice_i32(&order)?;
        let sorted = all_weighted.take(&order_idx, 0)?;
        let mut out = sorted
            .reshape(&[num_tokens as i32, k as i32, hidden])?
            .sum_axis(1, false)?;

        if let Some(ref shared) = self.shared_expert {
            let stage_t0 = Instant::now();
            let mut shared_out = shared.forward(&flat)?;
            if let Some(ref shared_gate) = self.shared_expert_gate {
                let gate = shared_gate.forward(&flat)?.sigmoid()?;
                shared_out = shared_out.multiply(&gate)?;
            }
            with_moe_profile_stats_mut(|stats| {
                stats.shared_expert_s += stage_t0.elapsed().as_secs_f64();
            });
            out = out.add(&shared_out)?;
        }

        out.reshape(&orig_shape)
    }
}

fn select_top_k(row: &[f32], k: usize) -> Vec<(usize, f32)> {
    debug_assert!(k > 0);
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(k);
    for (idx, &prob) in row.iter().enumerate() {
        let pos = top.iter().position(|&(cur_idx, cur_prob)| {
            prob > cur_prob || (prob == cur_prob && idx < cur_idx)
        });
        match pos {
            Some(insert_at) => {
                top.insert(insert_at, (idx, prob));
                if top.len() > k {
                    top.pop();
                }
            }
            None if top.len() < k => top.push((idx, prob)),
            None => {}
        }
    }
    top
}

fn softmax_row(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_v = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut exps = Vec::with_capacity(logits.len());
    let mut sum = 0.0f32;
    for &v in logits {
        let e = (v - max_v).exp();
        exps.push(e);
        sum += e;
    }
    if sum <= 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

// ------------------------------------------------------------------
// Dense MLP (for non-MoE layers)
// ------------------------------------------------------------------

struct DenseMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl DenseMlp {
    fn load(vb: &VarBuilder, cfg: &Qwen3MoeConfig) -> anyhow::Result<Self> {
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
// Feed-forward: either dense MLP or sparse MoE
// ------------------------------------------------------------------

enum FeedForward {
    Dense(DenseMlp),
    Moe(SparseMoeBlock),
}

impl FeedForward {
    fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            FeedForward::Dense(mlp) => mlp.forward(x),
            FeedForward::Moe(moe) => moe.forward(x),
        }
    }
}

// ------------------------------------------------------------------
// Attention
// ------------------------------------------------------------------

struct MoeAttention {
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

impl MoeAttention {
    fn load(vb: &VarBuilder, cfg: &Qwen3MoeConfig) -> anyhow::Result<Self> {
        let qc = cfg.quant_config();
        let q_proj = Linear::new(&vb.pp("q_proj"), &qc)?;
        let k_proj = Linear::new(&vb.pp("k_proj"), &qc)?;
        let v_proj = Linear::new(&vb.pp("v_proj"), &qc)?;
        let o_proj = Linear::new(&vb.pp("o_proj"), &qc)?;

        let q_norm = if vb.pp("q_norm").contains("weight") {
            Some(RmsNorm::new(cfg.rms_norm_eps, &vb.pp("q_norm"))?)
        } else { None };
        let k_norm = if vb.pp("k_norm").contains("weight") {
            Some(RmsNorm::new(cfg.rms_norm_eps, &vb.pp("k_norm"))?)
        } else { None };

        let head_dim = cfg.head_dim();
        let rope = match &cfg.rope_scaling {
            Some(scaling) => RoPE::with_scaling(head_dim as i32, cfg.rope_theta, false, scaling, cfg.max_position_embeddings)?,
            None => RoPE::new(head_dim as i32, cfg.rope_theta, false),
        };

        Ok(Self {
            q_proj, k_proj, v_proj, o_proj,
            q_norm, k_norm, rope,
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

        let q = self.q_proj.forward(x)?
            .reshape(&[b, seq_len, self.num_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = self.k_proj.forward(x)?
            .reshape(&[b, seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = self.v_proj.forward(x)?
            .reshape(&[b, seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let q = match &self.q_norm { Some(n) => n.forward(&q)?, None => q };
        let k = match &self.k_norm { Some(n) => n.forward(&k)?, None => k };

        let q = self.rope.forward(&q, offset)?;
        let k = self.rope.forward(&k, offset)?;

        let (k, v) = self.kv_cache.update(&k, &v)?;

        // MLX's SDPA handles grouped-query attention without explicit KV expansion.
        let mask_mode = if seq_len > 1 { "causal" } else { "" };
        let attn = q.fast_scaled_dot_product_attention(&k, &v, self.scale, mask_mode, None)?;

        let attn = attn.transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, seq_len, (self.num_heads * self.head_dim) as i32])?;

        self.o_proj.forward(&attn)
    }

    fn clear_cache(&mut self) { self.kv_cache.reset(); }
}

// ------------------------------------------------------------------
// Decoder Block
// ------------------------------------------------------------------

struct MoeBlock {
    attn: MoeAttention,
    ff: FeedForward,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl MoeBlock {
    fn load(layer_idx: usize, vb: &VarBuilder, cfg: &Qwen3MoeConfig) -> anyhow::Result<Self> {
        let ff = if cfg.is_moe_layer(layer_idx) {
            FeedForward::Moe(SparseMoeBlock::load(&vb.pp("mlp"), cfg)?)
        } else {
            FeedForward::Dense(DenseMlp::load(&vb.pp("mlp"), cfg)?)
        };

        Ok(Self {
            attn: MoeAttention::load(&vb.pp("self_attn"), cfg)?,
            ff,
            input_layernorm: RmsNorm::new(cfg.rms_norm_eps, &vb.pp("input_layernorm"))?,
            post_attention_layernorm: RmsNorm::new(cfg.rms_norm_eps, &vb.pp("post_attention_layernorm"))?,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let h = self.attn.forward(&h)?;
        let x = residual.add(&h)?;

        let residual = x.clone();
        let h = self.post_attention_layernorm.forward(&x)?;
        let h = self.ff.forward(&h)?;
        residual.add(&h)
    }

    fn clear_cache(&mut self) { self.attn.clear_cache(); }
}

// ------------------------------------------------------------------
// Qwen3Moe Model
// ------------------------------------------------------------------

/// Qwen3-MoE language model.
pub struct Qwen3Moe {
    embed_tokens: Embedding,
    layers: Vec<MoeBlock>,
    norm: RmsNorm,
    lm_head: Linear,
}

impl Qwen3Moe {
    pub fn new(cfg: &Qwen3MoeConfig, vb: &VarBuilder) -> anyhow::Result<Self> {
        let qc = cfg.quant_config();
        let model_vb = vb.pp("model");

        let embed_tokens = Embedding::new(&model_vb.pp("embed_tokens"), &qc)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(MoeBlock::load(i, &model_vb.pp(format!("layers.{i}")), cfg)?);
        }

        let norm = RmsNorm::new(cfg.rms_norm_eps, &model_vb.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            embed_tokens.as_linear()
        } else {
            Linear::new(&vb.pp("lm_head"), &qc)?
        };

        Ok(Self { embed_tokens, layers, norm, lm_head })
    }

    pub fn forward(&mut self, input_ids: &Array) -> Result<Array> {
        let shape = input_ids.shape_raw();
        let seq_len = shape[shape.len() - 1];

        let mut h = self.embed_tokens.forward(input_ids)?;
        for layer in &mut self.layers {
            h = layer.forward(&h)?;
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
