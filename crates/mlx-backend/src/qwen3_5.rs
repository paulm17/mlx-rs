use std::collections::HashMap;

use crate::array::Array;
use crate::llama::{EmbeddingLayer, KvCache, LayerCache, LinearLayer, RmsNorm, RecurrentCache, resolve_weight_prefix};
use crate::model::Model;
use crate::ops;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Qwen3_5Config {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub max_position_embeddings: i32,
    pub tie_word_embeddings: bool,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,

    // Linear attention params
    pub linear_num_value_heads: i32,
    pub linear_num_key_heads: i32,
    pub linear_key_head_dim: i32,
    pub linear_value_head_dim: i32,
    pub linear_conv_kernel_dim: i32,

    // Layer types
    pub layer_types: Vec<String>,
    pub full_attention_interval: i32,

    // MoE params
    pub num_experts: i32,
    pub num_experts_per_tok: i32,
    pub shared_expert_intermediate_size: i32,
    pub moe_intermediate_size: i32,
    pub norm_topk_prob: bool,
    pub decoder_sparse_step: i32,
    pub mlp_only_layers: Vec<i32>,

    // Computed
    pub scale: f32,
    pub rope_dim: i32,
}

impl Qwen3_5Config {
    pub fn from_json(config: &serde_json::Value) -> anyhow::Result<Self> {
        // May have text_config nesting
        let c = config.get("text_config").unwrap_or(config);

        let get_i32 = |k: &str, d: i32| -> i32 { c.get(k).and_then(|v| v.as_i64()).unwrap_or(d as i64) as i32 };
        let get_f32 = |k: &str, d: f32| -> f32 { c.get(k).and_then(|v| v.as_f64()).unwrap_or(d as f64) as f32 };
        let get_bool = |k: &str, d: bool| -> bool { c.get(k).and_then(|v| v.as_bool()).unwrap_or(d) };

        let hidden_size = get_i32("hidden_size", 0);
        let num_hidden_layers = get_i32("num_hidden_layers", 0);
        let num_attention_heads = get_i32("num_attention_heads", 0);
        let head_dim = get_i32("head_dim", 0);
        let intermediate_size = get_i32("intermediate_size", 0);
        let num_key_value_heads = get_i32("num_key_value_heads", num_attention_heads);
        let rms_norm_eps = get_f32("rms_norm_eps", 1e-6);
        let vocab_size = get_i32("vocab_size", 0);
        let max_position_embeddings = get_i32("max_position_embeddings", 4096);
        let tie_word_embeddings = get_bool("tie_word_embeddings", false);

        // Rope parameters
        let mut rope_theta = get_f32("rope_theta", 100000.0);
        let mut partial_rotary_factor = get_f32("partial_rotary_factor", 0.25);
        if let Some(rp) = c.get("rope_parameters") {
            if let Some(t) = rp.get("rope_theta").and_then(|v| v.as_f64()) {
                if t > 0.0 { rope_theta = t as f32; }
            }
            if let Some(p) = rp.get("partial_rotary_factor").and_then(|v| v.as_f64()) {
                if p > 0.0 { partial_rotary_factor = p as f32; }
            }
        }
        if partial_rotary_factor <= 0.0 { partial_rotary_factor = 0.25; }

        // Linear attention params
        let linear_num_value_heads = get_i32("linear_num_value_heads", 0);
        let linear_num_key_heads = get_i32("linear_num_key_heads", 0);
        let linear_key_head_dim = get_i32("linear_key_head_dim", 0);
        let linear_value_head_dim = get_i32("linear_value_head_dim", 0);
        let linear_conv_kernel_dim = get_i32("linear_conv_kernel_dim", 4);

        // Layer types
        let layer_types: Vec<String> = c
            .get("layer_types")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();

        let full_attention_interval = get_i32("full_attention_interval", 4);

        // MoE params
        let num_experts = get_i32("num_experts", 0);
        let num_experts_per_tok = get_i32("num_experts_per_tok", 1);
        let shared_expert_intermediate_size = get_i32("shared_expert_intermediate_size", intermediate_size);
        let moe_intermediate_size = get_i32("moe_intermediate_size", intermediate_size);
        let norm_topk_prob = get_bool("norm_topk_prob", true);
        let decoder_sparse_step = get_i32("decoder_sparse_step", 1);
        let mlp_only_layers: Vec<i32> = c
            .get("mlp_only_layers")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_i64().map(|x| x as i32)).collect())
            .unwrap_or_default();

        if hidden_size <= 0 || num_attention_heads <= 0 {
            anyhow::bail!("invalid Qwen3.5 config: hidden_size={hidden_size} heads={num_attention_heads}");
        }

        let head_dim = if head_dim > 0 { head_dim } else { hidden_size / num_attention_heads };
        let scale = 1.0 / (head_dim as f32).sqrt();
        let rope_dim = ((head_dim as f32) * partial_rotary_factor) as i32;
        let rope_dim = rope_dim.clamp(1, head_dim);

        Ok(Self {
            hidden_size,
            num_hidden_layers,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            vocab_size,
            max_position_embeddings,
            tie_word_embeddings,
            rope_theta,
            partial_rotary_factor,
            linear_num_value_heads,
            linear_num_key_heads,
            linear_key_head_dim,
            linear_value_head_dim,
            linear_conv_kernel_dim,
            layer_types,
            full_attention_interval,
            num_experts,
            num_experts_per_tok,
            shared_expert_intermediate_size,
            moe_intermediate_size,
            norm_topk_prob,
            decoder_sparse_step,
            mlp_only_layers,
            scale,
            rope_dim,
        })
    }

    pub fn layer_is_linear(&self, layer: usize) -> bool {
        if self.layer_types.len() == self.num_hidden_layers as usize {
            let t = &self.layer_types[layer];
            return !t.contains("full");
        }
        if self.full_attention_interval <= 0 {
            return true;
        }
        ((layer as i32) + 1) % self.full_attention_interval != 0
    }

    pub fn layer_uses_moe(&self, layer: usize) -> bool {
        if self.num_experts <= 0 {
            return false;
        }
        if self.mlp_only_layers.contains(&(layer as i32)) {
            return false;
        }
        if self.decoder_sparse_step <= 1 {
            return true;
        }
        ((layer as i32) + 1) % self.decoder_sparse_step == 0
    }
}

// ---------------------------------------------------------------------------
// FullAttention — standard transformer attention with Q/K norms + sigmoid gate
// ---------------------------------------------------------------------------

pub struct Qwen3_5Attention {
    q_proj: Box<dyn LinearLayer>,
    k_proj: Box<dyn LinearLayer>,
    v_proj: Box<dyn LinearLayer>,
    o_proj: Box<dyn LinearLayer>,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
}

impl Qwen3_5Attention {
    pub fn forward(
        &self,
        x: &Array,
        kv: &mut KvCache,
        positions: &Array,
        cfg: &Qwen3_5Config,
    ) -> anyhow::Result<Array> {
        let b = x.dim(0)?;
        let l = x.dim(1)?;
        let n_heads = cfg.num_attention_heads as usize;
        let hd = cfg.head_dim as usize;

        // Q projection: output is [B, L, n_heads, head_dim * 2] (q + gate)
        let qg = self.q_proj.forward(x)?;
        let qg = ops::reshape(&qg, &[b, l, n_heads, hd * 2])?;
        // Split into q and gate
        let q = ops::slice_last_dim(&qg, 0, hd)?;
        let gate = ops::slice_last_dim(&qg, hd, hd * 2)?;
        let gate = ops::reshape(&gate, &[b, l, n_heads * hd])?;

        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = ops::reshape(&q, &[b, l, n_heads, hd])?;
        let q = ops::transpose(&q, &[0, 2, 1, 3])?;

        let k = ops::reshape(&k, &[b, l, cfg.num_key_value_heads as usize, hd])?;
        let k = ops::transpose(&k, &[0, 2, 1, 3])?;

        let v = ops::reshape(&v, &[b, l, cfg.num_key_value_heads as usize, hd])?;
        let v = ops::transpose(&v, &[0, 2, 1, 3])?;

        // Q/K norms
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // RoPE (only on rope_dim dimensions)
        let q = ops::fast_rope_dynamic(&q, cfg.rope_dim, false, Some(cfg.rope_theta), 1.0, positions, None)?;
        let k = ops::fast_rope_dynamic(&k, cfg.rope_dim, false, Some(cfg.rope_theta), 1.0, positions, None)?;

        // KV cache update
        let (k, v) = kv.update(&k, &v)?;

        // SDPA
        let sdpa_mode = if l > 1 { "causal" } else { "" };
        let out = ops::fast_sdpa(&q, &k, &v, cfg.scale, sdpa_mode, None)?;

        let out = ops::transpose(&out, &[0, 2, 1, 3])?;
        let out = ops::reshape(&out, &[b, l, n_heads * hd])?;

        // Sigmoid gate
        let gate_sigmoid = ops::sigmoid(&gate)?;
        let out = ops::multiply(&out, &gate_sigmoid)?;

        self.o_proj.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// GatedDeltaNet — linear attention with recurrent state
// ---------------------------------------------------------------------------

pub struct Qwen3_5GatedDeltaNet {
    in_proj_qkv: Option<Box<dyn LinearLayer>>,
    in_proj_z: Option<Box<dyn LinearLayer>>,
    in_proj_b: Option<Box<dyn LinearLayer>>,
    in_proj_a: Option<Box<dyn LinearLayer>>,
    in_proj_qkvz: Option<Box<dyn LinearLayer>>,
    in_proj_ba: Option<Box<dyn LinearLayer>>,
    out_proj: Box<dyn LinearLayer>,
    conv_weight: Array,
    norm_weight: Array,
    dt_bias: Array,
    #[allow(dead_code)]
    a_log: Array,
    a_exp: Array,
}

impl Qwen3_5GatedDeltaNet {
    pub fn forward(
        &self,
        x: &Array,
        rc: &mut RecurrentCache,
        cfg: &Qwen3_5Config,
    ) -> anyhow::Result<Array> {
        let b = x.dim(0)?;
        let l = x.dim(1)?;
        let nk = cfg.linear_num_key_heads as usize;
        let nv = cfg.linear_num_value_heads as usize;
        let dk = cfg.linear_key_head_dim as usize;
        let dv = cfg.linear_value_head_dim as usize;
        let conv_tail = cfg.linear_conv_kernel_dim as usize - 1;
        let v_per_k = nv / nk;

        // Project to QKV and BA, then conv1d, then split
        let key_dim = nk * dk;
        let value_dim = nv * dv;
        let conv_dim = 2 * key_dim + value_dim;

        // Get recurrent states
        let (conv_state, delta_state) = rc.get_states(b, x.dtype()?)?;

        #[allow(unused_assignments)]
        let mut qkv_out: Option<(Array, Array, Array, Array)> = None;

        let (z, beta, alpha) = if let (Some(qkv_proj), Some(z_proj), Some(b_proj), Some(a_proj)) =
            (&self.in_proj_qkv, &self.in_proj_z, &self.in_proj_b, &self.in_proj_a)
        {
            // Split path: conv on raw flat qkv (interleaved layout), then split
            let qkv = qkv_proj.forward(x)?;
            let z = z_proj.forward(x)?;
            let z = ops::reshape(&z, &[b, l, nv, dv])?;
            let beta = b_proj.forward(x)?;
            let alpha = a_proj.forward(x)?;

            // Causal conv1d on raw qkv
            let qkv = ops::reshape(&qkv, &[b, l, conv_dim])?;
            let conv_in = ops::concatenate(&[conv_state, &qkv], 1)?;
            let conv_out = depthwise_conv1d(&conv_in, &self.conv_weight, l as usize)?;
            let conv_out = ops::silu(&conv_out)?;

            // Update conv state (tail of conv_in)
            let total_len = conv_in.dim(1)?;
            let new_conv_state = ops::slice_axis1(&conv_in, total_len - conv_tail, total_len)?;

            // Flat slicing after conv: [all_q, all_k, all_v]
            let q_out = ops::slice_last_dim(&conv_out, 0, key_dim)?;
            let k_out = ops::slice_last_dim(&conv_out, key_dim, 2 * key_dim)?;
            let v_out = ops::slice_last_dim(&conv_out, 2 * key_dim, 2 * key_dim + value_dim)?;
            let q_out = ops::reshape(&q_out, &[b, l, nk, dk])?;
            let k_out = ops::reshape(&k_out, &[b, l, nk, dk])?;
            let v_out = ops::reshape(&v_out, &[b, l, nv, dv])?;

            // Store results in outer scope via closure workaround
            // We'll use a flag to pass q/k/v out
            qkv_out = Some((q_out, k_out, v_out, new_conv_state));
            (z, beta, alpha)
        } else if let (Some(qkvz_proj), Some(ba_proj)) = (&self.in_proj_qkvz, &self.in_proj_ba) {
            // Combined path: reshape+slice, reorder to [all_q, all_k, all_v], then conv
            let qkvz = qkvz_proj.forward(x)?;
            let ba = ba_proj.forward(x)?;
            let qkvz = ops::reshape(&qkvz, &[b, l, nk, 2 * dk + 2 * v_per_k * dv])?;
            let q = ops::slice_last_dim(&qkvz, 0, dk)?;
            let k = ops::slice_last_dim(&qkvz, dk, 2 * dk)?;
            let v_part = ops::slice_last_dim(&qkvz, 2 * dk, 2 * dk + v_per_k * dv)?;
            let z = ops::slice_last_dim(&qkvz, 2 * dk + v_per_k * dv, 2 * dk + 2 * v_per_k * dv)?;
            let v = ops::reshape(&v_part, &[b, l, nv, dv])?;
            let z = ops::reshape(&z, &[b, l, nv, dv])?;
            let ba = ops::reshape(&ba, &[b, l, nk, 2 * v_per_k])?;
            let beta = ops::slice_last_dim(&ba, 0, v_per_k)?;
            let alpha = ops::slice_last_dim(&ba, v_per_k, 2 * v_per_k)?;
            let beta = ops::reshape(&beta, &[b, l, nv])?;
            let alpha = ops::reshape(&alpha, &[b, l, nv])?;

            // Concatenate QKV for conv: [all_q, all_k, all_v]
            let q_flat = ops::reshape(&q, &[b, l, key_dim])?;
            let k_flat = ops::reshape(&k, &[b, l, key_dim])?;
            let v_flat = ops::reshape(&v, &[b, l, value_dim])?;
            let qkv = ops::concatenate(&[&q_flat, &k_flat, &v_flat], -1)?;
            let qkv = ops::reshape(&qkv, &[b, l, conv_dim])?;

            // Causal conv1d
            let conv_in = ops::concatenate(&[conv_state, &qkv], 1)?;
            let conv_out = depthwise_conv1d(&conv_in, &self.conv_weight, l as usize)?;
            let conv_out = ops::silu(&conv_out)?;

            // Update conv state (tail of conv_in)
            let total_len = conv_in.dim(1)?;
            let new_conv_state = ops::slice_axis1(&conv_in, total_len - conv_tail, total_len)?;

            // Flat slicing after conv
            let q_out = ops::slice_last_dim(&conv_out, 0, key_dim)?;
            let k_out = ops::slice_last_dim(&conv_out, key_dim, 2 * key_dim)?;
            let v_out = ops::slice_last_dim(&conv_out, 2 * key_dim, 2 * key_dim + value_dim)?;
            let q_out = ops::reshape(&q_out, &[b, l, nk, dk])?;
            let k_out = ops::reshape(&k_out, &[b, l, nk, dk])?;
            let v_out = ops::reshape(&v_out, &[b, l, nv, dv])?;

            qkv_out = Some((q_out, k_out, v_out, new_conv_state));
            (z, beta, alpha)
        } else {
            anyhow::bail!("GatedDeltaNet: missing projections");
        };

        let (q_out, k_out, v_out, new_conv_state) = qkv_out.take().unwrap();

        // RMSNorm + scale
        let inv_scale_k = 1.0 / (dk as f32).sqrt();
        let inv_scale_q = inv_scale_k * inv_scale_k;
        let q_out = ops::rms_norm_weightless(&q_out, 1e-6)?;
        let q_out = ops::multiply_scalar(&q_out, inv_scale_q)?;
        let k_out = ops::rms_norm_weightless(&k_out, 1e-6)?;
        let k_out = ops::multiply_scalar(&k_out, inv_scale_k)?;

        // Decay: softplus(alpha + dt_bias) * a_exp
        let alpha_plus_bias = ops::add(&alpha, &self.dt_bias)?;
        let decay = ops::softplus(&alpha_plus_bias)?;
        let decay = ops::multiply(&decay, &self.a_exp)?;
        let decay = ops::neg_exp(&decay)?;
        let decay = ops::astype(&decay, alpha.dtype()?)?;

        let beta_gate = ops::sigmoid(&beta)?;

        // Gated delta scan (sequential over time)
        let (out, new_delta_state) = gated_delta_scan(
            &q_out, &k_out, &v_out, &decay, &beta_gate, delta_state,
        )?;

        // Update recurrent state
        rc.put_states(new_conv_state, new_delta_state);

        // Output norm + gating
        let out = ops::fast_rms_norm(&out, &self.norm_weight, cfg.rms_norm_eps)?;
        let z_silu = ops::silu(&z)?;
        let out_dtype = out.dtype()?;
        let out_f32 = ops::astype(&out, crate::ffi::MlxDtype::Float32)?;
        let z_f32 = ops::astype(&z_silu, crate::ffi::MlxDtype::Float32)?;
        let out = ops::multiply(&out_f32, &z_f32)?;
        let out = ops::astype(&out, out_dtype)?;
        let out = ops::reshape(&out, &[b, l, value_dim])?;

        self.out_proj.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// DenseMLP — SwiGLU feed-forward
// ---------------------------------------------------------------------------

pub struct Qwen3_5DenseMlp {
    gate_proj: Box<dyn LinearLayer>,
    up_proj: Box<dyn LinearLayer>,
    down_proj: Box<dyn LinearLayer>,
}

impl Qwen3_5DenseMlp {
    pub fn forward(&self, x: &Array) -> anyhow::Result<Array> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let gated = ops::silu(&gate)?;
        let gated = ops::multiply(&gated, &up)?;
        self.down_proj.forward(&gated)
    }
}

// ---------------------------------------------------------------------------
// SparseMoE — expert routing with shared expert
// ---------------------------------------------------------------------------

pub struct Qwen3_5SparseMoe {
    gate: Box<dyn LinearLayer>,
    experts_gate: Vec<Box<dyn LinearLayer>>,
    experts_up: Vec<Box<dyn LinearLayer>>,
    experts_down: Vec<Box<dyn LinearLayer>>,
    shared_expert: Option<Qwen3_5DenseMlp>,
    shared_expert_gate: Option<Box<dyn LinearLayer>>,
    num_experts_per_tok: i32,
    norm_topk_prob: bool,
}

impl Qwen3_5SparseMoe {
    pub fn forward(&self, x: &Array, cfg: &Qwen3_5Config) -> anyhow::Result<Array> {
        let b = x.dim(0)?;
        let l = x.dim(1)?;
        let top_k = self.num_experts_per_tok as usize;

        // Gate logits and top-k selection
        let gate_logits = self.gate.forward(x)?;
        let probs = ops::softmax_axis(&gate_logits, -1, true)?;
        let neg_probs = ops::negative(&probs)?;
        let indices = ops::argpartition(&neg_probs, top_k as i32 - 1, -1)?;
        let indices = ops::slice_last_dim(&indices, 0, top_k)?;

        let scores = ops::take_along_axis(&probs, &indices, -1)?;
        let scores = if self.norm_topk_prob && top_k > 1 {
            let sum = ops::sum_axis(&scores, scores.ndim() - 1, true)?;
            ops::divide(&scores, &sum)?
        } else {
            scores
        };

        // Expert computation using gather_mm
        let x_flat = ops::reshape(&x, &[b * l, cfg.hidden_size as usize])?;
        let idx_flat = ops::reshape(&indices, &[b * l, top_k])?;

        let mut expert_outputs = Vec::with_capacity(top_k);
        for k in 0..top_k {
            let k_idx = ops::slice_last_dim(&idx_flat, k, k + 1)?;
            let gate_w = self.experts_gate[k].as_dense_weight()
                .ok_or_else(|| anyhow::anyhow!("MoE expert gate weights must be dense for gather_mm"))?;
            let up_w = self.experts_up[k].as_dense_weight()
                .ok_or_else(|| anyhow::anyhow!("MoE expert up weights must be dense for gather_mm"))?;
            let down_w = self.experts_down[k].as_dense_weight()
                .ok_or_else(|| anyhow::anyhow!("MoE expert down weights must be dense for gather_mm"))?;
            let gate_out = ops::gather_mm(&x_flat, gate_w, Some(&k_idx), None, true)?;
            let up_out = ops::gather_mm(&x_flat, up_w, Some(&k_idx), None, true)?;
            let gated = ops::silu(&gate_out)?;
            let gated = ops::multiply(&gated, &up_out)?;
            let down_out = ops::gather_mm(&gated, down_w, Some(&k_idx), None, true)?;
            expert_outputs.push(down_out);
        }

        // Weighted sum of expert outputs
        let mut y = ops::zeros(&[b * l, cfg.hidden_size as usize], x.dtype()?)?;
        for (k, expert_out) in expert_outputs.iter().enumerate() {
            let score_k = ops::slice_last_dim(&scores, k, k + 1)?;
            let score_k = ops::reshape(&score_k, &[b * l, 1])?;
            let weighted = ops::multiply(expert_out, &score_k)?;
            y = ops::add(&y, &weighted)?;
        }
        let y = ops::reshape(&y, &[b, l, cfg.hidden_size as usize])?;

        // Shared expert
        if let Some(ref shared) = self.shared_expert {
            let shared_out = shared.forward(x)?;
            if let Some(ref gate) = self.shared_expert_gate {
                let gate_val = ops::sigmoid(&gate.forward(x)?)?;
                let shared_out = ops::multiply(&shared_out, &gate_val)?;
                return ops::add(&y, &shared_out);
            }
            return ops::add(&y, &shared_out);
        }

        Ok(y)
    }
}

// ---------------------------------------------------------------------------
// MLP block enum (Dense or MoE)
// ---------------------------------------------------------------------------

pub enum Qwen3_5MlpBlock {
    Dense(Qwen3_5DenseMlp),
    Moe(Qwen3_5SparseMoe),
}

impl Qwen3_5MlpBlock {
    pub fn forward(&self, x: &Array, cfg: &Qwen3_5Config) -> anyhow::Result<Array> {
        match self {
            Qwen3_5MlpBlock::Dense(mlp) => mlp.forward(x),
            Qwen3_5MlpBlock::Moe(moe) => moe.forward(x, cfg),
        }
    }
}

// ---------------------------------------------------------------------------
// Qwen3_5Layer — decoder layer
// ---------------------------------------------------------------------------

pub enum Qwen3_5AttentionBlock {
    Full(Qwen3_5Attention),
    Linear(Qwen3_5GatedDeltaNet),
}

pub struct Qwen3_5Layer {
    attention: Qwen3_5AttentionBlock,
    mlp: Qwen3_5MlpBlock,
    input_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    is_linear: bool,
}

impl Qwen3_5Layer {
    pub fn forward(
        &self,
        x: &Array,
        cache: &mut LayerCache,
        positions: &Array,
        cfg: &Qwen3_5Config,
    ) -> anyhow::Result<Array> {
        let normed = self.input_norm.forward(x)?;

        let r = match (&self.attention, cache) {
            (Qwen3_5AttentionBlock::Full(attn), LayerCache::Attention(kv)) => {
                attn.forward(&normed, kv, positions, cfg)?
            }
            (Qwen3_5AttentionBlock::Linear(lin), LayerCache::Recurrent(rc)) => {
                lin.forward(&normed, rc, cfg)?
            }
            _ => anyhow::bail!("cache type mismatch for layer"),
        };

        let h = ops::add(x, &r)?;
        let normed = self.post_attention_norm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed, cfg)?;
        ops::add(&h, &mlp_out)
    }
}

// ---------------------------------------------------------------------------
// Qwen3_5Model
// ---------------------------------------------------------------------------

pub struct Qwen3_5Model {
    embed_tokens: Box<dyn EmbeddingLayer>,
    layers: Vec<Qwen3_5Layer>,
    norm: RmsNorm,
    lm_head: Box<dyn LinearLayer>,
    config: Qwen3_5Config,
}

impl Model for Qwen3_5Model {
    fn forward(
        &self,
        input_ids: &Array,
        caches: &mut [LayerCache],
        positions: &Array,
    ) -> anyhow::Result<Array> {
        let mut h = self.embed_tokens.forward(input_ids)?;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], positions, &self.config)?;
        }
        let h = self.norm.forward(&h)?;
        self.lm_head.forward(&h)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn max_position_embeddings(&self) -> i32 {
        self.config.max_position_embeddings
    }

    fn hidden_size(&self) -> i32 {
        self.config.hidden_size
    }

    fn vocab_size(&self) -> i32 {
        self.config.vocab_size
    }

    fn new_caches(&self) -> Vec<LayerCache> {
        self.layers
            .iter()
            .map(|layer| {
                if layer.is_linear {
                    let conv_tail = self.config.linear_conv_kernel_dim as usize - 1;
                    let nk = self.config.linear_num_key_heads as usize;
                    let dk = self.config.linear_key_head_dim as usize;
                    let nv = self.config.linear_num_value_heads as usize;
                    let dv = self.config.linear_value_head_dim as usize;
                    let conv_dim = 2 * nk * dk + nv * dv;
                    LayerCache::Recurrent(RecurrentCache::new(conv_tail, conv_dim, nv, dv, dk))
                } else {
                    LayerCache::Attention(KvCache::new())
                }
            })
            .collect()
    }
}

impl Qwen3_5Model {
    pub fn config(&self) -> &Qwen3_5Config {
        &self.config
    }

    pub fn load_from_tensors(
        mut tensors: HashMap<String, Array>,
        config: Qwen3_5Config,
    ) -> anyhow::Result<Self> {
        let prefix = resolve_weight_prefix(&tensors);

        // Embeddings and LM head (possibly quantized, possibly tied)
        let embed_key = format!("{prefix}model.embed_tokens.weight");
        let (embed_tokens, tied_lm_head) = make_embedding_and_linear(&mut tensors, &embed_key)?;

        // Final norm
        let norm_w = take(&mut tensors, &format!("{prefix}model.norm.weight"))?;
        let norm = RmsNorm::new(norm_w, config.rms_norm_eps);

        // LM head (tied to embeddings by default)
        let lm_head: Box<dyn LinearLayer> = if config.tie_word_embeddings {
            tied_lm_head
        } else if tensors.contains_key(&format!("{prefix}lm_head.weight")) {
            make_linear(&mut tensors, &format!("{prefix}lm_head.weight"))?
        } else {
            tied_lm_head
        };

        // Layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for i in 0..config.num_hidden_layers as usize {
            let lp = format!("{prefix}model.layers.{i}");
            let is_linear = config.layer_is_linear(i);
            let uses_moe = config.layer_uses_moe(i);

            // Norms
            let input_norm_w = take(&mut tensors, &format!("{lp}.input_layernorm.weight"))?;
            let post_attn_norm_w = take(&mut tensors, &format!("{lp}.post_attention_layernorm.weight"))?;
            let input_norm = RmsNorm::new(input_norm_w, config.rms_norm_eps);
            let post_attention_norm = RmsNorm::new(post_attn_norm_w, config.rms_norm_eps);

            // Attention block
            let attention = if is_linear {
                let lin = load_gated_delta_net(&mut tensors, &lp)?;
                Qwen3_5AttentionBlock::Linear(lin)
            } else {
                let attn = load_full_attention(&mut tensors, &lp)?;
                Qwen3_5AttentionBlock::Full(attn)
            };

            // MLP block
            let mlp = if uses_moe {
                let moe = load_sparse_moe(&mut tensors, &lp, &config)?;
                Qwen3_5MlpBlock::Moe(moe)
            } else {
                let dense = load_dense_mlp(&mut tensors, &lp)?;
                Qwen3_5MlpBlock::Dense(dense)
            };

            layers.push(Qwen3_5Layer {
                attention,
                mlp,
                input_norm,
                post_attention_norm,
                is_linear,
            });
        }

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config,
        })
    }
}

// ---------------------------------------------------------------------------
// Weight loading helpers
// ---------------------------------------------------------------------------

fn take(tensors: &mut HashMap<String, Array>, key: &str) -> anyhow::Result<Array> {
    tensors
        .remove(key)
        .ok_or_else(|| anyhow::anyhow!("missing tensor: {key}"))
}

fn infer_quant_params(weight_cols: i32, scale_cols: i32) -> (i32, i32, String) {
    if scale_cols == 0 {
        return (64, 4, "affine".to_string());
    }
    let group_size_4 = weight_cols * 8 / scale_cols;
    let group_size_8 = weight_cols * 4 / scale_cols;
    if group_size_4 == 32 {
        (32, 4, "mxfp4".to_string())
    } else if group_size_4 == 64 {
        (64, 4, "affine".to_string())
    } else if group_size_8 == 64 {
        (64, 8, "affine".to_string())
    } else if group_size_8 == 32 {
        (32, 8, "mxfp8".to_string())
    } else {
        (64, 4, "affine".to_string())
    }
}

fn make_linear(
    tensors: &mut HashMap<String, Array>,
    base_key: &str,
) -> anyhow::Result<Box<dyn LinearLayer>> {
    let base = base_key.strip_suffix(".weight").unwrap_or(base_key);
    let scale_key = format!("{base}_scale");
    if tensors.contains_key(&scale_key) {
        let weight = take(tensors, base_key)?;
        let scales = take(tensors, &scale_key)?;
        let qbias_key = format!("{base}_qbias");
        let biases = tensors.remove(&qbias_key);
        let w_cols = weight.dim(weight.ndim() - 1)? as i32;
        let s_cols = scales.dim(scales.ndim() - 1)? as i32;
        let (group_size, bits, mode) = infer_quant_params(w_cols, s_cols);
        Ok(Box::new(crate::llama::QuantizedLinear::new(
            weight, scales, biases,
            Some(group_size), Some(bits), mode,
        )))
    } else {
        let weight = take(tensors, base_key)?;
        Ok(Box::new(crate::llama::Linear::new(weight, None)))
    }
}

#[allow(dead_code)]
fn make_embedding(
    tensors: &mut HashMap<String, Array>,
    base_key: &str,
) -> anyhow::Result<Box<dyn EmbeddingLayer>> {
    let base = base_key.strip_suffix(".weight").unwrap_or(base_key);
    let scale_key = format!("{base}_scale");
    if tensors.contains_key(&scale_key) {
        let weight = take(tensors, base_key)?;
        let scales = take(tensors, &scale_key)?;
        let qbias_key = format!("{base}_qbias");
        let biases = tensors.remove(&qbias_key);
        let w_cols = weight.dim(weight.ndim() - 1)? as i32;
        let s_cols = scales.dim(scales.ndim() - 1)? as i32;
        let (group_size, bits, mode) = infer_quant_params(w_cols, s_cols);
        Ok(Box::new(crate::llama::QuantizedEmbedding::new(
            weight, scales, biases,
            Some(group_size), Some(bits), mode,
        )))
    } else {
        let weight = take(tensors, base_key)?;
        Ok(Box::new(crate::llama::Embedding::new(weight)))
    }
}

fn make_embedding_and_linear(
    tensors: &mut HashMap<String, Array>,
    base_key: &str,
) -> anyhow::Result<(Box<dyn EmbeddingLayer>, Box<dyn LinearLayer>)> {
    let base = base_key.strip_suffix(".weight").unwrap_or(base_key);
    let scale_key = format!("{base}_scale");
    if tensors.contains_key(&scale_key) {
        let weight = take(tensors, base_key)?;
        let scales = take(tensors, &scale_key)?;
        let qbias_key = format!("{base}_qbias");
        let biases = tensors.remove(&qbias_key);
        let w_cols = weight.dim(weight.ndim() - 1)? as i32;
        let s_cols = scales.dim(scales.ndim() - 1)? as i32;
        let (group_size, bits, mode) = infer_quant_params(w_cols, s_cols);
        let emb = Box::new(crate::llama::QuantizedEmbedding::new(
            weight.clone(), scales.clone(), biases.clone(),
            Some(group_size), Some(bits), mode.clone(),
        )) as Box<dyn EmbeddingLayer>;
        let lin = Box::new(crate::llama::QuantizedLinear::new(
            weight, scales, biases,
            Some(group_size), Some(bits), mode,
        )) as Box<dyn LinearLayer>;
        Ok((emb, lin))
    } else {
        let weight = take(tensors, base_key)?;
        let emb = Box::new(crate::llama::Embedding::new(weight.clone())) as Box<dyn EmbeddingLayer>;
        let lin = Box::new(crate::llama::Linear::new(weight, None)) as Box<dyn LinearLayer>;
        Ok((emb, lin))
    }
}

fn maybe_shift_norm(_weights: &HashMap<String, Array>, norm_w: Array, _prefix: &str) -> Array {
    // Ollama shifts norm weights (+1.0) for linear attention layers if the model has MTP keys
    let has_mtp = _weights.keys().any(|k: &String| k.contains("mtp."));
    if has_mtp {
        ops::add(&norm_w, &Array::from_f32(1.0).unwrap()).unwrap_or(norm_w)
    } else {
        norm_w
    }
}

fn load_full_attention(tensors: &mut HashMap<String, Array>, lp: &str) -> anyhow::Result<Qwen3_5Attention> {
    Ok(Qwen3_5Attention {
        q_proj: make_linear(tensors, &format!("{lp}.self_attn.q_proj.weight"))?,
        k_proj: make_linear(tensors, &format!("{lp}.self_attn.k_proj.weight"))?,
        v_proj: make_linear(tensors, &format!("{lp}.self_attn.v_proj.weight"))?,
        o_proj: make_linear(tensors, &format!("{lp}.self_attn.o_proj.weight"))?,
        q_norm: RmsNorm::new(take(tensors, &format!("{lp}.self_attn.q_norm.weight"))?, 1e-6),
        k_norm: RmsNorm::new(take(tensors, &format!("{lp}.self_attn.k_norm.weight"))?, 1e-6),
    })
}

fn load_gated_delta_net(tensors: &mut HashMap<String, Array>, lp: &str) -> anyhow::Result<Qwen3_5GatedDeltaNet> {
    let try_make = |tensors: &mut HashMap<String, Array>, key: &str| -> anyhow::Result<Option<Box<dyn LinearLayer>>> {
        if tensors.contains_key(key) {
            Ok(Some(make_linear(tensors, key)?))
        } else {
            Ok(None)
        }
    };

    let in_proj_qkv = try_make(tensors, &format!("{lp}.linear_attn.in_proj_qkv.weight"))?;
    let in_proj_z = try_make(tensors, &format!("{lp}.linear_attn.in_proj_z.weight"))?;
    let in_proj_b = try_make(tensors, &format!("{lp}.linear_attn.in_proj_b.weight"))?;
    let in_proj_a = try_make(tensors, &format!("{lp}.linear_attn.in_proj_a.weight"))?;
    let in_proj_qkvz = try_make(tensors, &format!("{lp}.linear_attn.in_proj_qkvz.weight"))?;
    let in_proj_ba = try_make(tensors, &format!("{lp}.linear_attn.in_proj_ba.weight"))?;

    let out_proj = make_linear(tensors, &format!("{lp}.linear_attn.out_proj.weight"))?;

    let conv_weight_key = format!("{lp}.linear_attn.conv1d.weight");
    let conv_weight = if tensors.contains_key(&conv_weight_key) {
        take(tensors, &conv_weight_key)?
    } else {
        take(tensors, &format!("{lp}.linear_attn.conv1d"))?
    };
    // Ensure 2D
    let conv_weight = if conv_weight.ndim() == 3 {
        if conv_weight.dim(1)? == 1 {
            ops::squeeze(&conv_weight, 1)?
        } else if conv_weight.dim(2)? == 1 {
            ops::squeeze(&conv_weight, 2)?
        } else {
            conv_weight
        }
    } else {
        conv_weight
    };

    let norm_weight = if tensors.contains_key(&format!("{lp}.linear_attn.norm.weight")) {
        take(tensors, &format!("{lp}.linear_attn.norm.weight"))?
    } else {
        take(tensors, &format!("{lp}.linear_attn.norm"))?
    };
    let norm_weight = maybe_shift_norm(tensors, norm_weight, lp);

    let dt_bias = if tensors.contains_key(&format!("{lp}.linear_attn.dt_bias")) {
        take(tensors, &format!("{lp}.linear_attn.dt_bias"))?
    } else {
        take(tensors, &format!("{lp}.linear_attn.dt_proj"))?
    };

    let a_log = if tensors.contains_key(&format!("{lp}.linear_attn.A_log")) {
        take(tensors, &format!("{lp}.linear_attn.A_log"))?
    } else {
        take(tensors, &format!("{lp}.linear_attn.a_log"))?
    };
    let a_exp = ops::exp(&ops::astype(&a_log, crate::ffi::MlxDtype::Float32)?)?;

    Ok(Qwen3_5GatedDeltaNet {
        in_proj_qkv,
        in_proj_z,
        in_proj_b,
        in_proj_a,
        in_proj_qkvz,
        in_proj_ba,
        out_proj,
        conv_weight,
        norm_weight,
        dt_bias,
        a_log,
        a_exp,
    })
}

fn load_dense_mlp(tensors: &mut HashMap<String, Array>, lp: &str) -> anyhow::Result<Qwen3_5DenseMlp> {
    Ok(Qwen3_5DenseMlp {
        gate_proj: make_linear(tensors, &format!("{lp}.mlp.gate_proj.weight"))?,
        up_proj: make_linear(tensors, &format!("{lp}.mlp.up_proj.weight"))?,
        down_proj: make_linear(tensors, &format!("{lp}.mlp.down_proj.weight"))?,
    })
}

fn load_sparse_moe(
    tensors: &mut HashMap<String, Array>,
    lp: &str,
    cfg: &Qwen3_5Config,
) -> anyhow::Result<Qwen3_5SparseMoe> {
    let gate = make_linear(tensors, &format!("{lp}.mlp.gate.weight"))?;

    let num_experts = cfg.num_experts as usize;
    let mut experts_gate = Vec::with_capacity(num_experts);
    let mut experts_up = Vec::with_capacity(num_experts);
    let mut experts_down = Vec::with_capacity(num_experts);

    for e in 0..num_experts {
        experts_gate.push(make_linear(tensors, &format!("{lp}.mlp.experts.{e}.gate_proj.weight"))?);
        experts_up.push(make_linear(tensors, &format!("{lp}.mlp.experts.{e}.up_proj.weight"))?);
        experts_down.push(make_linear(tensors, &format!("{lp}.mlp.experts.{e}.down_proj.weight"))?);
    }

    let shared_expert = if tensors.contains_key(&format!("{lp}.mlp.shared_expert.gate_proj.weight")) {
        Some(Qwen3_5DenseMlp {
            gate_proj: make_linear(tensors, &format!("{lp}.mlp.shared_expert.gate_proj.weight"))?,
            up_proj: make_linear(tensors, &format!("{lp}.mlp.shared_expert.up_proj.weight"))?,
            down_proj: make_linear(tensors, &format!("{lp}.mlp.shared_expert.down_proj.weight"))?,
        })
    } else {
        None
    };

    let shared_expert_gate = if tensors.contains_key(&format!("{lp}.mlp.shared_expert_gate.weight")) {
        Some(make_linear(tensors, &format!("{lp}.mlp.shared_expert_gate.weight"))?)
    } else {
        None
    };

    Ok(Qwen3_5SparseMoe {
        gate,
        experts_gate,
        experts_up,
        experts_down,
        shared_expert,
        shared_expert_gate,
        num_experts_per_tok: cfg.num_experts_per_tok,
        norm_topk_prob: cfg.norm_topk_prob,
    })
}

// ---------------------------------------------------------------------------
// Helper ops
// ---------------------------------------------------------------------------

fn depthwise_conv1d(x: &Array, weight: &Array, out_len: usize) -> anyhow::Result<Array> {
    let b = x.dim(0)?;
    let c = weight.dim(0)?;
    let k = weight.dim(1)?;

    let mut out: Option<Array> = None;
    for i in 0..k {
        let seg = ops::slice_axis1(x, i, i + out_len)?;
        let wi = ops::slice_last_dim(weight, i, i + 1)?;
        let wi = ops::reshape(&wi, &[1, 1, c])?;
        let term = ops::multiply(&seg, &wi)?;
        out = match out {
            None => Some(term),
            Some(prev) => Some(ops::add(&prev, &term)?),
        };
    }
    let result = out.ok_or_else(|| anyhow::anyhow!("empty conv kernel"))?;
    ops::reshape(&result, &[b, out_len, c])
}

fn gated_delta_scan(
    q: &Array,
    k: &Array,
    v: &Array,
    g_decay: &Array,
    beta_gate: &Array,
    state: &Array,
) -> anyhow::Result<(Array, Array)> {
    let b = q.dim(0)?;
    let t = q.dim(1)?;
    let hk = k.dim(2)?;
    let hv = v.dim(2)?;
    let _dk = q.dim(3)?;
    let dv = v.dim(3)?;
    let v_per_k = hv / hk;

    // Expand q/k to hv heads for grouped-head attention
    let q = if v_per_k > 1 { ops::repeat_heads(q, v_per_k)? } else { q.clone() };
    let k = if v_per_k > 1 { ops::repeat_heads(k, v_per_k)? } else { k.clone() };

    let mut next_state = state.clone();
    let mut outs = Vec::with_capacity(t);

    for t_idx in 0..t {
        let qt = ops::slice_axis1(&q, t_idx, t_idx + 1)?;
        let qt = ops::squeeze(&qt, 1)?;
        let kt = ops::slice_axis1(&k, t_idx, t_idx + 1)?;
        let kt = ops::squeeze(&kt, 1)?;
        let vt = ops::slice_axis1(&v, t_idx, t_idx + 1)?;
        let vt = ops::squeeze(&vt, 1)?;
        let gt = ops::slice_axis1(g_decay, t_idx, t_idx + 1)?;
        let gt = ops::squeeze(&gt, 1)?;
        let bt = ops::slice_axis1(beta_gate, t_idx, t_idx + 1)?;
        let bt = ops::squeeze(&bt, 1)?;

        // state = state * g
        let gt_2d = ops::expand_dims(&ops::expand_dims(&gt, -1)?, -1)?;
        next_state = ops::multiply(&next_state, &gt_2d)?;

        // kv_mem = sum(state * k, dim=-1)
        let kt_3d = ops::expand_dims(&kt, 2)?;
        let kv_mem = ops::sum_axis(&ops::multiply(&next_state, &kt_3d)?, 3, false)?;

        // delta = (v - kv_mem) * beta
        let delta = ops::multiply(&ops::subtract(&vt, &kv_mem)?, &ops::expand_dims(&bt, -1)?)?;

        // state = state + k * delta
        let state_update = ops::multiply(&kt_3d, &ops::expand_dims(&delta, -1)?)?;
        next_state = ops::add(&next_state, &state_update)?;

        // out = sum(state * q, dim=-1)
        let qt_3d = ops::expand_dims(&qt, 2)?;
        let yt = ops::sum_axis(&ops::multiply(&next_state, &qt_3d)?, 3, false)?;
        let yt = ops::reshape(&yt, &[b, 1, hv, dv])?;
        outs.push(yt);
    }

    let out_refs: Vec<&Array> = outs.iter().collect();
    let out = ops::concatenate(&out_refs, 1)?;
    Ok((out, next_state))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen3_5_config_basic() {
        let json = serde_json::json!({
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 100,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 128,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "layer_types": ["full", "linear", "linear", "full"],
            "full_attention_interval": 4,
        });
        let cfg = Qwen3_5Config::from_json(&json).unwrap();
        assert_eq!(cfg.hidden_size, 64);
        assert_eq!(cfg.rope_theta, 1000000.0);
        assert!(cfg.layer_is_linear(1));
        assert!(!cfg.layer_is_linear(0));
        assert!(!cfg.layer_is_linear(3));
    }
}
