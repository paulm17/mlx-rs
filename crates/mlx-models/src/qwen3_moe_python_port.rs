//! Qwen3-MoE (Mixture of Experts) model implementation.

use mlx_core::{Array, Module, Result};
use mlx_nn::{Embedding, KvCache, Linear, QuantConfig, RmsNorm, RoPE, RopeScaling, VarBuilder};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

// ------------------------------------------------------------------
// Config
// ------------------------------------------------------------------

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen3MoePythonPortConfig {
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

fn default_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    1_000_000.0
}
fn default_max_pos() -> usize {
    32768
}

#[derive(Debug, Clone, Default)]
pub struct MoeProfileStats {
    pub router_host_s: f64,
    pub routing_build_s: f64,
    pub expert_forward_s: f64,
    pub shared_expert_s: f64,
    pub single_token_fast_path_hits: usize,
    pub device_router_shadow_checks: usize,
    pub device_router_shadow_mismatches: usize,
}

static MOE_PROFILE_STATS: OnceLock<Mutex<MoeProfileStats>> = OnceLock::new();

fn qwen3_moe_python_port_profile_stats_store() -> &'static Mutex<MoeProfileStats> {
    MOE_PROFILE_STATS.get_or_init(|| Mutex::new(MoeProfileStats::default()))
}

fn trace_qwen3_moe_python_port_generation_enabled() -> bool {
    matches!(
        std::env::var("MLX_TRACE_GENERATION").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn validate_device_router_enabled(target: &str) -> bool {
    match std::env::var("MLX_VALIDATE_MOE_DEVICE_ROUTER")
        .ok()
        .as_deref()
    {
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES") => true,
        Some("all") | Some("ALL") => true,
        Some(v) if v.eq_ignore_ascii_case(target) => true,
        _ => false,
    }
}

fn sort_top_pairs(pairs: &mut [(usize, f32)]) {
    pairs.sort_unstable_by(|(a_idx, a_prob), (b_idx, b_prob)| {
        b_prob.total_cmp(a_prob).then_with(|| a_idx.cmp(b_idx))
    });
}

fn top_pairs_match(lhs: &[(usize, f32)], rhs: &[(usize, f32)], tol: f32) -> bool {
    if lhs.len() != rhs.len() {
        return false;
    }
    lhs.iter()
        .zip(rhs.iter())
        .all(|((l_idx, l_prob), (r_idx, r_prob))| l_idx == r_idx && (l_prob - r_prob).abs() <= tol)
}

fn validate_qwen3_moe_python_port_shadow_top_k(
    target: &str,
    router_probs: &Array,
    num_experts: usize,
    k: usize,
    host_top: &[(usize, f32)],
) -> Result<()> {
    if !validate_device_router_enabled(target) {
        return Ok(());
    }

    let partition = router_probs.argpartition((num_experts - k) as i32, -1)?;
    let start = num_experts as i32 - k as i32;
    let top_idx = partition.slice(&[0, start], &[1, num_experts as i32])?;
    let top_probs = router_probs.take_along_axis(&top_idx, -1)?;

    let idxs = top_idx.to_vec_i32()?;
    let probs = top_probs.to_vec_f32()?;
    let mut device_top: Vec<(usize, f32)> = idxs
        .into_iter()
        .zip(probs.into_iter())
        .map(|(idx, prob)| (idx as usize, prob))
        .collect();
    sort_top_pairs(&mut device_top);

    with_qwen3_moe_python_port_profile_stats_mut(|stats| {
        stats.device_router_shadow_checks += 1;
        if !top_pairs_match(host_top, &device_top, 1e-4) {
            stats.device_router_shadow_mismatches += 1;
        }
    });
    Ok(())
}

fn with_qwen3_moe_python_port_profile_stats_mut<F: FnOnce(&mut MoeProfileStats)>(f: F) {
    if !trace_qwen3_moe_python_port_generation_enabled() {
        return;
    }
    if let Ok(mut stats) = qwen3_moe_python_port_profile_stats_store().lock() {
        f(&mut stats);
    }
}

pub fn reset_qwen3_moe_python_port_profile_stats() {
    if let Ok(mut stats) = qwen3_moe_python_port_profile_stats_store().lock() {
        *stats = MoeProfileStats::default();
    }
}

pub fn qwen3_moe_python_port_profile_stats() -> MoeProfileStats {
    qwen3_moe_python_port_profile_stats_store()
        .lock()
        .map(|stats| stats.clone())
        .unwrap_or_default()
}

impl Qwen3MoePythonPortConfig {
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
    fn load(vb: &VarBuilder, cfg: &Qwen3MoePythonPortConfig) -> anyhow::Result<Self> {
        let qc = cfg.quant_config();
        Ok(Self {
            gate: Linear::new(vb, &qc)?,
            num_experts_per_tok: cfg.num_experts_per_tok(),
            norm_topk_prob: cfg.norm_topk_prob,
        })
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

fn stack_arrays(arrays: &[Array]) -> Result<Array> {
    let expanded: Result<Vec<Array>> = arrays.iter().map(|a| a.expand_dims(0)).collect();
    let expanded = expanded?;
    let refs: Vec<&Array> = expanded.iter().collect();
    Array::concatenate(&refs, 0)
}

impl SwitchLinear {
    fn load(vb: &VarBuilder, proj: &str, cfg: &Qwen3MoePythonPortConfig) -> anyhow::Result<Self> {
        let mut weights = Vec::with_capacity(cfg.num_experts());
        let mut scales_vec = Vec::new();
        let mut biases_vec = Vec::new();
        let mut bias_vec = Vec::new();
        let qc = cfg.quant_config();

        for i in 0..cfg.num_experts() {
            let expert_vb = vb.pp(i).pp(proj);
            weights.push(expert_vb.get("weight")?);
            if expert_vb.contains("scales") {
                scales_vec.push(expert_vb.get("scales")?);
            }
            if expert_vb.contains("biases") {
                biases_vec.push(expert_vb.get("biases")?);
            }
            if expert_vb.contains("bias") {
                bias_vec.push(expert_vb.get("bias")?);
            }
        }

        let weight = stack_arrays(&weights)?;
        let scales = if scales_vec.is_empty() {
            None
        } else {
            Some(stack_arrays(&scales_vec)?)
        };
        let biases = if biases_vec.is_empty() {
            None
        } else {
            Some(stack_arrays(&biases_vec)?)
        };
        let bias = if bias_vec.is_empty() {
            None
        } else {
            Some(stack_arrays(&bias_vec)?)
        };
        let bits = if let Some(ref s) = scales {
            infer_bits(&weight.shape_raw(), &s.shape_raw(), qc.group_size, qc.bits)
        } else {
            0
        };

        Ok(Self {
            weight,
            scales,
            biases,
            bias,
            group_size: qc.group_size,
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
    fn load(vb: &VarBuilder, cfg: &Qwen3MoePythonPortConfig) -> anyhow::Result<Self> {
        Ok(Self {
            gate_proj: SwitchLinear::load(&vb.pp("experts"), "gate_proj", cfg)?,
            up_proj: SwitchLinear::load(&vb.pp("experts"), "up_proj", cfg)?,
            down_proj: SwitchLinear::load(&vb.pp("experts"), "down_proj", cfg)?,
        })
    }

    fn forward(&self, x: &Array, indices: &Array) -> Result<Array> {
        let x = x.expand_dims(-2)?.expand_dims(-2)?;
        let x_up = self.up_proj.forward(&x, indices)?;
        let x_gate = self.gate_proj.forward(&x, indices)?;
        let hidden = x_gate.multiply(&x_gate.sigmoid()?)?.multiply(&x_up)?;
        let out = self.down_proj.forward(&hidden, indices)?;
        out.squeeze(2)
    }
}

struct SparseMoeBlock {
    gate: MoeGate,
    switch_mlp: SwitchGlu,
    shared_expert: Option<SharedExpert>,
    shared_expert_gate: Option<Linear>,
}

struct SharedExpert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl SharedExpert {
    fn load(vb: &VarBuilder, cfg: &Qwen3MoePythonPortConfig) -> anyhow::Result<Self> {
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
    fn load(vb: &VarBuilder, cfg: &Qwen3MoePythonPortConfig) -> anyhow::Result<Self> {
        let gate = MoeGate::load(&vb.pp("gate"), cfg)?;
        let switch_mlp = SwitchGlu::load(vb, cfg)?;
        let shared_expert = if cfg.shared_expert_intermediate_size.is_some() {
            Some(SharedExpert::load(&vb.pp("shared_expert"), cfg)?)
        } else {
            None
        };

        let shared_expert_gate = if vb.pp("shared_expert_gate").contains("weight") {
            Some(Linear::new(
                &vb.pp("shared_expert_gate"),
                &cfg.quant_config(),
            )?)
        } else {
            None
        };

        Ok(Self {
            gate,
            switch_mlp,
            shared_expert,
            shared_expert_gate,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let shape = x.shape_raw();
        let hidden = shape[shape.len() - 1];
        let orig_shape = shape.clone();
        let flat = x.reshape(&[-1, hidden])?;
        let num_tokens = flat.shape_raw()[0] as usize;
        let num_experts = self.gate.gate.weight().shape_raw()[0] as usize;
        let k = self.gate.num_experts_per_tok.min(num_experts).max(1);

        if num_tokens == 1 {
            with_qwen3_moe_python_port_profile_stats_mut(|stats| {
                stats.single_token_fast_path_hits += 1;
            });
        }

        let stage_t0 = Instant::now();
        let router_logits = self.gate.gate.forward(&flat)?;
        let router_probs = router_logits.softmax(-1)?;
        let neg_router_probs = router_probs.negative()?;
        let partition = neg_router_probs.argpartition((k - 1) as i32, -1)?;
        let top_idx = partition.slice(&[0, 0], &[num_tokens as i32, k as i32])?;
        let mut top_probs = router_probs.take_along_axis(&top_idx, -1)?;
        if self.gate.norm_topk_prob {
            let denom = top_probs.sum_axis(-1, true)?;
            top_probs = top_probs.divide(&denom)?;
        }
        with_qwen3_moe_python_port_profile_stats_mut(|stats| {
            stats.router_host_s += stage_t0.elapsed().as_secs_f64();
        });
        if validate_device_router_enabled("qwen") {
            let host_logits = router_logits.to_vec_f32()?;
            let row_logits = &host_logits[..num_experts];
            let row_probs = softmax_row(row_logits);
            let top: Vec<(usize, f32)> = select_top_k(row_logits, k)
                .into_iter()
                .map(|(idx, _)| (idx, row_probs[idx]))
                .collect();
            validate_qwen3_moe_python_port_shadow_top_k(
                "qwen",
                &router_probs,
                num_experts,
                k,
                &top,
            )?;
        }

        let stage_t0 = Instant::now();
        let expert_out = self.switch_mlp.forward(&flat, &top_idx)?;
        with_qwen3_moe_python_port_profile_stats_mut(|stats| {
            stats.expert_forward_s += stage_t0.elapsed().as_secs_f64();
        });
        let score_arr = top_probs.expand_dims(-1)?.as_type(expert_out.dtype())?;
        let mut out = expert_out.multiply(&score_arr)?.sum_axis(1, false)?;

        if let Some(ref shared) = self.shared_expert {
            let stage_t0 = Instant::now();
            let mut shared_out = shared.forward(&flat)?;
            if let Some(ref shared_gate) = self.shared_expert_gate {
                let gate = shared_gate.forward(&flat)?.sigmoid()?;
                shared_out = shared_out.multiply(&gate)?;
            }
            with_qwen3_moe_python_port_profile_stats_mut(|stats| {
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
    let max_v = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
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
    fn load(vb: &VarBuilder, cfg: &Qwen3MoePythonPortConfig) -> anyhow::Result<Self> {
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
    fn load(vb: &VarBuilder, cfg: &Qwen3MoePythonPortConfig) -> anyhow::Result<Self> {
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
        let (b, seq_len) = (shape[0], shape[1]);
        let offset = self.kv_cache.offset() as i32;

        let q = self
            .q_proj
            .forward(x)?
            .reshape(&[b, seq_len, self.num_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape(&[b, seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape(&[b, seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let q = match &self.q_norm {
            Some(n) => n.forward(&q)?,
            None => q,
        };
        let k = match &self.k_norm {
            Some(n) => n.forward(&k)?,
            None => k,
        };

        let q = self.rope.forward(&q, offset)?;
        let k = self.rope.forward(&k, offset)?;

        let (k, v) = self.kv_cache.update(&k, &v)?;

        // MLX's SDPA handles grouped-query attention without explicit KV expansion.
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
// Decoder Block
// ------------------------------------------------------------------

struct MoeBlock {
    attn: MoeAttention,
    ff: FeedForward,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl MoeBlock {
    fn load(
        layer_idx: usize,
        vb: &VarBuilder,
        cfg: &Qwen3MoePythonPortConfig,
    ) -> anyhow::Result<Self> {
        let ff = if cfg.is_moe_layer(layer_idx) {
            FeedForward::Moe(SparseMoeBlock::load(&vb.pp("mlp"), cfg)?)
        } else {
            FeedForward::Dense(DenseMlp::load(&vb.pp("mlp"), cfg)?)
        };

        Ok(Self {
            attn: MoeAttention::load(&vb.pp("self_attn"), cfg)?,
            ff,
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
        let h = self.ff.forward(&h)?;
        residual.add(&h)
    }

    fn clear_cache(&mut self) {
        self.attn.clear_cache();
    }
}

// ------------------------------------------------------------------
// Qwen3MoePythonPort Model
// ------------------------------------------------------------------

/// Qwen3-MoE language model.
pub struct Qwen3MoePythonPort {
    embed_tokens: Embedding,
    layers: Vec<MoeBlock>,
    norm: RmsNorm,
    lm_head: Linear,
}

impl Qwen3MoePythonPort {
    pub fn new(cfg: &Qwen3MoePythonPortConfig, vb: &VarBuilder) -> anyhow::Result<Self> {
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
