use mlx_core::{Array, Module, Result};
use mlx_nn::{
    Embedding, KvCache, Linear, QuantConfig, RmsNorm, RoPE, VarBuilder,
};
use std::fs::{create_dir_all, OpenOptions};
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use crate::qwen3_moe::MoeProfileStats;

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

static LFM2_MOE_PROFILE_STATS: OnceLock<Mutex<MoeProfileStats>> = OnceLock::new();
static TRACE_GENERATION_ENABLED: OnceLock<bool> = OnceLock::new();
static TRACE_PORT_ENABLED: OnceLock<bool> = OnceLock::new();
static TRACE_LINE_COUNT: AtomicUsize = AtomicUsize::new(0);
static TRACE_LIMIT_REACHED: AtomicBool = AtomicBool::new(false);
const TRACE_LINE_LIMIT: usize = 2000;

fn lfm2_moe_profile_stats_store() -> &'static Mutex<MoeProfileStats> {
    LFM2_MOE_PROFILE_STATS.get_or_init(|| Mutex::new(MoeProfileStats::default()))
}

fn trace_generation_enabled() -> bool {
    *TRACE_GENERATION_ENABLED.get_or_init(|| {
        matches!(
            std::env::var("MLX_TRACE_GENERATION").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
    })
}

fn trace_port_enabled() -> bool {
    *TRACE_PORT_ENABLED.get_or_init(|| {
        matches!(
            std::env::var("MLX_TRACE_LFM2_PYTHON_PORT").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
    })
}

fn trace_port(stage: &str, detail: impl AsRef<str>) {
    if !trace_port_enabled() {
        return;
    }
    let prev = TRACE_LINE_COUNT.fetch_add(1, Ordering::Relaxed);
    if prev >= TRACE_LINE_LIMIT {
        if !TRACE_LIMIT_REACHED.swap(true, Ordering::Relaxed) {
            let line = format!(
                "[lfm2-python-port] stage=trace_limit_reached limit={TRACE_LINE_LIMIT}\n"
            );
            eprint!("{line}");
            let _ = std::io::stderr().flush();
            let _ = create_dir_all("logs");
            let path = format!("logs/lfm2_python_port_{}.log", std::process::id());
            if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) {
                let _ = file.write_all(line.as_bytes());
                let _ = file.flush();
            }
        }
        return;
    }
    let line = format!("[lfm2-python-port] stage={stage} {}\n", detail.as_ref());
    eprint!("{line}");
    let _ = std::io::stderr().flush();
    let _ = create_dir_all("logs");
    let path = format!("logs/lfm2_python_port_{}.log", std::process::id());
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) {
        let _ = file.write_all(line.as_bytes());
        let _ = file.flush();
    }
}

macro_rules! trace_portf {
    ($stage:expr, $($arg:tt)*) => {
        if trace_port_enabled() {
            trace_port($stage, format!($($arg)*));
        }
    };
}

fn with_lfm2_moe_profile_stats_mut<F: FnOnce(&mut MoeProfileStats)>(f: F) {
    if !trace_generation_enabled() {
        return;
    }
    if let Ok(mut stats) = lfm2_moe_profile_stats_store().lock() {
        f(&mut stats);
    }
}

pub(crate) fn reset_lfm2_moe_python_port_profile_stats() {
    if let Ok(mut stats) = lfm2_moe_profile_stats_store().lock() {
        *stats = MoeProfileStats::default();
    }
}

pub(crate) fn lfm2_moe_python_port_profile_stats() -> MoeProfileStats {
    lfm2_moe_profile_stats_store()
        .lock()
        .map(|stats| stats.clone())
        .unwrap_or_default()
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
    lhs.iter().zip(rhs.iter()).all(|((l_idx, l_prob), (r_idx, r_prob))| {
        l_idx == r_idx && (l_prob - r_prob).abs() <= tol
    })
}

fn validate_device_router_enabled() -> bool {
    matches!(
        std::env::var("MLX_VALIDATE_MOE_DEVICE_ROUTER").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
            | Ok("lfm2") | Ok("LFM2") | Ok("all") | Ok("ALL")
    )
}

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

        // MLX's SDPA handles grouped-query attention without explicit KV expansion.
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

    fn forward_switch(&self, x: &Array, expert_indices: &Array, sorted_indices: bool) -> Result<Array> {
        if let Some(scales) = &self.scales {
            return x.gather_qmm(
                &self.weight,
                scales,
                self.biases.as_ref(),
                None,
                Some(expert_indices),
                true,
                self.group_size,
                self.bits,
                sorted_indices,
            );
        }

        let wt = self.weight.transpose_axes(&[0, 2, 1])?;
        x.gather_mm(&wt, None, Some(expert_indices), sorted_indices)
    }

}

struct SwitchGlu {
    gate_proj: ExpertLinear,
    up_proj: ExpertLinear,
    down_proj: ExpertLinear,
}

impl SwitchGlu {
    fn load(vb: &VarBuilder, cfg: &Lfm2MoeConfig) -> anyhow::Result<Self> {
        Ok(Self {
            gate_proj: ExpertLinear::load(&vb.pp("gate_proj"), cfg)?,
            up_proj: ExpertLinear::load(&vb.pp("up_proj"), cfg)?,
            down_proj: ExpertLinear::load(&vb.pp("down_proj"), cfg)?,
        })
    }

    fn forward(&self, x: &Array, indices: &Array) -> Result<Array> {
        let x = x.expand_dims(-2)?.expand_dims(-2)?;
        let do_sort = false;
        let x_up = self.up_proj.forward_switch(&x, indices, do_sort)?;
        let x_gate = self.gate_proj.forward_switch(&x, indices, do_sort)?;
        let x = x_gate.multiply(&x_gate.sigmoid()?)?.multiply(&x_up)?;
        let x = self.down_proj.forward_switch(&x, indices, do_sort)?;
        x.squeeze(2)
    }
}

struct MoeFeedForward {
    layer_idx: usize,
    router: Linear,
    switch_mlp: SwitchGlu,
    expert_bias: Option<Array>,
    num_experts: usize,
    top_k: usize,
    norm_topk_prob: bool,
}

impl MoeFeedForward {
    fn load(layer_idx: usize, vb: &VarBuilder, cfg: &Lfm2MoeConfig) -> anyhow::Result<Self> {
        let qc = cfg.quant_config();
        Ok(Self {
            layer_idx,
            router: Linear::new(&vb.pp("gate"), &qc)?,
            switch_mlp: SwitchGlu::load(&vb.pp("switch_mlp"), cfg)?,
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
        trace_portf!("moe.enter", "layer={} input_shape={:?}", self.layer_idx, orig_shape);
        let flat = x.reshape(&[-1, hidden])?;
        trace_portf!("moe.flat", "layer={} flat_shape={:?}", self.layer_idx, flat.shape_raw());

        let mut router_logits = self.router.forward(&flat)?;
        if let Some(expert_bias) = &self.expert_bias {
            router_logits = router_logits.add(expert_bias)?;
        }
        trace_portf!("moe.router_logits", "layer={} router_logits_shape={:?}", self.layer_idx, router_logits.shape_raw());

        let num_tokens = flat.shape_raw()[0] as usize;
        let num_experts = self.num_experts;
        let k = self.top_k.min(num_experts).max(1);

        let router_probs = router_logits.softmax(-1)?;

        let stage_t0 = Instant::now();
        let mut shadow_checks = 0usize;
        let mut shadow_mismatches = 0usize;
        let start = num_experts as i32 - k as i32;

        if num_tokens == 1 {
            let partition = router_probs.argpartition((num_experts - k) as i32, -1)?;
            let expert_idx = partition.slice(&[0, start], &[1, num_experts as i32])?;
            let mut top_scores = router_probs.take_along_axis(&expert_idx, -1)?;
            if self.norm_topk_prob {
                let denom = top_scores.sum_axis(-1, true)?;
                top_scores = top_scores.divide(&denom)?;
            }

            if validate_device_router_enabled() {
                let stage_t0 = Instant::now();
                let row = router_probs.to_vec_f32()?;
                let row = &row[..num_experts];
                let mut expert_ids: Vec<usize> = (0..num_experts).collect();
                expert_ids.sort_unstable_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
                let top = &expert_ids[..k];
                let mut host_top: Vec<(usize, f32)> =
                    top.iter().copied().map(|idx| (idx, row[idx])).collect();
                sort_top_pairs(&mut host_top);
                let idxs = expert_idx.to_vec_i32()?;
                let probs = top_scores.to_vec_f32()?;
                let mut device_top: Vec<(usize, f32)> = idxs
                    .into_iter()
                    .zip(probs.into_iter())
                    .map(|(idx, prob)| (idx as usize, prob))
                    .collect();
                sort_top_pairs(&mut device_top);
                shadow_checks += 1;
                if !top_pairs_match(&host_top, &device_top, 1e-4) {
                    shadow_mismatches += 1;
                    trace_portf!("moe.shadow_mismatch", "layer={} host_top={:?} device_top={:?}", self.layer_idx, host_top, device_top);
                }
                with_lfm2_moe_profile_stats_mut(|stats| {
                    stats.router_host_s += stage_t0.elapsed().as_secs_f64();
                });
            }

            trace_portf!("moe.routing_built", "layer={} num_tokens=1 top_k={}", self.layer_idx, k);
            with_lfm2_moe_profile_stats_mut(|stats| {
                stats.routing_build_s += stage_t0.elapsed().as_secs_f64();
                stats.single_token_fast_path_hits += 1;
                stats.device_router_shadow_checks += shadow_checks;
                stats.device_router_shadow_mismatches += shadow_mismatches;
            });

            let stage_t0 = Instant::now();
            let x_switch = flat.reshape(&[1, hidden])?;
            trace_portf!("moe.expert_idx", "layer={} expert_idx_shape={:?}", self.layer_idx, expert_idx.shape_raw());
            let expert_out = self.switch_mlp.forward(&x_switch, &expert_idx)?;
            trace_portf!("moe.down_done", "layer={} expert_out_shape={:?}", self.layer_idx, expert_out.shape_raw());
            with_lfm2_moe_profile_stats_mut(|stats| {
                stats.expert_forward_s += stage_t0.elapsed().as_secs_f64();
            });

            let score_arr = top_scores.expand_dims(-1)?.as_type(expert_out.dtype())?;
            trace_portf!("moe.score_arr", "layer={} score_arr_shape={:?}", self.layer_idx, score_arr.shape_raw());
            let weighted = expert_out.multiply(&score_arr)?;
            trace_portf!("moe.weighted", "layer={} weighted_shape={:?}", self.layer_idx, weighted.shape_raw());
            let out = weighted.sum_axis(1, false)?.reshape(&[1, hidden])?;
            trace_portf!("moe.exit", "layer={} out_shape={:?}", self.layer_idx, out.shape_raw());
            return out.reshape(&orig_shape);
        }

        let partition = router_probs.argpartition((num_experts - k) as i32, -1)?;
        let expert_idx = partition.slice(&[0, start], &[num_tokens as i32, num_experts as i32])?;
        let mut top_scores = router_probs.take_along_axis(&expert_idx, -1)?;
        if self.norm_topk_prob {
            let denom = top_scores.sum_axis(-1, true)?;
            top_scores = top_scores.divide(&denom)?;
        }
        trace_portf!("moe.routing_built", "layer={} num_tokens={} top_k={}", self.layer_idx, num_tokens, k);
        with_lfm2_moe_profile_stats_mut(|stats| {
            stats.routing_build_s += stage_t0.elapsed().as_secs_f64();
            stats.device_router_shadow_checks += shadow_checks;
            stats.device_router_shadow_mismatches += shadow_mismatches;
        });

        let stage_t0 = Instant::now();
        trace_portf!("moe.expert_idx", "layer={} expert_idx_shape={:?}", self.layer_idx, expert_idx.shape_raw());
        let expert_out = self.switch_mlp.forward(&flat, &expert_idx)?;
        trace_portf!("moe.down_done", "layer={} expert_out_shape={:?}", self.layer_idx, expert_out.shape_raw());
        with_lfm2_moe_profile_stats_mut(|stats| {
            stats.expert_forward_s += stage_t0.elapsed().as_secs_f64();
        });

        let score_arr = top_scores.expand_dims(-1)?.as_type(expert_out.dtype())?;
        trace_portf!("moe.score_arr", "layer={} score_arr_shape={:?}", self.layer_idx, score_arr.shape_raw());
        let weighted = expert_out.multiply(&score_arr)?;
        trace_portf!("moe.weighted", "layer={} weighted_shape={:?}", self.layer_idx, weighted.shape_raw());
        let out = weighted.sum_axis(1, false)?;
        trace_portf!("moe.exit", "layer={} out_shape={:?}", self.layer_idx, out.shape_raw());
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
    layer_idx: usize,
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
            LayerFfn::Moe(MoeFeedForward::load(idx, &vb.pp("feed_forward"), cfg)?)
        } else {
            LayerFfn::Dense(DenseMlp::load(&vb.pp("feed_forward"), cfg)?)
        };

        Ok(Self {
            layer_idx: idx,
            operator_norm: RmsNorm::new(cfg.norm_eps, &vb.pp("operator_norm"))?,
            ffn_norm: RmsNorm::new(cfg.norm_eps, &vb.pp("ffn_norm"))?,
            operator,
            ffn,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        trace_portf!("layer.enter", "layer={} x_shape={:?}", self.layer_idx, x.shape_raw());
        let residual = x.clone();
        let h = self.operator_norm.forward(x)?;
        let h = self.operator.forward(&h)?;
        let x = residual.add(&h)?;
        trace_portf!("layer.after_operator", "layer={} x_shape={:?}", self.layer_idx, x.shape_raw());

        let residual = x.clone();
        let h = self.ffn_norm.forward(&x)?;
        let h = self.ffn.forward(&h)?;
        trace_portf!("layer.after_ffn", "layer={} h_shape={:?}", self.layer_idx, h.shape_raw());
        let out = residual.add(&h)?;
        trace_portf!("layer.exit", "layer={} out_shape={:?}", self.layer_idx, out.shape_raw());
        Ok(out)
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
        trace_portf!("model.enter", "input_shape={:?} seq_len={}", shape, seq_len);

        let mut h = self.embed_tokens.forward(input_ids)?;
        trace_portf!("model.embed", "embed_shape={:?}", h.shape_raw());
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            trace_portf!("model.layer_start", "layer={idx}");
            h = layer.forward(&h)?;
            trace_portf!("model.layer_done", "layer={idx} h_shape={:?}", h.shape_raw());
        }
        h = self.norm.forward(&h)?;
        trace_portf!("model.norm", "norm_shape={:?}", h.shape_raw());

        if seq_len > 1 {
            let h_shape = h.shape_raw();
            let mut start = vec![0i32; h_shape.len()];
            let mut stop = h_shape.clone();
            start[h_shape.len() - 2] = seq_len - 1;
            stop[h_shape.len() - 2] = seq_len;
            h = h.slice(&start, &stop)?;
            trace_portf!("model.last_token", "slice_shape={:?}", h.shape_raw());
        }

        let logits = self.lm_head.forward(&h)?;
        trace_portf!("model.exit", "logits_shape={:?}", logits.shape_raw());
        Ok(logits)
    }

    pub fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }
}

pub type Lfm2MoePythonPort = Lfm2Moe;
pub type Lfm2MoePythonPortConfig = Lfm2MoeConfig;
