use std::collections::HashMap;

use crate::array::Array;
use crate::llama::{
    Embedding, EmbeddingLayer, KvCache, LayerCache, Linear, LinearLayer, RmsNorm,
    resolve_weight_prefix,
};
use crate::model::Model;
use crate::ops;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct QwenMoeConfig {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub moe_intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: i32,
    pub tie_word_embeddings: bool,
    pub num_experts: i32,
    pub num_experts_per_tok: i32,
    pub norm_topk_prob: bool,
    pub decoder_sparse_step: i32,
    pub shared_expert_intermediate_size: i32,
}

impl QwenMoeConfig {
    pub fn from_json(config: &serde_json::Value) -> anyhow::Result<Self> {
        let c = config.get("text_config").unwrap_or(config);

        let get_i32 =
            |k: &str, d: i32| -> i32 { c.get(k).and_then(|v| v.as_i64()).unwrap_or(d as i64) as i32 };
        let get_f32 =
            |k: &str, d: f32| -> f32 { c.get(k).and_then(|v| v.as_f64()).unwrap_or(d as f64) as f32 };
        let get_bool = |k: &str, d: bool| -> bool { c.get(k).and_then(|v| v.as_bool()).unwrap_or(d) };

        let hidden_size = get_i32("hidden_size", 0);
        let num_hidden_layers = get_i32("num_hidden_layers", 0);
        let intermediate_size = get_i32("intermediate_size", 0);
        let num_attention_heads = get_i32("num_attention_heads", 0);
        let num_key_value_heads = get_i32("num_key_value_heads", num_attention_heads);
        let head_dim = get_i32("head_dim", 0);
        let vocab_size = get_i32("vocab_size", 0);
        let rms_norm_eps = get_f32("rms_norm_eps", 1e-6);
        let rope_theta = get_f32("rope_theta", 1000000.0);
        let max_position_embeddings = get_i32("max_position_embeddings", 32768);
        let tie_word_embeddings = get_bool("tie_word_embeddings", false);

        let num_experts = get_i32("num_experts", 8);
        let num_experts_per_tok = get_i32("num_experts_per_tok", 2);
        let norm_topk_prob = get_bool("norm_topk_prob", false);
        let decoder_sparse_step = get_i32("decoder_sparse_step", 1);
        let moe_intermediate_size = get_i32("moe_intermediate_size", intermediate_size);
        let shared_expert_intermediate_size =
            get_i32("shared_expert_intermediate_size", intermediate_size);

        if hidden_size <= 0 || num_attention_heads <= 0 {
            anyhow::bail!("invalid QwenMoe config: hidden_size={hidden_size} heads={num_attention_heads}");
        }

        let head_dim = if head_dim > 0 {
            head_dim
        } else {
            hidden_size / num_attention_heads
        };

        Ok(Self {
            hidden_size,
            num_hidden_layers,
            intermediate_size,
            moe_intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            vocab_size,
            rms_norm_eps,
            rope_theta,
            max_position_embeddings,
            tie_word_embeddings,
            num_experts,
            num_experts_per_tok,
            norm_topk_prob,
            decoder_sparse_step,
            shared_expert_intermediate_size,
        })
    }

    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        if self.decoder_sparse_step <= 0 {
            return false;
        }
        ((layer_idx as i32) + 1) % self.decoder_sparse_step == 0
    }
}

// ---------------------------------------------------------------------------
// SwitchLinear — stacked expert weights for gather_qmm
// ---------------------------------------------------------------------------

struct SwitchLinear {
    weight: Array,
    scales: Option<Array>,
    biases: Option<Array>,
    bias: Option<Array>,
    group_size: i32,
    bits: i32,
    mode: String,
}

impl SwitchLinear {
    fn forward(&self, x: &Array, indices: &Array) -> anyhow::Result<Array> {
        let sorted_indices = false;
        let mut out = if let Some(scales) = &self.scales {
            ops::gather_qmm(
                x,
                &self.weight,
                scales,
                self.biases.as_ref(),
                None,
                Some(indices),
                true,
                Some(self.group_size),
                Some(self.bits),
                &self.mode,
                sorted_indices,
            )?
        } else {
            let wt = ops::transpose(&self.weight, &[0, 2, 1])?;
            ops::gather_mm(x, &wt, None, Some(indices), sorted_indices)?
        };
        if let Some(bias) = &self.bias {
            let gathered = ops::take(bias, indices, 0)?;
            let gathered = ops::expand_dims(&gathered, -2)?;
            out = ops::add(&out, &gathered)?;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// SwitchGlu — expert-parallel SwiGLU using gather_mm
// ---------------------------------------------------------------------------

struct SwitchGlu {
    gate_proj: SwitchLinear,
    up_proj: SwitchLinear,
    down_proj: SwitchLinear,
}

impl SwitchGlu {
    fn forward(&self, x: &Array, indices: &Array) -> anyhow::Result<Array> {
        let x = ops::expand_dims(&ops::expand_dims(x, -2)?, -2)?;
        let x_gate = self.gate_proj.forward(&x, indices)?;
        let x_up = self.up_proj.forward(&x, indices)?;
        let hidden = ops::multiply(&ops::silu(&x_gate)?, &x_up)?;
        let out = self.down_proj.forward(&hidden, indices)?;
        ops::squeeze(&out, 2)
    }
}

// ---------------------------------------------------------------------------
// SparseMoeBlock — router + SwitchGlu + optional shared expert
// ---------------------------------------------------------------------------

struct SharedExpert {
    gate_proj: Box<dyn LinearLayer>,
    up_proj: Box<dyn LinearLayer>,
    down_proj: Box<dyn LinearLayer>,
}

impl SharedExpert {
    fn forward(&self, x: &Array) -> anyhow::Result<Array> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let gate_silu = ops::silu(&gate)?;
        let gated = ops::multiply(&gate_silu, &up)?;
        self.down_proj.forward(&gated)
    }
}

struct SparseMoeBlock {
    gate: Box<dyn LinearLayer>,
    switch_mlp: SwitchGlu,
    shared_expert: Option<SharedExpert>,
    shared_expert_gate: Option<Box<dyn LinearLayer>>,
    num_experts_per_tok: i32,
    norm_topk_prob: bool,
}

impl SparseMoeBlock {
    fn forward(&self, x: &Array, cfg: &QwenMoeConfig) -> anyhow::Result<Array> {
        let shape = x.shape();
        let ndim = shape.len();
        let hidden = shape[ndim - 1];
        let flat_size: usize = shape[..ndim - 1].iter().product();
        let flat = ops::reshape(x, &[flat_size, hidden])?;
        let num_experts = cfg.num_experts as usize;
        let k = (self.num_experts_per_tok as usize).min(num_experts).max(1);

        let router_logits = self.gate.forward(&flat)?;
        let router_probs = ops::softmax_axis(&router_logits, -1, true)?;
        let neg_probs = ops::negative(&router_probs)?;
        let partition = ops::argpartition(&neg_probs, k as i32 - 1, -1)?;
        let top_idx = ops::slice_last_dim(&partition, 0, k)?;
        let mut top_probs = ops::take_along_axis(&router_probs, &top_idx, -1)?;

        if self.norm_topk_prob && k > 1 {
            let sum = ops::sum_axis(&top_probs, top_probs.ndim() - 1, true)?;
            top_probs = ops::divide(&top_probs, &sum)?;
        }

        let expert_out = self.switch_mlp.forward(&flat, &top_idx)?;

        let score = ops::astype(
            &ops::expand_dims(&top_probs, -1)?,
            expert_out.dtype()?,
        )?;
        let mut out = ops::sum_axis(&ops::multiply(&expert_out, &score)?, 1, false)?;

        if let Some(ref shared) = self.shared_expert {
            let shared_out = shared.forward(&flat)?;
            if let Some(ref gate) = self.shared_expert_gate {
                let gate_raw = gate.forward(&flat)?;
                let gate_val = ops::sigmoid(&gate_raw)?;
                out = ops::add(&out, &ops::multiply(&shared_out, &gate_val)?)?;
            } else {
                out = ops::add(&out, &shared_out)?;
            }
        }

        ops::reshape(&out, &shape)
    }
}

// ---------------------------------------------------------------------------
// DenseMlp — standard SwiGLU (for non-MoE layers)
// ---------------------------------------------------------------------------

struct DenseMlp {
    gate_proj: Box<dyn LinearLayer>,
    up_proj: Box<dyn LinearLayer>,
    down_proj: Box<dyn LinearLayer>,
}

impl DenseMlp {
    fn forward(&self, x: &Array) -> anyhow::Result<Array> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let gate_silu = ops::silu(&gate)?;
        let gated = ops::multiply(&gate_silu, &up)?;
        self.down_proj.forward(&gated)
    }
}

// ---------------------------------------------------------------------------
// FeedForward — enum dispatch
// ---------------------------------------------------------------------------

enum FeedForward {
    Dense(DenseMlp),
    Moe(SparseMoeBlock),
}

impl FeedForward {
    fn forward(&self, x: &Array, cfg: &QwenMoeConfig) -> anyhow::Result<Array> {
        match self {
            FeedForward::Dense(mlp) => mlp.forward(x),
            FeedForward::Moe(moe) => moe.forward(x, cfg),
        }
    }
}

// ---------------------------------------------------------------------------
// QwenMoeAttention — Q/K/V/O with optional Q/K norms + RoPE + KvCache
// ---------------------------------------------------------------------------

struct QwenMoeAttention {
    q_proj: Box<dyn LinearLayer>,
    k_proj: Box<dyn LinearLayer>,
    v_proj: Box<dyn LinearLayer>,
    o_proj: Box<dyn LinearLayer>,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
}

impl QwenMoeAttention {
    fn forward(
        &self,
        x: &Array,
        kv: &mut KvCache,
        positions: &Array,
        cfg: &QwenMoeConfig,
    ) -> anyhow::Result<Array> {
        let b = x.dim(0)?;
        let l = x.dim(1)?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = ops::reshape(&q, &[b, l, cfg.num_attention_heads as usize, cfg.head_dim as usize])?;
        let q = ops::transpose(&q, &[0, 2, 1, 3])?;

        let k = ops::reshape(&k, &[b, l, cfg.num_key_value_heads as usize, cfg.head_dim as usize])?;
        let k = ops::transpose(&k, &[0, 2, 1, 3])?;

        let v = ops::reshape(&v, &[b, l, cfg.num_key_value_heads as usize, cfg.head_dim as usize])?;
        let v = ops::transpose(&v, &[0, 2, 1, 3])?;

        let (q, k) = if let (Some(ref qn), Some(ref kn)) = (&self.q_norm, &self.k_norm) {
            (qn.forward(&q)?, kn.forward(&k)?)
        } else {
            (q, k)
        };

        let q = ops::fast_rope_dynamic(&q, cfg.head_dim, false, Some(cfg.rope_theta), 1.0, positions, None)?;
        let k = ops::fast_rope_dynamic(&k, cfg.head_dim, false, Some(cfg.rope_theta), 1.0, positions, None)?;

        let (k, v) = kv.update(&k, &v)?;

        let scale = cfg.scale();
        let sdpa_mode = if l > 1 { "causal" } else { "" };
        let out = ops::fast_sdpa(&q, &k, &v, scale, sdpa_mode, None)?;

        let out = ops::transpose(&out, &[0, 2, 1, 3])?;
        let out = ops::reshape(&out, &[b, l, (cfg.num_attention_heads * cfg.head_dim) as usize])?;
        self.o_proj.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// QwenMoeLayer — decoder layer
// ---------------------------------------------------------------------------

struct QwenMoeLayer {
    attention: QwenMoeAttention,
    ff: FeedForward,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl QwenMoeLayer {
    fn forward(
        &self,
        x: &Array,
        kv: &mut KvCache,
        positions: &Array,
        cfg: &QwenMoeConfig,
    ) -> anyhow::Result<Array> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.attention.forward(&normed, kv, positions, cfg)?;
        let h = ops::add(x, &attn_out)?;
        let normed = self.post_attention_layernorm.forward(&h)?;
        let ff_out = self.ff.forward(&normed, cfg)?;
        ops::add(&h, &ff_out)
    }
}

// ---------------------------------------------------------------------------
// QwenMoeModel
// ---------------------------------------------------------------------------

pub struct QwenMoeModel {
    embed_tokens: Box<dyn EmbeddingLayer>,
    layers: Vec<QwenMoeLayer>,
    norm: RmsNorm,
    lm_head: Box<dyn LinearLayer>,
    config: QwenMoeConfig,
}

impl Model for QwenMoeModel {
    fn forward(
        &self,
        input_ids: &Array,
        caches: &mut [LayerCache],
        positions: &Array,
    ) -> anyhow::Result<Array> {
        let mut h = self.embed_tokens.forward(input_ids)?;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(LayerCache::Attention(kv)) = caches.get_mut(i) {
                h = layer.forward(&h, kv, positions, &self.config)?;
            }
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
        (0..self.layers.len())
            .map(|_| LayerCache::Attention(KvCache::new()))
            .collect()
    }
}

impl QwenMoeModel {
    pub fn config(&self) -> &QwenMoeConfig {
        &self.config
    }

    pub fn load_from_tensors(
        mut tensors: HashMap<String, Array>,
        config: QwenMoeConfig,
    ) -> anyhow::Result<Self> {
        let prefix = resolve_weight_prefix(&tensors);

        let embed_key = format!("{prefix}model.embed_tokens.weight");
        let embed_tokens = make_embedding(&mut tensors, &embed_key)?;

        let norm_w = take(&mut tensors, &format!("{prefix}model.norm.weight"))?;
        let norm = RmsNorm::new(norm_w, config.rms_norm_eps);

        let lm_head: Box<dyn LinearLayer> = if config.tie_word_embeddings {
            make_linear(&mut tensors, &embed_key)?
        } else if tensors.contains_key(&format!("{prefix}lm_head.weight")) {
            make_linear(&mut tensors, &format!("{prefix}lm_head.weight"))?
        } else {
            make_linear(&mut tensors, &embed_key)?
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for i in 0..config.num_hidden_layers as usize {
            let lp = format!("{prefix}model.layers.{i}");

            let attention = QwenMoeAttention {
                q_proj: make_linear_with_bias(&mut tensors, &format!("{lp}.self_attn.q_proj.weight"))?,
                k_proj: make_linear_with_bias(&mut tensors, &format!("{lp}.self_attn.k_proj.weight"))?,
                v_proj: make_linear_with_bias(&mut tensors, &format!("{lp}.self_attn.v_proj.weight"))?,
                o_proj: make_linear_with_bias(&mut tensors, &format!("{lp}.self_attn.o_proj.weight"))?,
                q_norm: if tensors.contains_key(&format!("{lp}.self_attn.q_norm.weight")) {
                    Some(RmsNorm::new(
                        take(&mut tensors, &format!("{lp}.self_attn.q_norm.weight"))?,
                        config.rms_norm_eps,
                    ))
                } else {
                    None
                },
                k_norm: if tensors.contains_key(&format!("{lp}.self_attn.k_norm.weight")) {
                    Some(RmsNorm::new(
                        take(&mut tensors, &format!("{lp}.self_attn.k_norm.weight"))?,
                        config.rms_norm_eps,
                    ))
                } else {
                    None
                },
            };

            let ff = if config.is_moe_layer(i) {
                let num_experts = config.num_experts as usize;
                let experts_prefix = format!("{lp}.mlp.experts");

                let gate = make_linear(&mut tensors, &format!("{lp}.mlp.gate.weight"))?;

                let switch_mlp = SwitchGlu {
                    gate_proj: load_switch_linear(&mut tensors, &experts_prefix, "gate_proj", num_experts)?,
                    up_proj: load_switch_linear(&mut tensors, &experts_prefix, "up_proj", num_experts)?,
                    down_proj: load_switch_linear(&mut tensors, &experts_prefix, "down_proj", num_experts)?,
                };

                let shared_expert = if tensors
                    .contains_key(&format!("{lp}.mlp.shared_expert.gate_proj.weight"))
                {
                    Some(SharedExpert {
                        gate_proj: make_linear(
                            &mut tensors,
                            &format!("{lp}.mlp.shared_expert.gate_proj.weight"),
                        )?,
                        up_proj: make_linear(
                            &mut tensors,
                            &format!("{lp}.mlp.shared_expert.up_proj.weight"),
                        )?,
                        down_proj: make_linear(
                            &mut tensors,
                            &format!("{lp}.mlp.shared_expert.down_proj.weight"),
                        )?,
                    })
                } else {
                    None
                };

                let shared_expert_gate = if tensors
                    .contains_key(&format!("{lp}.mlp.shared_expert_gate.weight"))
                {
                    Some(make_linear(
                        &mut tensors,
                        &format!("{lp}.mlp.shared_expert_gate.weight"),
                    )?)
                } else {
                    None
                };

                FeedForward::Moe(SparseMoeBlock {
                    gate,
                    switch_mlp,
                    shared_expert,
                    shared_expert_gate,
                    num_experts_per_tok: config.num_experts_per_tok,
                    norm_topk_prob: config.norm_topk_prob,
                })
            } else {
                FeedForward::Dense(DenseMlp {
                    gate_proj: make_linear(&mut tensors, &format!("{lp}.mlp.gate_proj.weight"))?,
                    up_proj: make_linear(&mut tensors, &format!("{lp}.mlp.up_proj.weight"))?,
                    down_proj: make_linear(&mut tensors, &format!("{lp}.mlp.down_proj.weight"))?,
                })
            };

            let input_layernorm = RmsNorm::new(
                take(&mut tensors, &format!("{lp}.input_layernorm.weight"))?,
                config.rms_norm_eps,
            );
            let post_attention_layernorm = RmsNorm::new(
                take(&mut tensors, &format!("{lp}.post_attention_layernorm.weight"))?,
                config.rms_norm_eps,
            );

            layers.push(QwenMoeLayer {
                attention,
                ff,
                input_layernorm,
                post_attention_layernorm,
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

fn stack_arrays(arrays: &[Array]) -> anyhow::Result<Array> {
    let expanded: Vec<Array> = arrays
        .iter()
        .map(|a| ops::expand_dims(a, 0))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let refs: Vec<&Array> = expanded.iter().collect();
    ops::concatenate(&refs, 0)
}

/// Load expert weights and stack into 3D quantized arrays for gather_qmm.
fn load_switch_linear(
    tensors: &mut HashMap<String, Array>,
    experts_prefix: &str,
    proj: &str,
    num_experts: usize,
) -> anyhow::Result<SwitchLinear> {
    let mut weights = Vec::with_capacity(num_experts);
    let mut scales_vec: Vec<Array> = Vec::new();
    let mut biases_vec: Vec<Array> = Vec::new();
    let mut bias_vec: Vec<Array> = Vec::new();

    for e in 0..num_experts {
        let base_key = format!("{experts_prefix}.{e}.{proj}.weight");
        let base = base_key.strip_suffix(".weight").unwrap_or(&base_key);
        let scale_key = format!("{base}_scale");

        weights.push(take(tensors, &base_key)?);

        if tensors.contains_key(&scale_key) {
            scales_vec.push(take(tensors, &scale_key)?);
        }

        let qbias_key = format!("{base}_qbias");
        if let Some(b) = tensors.remove(&qbias_key) {
            biases_vec.push(b);
        }

        let attn_bias_key = format!("{base}.bias");
        if let Some(b) = tensors.remove(&attn_bias_key) {
            bias_vec.push(b);
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

    let (group_size, bits, mode) = if let Some(ref s) = scales {
        let w_cols = weight.dim(weight.ndim() - 1)? as i32;
        let s_cols = s.dim(s.ndim() - 1)? as i32;
        infer_quant_params(w_cols, s_cols)
    } else {
        (0, 0, String::new())
    };

    Ok(SwitchLinear {
        weight,
        scales,
        biases,
        bias,
        group_size,
        bits,
        mode,
    })
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
            weight,
            scales,
            biases,
            Some(group_size),
            Some(bits),
            mode,
        )))
    } else {
        let weight = take(tensors, base_key)?;
        Ok(Box::new(Linear::new(weight, None)))
    }
}

fn make_linear_with_bias(
    tensors: &mut HashMap<String, Array>,
    base_key: &str,
) -> anyhow::Result<Box<dyn LinearLayer>> {
    let base = base_key.strip_suffix(".weight").unwrap_or(base_key);
    let scale_key = format!("{base}_scale");
    let attn_bias_key = format!("{base}.bias");
    let attn_bias = tensors.remove(&attn_bias_key);
    if tensors.contains_key(&scale_key) {
        let weight = take(tensors, base_key)?;
        let scales = take(tensors, &scale_key)?;
        let qbias_key = format!("{base}_qbias");
        let biases = tensors.remove(&qbias_key);
        let w_cols = weight.dim(weight.ndim() - 1)? as i32;
        let s_cols = scales.dim(scales.ndim() - 1)? as i32;
        let (group_size, bits, mode) = infer_quant_params(w_cols, s_cols);
        let ql = crate::llama::QuantizedLinear::new(
            weight,
            scales,
            biases,
            Some(group_size),
            Some(bits),
            mode,
        );
        Ok(Box::new(match attn_bias {
            Some(b) => ql.with_attn_bias(b),
            None => ql,
        }))
    } else {
        let weight = take(tensors, base_key)?;
        Ok(Box::new(Linear::new(weight, attn_bias)))
    }
}

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
            weight,
            scales,
            biases,
            Some(group_size),
            Some(bits),
            mode,
        )))
    } else {
        let weight = take(tensors, base_key)?;
        Ok(Box::new(Embedding::new(weight)))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen_moe_config() {
        let json = serde_json::json!({
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "moe_intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 100,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 128,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "norm_topk_prob": false,
            "decoder_sparse_step": 1,
            "shared_expert_intermediate_size": 128,
        });
        let cfg = QwenMoeConfig::from_json(&json).unwrap();
        assert_eq!(cfg.hidden_size, 64);
        assert_eq!(cfg.num_experts, 8);
        assert_eq!(cfg.num_experts_per_tok, 2);
        assert!(cfg.is_moe_layer(0));
        assert!(cfg.is_moe_layer(1));
    }
}
