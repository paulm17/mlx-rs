use std::collections::HashMap;

use crate::array::Array;
use crate::llama::{EmbeddingLayer, KvCache, LinearLayer, RmsNorm, resolve_weight_prefix};
use crate::model::Model;
use crate::ops;

// ---------------------------------------------------------------------------
// GeLU helper (tanh approximation) — matches mlx.GeGLU gate behavior
// ---------------------------------------------------------------------------

fn gelu(x: &Array) -> anyhow::Result<Array> {
    let half = ops::multiply(x, &Array::from_f32(0.5)?)?;
    let x_cubed = ops::multiply(x, &ops::multiply(x, x)?)?;
    let inner = ops::add(x, &ops::multiply(&x_cubed, &Array::from_f32(0.044715)?)?)?;
    let scaled = ops::multiply(&inner, &Array::from_f32((2.0 / std::f32::consts::PI).sqrt())?)?;
    let tanh_val = ops::tanh(&scaled)?;
    let one_plus_tanh = ops::add(&Array::from_f32(1.0)?, &tanh_val)?;
    ops::multiply(&half, &one_plus_tanh)
}

// ---------------------------------------------------------------------------
// Weightless RMSNorm (V normalization — no learnable parameter)
// Operates on the last dimension: rms_norm(x) = x / sqrt(mean(x^2) + eps)
// ---------------------------------------------------------------------------

fn rms_norm_weightless(x: &Array, eps: f32) -> anyhow::Result<Array> {
    let ndim = x.ndim();
    let last_dim = x.dim(ndim - 1)? as f32;
    let x_sq = ops::multiply(x, x)?;
    let sum_sq = ops::sum_axis(&x_sq, ndim - 1, true)?;
    let mean_sq = ops::divide(&sum_sq, &Array::from_f32(last_dim)?)?;
    let mean_sq_eps = ops::add(&mean_sq, &Array::from_f32(eps)?)?;
    let rms = ops::sqrt(&mean_sq_eps)?;
    ops::divide(x, &rms)
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Gemma4Config {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub num_global_key_value_heads: i32,
    pub head_dim: i32,
    pub global_head_dim: i32,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: i32,
    pub tie_word_embeddings: bool,
    pub sliding_window: i32,
    pub sliding_window_pattern: i32,
    pub final_logit_softcapping: f32,
    pub attention_k_eq_v: bool,
    pub num_kv_shared_layers: i32,
    pub hidden_size_per_layer_input: i32,
    pub layer_types: Vec<String>,

    // Computed RoPE settings
    pub sliding_rope_base: f32,
    pub full_rope_base: f32,
    pub sliding_rope_dims: i32,
    pub full_rope_dims: i32,
    pub full_rope_freqs: Option<Array>, // Precomputed for partial_rotary_factor

    // Computed scale factors (Ollama uses 1.0 — Q/K norms handle magnitude)
    pub sliding_scale: f32,
    pub full_scale: f32,

    // Computed PLE scales
    pub embed_scale: f32,
    pub ple_scale: f32,
    pub ple_proj_scale: f32,
    pub ple_combine_scale: f32,

    // KV sharing
    pub kv_share_map: HashMap<i32, i32>,
    pub kv_donors: HashMap<i32, bool>,
}

impl Gemma4Config {
    pub fn from_json(config: &serde_json::Value) -> anyhow::Result<Self> {
        // Gemma4 models may have a top-level text_config sub-object
        let c = config.get("text_config").unwrap_or(config);

        let get_i32 = |k: &str, d: i32| -> i32 { c.get(k).and_then(|v| v.as_i64()).unwrap_or(d as i64) as i32 };
        let get_f32 = |k: &str, d: f32| -> f32 { c.get(k).and_then(|v| v.as_f64()).unwrap_or(d as f64) as f32 };
        let get_bool = |k: &str, d: bool| -> bool { c.get(k).and_then(|v| v.as_bool()).unwrap_or(d) };

        let hidden_size = get_i32("hidden_size", 0);
        let num_hidden_layers = get_i32("num_hidden_layers", 0);
        let intermediate_size = get_i32("intermediate_size", 0);
        let num_attention_heads = get_i32("num_attention_heads", 8);
        let num_key_value_heads = get_i32("num_key_value_heads", 1);
        let head_dim = get_i32("head_dim", 256);
        let global_head_dim = get_i32("global_head_dim", head_dim);
        let vocab_size = get_i32("vocab_size", 262144);
        let rms_norm_eps = get_f32("rms_norm_eps", 1e-6);
        let max_position_embeddings = get_i32("max_position_embeddings", 131072);
        let tie_word_embeddings = get_bool("tie_word_embeddings", false);
        let sliding_window = get_i32("sliding_window", 0);
        let sliding_window_pattern = get_i32("sliding_window_pattern", 5);
        let final_logit_softcapping = get_f32("final_logit_softcapping", 0.0);
        let attention_k_eq_v = get_bool("attention_k_eq_v", false);
        let num_kv_shared_layers = get_i32("num_kv_shared_layers", 0);
        let hidden_size_per_layer_input = get_i32("hidden_size_per_layer_input", 0);
        let num_global_key_value_heads = if attention_k_eq_v {
            get_i32("num_global_key_value_heads", 0)
        } else {
            0
        };

        let layer_types: Vec<String> = c
            .get("layer_types")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();

        if hidden_size <= 0 || num_attention_heads <= 0 {
            anyhow::bail!("invalid Gemma4 config: hidden_size={hidden_size} heads={num_attention_heads}");
        }

        // Compute RoPE settings from rope_parameters (matches Ollama's parseTextConfig)
        let mut sliding_rope_base: f32 = 10000.0;
        let mut full_rope_base: f32 = 1000000.0;
        let sliding_rope_dims = head_dim; // full rotation for sliding
        let mut full_rope_dims = head_dim; // default: full rotation
        let mut full_rope_freqs: Option<Array> = None;

        if let Some(rp) = c.get("rope_parameters") {
            if let Some(sp) = rp.get("sliding_attention") {
                if let Some(theta) = sp.get("rope_theta").and_then(|v| v.as_f64()) {
                    if theta > 0.0 {
                        sliding_rope_base = theta as f32;
                    }
                }
            }
            if let Some(fp) = rp.get("full_attention") {
                if let Some(theta) = fp.get("rope_theta").and_then(|v| v.as_f64()) {
                    if theta > 0.0 {
                        full_rope_base = theta as f32;
                    }
                }
                // Partial rotary factor: some dims use standard RoPE, others use 1e10 (identity)
                if let Some(partial) = fp.get("partial_rotary_factor").and_then(|v| v.as_f64()) {
                    if partial > 0.0 {
                        let ghd = global_head_dim as usize;
                        full_rope_dims = global_head_dim;
                        let half_dim = ghd / 2;
                        let rope_angles = (partial * ghd as f64 / 2.0) as usize;
                        let mut freqs = Vec::with_capacity(half_dim);
                        for i in 0..rope_angles {
                            freqs.push(full_rope_base.powf(2.0 * i as f32 / ghd as f32));
                        }
                        for _ in rope_angles..half_dim {
                            freqs.push(1e10);
                        }
                        full_rope_freqs = Some(Array::from_data_f32(&freqs, &[half_dim])?);
                    }
                }
            }
        }

        // Gemma 4 uses scale=1.0 — Q/K norms handle magnitude control
        let sliding_scale = 1.0f32;
        let full_scale = 1.0f32;

        // Precompute PLE scale factors
        let embed_scale = (hidden_size as f32).sqrt();
        let (ple_scale, ple_proj_scale, ple_combine_scale) = if hidden_size_per_layer_input > 0 {
            (
                (hidden_size_per_layer_input as f32).sqrt(),
                1.0 / (hidden_size as f32).sqrt(),
                2.0_f32.powf(-0.5),
            )
        } else {
            (0.0, 0.0, 0.0)
        };

        // Build effective layer_types for KV sharing (synthesize from pattern if not in config)
        let effective_layer_types: Vec<String> = if !layer_types.is_empty() {
            layer_types.clone()
        } else if sliding_window_pattern > 0 {
            (0..num_hidden_layers)
                .map(|i| {
                    if (i + 1) % sliding_window_pattern == 0 {
                        "full_attention".to_string()
                    } else {
                        "sliding_attention".to_string()
                    }
                })
                .collect()
        } else {
            vec![]
        };

        // Compute KV sharing map (matches Ollama's logic)
        let mut kv_share_map = HashMap::new();
        let mut kv_donors: HashMap<i32, bool> = HashMap::new();
        if num_kv_shared_layers > 0 && !effective_layer_types.is_empty() {
            let first_shared = num_hidden_layers - num_kv_shared_layers;
            let prev_layers = &effective_layer_types[..first_shared as usize];

            for i in first_shared..num_hidden_layers {
                let layer_type = &effective_layer_types[i as usize];
                // Find the last non-shared layer of the same type
                if let Some(donor) = prev_layers.iter().enumerate().rev()
                    .find(|(_, t)| *t == layer_type)
                    .map(|(j, _)| j as i32)
                {
                    kv_share_map.insert(i, donor);
                    kv_donors.insert(donor, true);
                }
            }
        }

        Ok(Self {
            hidden_size,
            num_hidden_layers,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            num_global_key_value_heads,
            head_dim,
            global_head_dim,
            vocab_size,
            rms_norm_eps,
            max_position_embeddings,
            tie_word_embeddings,
            sliding_window,
            sliding_window_pattern,
            final_logit_softcapping,
            attention_k_eq_v,
            num_kv_shared_layers,
            hidden_size_per_layer_input,
            layer_types,
            sliding_rope_base,
            full_rope_base,
            sliding_rope_dims,
            full_rope_dims,
            full_rope_freqs,
            sliding_scale,
            full_scale,
            embed_scale,
            ple_scale,
            ple_proj_scale,
            ple_combine_scale,
            kv_share_map,
            kv_donors,
        })
    }
}

// ---------------------------------------------------------------------------
// Layer type helpers
// ---------------------------------------------------------------------------

pub fn is_layer_sliding(layer_idx: i32, pattern: i32, layer_types: &[String]) -> bool {
    if !layer_types.is_empty() && (layer_idx as usize) < layer_types.len() {
        return layer_types[layer_idx as usize] == "sliding_attention";
    }
    if pattern <= 0 {
        return false;
    }
    (layer_idx + 1) % pattern != 0
}

// ---------------------------------------------------------------------------
// Gemma4Attention — Q/K norm, V norm, optional K=V for global layers
// ---------------------------------------------------------------------------

pub struct Gemma4Attention {
    q_proj: Box<dyn LinearLayer>,
    k_proj: Box<dyn LinearLayer>,
    v_proj: Option<Box<dyn LinearLayer>>,
    o_proj: Box<dyn LinearLayer>,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    is_sliding: bool,
    head_dim: i32,
    kv_heads: i32,
    n_heads: i32,
    rope_theta: f32,
    rope_dims: i32,
    full_rope_freqs: Option<Array>,
    scale: f32,
    eps: f32,
}

impl Gemma4Attention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q: Box<dyn LinearLayer>,
        k: Box<dyn LinearLayer>,
        v: Option<Box<dyn LinearLayer>>,
        o: Box<dyn LinearLayer>,
        qn: RmsNorm,
        kn: RmsNorm,
        is_sliding: bool,
        head_dim: i32,
        kv_heads: i32,
        n_heads: i32,
        rope_theta: f32,
        rope_dims: i32,
        full_rope_freqs: Option<Array>,
        scale: f32,
        eps: f32,
    ) -> Self {
        Self {
            q_proj: q,
            k_proj: k,
            v_proj: v,
            o_proj: o,
            q_norm: qn,
            k_norm: kn,
            is_sliding,
            head_dim,
            kv_heads,
            n_heads,
            rope_theta,
            rope_dims,
            full_rope_freqs,
            scale,
            eps,
        }
    }

    pub fn forward(
        &self,
        x: &Array,
        kv: &mut KvCache,
        positions: &Array,
    ) -> anyhow::Result<Array> {
        let b = x.dim(0)?;
        let l = x.dim(1)?;
        let hd = self.head_dim as usize;

        // Q projection and norm
        let q = self.q_proj.forward(x)?;
        let q = ops::reshape(&q, &[b, l, self.n_heads as usize, hd])?;
        let q = ops::transpose(&q, &[0, 2, 1, 3])?;
        let q = self.q_norm.forward(&q)?;

        // RoPE for Q — use custom freqs for full-attention with partial_rotary_factor
        let freqs_ref = if !self.is_sliding {
            self.full_rope_freqs.as_ref()
        } else {
            None
        };
        let rope_base = if freqs_ref.is_some() { None } else { Some(self.rope_theta) };
        let q = ops::fast_rope_dynamic(
            &q,
            self.rope_dims,
            false,
            rope_base,
            1.0,
            positions,
            freqs_ref,
        )?;

        // Get or compute K/V
        let (ks, vs) = if let Some((dk, dv, _)) = &kv.donor {
            (dk.clone(), dv.clone())
        } else {
            let k = self.k_proj.forward(x)?;
            let k = ops::reshape(&k, &[b, l, self.kv_heads as usize, hd])?;
            let k = ops::transpose(&k, &[0, 2, 1, 3])?;

            // V: use v_proj if present, else V=K (attention_k_eq_v for global layers)
            let v = match &self.v_proj {
                Some(vp) => {
                    let v = vp.forward(x)?;
                    let v = ops::reshape(&v, &[b, l, self.kv_heads as usize, hd])?;
                    ops::transpose(&v, &[0, 2, 1, 3])?
                }
                None => k.clone(),
            };

            // K norm + RoPE
            let k_normed = self.k_norm.forward(&k)?;
            let k_roped = ops::fast_rope_dynamic(
                &k_normed,
                self.rope_dims,
                false,
                rope_base,
                1.0,
                positions,
                freqs_ref,
            )?;

            // V normalization (weightless RMSNorm — matches Ollama's RMSNormFn(v, nil, eps))
            let v_normed = rms_norm_weightless(&v, self.eps)?;

            let (k_cached, v_cached) = kv.update(&k_roped, &v_normed)?;
            (k_cached, v_cached)
        };

        // Scaled dot-product attention
        // Ollama: causal mask for prefill (L>1), no mask for decode (L=1)
        let sdpa_mode = if l > 1 { "causal" } else { "" };
        let out = ops::fast_sdpa(&q, &ks, &vs, self.scale, sdpa_mode, None)?;
        let out = ops::transpose(&out, &[0, 2, 1, 3])?;
        let out = ops::reshape(&out, &[b, l, self.n_heads as usize * hd])?;
        self.o_proj.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// Gemma4Mlp (GeGLU — gate * gelu(up))
// ---------------------------------------------------------------------------

pub struct Gemma4Mlp {
    gate_proj: Box<dyn LinearLayer>,
    up_proj: Box<dyn LinearLayer>,
    down_proj: Box<dyn LinearLayer>,
}

impl Gemma4Mlp {
    pub fn forward(&self, x: &Array) -> anyhow::Result<Array> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let gate_gelu = gelu(&gate)?;
        let gated = ops::multiply(&gate_gelu, &up)?;
        self.down_proj.forward(&gated)
    }
}

// ---------------------------------------------------------------------------
// PLE (Per-Layer Embeddings)
// ---------------------------------------------------------------------------

pub struct PleLayer {
    input_gate: Box<dyn LinearLayer>,
    projection: Box<dyn LinearLayer>,
    post_norm: RmsNorm,
}

impl PleLayer {
    pub fn forward(&self, h: &Array, ple_input: &Array) -> anyhow::Result<Array> {
        let gate = self.input_gate.forward(h)?;
        let gated = gelu(&gate)?;
        let gated = ops::multiply(&gated, ple_input)?;
        let proj = self.projection.forward(&gated)?;
        self.post_norm.forward(&proj)
    }
}

pub struct PleProjection {
    pl_embed: Box<dyn EmbeddingLayer>,
    per_layer_proj: Box<dyn LinearLayer>,
    proj_norm: RmsNorm,
    embed_scale: f32,
    proj_scale: f32,
    combine_scale: f32,
    num_layers: usize,
    ple_dim: usize,
}

impl PleProjection {
    pub fn compute(&self, input_ids: &Array, h: &Array) -> anyhow::Result<Array> {
        let b = h.dim(0)?;
        let l = h.dim(1)?;

        // Token-based per-layer embeddings: [B, L, NumLayers*PLEDim]
        let ple_emb = self.pl_embed.forward(input_ids)?;
        let ple_emb = ops::multiply(&ple_emb, &Array::from_f32(self.embed_scale)?)?;
        // Reshape to [B, L, NumLayers, PLEDim]
        let ple_emb = ops::reshape(&ple_emb, &[b, l, self.num_layers, self.ple_dim])?;

        // Hidden-state projection: [B, L, NumLayers*PLEDim]
        let ple_proj = self.per_layer_proj.forward(h)?;
        let ple_proj = ops::multiply(&ple_proj, &Array::from_f32(self.proj_scale)?)?;
        // Reshape to [B, L, NumLayers, PLEDim]
        let ple_proj = ops::reshape(&ple_proj, &[b, l, self.num_layers, self.ple_dim])?;
        let ple_proj = self.proj_norm.forward(&ple_proj)?;

        // Combine: (proj + emb) * 2^(-0.5)
        let combined = ops::add(&ple_proj, &ple_emb)?;
        ops::multiply(&combined, &Array::from_f32(self.combine_scale)?)
    }

    pub fn ple_dim(&self) -> usize {
        self.ple_dim
    }
}

// ---------------------------------------------------------------------------
// Gemma4Layer
// ---------------------------------------------------------------------------

pub struct Gemma4Layer {
    attention: Gemma4Attention,
    mlp: Gemma4Mlp,
    attention_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    pre_feedforward_norm: RmsNorm,
    post_feedforward_norm: RmsNorm,
    ple: Option<PleLayer>,
    layer_scalar: Option<Array>,
}

impl Gemma4Layer {
    pub fn forward(
        &self,
        x: &Array,
        kv: &mut KvCache,
        positions: &Array,
        ple_input: Option<&Array>,
    ) -> anyhow::Result<Array> {
        // Pre-attention norm
        let normed = self.attention_norm.forward(x)?;
        let attn_out = self.attention.forward(&normed, kv, positions)?;
        let attn_normed = self.post_attention_norm.forward(&attn_out)?;
        let mut h = ops::add(x, &attn_normed)?;

        // MLP with pre/post norms
        let pre_norm = self.pre_feedforward_norm.forward(&h)?;
        let mlp_out = self.mlp.forward(&pre_norm)?;
        let mlp_normed = self.post_feedforward_norm.forward(&mlp_out)?;
        h = ops::add(&h, &mlp_normed)?;

        // PLE injection
        if let (Some(ple), Some(pi)) = (&self.ple, ple_input) {
            let ple_out = ple.forward(&h, pi)?;
            h = ops::add(&h, &ple_out)?;
        }

        // Layer scalar (for full-attention layers)
        if let Some(scalar) = &self.layer_scalar {
            h = ops::multiply(&h, scalar)?;
        }

        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// Gemma4Model
// ---------------------------------------------------------------------------

pub struct Gemma4Model {
    embed_tokens: Box<dyn EmbeddingLayer>,
    layers: Vec<Gemma4Layer>,
    norm: RmsNorm,
    lm_head: Box<dyn LinearLayer>,
    config: Gemma4Config,
    ple: Option<PleProjection>,
}

impl Model for Gemma4Model {
    fn forward(
        &self,
        input_ids: &Array,
        caches: &mut [KvCache],
        positions: &Array,
    ) -> anyhow::Result<Array> {
        let b = input_ids.dim(0)?;
        let l = input_ids.dim(1)?;

        let mut h = self.embed_tokens.forward(input_ids)?;
        h = ops::multiply(&h, &Array::from_f32(self.config.embed_scale)?)?;

        // Precompute PLE inputs if configured
        let ple_tensor: Option<Array> = if let Some(ple) = &self.ple {
            Some(ple.compute(input_ids, &h)?)
        } else {
            None
        };

        // KV sharing: donor layer index -> (k, v)
        let mut shared_kv: HashMap<i32, (Array, Array)> = HashMap::new();

        for i in 0..self.config.num_hidden_layers as usize {
            // Set up donor KV for shared layers
            if let Some(&donor) = self.config.kv_share_map.get(&(i as i32)) {
                if let Some((dk, dv)) = shared_kv.get(&donor) {
                    caches[i].donor = Some((dk.clone(), dv.clone(), 1.0));
                }
            }

            // Slice PLE input for this layer: [B,L,NumLayers,PLEDim] -> [B,L,PLEDim]
            let ple_input: Option<Array> = ple_tensor.as_ref().map(|pt| {
                let dim = self.ple.as_ref().unwrap().ple_dim();
                let flat = ops::reshape(
                    pt,
                    &[b * l, self.config.num_hidden_layers as usize, dim],
                )
                .unwrap();
                let idx = Array::from_i32(i as i32).unwrap();
                let sliced = ops::take(&flat, &idx, 1).unwrap();
                ops::reshape(&sliced, &[b, l, dim]).unwrap()
            });

            h = self.layers[i].forward(&h, &mut caches[i], positions, ple_input.as_ref())?;
            let _is_sliding = is_layer_sliding(i as i32, self.config.sliding_window_pattern, &self.config.layer_types);

            // If this is a donor layer, store its KV for donees
            if self
                .config
                .kv_donors
                .get(&(i as i32))
                .copied()
                .unwrap_or(false)
                && caches[i].donor.is_none()
            {
                if let (Some(k), Some(v)) = (&caches[i].k_cache, &caches[i].v_cache) {
                    shared_kv.insert(i as i32, (k.clone(), v.clone()));
                }
            }
        }

        // Final norm + LM head
        let h = self.norm.forward(&h)?;
        let logits = self.lm_head.forward(&h)?;

        // Logit softcapping
        if self.config.final_logit_softcapping > 0.0 {
            let cap = Array::from_f32(self.config.final_logit_softcapping)?;
            let scaled = ops::divide(&logits, &cap)?;
            let tanh_val = ops::tanh(&scaled)?;
            ops::multiply(&cap, &tanh_val)
        } else {
            Ok(logits)
        }
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

    fn new_caches(&self) -> Vec<KvCache> {
        let sw = self.config.sliding_window as usize;
        (0..self.config.num_hidden_layers as usize)
            .map(|i| {
                let is_sliding = is_layer_sliding(i as i32, self.config.sliding_window_pattern, &self.config.layer_types);
                if is_sliding && sw > 0 {
                    KvCache::new_rotating(sw)
                } else {
                    KvCache::new()
                }
            })
            .collect()
    }
}

impl Gemma4Model {
    pub fn config(&self) -> &Gemma4Config {
        &self.config
    }

    pub fn load_from_tensors(
        mut tensors: HashMap<String, Array>,
        config: Gemma4Config,
    ) -> anyhow::Result<Self> {
        let prefix = resolve_weight_prefix(&tensors);

        // Embeddings and LM head (tied by default for Gemma4)
        let embed_key = format!("{prefix}model.embed_tokens.weight");
        let (embed_tokens, tied_lm_head) = make_embedding_and_linear(&mut tensors, &embed_key)?;

        // Final norm
        let norm_w = tensors
            .remove(&format!("{prefix}model.norm.weight"))
            .ok_or_else(|| anyhow::anyhow!("missing {prefix}model.norm.weight"))?;
        let norm = RmsNorm::new(norm_w, config.rms_norm_eps);

        // LM head (tied to embeddings by default)
        let lm_head: Box<dyn LinearLayer> = if config.tie_word_embeddings {
            tied_lm_head
        } else if tensors.contains_key(&format!("{prefix}lm_head.weight")) {
            make_linear(&mut tensors, &format!("{prefix}lm_head.weight"))?
        } else {
            tied_lm_head
        };

        // PLE model-level weights
        let ple = if config.hidden_size_per_layer_input > 0 {
            let pl_embed_key = format!("{prefix}model.embed_tokens_per_layer.weight");
            let per_layer_proj_key = format!("{prefix}model.per_layer_model_projection.weight");
            let proj_norm_w = tensors
                .remove(&format!("{prefix}model.per_layer_projection_norm.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing per_layer_projection_norm.weight"))?;

            let ple_dim = config.hidden_size_per_layer_input as usize;
            let num_layers = config.num_hidden_layers as usize;
            Some(PleProjection {
                pl_embed: make_embedding(&mut tensors, &pl_embed_key)?,
                per_layer_proj: make_linear(&mut tensors, &per_layer_proj_key)?,
                proj_norm: RmsNorm::new(proj_norm_w, config.rms_norm_eps),
                embed_scale: config.ple_scale,
                proj_scale: config.ple_proj_scale,
                combine_scale: config.ple_combine_scale,
                num_layers,
                ple_dim,
            })
        } else {
            None
        };

        // Decoder layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for i in 0..config.num_hidden_layers as usize {
            let lp = format!("{prefix}model.layers.{i}");
            let is_sliding = is_layer_sliding(
                i as i32,
                config.sliding_window_pattern,
                &config.layer_types,
            );

            // Per-layer RoPE settings
            let (head_dim, rope_theta, rope_dims) = if is_sliding {
                (config.head_dim, config.sliding_rope_base, config.sliding_rope_dims)
            } else {
                (config.global_head_dim, config.full_rope_base, config.full_rope_dims)
            };

            // KV heads: global K=V layers may use different head count
            let kv_heads = if config.attention_k_eq_v && !is_sliding && config.num_global_key_value_heads > 0 {
                config.num_global_key_value_heads
            } else {
                config.num_key_value_heads
            };

            let scale = if is_sliding { config.sliding_scale } else { config.full_scale };

            // Attention
            let attn = {
                let q = make_linear(&mut tensors, &format!("{lp}.self_attn.q_proj.weight"))?;
                let k = make_linear(&mut tensors, &format!("{lp}.self_attn.k_proj.weight"))?;
                let o = make_linear(&mut tensors, &format!("{lp}.self_attn.o_proj.weight"))?;
                let v = if config.attention_k_eq_v && !is_sliding {
                    None
                } else {
                    Some(make_linear(&mut tensors, &format!("{lp}.self_attn.v_proj.weight"))?)
                };
                let qn = RmsNorm::new(
                    take(&mut tensors, &format!("{lp}.self_attn.q_norm.weight"))?,
                    config.rms_norm_eps,
                );
                let kn = RmsNorm::new(
                    take(&mut tensors, &format!("{lp}.self_attn.k_norm.weight"))?,
                    config.rms_norm_eps,
                );
                let freqs = if !is_sliding {
                    config.full_rope_freqs.clone()
                } else {
                    None
                };
                Gemma4Attention::new(
                    q, k, v, o, qn, kn, is_sliding, head_dim, kv_heads,
                    config.num_attention_heads, rope_theta, rope_dims, freqs, scale, config.rms_norm_eps,
                )
            };

            // MLP
            let mlp = {
                let gate = make_linear(&mut tensors, &format!("{lp}.mlp.gate_proj.weight"))?;
                let up = make_linear(&mut tensors, &format!("{lp}.mlp.up_proj.weight"))?;
                let down = make_linear(&mut tensors, &format!("{lp}.mlp.down_proj.weight"))?;
                Gemma4Mlp { gate_proj: gate, up_proj: up, down_proj: down }
            };

            // Norms
            let attn_norm = RmsNorm::new(
                take(&mut tensors, &format!("{lp}.input_layernorm.weight"))?,
                config.rms_norm_eps,
            );
            let post_attn_norm = RmsNorm::new(
                take(&mut tensors, &format!("{lp}.post_attention_layernorm.weight"))?,
                config.rms_norm_eps,
            );
            let pre_ff_norm = RmsNorm::new(
                take(&mut tensors, &format!("{lp}.pre_feedforward_layernorm.weight"))?,
                config.rms_norm_eps,
            );
            let post_ff_norm = RmsNorm::new(
                take(&mut tensors, &format!("{lp}.post_feedforward_layernorm.weight"))?,
                config.rms_norm_eps,
            );

            // PLE per-layer
            let ple_layer = if config.hidden_size_per_layer_input > 0 {
                Some(PleLayer {
                    input_gate: make_linear(&mut tensors, &format!("{lp}.per_layer_input_gate.weight"))?,
                    projection: make_linear(&mut tensors, &format!("{lp}.per_layer_projection.weight"))?,
                    post_norm: RmsNorm::new(
                        take(&mut tensors, &format!("{lp}.post_per_layer_input_norm.weight"))?,
                        config.rms_norm_eps,
                    ),
                })
            } else {
                None
            };

            let layer_scalar = tensors.remove(&format!("{lp}.layer_scalar"));

            layers.push(Gemma4Layer {
                attention: attn,
                mlp,
                attention_norm: attn_norm,
                post_attention_norm: post_attn_norm,
                pre_feedforward_norm: pre_ff_norm,
                post_feedforward_norm: post_ff_norm,
                ple: ple_layer,
                layer_scalar,
            });
        }

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config,
            ple,
        })
    }
}

fn take(tensors: &mut HashMap<String, Array>, key: &str) -> anyhow::Result<Array> {
    tensors
        .remove(key)
        .ok_or_else(|| anyhow::anyhow!("missing tensor: {key}"))
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
        // Tied weight: embed and lm_head share packed arrays
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

fn infer_quant_params(weight_cols: i32, scale_cols: i32) -> (i32, i32, String) {
    // Ollama's shape-based inference: groupSize = weightCols * 8 / scaleCols for 4-bit,
    // weightCols * 4 / scaleCols for 8-bit
    if scale_cols == 0 {
        return (64, 4, "affine".to_string());
    }
    let group_size_4 = weight_cols * 8 / scale_cols;
    let group_size_8 = weight_cols * 4 / scale_cols;
    // Prefer 4-bit interpretation
    if group_size_4 == 32 {
        (32, 4, "mxfp4".to_string())
    } else if group_size_4 == 64 {
        (64, 4, "affine".to_string())
    } else if group_size_8 == 64 {
        (64, 8, "affine".to_string())
    } else if group_size_8 == 32 {
        (32, 8, "mxfp8".to_string())
    } else {
        // Default fallback
        (64, 4, "affine".to_string())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_e2b() {
        let json = serde_json::json!({
            "hidden_size": 1536, "num_hidden_layers": 35, "intermediate_size": 6144,
            "num_attention_heads": 8, "num_key_value_heads": 1,
            "head_dim": 256, "global_head_dim": 512,
            "rope_parameters": {
                "sliding_attention": { "rope_theta": 10000.0 },
                "full_attention": { "rope_theta": 1000000.0 }
            },
            "sliding_window": 512, "sliding_window_pattern": 5,
            "final_logit_softcapping": 30.0,
            "num_kv_shared_layers": 20,
            "hidden_size_per_layer_input": 256
        });
        let cfg = Gemma4Config::from_json(&json).unwrap();
        assert_eq!(cfg.hidden_size, 1536);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.global_head_dim, 512);
        assert_eq!(cfg.sliding_rope_base, 10000.0);
        assert_eq!(cfg.full_rope_base, 1000000.0);
        assert!(!cfg.kv_share_map.is_empty());
        assert_eq!(cfg.sliding_scale, 1.0);
        assert_eq!(cfg.full_scale, 1.0);
    }

    #[test]
    fn test_sliding_pattern() {
        let empty: Vec<String> = vec![];
        // Pattern 5: layers 0-3 sliding, 4 full, 5-8 sliding, 9 full, ...
        assert!(is_layer_sliding(0, 5, &empty));
        assert!(is_layer_sliding(3, 5, &empty));
        assert!(!is_layer_sliding(4, 5, &empty));
        assert!(is_layer_sliding(5, 5, &empty));
        assert!(!is_layer_sliding(9, 5, &empty));

        // Test with explicit layer_types
        let types: Vec<String> = vec![
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
        ].into_iter().map(String::from).collect();
        assert!(!is_layer_sliding(4, 0, &types));
        assert!(is_layer_sliding(0, 0, &types));
        assert!(!is_layer_sliding(9, 0, &types));
    }

    #[test]
    fn test_kv_share_map() {
        let types: Vec<String> = (0..35)
            .map(|i| {
                if (i + 1) % 5 == 0 {
                    "full_attention".to_string()
                } else {
                    "sliding_attention".to_string()
                }
            })
            .collect();
        let cfg_json = serde_json::json!({
            "hidden_size": 1536, "num_hidden_layers": 35,
            "num_attention_heads": 8, "num_key_value_heads": 1,
            "head_dim": 256, "sliding_window_pattern": 5,
            "num_kv_shared_layers": 20,
            "layer_types": types,
        });
        let cfg = Gemma4Config::from_json(&cfg_json).unwrap();
        // Layers 15-34 are shared, each should have a donor < 15
        for i in 15..35 {
            if let Some(&donor) = cfg.kv_share_map.get(&i) {
                assert!(donor < 15, "donor {donor} for layer {i} should be < 15");
                // Same type
                assert_eq!(
                    is_layer_sliding(i, 5, &cfg.layer_types),
                    is_layer_sliding(donor, 5, &cfg.layer_types),
                    "layer {i} and donor {donor} should have same type"
                );
            }
        }
    }

    #[test]
    fn test_config_defaults() {
        let json = serde_json::json!({
            "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4
        });
        let cfg = Gemma4Config::from_json(&json).unwrap();
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.global_head_dim, 256);
        assert_eq!(cfg.full_rope_base, 1000000.0);
        assert_eq!(cfg.sliding_rope_base, 10000.0);
        assert_eq!(cfg.sliding_window_pattern, 5);
        assert_eq!(cfg.final_logit_softcapping, 0.0);
        assert_eq!(cfg.sliding_scale, 1.0);
        assert_eq!(cfg.full_scale, 1.0);
    }

    #[test]
    fn test_partial_rotary_factor() {
        if crate::loader::check_init().is_err() {
            return; // MLX runtime not available
        }
        let json = serde_json::json!({
            "hidden_size": 1536, "num_hidden_layers": 4,
            "num_attention_heads": 8, "head_dim": 256, "global_head_dim": 512,
            "rope_parameters": {
                "full_attention": {
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 0.5
                },
                "sliding_attention": { "rope_theta": 10000.0 }
            }
        });
        let cfg = Gemma4Config::from_json(&json).unwrap();
        assert_eq!(cfg.full_rope_dims, 512);
        assert!(cfg.full_rope_freqs.is_some());
        assert_eq!(cfg.full_rope_base, 1000000.0);
        assert_eq!(cfg.sliding_rope_base, 10000.0);
    }
}
