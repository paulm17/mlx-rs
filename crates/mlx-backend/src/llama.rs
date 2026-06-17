use crate::array::Array;
use crate::model::Model;
use crate::ops;

pub trait LinearLayer: Send {
    fn forward(&self, x: &Array) -> anyhow::Result<Array>;
    fn as_dense_weight(&self) -> Option<&Array> { None }
}

pub trait EmbeddingLayer: Send {
    fn forward(&self, indices: &Array) -> anyhow::Result<Array>;
    fn weight(&self) -> &Array;
}

#[derive(Clone)]
pub enum LayerCache {
    Attention(KvCache),
    Recurrent(RecurrentCache),
}

impl LayerCache {
    pub fn as_attention_mut(&mut self) -> Option<&mut KvCache> {
        match self {
            LayerCache::Attention(kv) => Some(kv),
            _ => None,
        }
    }

    pub fn as_recurrent_mut(&mut self) -> Option<&mut RecurrentCache> {
        match self {
            LayerCache::Recurrent(rc) => Some(rc),
            _ => None,
        }
    }
}

#[derive(Clone)]
pub struct RecurrentCache {
    pub conv_state: Option<Array>,
    pub delta_state: Option<Array>,
    pub conv_tail: usize,
    pub conv_dim: usize,
    pub num_v_heads: usize,
    pub head_v_dim: usize,
    pub head_k_dim: usize,
}

impl RecurrentCache {
    pub fn new(conv_tail: usize, conv_dim: usize, num_v_heads: usize, head_v_dim: usize, head_k_dim: usize) -> Self {
        Self {
            conv_state: None,
            delta_state: None,
            conv_tail,
            conv_dim,
            num_v_heads,
            head_v_dim,
            head_k_dim,
        }
    }

    pub fn get_states(&mut self, batch_size: usize, dtype: crate::ffi::MlxDtype) -> anyhow::Result<(&Array, &Array)> {
        if self.conv_state.is_none() {
            self.conv_state = Some(ops::zeros(
                &[batch_size, self.conv_tail, self.conv_dim],
                dtype,
            )?);
            self.delta_state = Some(ops::zeros(
                &[batch_size, self.num_v_heads, self.head_v_dim, self.head_k_dim],
                crate::ffi::MlxDtype::Float32,
            )?);
        }
        Ok((self.conv_state.as_ref().unwrap(), self.delta_state.as_ref().unwrap()))
    }

    pub fn put_states(&mut self, conv_state: Array, delta_state: Array) {
        self.conv_state = Some(conv_state);
        self.delta_state = Some(delta_state);
    }
}

pub struct Linear {
    weight: Array,
    bias: Option<Array>,
}

impl Linear {
    pub fn new(weight: Array, bias: Option<Array>) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Array) -> anyhow::Result<Array> {
        <Self as LinearLayer>::forward(self, x)
    }

    pub fn weight(&self) -> &Array {
        &self.weight
    }
}

impl LinearLayer for Linear {
    fn forward(&self, x: &Array) -> anyhow::Result<Array> {
        let w_t = ops::transpose(&self.weight, &[1, 0])?;
        let out = ops::matmul(x, &w_t)?;
        match &self.bias {
            Some(b) => ops::add(&out, b),
            None => Ok(out),
        }
    }

    fn as_dense_weight(&self) -> Option<&Array> {
        Some(&self.weight)
    }
}

pub struct Embedding {
    weight: Array,
}

impl Embedding {
    pub fn new(weight: Array) -> Self {
        Self { weight }
    }

    pub fn forward(&self, indices: &Array) -> anyhow::Result<Array> {
        <Self as EmbeddingLayer>::forward(self, indices)
    }

    pub fn weight(&self) -> &Array {
        &self.weight
    }
}

impl EmbeddingLayer for Embedding {
    fn forward(&self, indices: &Array) -> anyhow::Result<Array> {
        ops::take(&self.weight, indices, 0)
    }

    fn weight(&self) -> &Array {
        &self.weight
    }
}

pub struct RmsNorm {
    weight: Array,
    eps: f32,
}

impl RmsNorm {
    pub fn new(weight: Array, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Array) -> anyhow::Result<Array> {
        ops::fast_rms_norm(x, &self.weight, self.eps)
    }
}

pub struct QuantizedLinear {
    weight: Array,
    scales: Array,
    biases: Option<Array>,
    attn_bias: Option<Array>,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: String,
}

impl QuantizedLinear {
    pub fn new(
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        group_size: Option<i32>,
        bits: Option<i32>,
        mode: String,
    ) -> Self {
        Self { weight, scales, biases, attn_bias: None, group_size, bits, mode }
    }

    pub fn with_attn_bias(mut self, bias: Array) -> Self {
        self.attn_bias = Some(bias);
        self
    }
}

impl LinearLayer for QuantizedLinear {
    fn forward(&self, x: &Array) -> anyhow::Result<Array> {
        let out = ops::quantized_matmul(
            x,
            &self.weight,
            &self.scales,
            self.biases.as_ref(),
            true, // transpose=true because weight is [out, in] packed
            self.group_size,
            self.bits,
            &self.mode,
        )?;
        if let Some(ref bias) = self.attn_bias {
            ops::add(&out, bias)
        } else {
            Ok(out)
        }
    }
}

pub struct QuantizedEmbedding {
    weight: Array,
    scales: Array,
    biases: Option<Array>,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: String,
}

impl QuantizedEmbedding {
    pub fn new(
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        group_size: Option<i32>,
        bits: Option<i32>,
        mode: String,
    ) -> Self {
        Self { weight, scales, biases, group_size, bits, mode }
    }
}

impl EmbeddingLayer for QuantizedEmbedding {
    fn forward(&self, indices: &Array) -> anyhow::Result<Array> {
        let w = ops::take(&self.weight, indices, 0)?;
        let s = ops::take(&self.scales, indices, 0)?;
        let b = match &self.biases {
            Some(bias) => Some(ops::take(bias, indices, 0)?),
            None => None,
        };
        ops::dequantize(
            &w,
            &s,
            b.as_ref(),
            self.group_size,
            self.bits,
            &self.mode,
            None,
        )
    }

    fn weight(&self) -> &Array {
        &self.weight
    }
}

pub struct Attention {
    q_proj: Box<dyn LinearLayer>,
    k_proj: Box<dyn LinearLayer>,
    v_proj: Box<dyn LinearLayer>,
    o_proj: Box<dyn LinearLayer>,
}

impl Attention {
    pub fn new(
        q_proj: Box<dyn LinearLayer>,
        k_proj: Box<dyn LinearLayer>,
        v_proj: Box<dyn LinearLayer>,
        o_proj: Box<dyn LinearLayer>,
    ) -> Self {
        Self { q_proj, k_proj, v_proj, o_proj }
    }

    pub fn forward(
        &self,
        x: &Array,
        kv: &mut KvCache,
        positions: &Array,
        cfg: &LlamaConfig,
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

        let head_dim = cfg.head_dim as i32;
        let rope_theta = Some(cfg.rope_theta);
        let q = ops::fast_rope_dynamic(&q, head_dim, false, rope_theta, 1.0, positions, None)?;
        let k = ops::fast_rope_dynamic(&k, head_dim, false, rope_theta, 1.0, positions, None)?;

        let (k, v) = kv.update(&k, &v)?;

        let scale = cfg.scale();
        let sdpa_mode = if l > 1 { "causal" } else { "" };
        let out = ops::fast_sdpa(&q, &k, &v, scale, sdpa_mode, None)?;

        let out = ops::transpose(&out, &[0, 2, 1, 3])?;
        let out = ops::reshape(&out, &[b, l, (cfg.num_attention_heads * cfg.head_dim) as usize])?;
        self.o_proj.forward(&out)
    }
}

pub struct Mlp {
    gate_proj: Box<dyn LinearLayer>,
    up_proj: Box<dyn LinearLayer>,
    down_proj: Box<dyn LinearLayer>,
}

impl Mlp {
    pub fn new(
        gate_proj: Box<dyn LinearLayer>,
        up_proj: Box<dyn LinearLayer>,
        down_proj: Box<dyn LinearLayer>,
    ) -> Self {
        Self { gate_proj, up_proj, down_proj }
    }

    pub fn forward(&self, x: &Array) -> anyhow::Result<Array> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let gate_silu = ops::multiply(&gate, &ops::sigmoid(&gate)?)?;
        let gated = ops::multiply(&gate_silu, &up)?;
        self.down_proj.forward(&gated)
    }
}

pub struct Layer {
    attention: Attention,
    mlp: Mlp,
    attention_norm: RmsNorm,
    mlp_norm: RmsNorm,
}

impl Layer {
    pub fn forward(
        &self,
        x: &Array,
        kv: &mut KvCache,
        positions: &Array,
        cfg: &LlamaConfig,
    ) -> anyhow::Result<Array> {
        let normed = self.attention_norm.forward(x)?;
        let attn_out = self.attention.forward(&normed, kv, positions, cfg)?;
        let h = ops::add(x, &attn_out)?;
        let normed = self.mlp_norm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        ops::add(&h, &mlp_out)
    }
}

#[derive(Clone)]
pub struct KvCache {
    pub(crate) k_cache: Option<Array>,
    pub(crate) v_cache: Option<Array>,
    pub donor: Option<(Array, Array, f32)>,
    /// Maximum sequence length for sliding-window layers. When the cache grows
    /// beyond this, oldest tokens are dropped from the sequence dimension.
    pub max_len: Option<usize>,
}

impl KvCache {
    pub fn new() -> Self {
        Self { k_cache: None, v_cache: None, donor: None, max_len: None }
    }

    pub fn new_rotating(max_len: usize) -> Self {
        Self { k_cache: None, v_cache: None, donor: None, max_len: Some(max_len) }
    }

    pub fn update(&mut self, k: &Array, v: &Array) -> anyhow::Result<(Array, Array)> {
        let _offset_before = self.len();
        let (mut new_k, mut new_v) = match (&self.k_cache, &self.v_cache) {
            (Some(ck), Some(cv)) => {
                (ops::concatenate(&[ck, k], 2)?, ops::concatenate(&[cv, v], 2)?)
            }
            _ => (k.clone(), v.clone()),
        };

        // Sliding-window eviction: keep only the most recent max_len positions.
        if let Some(max_len) = self.max_len {
            let seq_len = new_k.dim(2)?;
            if seq_len > max_len {
                let start = seq_len - max_len;
                let indices: Vec<i32> = (start..seq_len).map(|i| i as i32).collect();
                let idx = Array::from_data_i32(&indices, &[max_len])?;
                new_k = ops::take(&new_k, &idx, 2)?;
                new_v = ops::take(&new_v, &idx, 2)?;
            }
        }

        self.k_cache = Some(new_k.clone());
        self.v_cache = Some(new_v.clone());
        Ok((new_k, new_v))
    }

    pub fn len(&self) -> usize {
        self.k_cache.as_ref().map(|k| k.dim(2).unwrap_or(0)).unwrap_or(0)
    }

    /// Collect the current K and V arrays for explicit evaluation.
    pub fn arrays(&self) -> Vec<&Array> {
        let mut out = Vec::with_capacity(2);
        if let Some(k) = &self.k_cache { out.push(k); }
        if let Some(v) = &self.v_cache { out.push(v); }
        out
    }
}

#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: i32,
    pub tie_word_embeddings: bool,
    pub head_dim: i32,
}

impl LlamaConfig {
    pub fn from_json(config: &serde_json::Value) -> anyhow::Result<Self> {
        let hidden_size = config.get("hidden_size").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        let num_hidden_layers = config.get("num_hidden_layers").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        let intermediate_size = config.get("intermediate_size").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        let num_attention_heads = config.get("num_attention_heads").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        let num_key_value_heads = config.get("num_key_value_heads").and_then(|v| v.as_i64()).unwrap_or(num_attention_heads as i64) as i32;
        let vocab_size = config.get("vocab_size").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        let rms_norm_eps = config.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-5) as f32;
        let rope_theta = config.get("rope_theta").and_then(|v| v.as_f64()).unwrap_or(10000.0) as f32;
        let max_position_embeddings = config.get("max_position_embeddings").and_then(|v| v.as_i64()).unwrap_or(2048) as i32;
        let tie_word_embeddings = config.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(false);
        let head_dim = config.get("head_dim").and_then(|v| v.as_i64()).unwrap_or((hidden_size / num_attention_heads) as i64) as i32;

        if hidden_size <= 0 || num_attention_heads <= 0 || num_key_value_heads <= 0 {
            anyhow::bail!("invalid model config: hidden_size={hidden_size} heads={num_attention_heads}/{num_key_value_heads}");
        }

        Ok(Self {
            hidden_size,
            num_hidden_layers,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            vocab_size,
            rms_norm_eps,
            rope_theta,
            max_position_embeddings,
            tie_word_embeddings,
            head_dim,
        })
    }

    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

pub struct LlamaModel {
    embed_tokens: Box<dyn EmbeddingLayer>,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: Box<dyn LinearLayer>,
    config: LlamaConfig,
}

impl Model for LlamaModel {
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
        let logits = self.lm_head.forward(&h)?;
        Ok(logits)
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
        (0..self.layers.len()).map(|_| LayerCache::Attention(KvCache::new())).collect()
    }
}

impl LlamaModel {
    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }

    pub fn load_from_tensors(
        tensors: std::collections::HashMap<String, Array>,
        config: LlamaConfig,
    ) -> anyhow::Result<Self> {
        let prefix = resolve_weight_prefix(&tensors);

        let embed_tokens = make_embedding(&tensors, &format!("{prefix}model.embed_tokens.weight"))?;

        let norm_w = tensors.get(&format!("{prefix}model.norm.weight"))
            .ok_or_else(|| anyhow::anyhow!("missing {prefix}model.norm.weight"))?;
        let norm = RmsNorm::new(norm_w.clone(), config.rms_norm_eps);

        let lm_head: Box<dyn LinearLayer> = if config.tie_word_embeddings {
            let base = format!("{prefix}model.embed_tokens.weight");
            let base_trimmed = base.strip_suffix(".weight").unwrap_or(&base);
            let scale_key = format!("{base_trimmed}_scale");
            if let Some(scales) = tensors.get(&scale_key) {
                let weight = tensors.get(&base)
                    .ok_or_else(|| anyhow::anyhow!("missing {base}"))?;
                let biases = tensors.get(&format!("{base_trimmed}_qbias")).cloned();
                let w_cols = weight.dim(weight.ndim() - 1)? as i32;
                let s_cols = scales.dim(scales.ndim() - 1)? as i32;
                let (group_size, bits, mode) = infer_quant_params(w_cols, s_cols);
                Box::new(QuantizedLinear::new(
                    weight.clone(), scales.clone(), biases,
                    Some(group_size), Some(bits), mode,
                ))
            } else {
                let weight = tensors.get(&base)
                    .ok_or_else(|| anyhow::anyhow!("missing {base}"))?;
                Box::new(Linear::new(weight.clone(), None))
            }
        } else {
            make_linear(&tensors, &format!("{prefix}lm_head.weight"))
                .or_else(|_| make_linear(&tensors, "lm_head.weight"))
                .or_else(|_| make_linear(&tensors, &format!("{prefix}model.embed_tokens.weight")))?
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for i in 0..config.num_hidden_layers as usize {
            let lp = format!("{prefix}model.layers.{i}");

            let q_proj = make_linear_with_bias(&tensors, &format!("{lp}.self_attn.q_proj.weight"))?;
            let k_proj = make_linear_with_bias(&tensors, &format!("{lp}.self_attn.k_proj.weight"))?;
            let v_proj = make_linear_with_bias(&tensors, &format!("{lp}.self_attn.v_proj.weight"))?;
            let o_proj = make_linear_with_bias(&tensors, &format!("{lp}.self_attn.o_proj.weight"))?;

            let attention = Attention::new(q_proj, k_proj, v_proj, o_proj);

            let gate_proj = make_linear(&tensors, &format!("{lp}.mlp.gate_proj.weight"))?;
            let up_proj = make_linear(&tensors, &format!("{lp}.mlp.up_proj.weight"))?;
            let down_proj = make_linear(&tensors, &format!("{lp}.mlp.down_proj.weight"))?;

            let mlp = Mlp::new(gate_proj, up_proj, down_proj);

            let attn_norm_w = tensors.get(&format!("{lp}.input_layernorm.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {lp}.input_layernorm.weight"))?;
            let mlp_norm_w = tensors.get(&format!("{lp}.post_attention_layernorm.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {lp}.post_attention_layernorm.weight"))?;

            layers.push(Layer {
                attention,
                mlp,
                attention_norm: RmsNorm::new(attn_norm_w.clone(), config.rms_norm_eps),
                mlp_norm: RmsNorm::new(mlp_norm_w.clone(), config.rms_norm_eps),
            });
        }

        Ok(Self { embed_tokens, layers, norm, lm_head, config })
    }
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
    tensors: &std::collections::HashMap<String, Array>,
    base_key: &str,
) -> anyhow::Result<Box<dyn LinearLayer>> {
    let base = base_key.strip_suffix(".weight").unwrap_or(base_key);
    let scale_key = format!("{base}_scale");
    if let Some(scales) = tensors.get(&scale_key) {
        let weight = tensors.get(base_key)
            .ok_or_else(|| anyhow::anyhow!("missing {base_key}"))?;
        let biases = tensors.get(&format!("{base}_qbias")).cloned();
        let w_cols = weight.dim(weight.ndim() - 1)? as i32;
        let s_cols = scales.dim(scales.ndim() - 1)? as i32;
        let (group_size, bits, mode) = infer_quant_params(w_cols, s_cols);
        Ok(Box::new(QuantizedLinear::new(
            weight.clone(), scales.clone(), biases,
            Some(group_size), Some(bits), mode,
        )))
    } else {
        let weight = tensors.get(base_key)
            .ok_or_else(|| anyhow::anyhow!("missing {base_key}"))?;
        Ok(Box::new(Linear::new(weight.clone(), None)))
    }
}

fn make_linear_with_bias(
    tensors: &std::collections::HashMap<String, Array>,
    base_key: &str,
) -> anyhow::Result<Box<dyn LinearLayer>> {
    let base = base_key.strip_suffix(".weight").unwrap_or(base_key);
    let scale_key = format!("{base}_scale");
    let attn_bias = tensors.get(&format!("{base}.bias")).cloned();
    if let Some(scales) = tensors.get(&scale_key) {
        let weight = tensors.get(base_key)
            .ok_or_else(|| anyhow::anyhow!("missing {base_key}"))?;
        let biases = tensors.get(&format!("{base}_qbias")).cloned();
        let w_cols = weight.dim(weight.ndim() - 1)? as i32;
        let s_cols = scales.dim(scales.ndim() - 1)? as i32;
        let (group_size, bits, mode) = infer_quant_params(w_cols, s_cols);
        let ql = QuantizedLinear::new(
            weight.clone(), scales.clone(), biases,
            Some(group_size), Some(bits), mode,
        );
        Ok(Box::new(match attn_bias {
            Some(b) => ql.with_attn_bias(b),
            None => ql,
        }))
    } else {
        let weight = tensors.get(base_key)
            .ok_or_else(|| anyhow::anyhow!("missing {base_key}"))?;
        Ok(Box::new(Linear::new(weight.clone(), attn_bias)))
    }
}

fn make_embedding(
    tensors: &std::collections::HashMap<String, Array>,
    base_key: &str,
) -> anyhow::Result<Box<dyn EmbeddingLayer>> {
    let base = base_key.strip_suffix(".weight").unwrap_or(base_key);
    let scale_key = format!("{base}_scale");
    if let Some(scales) = tensors.get(&scale_key) {
        let weight = tensors.get(base_key)
            .ok_or_else(|| anyhow::anyhow!("missing {base_key}"))?;
        let biases = tensors.get(&format!("{base}_qbias")).cloned();
        let w_cols = weight.dim(weight.ndim() - 1)? as i32;
        let s_cols = scales.dim(scales.ndim() - 1)? as i32;
        let (group_size, bits, mode) = infer_quant_params(w_cols, s_cols);
        Ok(Box::new(QuantizedEmbedding::new(
            weight.clone(), scales.clone(), biases,
            Some(group_size), Some(bits), mode,
        )))
    } else {
        let weight = tensors.get(base_key)
            .ok_or_else(|| anyhow::anyhow!("missing {base_key}"))?;
        Ok(Box::new(Embedding::new(weight.clone())))
    }
}

pub fn resolve_weight_prefix(tensors: &std::collections::HashMap<String, Array>) -> String {
    if tensors.contains_key("model.embed_tokens.weight") {
        return String::new();
    }
    if tensors.contains_key("language_model.model.embed_tokens.weight") {
        return "language_model.".to_string();
    }
    String::new()
}

pub fn argmax(logits: &Array) -> anyhow::Result<i32> {
    let idx = crate::ops::argmax_axis(logits, -1, false)?;
    Ok(idx.item_i32()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader;

    fn mlx_available() -> bool {
        loader::check_init().is_ok()
    }

    #[test]
    fn test_llama_config_from_json() {
        let json = serde_json::json!({
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 100,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 128
        });
        let cfg = LlamaConfig::from_json(&json).unwrap();
        assert_eq!(cfg.hidden_size, 64);
        assert_eq!(cfg.num_hidden_layers, 2);
        assert_eq!(cfg.num_attention_heads, 4);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.head_dim, 16);
    }

    #[test]
    fn test_linear_forward() {
        if !mlx_available() { return; }
        let w = Array::from_data_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let linear = Linear::new(w, None);
        let x = Array::from_data_f32(&[2.0, 3.0], &[1, 2]).unwrap();
        let out = linear.forward(&x).unwrap();
        out.eval().unwrap();
        let data = out.data_f32().unwrap();
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_embedding_forward() {
        if !mlx_available() { return; }
        let w = Array::from_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let emb = Embedding::new(w);
        let idx = Array::from_i32(1).unwrap();
        let out = emb.forward(&idx).unwrap();
        out.eval().unwrap();
        let data = out.data_f32().unwrap();
        assert_eq!(data, &[3.0, 4.0]);
    }

    #[test]
    fn test_rms_norm_forward() {
        if !mlx_available() { return; }
        let w = Array::from_data_f32(&[1.0, 1.0], &[2]).unwrap();
        let norm = RmsNorm::new(w, 1e-5);
        let x = Array::from_data_f32(&[3.0, 4.0], &[1, 2]).unwrap();
        let out = norm.forward(&x).unwrap();
        out.eval().unwrap();
        let data = out.data_f32().unwrap();
        let expected_rms = ((3.0f32 * 3.0 + 4.0 * 4.0) / 2.0 + 1e-5).sqrt();
        assert!((data[0] - 3.0 / expected_rms).abs() < 1e-4);
        assert!((data[1] - 4.0 / expected_rms).abs() < 1e-4);
    }

    #[test]
    fn test_kv_cache_update() {
        if !mlx_available() { return; }
        let mut kv = KvCache::new();
        assert_eq!(kv.len(), 0);
        let k = Array::from_data_f32(&[1.0, 2.0], &[1, 2, 1]).unwrap();
        let v = Array::from_data_f32(&[3.0, 4.0], &[1, 2, 1]).unwrap();
        let (k1, _v1) = kv.update(&k, &v).unwrap();
        assert_eq!(k1.shape(), vec![1, 2, 1]);
        assert_eq!(kv.len(), 1);
        let k2 = Array::from_data_f32(&[5.0, 6.0], &[1, 2, 1]).unwrap();
        let v2 = Array::from_data_f32(&[7.0, 8.0], &[1, 2, 1]).unwrap();
        let (k3, _v3) = kv.update(&k2, &v2).unwrap();
        assert_eq!(k3.shape(), vec![1, 2, 2]);
        assert_eq!(kv.len(), 2);
    }

    #[test]
    fn test_argmax() {
        if !mlx_available() { return; }
        let logits = Array::from_data_f32(&[1.0, 3.0, 2.0], &[1, 3]).unwrap();
        let idx = argmax(&logits).unwrap();
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_swiglu() {
        if !mlx_available() { return; }
        let gate = Array::from_data_f32(&[1.0, -1.0], &[1, 2]).unwrap();
        let up = Array::from_data_f32(&[2.0, 3.0], &[1, 2]).unwrap();
        let gate_silu = ops::multiply(&gate, &ops::sigmoid(&gate).unwrap()).unwrap();
        let out = ops::multiply(&gate_silu, &up).unwrap();
        out.eval().unwrap();
        let data = out.data_f32().unwrap();
        let sigmoid_1 = 1.0 / (1.0 + (-1.0f32).exp());
        assert!((data[0] - 1.0 * sigmoid_1 * 2.0).abs() < 1e-5);
    }
}

