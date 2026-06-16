use crate::array::Array;
use crate::ops;

pub struct Linear {
    weight: Array,
    bias: Option<Array>,
}

impl Linear {
    pub fn new(weight: Array, bias: Option<Array>) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Array) -> anyhow::Result<Array> {
        let w_t = ops::transpose(&self.weight, &[1, 0])?;
        let out = ops::matmul(x, &w_t)?;
        match &self.bias {
            Some(b) => ops::add(&out, b),
            None => Ok(out),
        }
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
        ops::take(&self.weight, indices, 0)
    }

    pub fn weight(&self) -> &Array {
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

pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
}

impl Attention {
    pub fn new(q_proj: Linear, k_proj: Linear, v_proj: Linear, o_proj: Linear) -> Self {
        Self { q_proj, k_proj, v_proj, o_proj }
    }

    pub fn forward(
        &self,
        x: &Array,
        kv: &mut KvCache,
        _positions: &Array,
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
        let q = ops::fast_rope(&q, head_dim, false, rope_theta, 1.0, 0)?;
        let k = ops::fast_rope(&k, head_dim, false, rope_theta, 1.0, 0)?;

        let (k, v) = kv.update(&k, &v)?;

        let scale = cfg.scale();
        let out = ops::fast_sdpa(&q, &k, &v, scale, "causal", None)?;

        let out = ops::transpose(&out, &[0, 2, 1, 3])?;
        let out = ops::reshape(&out, &[b, l, (cfg.num_attention_heads * cfg.head_dim) as usize])?;
        self.o_proj.forward(&out)
    }
}

pub struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    pub fn new(gate_proj: Linear, up_proj: Linear, down_proj: Linear) -> Self {
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
    k_cache: Option<Array>,
    v_cache: Option<Array>,
}

impl KvCache {
    pub fn new() -> Self {
        Self { k_cache: None, v_cache: None }
    }

    pub fn update(&mut self, k: &Array, v: &Array) -> anyhow::Result<(Array, Array)> {
        match (&self.k_cache, &self.v_cache) {
            (Some(ck), Some(cv)) => {
                let new_k = ops::concatenate(&[ck, k], 2)?;
                let new_v = ops::concatenate(&[cv, v], 2)?;
                self.k_cache = Some(new_k.clone());
                self.v_cache = Some(new_v.clone());
                Ok((new_k, new_v))
            }
            _ => {
                self.k_cache = Some(k.clone());
                self.v_cache = Some(v.clone());
                Ok((k.clone(), v.clone()))
            }
        }
    }

    pub fn len(&self) -> usize {
        self.k_cache.as_ref().map(|k| k.dim(2).unwrap_or(0)).unwrap_or(0)
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
    embed_tokens: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: Linear,
    config: LlamaConfig,
}

impl LlamaModel {
    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn forward(
        &self,
        input_ids: &Array,
        caches: &mut [KvCache],
        positions: &Array,
    ) -> anyhow::Result<Array> {
        let mut h = self.embed_tokens.forward(input_ids)?;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], positions, &self.config)?;
        }
        let h = self.norm.forward(&h)?;
        self.lm_head.forward(&h)
    }

    pub fn embed(&self, input_ids: &Array) -> anyhow::Result<Array> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn unembed(&self, hidden: &Array) -> anyhow::Result<Array> {
        self.lm_head.forward(hidden)
    }

    pub fn load_from_tensors(
        tensors: std::collections::HashMap<String, Array>,
        config: LlamaConfig,
    ) -> anyhow::Result<Self> {
        let prefix = resolve_weight_prefix(&tensors);

        let embed_w = tensors.get(&format!("{prefix}model.embed_tokens.weight"))
            .ok_or_else(|| anyhow::anyhow!("missing {prefix}model.embed_tokens.weight"))?;
        let embed_tokens = Embedding::new(embed_w.clone());

        let norm_w = tensors.get(&format!("{prefix}model.norm.weight"))
            .ok_or_else(|| anyhow::anyhow!("missing {prefix}model.norm.weight"))?;
        let norm = RmsNorm::new(norm_w.clone(), config.rms_norm_eps);

        let lm_head = if config.tie_word_embeddings {
            Linear::new(embed_w.clone(), None)
        } else if let Some(w) = tensors.get(&format!("{prefix}lm_head.weight")) {
            Linear::new(w.clone(), None)
        } else if let Some(w) = tensors.get("lm_head.weight") {
            Linear::new(w.clone(), None)
        } else {
            Linear::new(embed_w.clone(), None)
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for i in 0..config.num_hidden_layers as usize {
            let lp = format!("{prefix}model.layers.{i}");

            let q_w = tensors.get(&format!("{lp}.self_attn.q_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {lp}.self_attn.q_proj.weight"))?;
            let k_w = tensors.get(&format!("{lp}.self_attn.k_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {lp}.self_attn.k_proj.weight"))?;
            let v_w = tensors.get(&format!("{lp}.self_attn.v_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {lp}.self_attn.v_proj.weight"))?;
            let o_w = tensors.get(&format!("{lp}.self_attn.o_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {lp}.self_attn.o_proj.weight"))?;

            let q_bias = tensors.get(&format!("{lp}.self_attn.q_proj.bias")).cloned();
            let k_bias = tensors.get(&format!("{lp}.self_attn.k_proj.bias")).cloned();
            let v_bias = tensors.get(&format!("{lp}.self_attn.v_proj.bias")).cloned();
            let o_bias = tensors.get(&format!("{lp}.self_attn.o_proj.bias")).cloned();

            let attention = Attention::new(
                Linear::new(q_w.clone(), q_bias),
                Linear::new(k_w.clone(), k_bias),
                Linear::new(v_w.clone(), v_bias),
                Linear::new(o_w.clone(), o_bias),
            );

            let gate_w = tensors.get(&format!("{lp}.mlp.gate_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {lp}.mlp.gate_proj.weight"))?;
            let up_w = tensors.get(&format!("{lp}.mlp.up_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {lp}.mlp.up_proj.weight"))?;
            let down_w = tensors.get(&format!("{lp}.mlp.down_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {lp}.mlp.down_proj.weight"))?;

            let mlp = Mlp::new(
                Linear::new(gate_w.clone(), None),
                Linear::new(up_w.clone(), None),
                Linear::new(down_w.clone(), None),
            );

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

fn resolve_weight_prefix(tensors: &std::collections::HashMap<String, Array>) -> String {
    if tensors.contains_key("model.embed_tokens.weight") {
        return String::new();
    }
    if tensors.contains_key("language_model.model.embed_tokens.weight") {
        return "language_model.".to_string();
    }
    String::new()
}

pub fn argmax(logits: &Array) -> anyhow::Result<i32> {
    let shape = logits.shape();
    let ndim = shape.len();
    let vocab_size = shape[ndim - 1];
    let data = logits.data_f32()?;
    let offset = data.len() - vocab_size;
    let mut best_val = f32::NEG_INFINITY;
    let mut best_idx = 0i32;
    for i in 0..vocab_size {
        if data[offset + i] > best_val {
            best_val = data[offset + i];
            best_idx = i as i32;
        }
    }
    Ok(best_idx)
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

