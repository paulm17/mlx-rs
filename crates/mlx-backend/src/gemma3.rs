use std::collections::HashMap;

use crate::array::Array;
use crate::llama::{Embedding, KvCache, Linear, RmsNorm, resolve_weight_prefix};
use crate::model::Model;
use crate::ops;

fn gelu(x: &Array) -> anyhow::Result<Array> {
    let half = ops::multiply(x, &Array::from_f32(0.5)?)?;
    let x_cubed = ops::multiply(x, &ops::multiply(x, x)?)?;
    let inner = ops::add(x, &ops::multiply(&x_cubed, &Array::from_f32(0.044715)?)?)?;
    let scaled = ops::multiply(&inner, &Array::from_f32((2.0 / std::f32::consts::PI).sqrt())?)?;
    let tanh_val = ops::tanh(&scaled)?;
    let one_plus_tanh = ops::add(&Array::from_f32(1.0)?, &tanh_val)?;
    ops::multiply(&half, &one_plus_tanh)
}

pub struct Gemma3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
}

impl Gemma3Attention {
    pub fn forward(
        &self,
        x: &Array,
        kv: &mut KvCache,
        _positions: &Array,
        cfg: &Gemma3Config,
        head_dim: i32,
        rope_theta: f32,
    ) -> anyhow::Result<Array> {
        let b = x.dim(0)?;
        let l = x.dim(1)?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = ops::reshape(&q, &[b, l, cfg.num_attention_heads as usize, head_dim as usize])?;
        let q = ops::transpose(&q, &[0, 2, 1, 3])?;

        let k = ops::reshape(&k, &[b, l, cfg.num_key_value_heads as usize, head_dim as usize])?;
        let k = ops::transpose(&k, &[0, 2, 1, 3])?;

        let v = ops::reshape(&v, &[b, l, cfg.num_key_value_heads as usize, head_dim as usize])?;
        let v = ops::transpose(&v, &[0, 2, 1, 3])?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let q = ops::fast_rope(&q, head_dim, false, Some(rope_theta), 1.0, 0)?;
        let k = ops::fast_rope(&k, head_dim, false, Some(rope_theta), 1.0, 0)?;

        let (k, v) = kv.update(&k, &v)?;

        let scale = 1.0 / (head_dim as f32).sqrt();
        let out = ops::fast_sdpa(&q, &k, &v, scale, "causal", None)?;

        let out = ops::transpose(&out, &[0, 2, 1, 3])?;
        let out = ops::reshape(&out, &[b, l, (cfg.num_attention_heads * head_dim) as usize])?;
        self.o_proj.forward(&out)
    }
}

pub struct Gemma3Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Gemma3Mlp {
    pub fn forward(&self, x: &Array) -> anyhow::Result<Array> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let gate_gelu = gelu(&gate)?;
        let gated = ops::multiply(&gate_gelu, &up)?;
        self.down_proj.forward(&gated)
    }
}

pub struct Gemma3Layer {
    attention: Gemma3Attention,
    mlp: Gemma3Mlp,
    attention_norm: RmsNorm,
    #[allow(dead_code)]
    mlp_norm: RmsNorm,
    pre_feedforward_norm: RmsNorm,
    post_feedforward_norm: RmsNorm,
    is_sliding: bool,
}

impl Gemma3Layer {
    pub fn forward(
        &self,
        x: &Array,
        kv: &mut KvCache,
        positions: &Array,
        cfg: &Gemma3Config,
    ) -> anyhow::Result<Array> {
        let (head_dim, rope_theta) = if self.is_sliding {
            (cfg.head_dim, cfg.rope_local_base_freq)
        } else {
            (cfg.head_dim, cfg.rope_theta)
        };

        let normed = self.attention_norm.forward(x)?;
        let attn_out = self.attention.forward(&normed, kv, positions, cfg, head_dim, rope_theta)?;
        let h = ops::add(x, &self.post_feedforward_norm.forward(&attn_out)?)?;

        let normed = self.pre_feedforward_norm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        ops::add(&h, &self.post_feedforward_norm.forward(&mlp_out)?)
    }
}

#[derive(Debug, Clone)]
pub struct Gemma3Config {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_local_base_freq: f32,
    pub max_position_embeddings: i32,
    pub tie_word_embeddings: bool,
    pub head_dim: i32,
    pub sliding_window: i32,
    pub sliding_window_pattern: i32,
}

impl Gemma3Config {
    pub fn from_json(config: &serde_json::Value) -> anyhow::Result<Self> {
        let hidden_size = config.get("hidden_size").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        let num_hidden_layers = config.get("num_hidden_layers").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        let intermediate_size = config.get("intermediate_size").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        let num_attention_heads = config.get("num_attention_heads").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        let num_key_value_heads = config.get("num_key_value_heads").and_then(|v| v.as_i64()).unwrap_or(num_attention_heads as i64) as i32;
        let vocab_size = config.get("vocab_size").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        let rms_norm_eps = config.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32;
        let rope_theta = config.get("rope_theta").and_then(|v| v.as_f64()).unwrap_or(10000.0) as f32;
        let rope_local_base_freq = config.get("rope_local_base_freq").and_then(|v| v.as_f64()).unwrap_or(10000.0) as f32;
        let max_position_embeddings = config.get("max_position_embeddings").and_then(|v| v.as_i64()).unwrap_or(4096) as i32;
        let tie_word_embeddings = config.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(false);
        let head_dim = config.get("head_dim").and_then(|v| v.as_i64()).unwrap_or(256) as i32;
        let sliding_window = config.get("sliding_window").and_then(|v| v.as_i64()).unwrap_or(4096) as i32;
        let sliding_window_pattern = config.get("sliding_window_pattern").and_then(|v| v.as_i64()).unwrap_or(4) as i32;

        if hidden_size <= 0 || num_attention_heads <= 0 {
            anyhow::bail!("invalid Gemma3 config: hidden_size={hidden_size} heads={num_attention_heads}");
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
            rope_local_base_freq,
            max_position_embeddings,
            tie_word_embeddings,
            head_dim,
            sliding_window,
            sliding_window_pattern,
        })
    }
}

pub struct Gemma3Model {
    embed_tokens: Embedding,
    layers: Vec<Gemma3Layer>,
    norm: RmsNorm,
    lm_head: Linear,
    config: Gemma3Config,
}

impl Model for Gemma3Model {
    fn forward(
        &self,
        input_ids: &Array,
        caches: &mut [KvCache],
        positions: &Array,
    ) -> anyhow::Result<Array> {
        let mut h = self.embed_tokens.forward(input_ids)?;
        let scale = (self.config.hidden_size as f32).sqrt();
        h = ops::multiply(&h, &Array::from_f32(scale)?)?;

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
}

impl Gemma3Model {
    pub fn config(&self) -> &Gemma3Config {
        &self.config
    }

    pub fn load_from_tensors(
        tensors: HashMap<String, Array>,
        config: Gemma3Config,
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
        } else {
            Linear::new(embed_w.clone(), None)
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for i in 0..config.num_hidden_layers as usize {
            let lp = format!("{prefix}model.layers.{i}");
            let is_sliding = (i as i32 % config.sliding_window_pattern) != 0;

            let q_w = tensors.get(&format!("{lp}.self_attn.q_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {lp}.self_attn.q_proj.weight"))?;
            let k_w = tensors.get(&format!("{lp}.self_attn.k_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {lp}.self_attn.k_proj.weight"))?;
            let v_w = tensors.get(&format!("{lp}.self_attn.v_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {lp}.self_attn.v_proj.weight"))?;
            let o_w = tensors.get(&format!("{lp}.self_attn.o_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {lp}.self_attn.o_proj.weight"))?;

            let q_norm_w = tensors.get(&format!("{lp}.self_attn.q_norm.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {lp}.self_attn.q_norm.weight"))?;
            let k_norm_w = tensors.get(&format!("{lp}.self_attn.k_norm.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {lp}.self_attn.k_norm.weight"))?;

            let attention = Gemma3Attention {
                q_proj: Linear::new(q_w.clone(), None),
                k_proj: Linear::new(k_w.clone(), None),
                v_proj: Linear::new(v_w.clone(), None),
                o_proj: Linear::new(o_w.clone(), None),
                q_norm: RmsNorm::new(q_norm_w.clone(), config.rms_norm_eps),
                k_norm: RmsNorm::new(k_norm_w.clone(), config.rms_norm_eps),
            };

            let gate_w = tensors.get(&format!("{lp}.mlp.gate_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {lp}.mlp.gate_proj.weight"))?;
            let up_w = tensors.get(&format!("{lp}.mlp.up_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {lp}.mlp.up_proj.weight"))?;
            let down_w = tensors.get(&format!("{lp}.mlp.down_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {lp}.mlp.down_proj.weight"))?;

            let mlp = Gemma3Mlp {
                gate_proj: Linear::new(gate_w.clone(), None),
                up_proj: Linear::new(up_w.clone(), None),
                down_proj: Linear::new(down_w.clone(), None),
            };

            let attn_norm_w = tensors.get(&format!("{lp}.input_layernorm.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {lp}.input_layernorm.weight"))?;
            let mlp_norm_w = tensors.get(&format!("{lp}.post_attention_layernorm.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {lp}.post_attention_layernorm.weight"))?;
            let pre_ff_norm_w = tensors.get(&format!("{lp}.pre_feedforward_layernorm.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {lp}.pre_feedforward_layernorm.weight"))?;
            let post_ff_norm_w = tensors.get(&format!("{lp}.post_feedforward_layernorm.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {lp}.post_feedforward_layernorm.weight"))?;

            layers.push(Gemma3Layer {
                attention,
                mlp,
                attention_norm: RmsNorm::new(attn_norm_w.clone(), config.rms_norm_eps),
                mlp_norm: RmsNorm::new(mlp_norm_w.clone(), config.rms_norm_eps),
                pre_feedforward_norm: RmsNorm::new(pre_ff_norm_w.clone(), config.rms_norm_eps),
                post_feedforward_norm: RmsNorm::new(post_ff_norm_w.clone(), config.rms_norm_eps),
                is_sliding,
            });
        }

        Ok(Self { embed_tokens, layers, norm, lm_head, config })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader;

    fn mlx_available() -> bool {
        loader::check_init().is_ok()
    }

    #[test]
    fn test_gemma3_config_from_json() {
        let json = serde_json::json!({
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 100,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "rope_local_base_freq": 10000.0,
            "max_position_embeddings": 128,
            "head_dim": 256,
            "sliding_window": 4096,
            "sliding_window_pattern": 4
        });
        let cfg = Gemma3Config::from_json(&json).unwrap();
        assert_eq!(cfg.hidden_size, 64);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.sliding_window_pattern, 4);
    }

    #[test]
    fn test_gelu() {
        if !mlx_available() { return; }
        let x = Array::from_data_f32(&[0.0, 1.0, -1.0], &[3]).unwrap();
        let out = gelu(&x).unwrap();
        out.eval().unwrap();
        let data = out.data_f32().unwrap();
        assert!((data[0]).abs() < 1e-5);
        assert!(data[1] > 0.68);
        assert!(data[2] < -0.15);
    }

    #[test]
    fn test_gemma3_sliding_window_detection() {
        let json = serde_json::json!({
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "sliding_window_pattern": 3
        });
        let cfg = Gemma3Config::from_json(&json).unwrap();
        assert_eq!(cfg.sliding_window_pattern, 3);
    }
}
