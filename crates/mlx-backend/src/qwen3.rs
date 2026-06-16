use std::collections::HashMap;

use crate::array::Array;
use crate::llama::{Embedding, KvCache, Linear, Mlp, RmsNorm, resolve_weight_prefix};
use crate::model::Model;
use crate::ops;

pub struct Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
}

impl Qwen3Attention {
    pub fn forward(
        &self,
        x: &Array,
        kv: &mut KvCache,
        _positions: &Array,
        cfg: &Qwen3Config,
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

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

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

pub struct Qwen3Layer {
    attention: Qwen3Attention,
    mlp: Mlp,
    attention_norm: RmsNorm,
    mlp_norm: RmsNorm,
}

impl Qwen3Layer {
    pub fn forward(
        &self,
        x: &Array,
        kv: &mut KvCache,
        positions: &Array,
        cfg: &Qwen3Config,
    ) -> anyhow::Result<Array> {
        let normed = self.attention_norm.forward(x)?;
        let attn_out = self.attention.forward(&normed, kv, positions, cfg)?;
        let h = ops::add(x, &attn_out)?;
        let normed = self.mlp_norm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        ops::add(&h, &mlp_out)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3Config {
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
    pub qk_norm_eps: f32,
}

impl Qwen3Config {
    pub fn from_json(config: &serde_json::Value) -> anyhow::Result<Self> {
        let hidden_size = config.get("hidden_size").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        let num_hidden_layers = config.get("num_hidden_layers").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        let intermediate_size = config.get("intermediate_size").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        let num_attention_heads = config.get("num_attention_heads").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        let num_key_value_heads = config.get("num_key_value_heads").and_then(|v| v.as_i64()).unwrap_or(num_attention_heads as i64) as i32;
        let vocab_size = config.get("vocab_size").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        let rms_norm_eps = config.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-5) as f32;
        let rope_theta = config.get("rope_theta").and_then(|v| v.as_f64()).unwrap_or(1000000.0) as f32;
        let max_position_embeddings = config.get("max_position_embeddings").and_then(|v| v.as_i64()).unwrap_or(4096) as i32;
        let tie_word_embeddings = config.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(false);
        let head_dim = config.get("head_dim").and_then(|v| v.as_i64()).unwrap_or((hidden_size / num_attention_heads) as i64) as i32;
        let qk_norm_eps = config.get("qk_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32;

        if hidden_size <= 0 || num_attention_heads <= 0 {
            anyhow::bail!("invalid Qwen3 config: hidden_size={hidden_size} heads={num_attention_heads}");
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
            qk_norm_eps,
        })
    }

    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

pub struct Qwen3Model {
    embed_tokens: Embedding,
    layers: Vec<Qwen3Layer>,
    norm: RmsNorm,
    lm_head: Linear,
    config: Qwen3Config,
}

impl Model for Qwen3Model {
    fn forward(
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

impl Qwen3Model {
    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }

    pub fn load_from_tensors(
        tensors: HashMap<String, Array>,
        config: Qwen3Config,
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

            let attention = Qwen3Attention {
                q_proj: Linear::new(q_w.clone(), None),
                k_proj: Linear::new(k_w.clone(), None),
                v_proj: Linear::new(v_w.clone(), None),
                o_proj: Linear::new(o_w.clone(), None),
                q_norm: RmsNorm::new(q_norm_w.clone(), config.qk_norm_eps),
                k_norm: RmsNorm::new(k_norm_w.clone(), config.qk_norm_eps),
            };

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

            layers.push(Qwen3Layer {
                attention,
                mlp,
                attention_norm: RmsNorm::new(attn_norm_w.clone(), config.rms_norm_eps),
                mlp_norm: RmsNorm::new(mlp_norm_w.clone(), config.rms_norm_eps),
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
    fn test_qwen3_config_from_json() {
        let json = serde_json::json!({
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 100,
            "rms_norm_eps": 1e-5,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 128,
            "head_dim": 16
        });
        let cfg = Qwen3Config::from_json(&json).unwrap();
        assert_eq!(cfg.hidden_size, 64);
        assert_eq!(cfg.rope_theta, 1000000.0);
        assert_eq!(cfg.head_dim, 16);
        assert!((cfg.qk_norm_eps - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_qwen3_qk_norm() {
        if !mlx_available() { return; }
        let w = Array::from_data_f32(&[1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
        let norm = RmsNorm::new(w, 1e-6);
        let x = Array::from_data_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let out = norm.forward(&x).unwrap();
        out.eval().unwrap();
        let data = out.data_f32().unwrap();
        assert!(data.iter().all(|v| v.is_finite()));
    }
}
