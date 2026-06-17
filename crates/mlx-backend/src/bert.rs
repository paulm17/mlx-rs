use std::collections::HashMap;

use crate::array::Array;
use crate::llama::{make_embedding, make_linear, EmbeddingLayer, LayerNorm, LinearLayer};
use crate::model::{EncoderModel, Model};
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

fn resolve_bert_prefix(tensors: &HashMap<String, Array>) -> String {
    if tensors.contains_key("bert.embeddings.word_embeddings.weight") {
        return "bert.".to_string();
    }
    String::new()
}

#[derive(Debug, Clone)]
pub struct BertConfig {
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub intermediate_size: i32,
    pub hidden_act: String,
    pub layer_norm_eps: f32,
    pub max_position_embeddings: i32,
    pub type_vocab_size: i32,
}

impl BertConfig {
    pub fn from_json(config: &serde_json::Value) -> anyhow::Result<Self> {
        let vocab_size = config.get("vocab_size").and_then(|v| v.as_i64()).unwrap_or(30522) as i32;
        let hidden_size = config.get("hidden_size").and_then(|v| v.as_i64()).unwrap_or(768) as i32;
        let num_hidden_layers = config.get("num_hidden_layers").and_then(|v| v.as_i64()).unwrap_or(12) as i32;
        let num_attention_heads = config.get("num_attention_heads").and_then(|v| v.as_i64()).unwrap_or(12) as i32;
        let intermediate_size = config.get("intermediate_size").and_then(|v| v.as_i64()).unwrap_or(3072) as i32;
        let hidden_act = config.get("hidden_act").and_then(|v| v.as_str()).unwrap_or("gelu").to_string();
        let layer_norm_eps = config.get("layer_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-12) as f32;
        let max_position_embeddings = config.get("max_position_embeddings").and_then(|v| v.as_i64()).unwrap_or(512) as i32;
        let type_vocab_size = config.get("type_vocab_size").and_then(|v| v.as_i64()).unwrap_or(2) as i32;

        if hidden_size <= 0 || num_attention_heads <= 0 {
            anyhow::bail!("invalid BERT config: hidden_size={hidden_size} heads={num_attention_heads}");
        }

        Ok(Self {
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            hidden_act,
            layer_norm_eps,
            max_position_embeddings,
            type_vocab_size,
        })
    }

    pub fn head_size(&self) -> i32 {
        self.hidden_size / self.num_attention_heads
    }
}

pub struct BertEmbeddings {
    word_embeddings: Box<dyn EmbeddingLayer>,
    position_embeddings: Box<dyn EmbeddingLayer>,
    token_type_embeddings: Box<dyn EmbeddingLayer>,
    layer_norm: LayerNorm,
}

impl BertEmbeddings {
    pub fn forward(
        &self,
        input_ids: &Array,
        token_type_ids: Option<&Array>,
    ) -> anyhow::Result<Array> {
        let seq_len = input_ids.dim(input_ids.ndim() - 1)?;
        let positions = ops::arange(0.0, seq_len as f64, 1.0, crate::ffi::MlxDtype::Int32)?;
        let position_ids = ops::reshape(&positions, &[1, seq_len])?;

        let word_embeds = self.word_embeddings.forward(input_ids)?;
        let position_embeds = self.position_embeddings.forward(&position_ids)?;

        let token_type = match token_type_ids {
            Some(ids) => ids.clone(),
            None => ops::zeros(&[1, seq_len], crate::ffi::MlxDtype::Int32)?,
        };
        let token_type_embeds = self.token_type_embeddings.forward(&token_type)?;

        let embeddings = ops::add(&word_embeds, &position_embeds)?;
        let embeddings = ops::add(&embeddings, &token_type_embeds)?;
        self.layer_norm.forward(&embeddings)
    }

    pub fn load_from_tensors(
        tensors: &HashMap<String, Array>,
        config: &BertConfig,
        prefix: &str,
    ) -> anyhow::Result<Self> {
        let word_embeddings = make_embedding(tensors, &format!("{prefix}embeddings.word_embeddings.weight"))?;
        let position_embeddings =
            make_embedding(tensors, &format!("{prefix}embeddings.position_embeddings.weight"))?;
        let token_type_embeddings =
            make_embedding(tensors, &format!("{prefix}embeddings.token_type_embeddings.weight"))?;

        let ln_weight = tensors
            .get(&format!("{prefix}embeddings.LayerNorm.weight"))
            .ok_or_else(|| anyhow::anyhow!("missing {prefix}embeddings.LayerNorm.weight"))?;
        let ln_bias = tensors
            .get(&format!("{prefix}embeddings.LayerNorm.bias"))
            .ok_or_else(|| anyhow::anyhow!("missing {prefix}embeddings.LayerNorm.bias"))?;
        let layer_norm = LayerNorm::new(ln_weight.clone(), Some(ln_bias.clone()), config.layer_norm_eps);

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
        })
    }
}

pub struct BertSelfAttention {
    query: Box<dyn LinearLayer>,
    key: Box<dyn LinearLayer>,
    value: Box<dyn LinearLayer>,
    num_heads: i32,
    head_size: i32,
}

impl BertSelfAttention {
    pub fn forward(
        &self,
        hidden_states: &Array,
        attention_mask: Option<&Array>,
    ) -> anyhow::Result<Array> {
        let b = hidden_states.dim(0)?;
        let l = hidden_states.dim(1)?;

        let q = self.query.forward(hidden_states)?;
        let k = self.key.forward(hidden_states)?;
        let v = self.value.forward(hidden_states)?;

        let q = ops::reshape(&q, &[b, l, self.num_heads as usize, self.head_size as usize])?;
        let q = ops::transpose(&q, &[0, 2, 1, 3])?;

        let k = ops::reshape(&k, &[b, l, self.num_heads as usize, self.head_size as usize])?;
        let k = ops::transpose(&k, &[0, 2, 1, 3])?;

        let v = ops::reshape(&v, &[b, l, self.num_heads as usize, self.head_size as usize])?;
        let v = ops::transpose(&v, &[0, 2, 1, 3])?;

        let scale = 1.0 / (self.head_size as f32).sqrt();
        let out = ops::fast_sdpa(&q, &k, &v, scale, "", attention_mask)?;

        let out = ops::transpose(&out, &[0, 2, 1, 3])?;
        ops::reshape(&out, &[b, l, (self.num_heads * self.head_size) as usize])
    }

    pub fn load_from_tensors(
        tensors: &HashMap<String, Array>,
        prefix: &str,
        num_heads: i32,
        head_size: i32,
    ) -> anyhow::Result<Self> {
        let query = make_linear(tensors, &format!("{prefix}query.weight"))?;
        let key = make_linear(tensors, &format!("{prefix}key.weight"))?;
        let value = make_linear(tensors, &format!("{prefix}value.weight"))?;

        Ok(Self {
            query,
            key,
            value,
            num_heads,
            head_size,
        })
    }
}

pub struct BertAttention {
    self_attention: BertSelfAttention,
    output_dense: Box<dyn LinearLayer>,
    layer_norm: LayerNorm,
}

impl BertAttention {
    pub fn forward(
        &self,
        hidden_states: &Array,
        attention_mask: Option<&Array>,
    ) -> anyhow::Result<Array> {
        let attn_out = self.self_attention.forward(hidden_states, attention_mask)?;
        let projected = self.output_dense.forward(&attn_out)?;
        let summed = ops::add(hidden_states, &projected)?;
        self.layer_norm.forward(&summed)
    }

    pub fn load_from_tensors(
        tensors: &HashMap<String, Array>,
        prefix: &str,
        config: &BertConfig,
    ) -> anyhow::Result<Self> {
        let self_attn_prefix = format!("{prefix}attention.self.");
        let self_attention = BertSelfAttention::load_from_tensors(
            tensors,
            &self_attn_prefix,
            config.num_attention_heads,
            config.head_size(),
        )?;

        let output_dense = make_linear(tensors, &format!("{prefix}attention.output.dense.weight"))?;

        let ln_weight = tensors
            .get(&format!("{prefix}attention.output.LayerNorm.weight"))
            .ok_or_else(|| anyhow::anyhow!("missing {prefix}attention.output.LayerNorm.weight"))?;
        let ln_bias = tensors
            .get(&format!("{prefix}attention.output.LayerNorm.bias"))
            .ok_or_else(|| anyhow::anyhow!("missing {prefix}attention.output.LayerNorm.bias"))?;
        let layer_norm = LayerNorm::new(ln_weight.clone(), Some(ln_bias.clone()), config.layer_norm_eps);

        Ok(Self {
            self_attention,
            output_dense,
            layer_norm,
        })
    }
}

pub struct BertMlp {
    intermediate_dense: Box<dyn LinearLayer>,
    output_dense: Box<dyn LinearLayer>,
    layer_norm: LayerNorm,
}

impl BertMlp {
    pub fn forward(&self, hidden_states: &Array) -> anyhow::Result<Array> {
        let intermediate = self.intermediate_dense.forward(hidden_states)?;
        let activated = gelu(&intermediate)?;
        let output = self.output_dense.forward(&activated)?;
        let summed = ops::add(hidden_states, &output)?;
        self.layer_norm.forward(&summed)
    }

    pub fn load_from_tensors(
        tensors: &HashMap<String, Array>,
        prefix: &str,
        config: &BertConfig,
    ) -> anyhow::Result<Self> {
        let intermediate_dense = make_linear(tensors, &format!("{prefix}intermediate.dense.weight"))?;
        let output_dense = make_linear(tensors, &format!("{prefix}output.dense.weight"))?;

        let ln_weight = tensors
            .get(&format!("{prefix}output.LayerNorm.weight"))
            .ok_or_else(|| anyhow::anyhow!("missing {prefix}output.LayerNorm.weight"))?;
        let ln_bias = tensors
            .get(&format!("{prefix}output.LayerNorm.bias"))
            .ok_or_else(|| anyhow::anyhow!("missing {prefix}output.LayerNorm.bias"))?;
        let layer_norm = LayerNorm::new(ln_weight.clone(), Some(ln_bias.clone()), config.layer_norm_eps);

        Ok(Self {
            intermediate_dense,
            output_dense,
            layer_norm,
        })
    }
}

pub struct BertLayer {
    attention: BertAttention,
    mlp: BertMlp,
}

impl BertLayer {
    pub fn forward(
        &self,
        hidden_states: &Array,
        attention_mask: Option<&Array>,
    ) -> anyhow::Result<Array> {
        let attention_output = self.attention.forward(hidden_states, attention_mask)?;
        self.mlp.forward(&attention_output)
    }

    pub fn load_from_tensors(
        tensors: &HashMap<String, Array>,
        layer_idx: usize,
        config: &BertConfig,
        prefix: &str,
    ) -> anyhow::Result<Self> {
        let layer_prefix = format!("{prefix}encoder.layer.{layer_idx}.");
        let attention = BertAttention::load_from_tensors(tensors, &layer_prefix, config)?;
        let mlp = BertMlp::load_from_tensors(tensors, &layer_prefix, config)?;

        Ok(Self { attention, mlp })
    }
}

pub struct BertModel {
    embeddings: BertEmbeddings,
    layers: Vec<BertLayer>,
    config: BertConfig,
}

impl BertModel {
    pub fn load_from_tensors(
        tensors: HashMap<String, Array>,
        config: BertConfig,
    ) -> anyhow::Result<Self> {
        let prefix = resolve_bert_prefix(&tensors);
        let embeddings = BertEmbeddings::load_from_tensors(&tensors, &config, &prefix)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for i in 0..config.num_hidden_layers as usize {
            layers.push(BertLayer::load_from_tensors(&tensors, i, &config, &prefix)?);
        }

        Ok(Self {
            embeddings,
            layers,
            config,
        })
    }

    pub fn forward_inner(
        &self,
        input_ids: &Array,
        attention_mask: Option<&Array>,
    ) -> anyhow::Result<Array> {
        let mut hidden_states = self.embeddings.forward(input_ids, None)?;

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }

        Ok(hidden_states)
    }
}

impl EncoderModel for BertModel {
    fn encode(&self, input_ids: &Array) -> anyhow::Result<Array> {
        self.forward_inner(input_ids, None)
    }

    fn encode_masked(
        &self,
        input_ids: &Array,
        attention_mask: Option<&Array>,
    ) -> anyhow::Result<Array> {
        self.forward_inner(input_ids, attention_mask)
    }

    fn hidden_size(&self) -> i32 {
        self.config.hidden_size
    }
}

impl Model for BertModel {
    fn forward(
        &self,
        _input_ids: &Array,
        _caches: &mut [crate::llama::LayerCache],
        _positions: &Array,
    ) -> anyhow::Result<Array> {
        anyhow::bail!("BERT is an encoder model; use EncoderModel::encode() instead")
    }

    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers as usize
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

    fn new_caches(&self) -> Vec<crate::llama::LayerCache> {
        Vec::new()
    }
}
