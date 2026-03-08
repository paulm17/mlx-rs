use mlx_core::{Array, Module, Result};
use mlx_nn::{Activation, Embedding, LayerNorm, Linear, QuantConfig, VarBuilder};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct BertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
    #[serde(default = "default_hidden_dropout_prob")]
    pub hidden_dropout_prob: f32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_type_vocab_size")]
    pub type_vocab_size: usize,
    #[serde(default = "default_initializer_range")]
    pub initializer_range: f32,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(default = "default_pad_token_id")]
    pub pad_token_id: usize,
    pub quantization: Option<QuantizationConfig>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct QuantizationConfig {
    #[serde(default = "default_group_size")]
    pub group_size: i32,
    #[serde(default = "default_bits")]
    pub bits: i32,
}

fn default_hidden_act() -> Activation {
    Activation::Gelu
}
fn default_hidden_dropout_prob() -> f32 {
    0.1
}
fn default_max_position_embeddings() -> usize {
    512
}
fn default_type_vocab_size() -> usize {
    2
}
fn default_initializer_range() -> f32 {
    0.02
}
fn default_layer_norm_eps() -> f32 {
    1e-12
}
fn default_pad_token_id() -> usize {
    0
}
fn default_group_size() -> i32 {
    64
}
fn default_bits() -> i32 {
    4
}

impl BertConfig {
    fn quant_config(&self) -> QuantConfig {
        match &self.quantization {
            Some(q) => QuantConfig {
                group_size: q.group_size,
                bits: q.bits,
            },
            None => QuantConfig::default(),
        }
    }
}

struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
}

impl BertEmbeddings {
    fn load(vb: &VarBuilder, cfg: &BertConfig) -> anyhow::Result<Self> {
        let qc = cfg.quant_config();
        Ok(Self {
            word_embeddings: Embedding::new(&vb.pp("word_embeddings"), &qc)?,
            position_embeddings: Embedding::new(&vb.pp("position_embeddings"), &qc)?,
            token_type_embeddings: Embedding::new(&vb.pp("token_type_embeddings"), &qc)?,
            layer_norm: LayerNorm::new(cfg.layer_norm_eps, &vb.pp("LayerNorm"))?,
        })
    }

    fn forward(&self, input_ids: &Array) -> Result<Array> {
        let shape = input_ids.shape_raw();
        let batch = shape[0];
        let seq_len = shape[1];

        let word_embeddings = self.word_embeddings.forward(input_ids)?;
        let position_ids =
            Array::from_slice_i32(&(0..seq_len).collect::<Vec<_>>())?.reshape(&[1, seq_len])?;
        let position_embeddings = self.position_embeddings.forward(&position_ids)?;
        let token_type_ids = Array::zeros(&[batch, seq_len], mlx_core::DType::Int32)?;
        let token_type_embeddings = self.token_type_embeddings.forward(&token_type_ids)?;
        let embeddings = word_embeddings
            .add(&position_embeddings)?
            .add(&token_type_embeddings)?;
        self.layer_norm.forward(&embeddings)
    }
}

struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dense: Linear,
    num_attention_heads: usize,
    attention_head_size: usize,
    scale: f32,
}

impl BertSelfAttention {
    fn load(vb: &VarBuilder, cfg: &BertConfig) -> anyhow::Result<Self> {
        let qc = cfg.quant_config();
        let attention_head_size = cfg.hidden_size / cfg.num_attention_heads;
        Ok(Self {
            query: Linear::new(&vb.pp("self.query"), &qc)?,
            key: Linear::new(&vb.pp("self.key"), &qc)?,
            value: Linear::new(&vb.pp("self.value"), &qc)?,
            dense: Linear::new(&vb.pp("output.dense"), &qc)?,
            num_attention_heads: cfg.num_attention_heads,
            attention_head_size,
            scale: 1.0 / (attention_head_size as f32).sqrt(),
        })
    }

    fn forward(&self, hidden_states: &Array, attention_mask: Option<&Array>) -> Result<Array> {
        let shape = hidden_states.shape_raw();
        let (batch, seq_len, hidden_size) = (shape[0], shape[1], shape[2]);

        let query = self
            .query
            .forward(hidden_states)?
            .reshape(&[
                batch,
                seq_len,
                self.num_attention_heads as i32,
                self.attention_head_size as i32,
            ])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let key = self
            .key
            .forward(hidden_states)?
            .reshape(&[
                batch,
                seq_len,
                self.num_attention_heads as i32,
                self.attention_head_size as i32,
            ])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let value = self
            .value
            .forward(hidden_states)?
            .reshape(&[
                batch,
                seq_len,
                self.num_attention_heads as i32,
                self.attention_head_size as i32,
            ])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let mask = if let Some(attention_mask) = attention_mask {
            let mask = attention_mask
                .reshape(&[batch, 1, 1, seq_len])?
                .as_type(hidden_states.dtype())?;
            let zero = Array::from_float(0.0)?.as_type(hidden_states.dtype())?;
            let neg_inf = Array::from_float(-1.0e9)?.as_type(hidden_states.dtype())?;
            Some(mask.greater(&zero)?.where_cond(&zero, &neg_inf)?)
        } else {
            None
        };

        let context = query.fast_scaled_dot_product_attention(
            &key,
            &value,
            self.scale,
            if mask.is_some() { "array" } else { "" },
            mask.as_ref(),
        )?;
        let context =
            context
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[batch, seq_len, hidden_size])?;
        self.dense.forward(&context)
    }
}

struct BertAttention {
    self_attention: BertSelfAttention,
    layer_norm: LayerNorm,
}

impl BertAttention {
    fn load(vb: &VarBuilder, cfg: &BertConfig) -> anyhow::Result<Self> {
        Ok(Self {
            self_attention: BertSelfAttention::load(vb, cfg)?,
            layer_norm: LayerNorm::new(cfg.layer_norm_eps, &vb.pp("output.LayerNorm"))?,
        })
    }

    fn forward(&self, hidden_states: &Array, attention_mask: Option<&Array>) -> Result<Array> {
        let residual = hidden_states.clone();
        let output = self.self_attention.forward(hidden_states, attention_mask)?;
        self.layer_norm.forward(&residual.add(&output)?)
    }
}

struct BertMlp {
    intermediate: Linear,
    output: Linear,
    output_layer_norm: LayerNorm,
    activation: Activation,
}

impl BertMlp {
    fn load(vb: &VarBuilder, cfg: &BertConfig) -> anyhow::Result<Self> {
        let qc = cfg.quant_config();
        Ok(Self {
            intermediate: Linear::new(&vb.pp("intermediate.dense"), &qc)?,
            output: Linear::new(&vb.pp("output.dense"), &qc)?,
            output_layer_norm: LayerNorm::new(cfg.layer_norm_eps, &vb.pp("output.LayerNorm"))?,
            activation: cfg.hidden_act,
        })
    }

    fn forward(&self, hidden_states: &Array) -> Result<Array> {
        let residual = hidden_states.clone();
        let hidden = self.intermediate.forward(hidden_states)?;
        let hidden = self.activation.forward(&hidden)?;
        let hidden = self.output.forward(&hidden)?;
        self.output_layer_norm.forward(&residual.add(&hidden)?)
    }
}

struct BertLayer {
    attention: BertAttention,
    mlp: BertMlp,
}

impl BertLayer {
    fn load(vb: &VarBuilder, cfg: &BertConfig) -> anyhow::Result<Self> {
        Ok(Self {
            attention: BertAttention::load(&vb.pp("attention"), cfg)?,
            mlp: BertMlp::load(vb, cfg)?,
        })
    }

    fn forward(&self, hidden_states: &Array, attention_mask: Option<&Array>) -> Result<Array> {
        let hidden_states = self.attention.forward(hidden_states, attention_mask)?;
        self.mlp.forward(&hidden_states)
    }
}

pub struct Bert {
    embeddings: BertEmbeddings,
    layers: Vec<BertLayer>,
}

impl Bert {
    pub fn new(cfg: &BertConfig, vb: &VarBuilder) -> anyhow::Result<Self> {
        let model_vb = if vb.contains("embeddings.word_embeddings.weight") {
            vb.clone()
        } else {
            vb.pp("bert")
        };
        let embeddings = BertEmbeddings::load(&model_vb.pp("embeddings"), cfg)?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|idx| BertLayer::load(&model_vb.pp(format!("encoder.layer.{idx}")), cfg))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self { embeddings, layers })
    }

    pub fn encode(&self, input_ids: &Array) -> Result<Array> {
        self.encode_masked(input_ids, None)
    }

    pub fn encode_masked(&self, input_ids: &Array, attention_mask: Option<&Array>) -> Result<Array> {
        let mut hidden_states = self.embeddings.forward(input_ids)?;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }
        Ok(hidden_states)
    }

    pub fn reset_state(&mut self) {}
}
