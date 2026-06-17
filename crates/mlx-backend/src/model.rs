use crate::array::Array;
use crate::llama::LayerCache;

pub trait Model: Send {
    fn forward(
        &self,
        input_ids: &Array,
        caches: &mut [LayerCache],
        positions: &Array,
    ) -> anyhow::Result<Array>;

    fn num_layers(&self) -> usize;

    fn max_position_embeddings(&self) -> i32;

    fn hidden_size(&self) -> i32;

    fn vocab_size(&self) -> i32;

    fn new_caches(&self) -> Vec<LayerCache>;
}

pub trait EncoderModel: Send {
    fn encode(&self, input_ids: &Array) -> anyhow::Result<Array>;

    fn encode_masked(
        &self,
        input_ids: &Array,
        attention_mask: Option<&Array>,
    ) -> anyhow::Result<Array>;

    fn hidden_size(&self) -> i32;
}
