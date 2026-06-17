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
