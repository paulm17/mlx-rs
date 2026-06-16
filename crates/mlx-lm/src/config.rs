use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LlamaCppConfig {
    pub n_ctx: Option<u32>,
    pub n_batch: Option<u32>,
    pub n_ubatch: Option<u32>,
    pub n_gpu_layers: Option<u32>,
    pub n_threads: Option<u32>,
    pub n_threads_batch: Option<u32>,
    pub embedding: Option<bool>,
    pub pooling: Option<String>,
    pub use_mmap: Option<bool>,
    pub use_mlock: Option<bool>,
    pub flash_attn: Option<bool>,
}
