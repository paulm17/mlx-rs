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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MlxConfig {
    pub cache_limit: Option<usize>,
    pub compile: Option<bool>,
    pub prefill_chunk_size: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llamacpp_config_default() {
        let config = LlamaCppConfig::default();
        assert!(config.n_ctx.is_none());
        assert!(config.n_batch.is_none());
        assert!(config.flash_attn.is_none());
    }

    #[test]
    fn test_mlx_config_default() {
        let config = MlxConfig::default();
        assert!(config.cache_limit.is_none());
        assert!(config.compile.is_none());
        assert!(config.prefill_chunk_size.is_none());
    }

    #[test]
    fn test_mlx_config_with_values() {
        let config = MlxConfig {
            cache_limit: Some(128),
            compile: Some(false),
            prefill_chunk_size: Some(1024),
        };
        assert_eq!(config.cache_limit, Some(128));
        assert_eq!(config.compile, Some(false));
        assert_eq!(config.prefill_chunk_size, Some(1024));
    }

    #[test]
    fn test_llamacpp_config_flash_attn_roundtrip() {
        let json = serde_json::to_string(&LlamaCppConfig {
            flash_attn: Some(true),
            ..Default::default()
        }).unwrap();
        let parsed: LlamaCppConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.flash_attn, Some(true));
    }
}
