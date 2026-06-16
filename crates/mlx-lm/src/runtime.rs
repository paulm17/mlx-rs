use anyhow::{bail, Result};

use crate::config::LlamaCppConfig;
use crate::types::LoadedModelInfo;

pub struct Runtime;

impl Runtime {
    pub fn new(_model_path: &str, _config: LlamaCppConfig) -> Result<Self> {
        bail!("llama.cpp runtime not yet implemented")
    }

    pub fn model_info(&self) -> Result<LoadedModelInfo> {
        bail!("llama.cpp runtime not yet implemented")
    }

    pub fn tokenize(&self, _text: &str, _add_bos: bool) -> Result<Vec<i32>> {
        bail!("llama.cpp runtime not yet implemented")
    }

    pub fn detokenize(&self, _tokens: &[i32]) -> Result<String> {
        bail!("llama.cpp runtime not yet implemented")
    }
}
