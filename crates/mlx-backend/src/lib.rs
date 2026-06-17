pub mod array;
pub mod cache;
pub mod chat_template;
pub mod ffi;
pub mod gemma3;
pub mod gemma4;
pub mod llama;
pub mod loader;
pub mod manifest;
pub mod memory;
pub mod mlx_backend;
pub mod model;
pub mod ops;
pub mod qwen3;
pub mod registry;
pub mod sampler;
pub mod tensors;

pub use array::Array;
pub use ffi::{MlxDtype, MlxDeviceType};
pub use loader::{check_init, default_device_available, device_count, loaded_library_path, symbols, version};
pub use manifest::ModelManifest;
pub use mlx_backend::MlxBackend;
pub use tensors::SafetensorsFile;

#[ctor::ctor]
fn register_mlx_backend() {
    llama_lm::register_safetensors_backend(|path: &std::path::Path| -> anyhow::Result<Box<dyn backend_trait::Backend>> {
        Ok(Box::new(MlxBackend::load(path)?))
    });
}
