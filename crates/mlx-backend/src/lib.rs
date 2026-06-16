pub mod array;
pub mod ffi;
pub mod loader;
pub mod manifest;
pub mod memory;
pub mod ops;
pub mod tensors;

pub use array::Array;
pub use ffi::{MlxDtype, MlxDeviceType};
pub use loader::{check_init, default_device_available, device_count, loaded_library_path, symbols, version};
pub use manifest::ModelManifest;
pub use tensors::SafetensorsFile;
