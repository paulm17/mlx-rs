pub mod ffi;
pub mod loader;

pub use ffi::MlxDeviceType;
pub use loader::{check_init, default_device_available, device_count, loaded_library_path, version};
