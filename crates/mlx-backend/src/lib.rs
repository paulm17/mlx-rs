pub mod array;
pub mod ffi;
pub mod loader;
pub mod memory;
pub mod ops;

pub use array::Array;
pub use ffi::{MlxDtype, MlxDeviceType};
pub use loader::{check_init, default_device_available, device_count, loaded_library_path, symbols, version};
