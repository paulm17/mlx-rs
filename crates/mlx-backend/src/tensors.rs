use std::collections::HashMap;
use std::path::Path;

use anyhow::Context;

use crate::array::Array;
use crate::ffi::{
    MlxArray, MlxMapStringToArray, MlxMapStringToString,
};
use crate::loader;

pub struct SafetensorsFile {
    arrays: MlxMapStringToArray,
    metadata: MlxMapStringToString,
}

impl Drop for SafetensorsFile {
    fn drop(&mut self) {
        let syms = loader::symbols().expect("MLX not initialized");
        unsafe {
            (syms.mlx_map_string_to_array_free)(self.arrays);
            (syms.mlx_map_string_to_string_free)(self.metadata);
        }
    }
}

impl SafetensorsFile {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let syms = loader::symbols()?;
        let c_path = std::ffi::CString::new(path.to_string_lossy().as_bytes())
            .context("path contains null byte")?;

        // Use the thread-local CPU stream. This stream was created by
        // init_streams() via mlx_default_cpu_stream_new() and is kept alive
        // in thread-local storage. Arrays loaded via mlx_load_safetensors
        // will reference this stream.
        let stream = crate::ops::cpu_stream()?;

        let mut arrays = MlxMapStringToArray {
            ctx: std::ptr::null_mut(),
        };
        let mut metadata = MlxMapStringToString {
            ctx: std::ptr::null_mut(),
        };

        let rc = unsafe {
            (syms.mlx_load_safetensors)(
                &mut arrays,
                &mut metadata,
                c_path.as_ptr(),
                stream,
            )
        };

        // DO NOT free the stream — it's stored in thread-local storage
        // and referenced by the loaded arrays.

        if rc != 0 {
            anyhow::bail!("mlx_load_safetensors failed for: {}", path.display());
        }

        Ok(Self { arrays, metadata })
    }

    pub fn load_all(&self) -> anyhow::Result<HashMap<String, Array>> {
        let syms = loader::symbols()?;
        let mut result = HashMap::new();

        let iter = unsafe {
            (syms.mlx_map_string_to_array_iterator_new)(self.arrays)
        };

        loop {
            let mut key_ptr: *const std::ffi::c_char = std::ptr::null();
            let mut value = MlxArray {
                ctx: std::ptr::null_mut(),
            };
            let rc = unsafe {
                (syms.mlx_map_string_to_array_iterator_next)(
                    &mut key_ptr,
                    &mut value,
                    iter,
                )
            };
            if rc != 0 {
                break;
            }
            if key_ptr.is_null() {
                break;
            }

            let name = unsafe { std::ffi::CStr::from_ptr(key_ptr) }
                .to_str()
                .context("invalid UTF-8 in tensor name")?
                .to_owned();

            result.insert(name, Array { ctx: value });
        }

        unsafe {
            (syms.mlx_map_string_to_array_iterator_free)(iter);
        }

        Ok(result)
    }

    pub fn tensor_names(&self) -> anyhow::Result<Vec<String>> {
        let syms = loader::symbols()?;
        let mut names = Vec::new();

        let iter = unsafe {
            (syms.mlx_map_string_to_array_iterator_new)(self.arrays)
        };

        loop {
            let mut key_ptr: *const std::ffi::c_char = std::ptr::null();
            let mut value = MlxArray {
                ctx: std::ptr::null_mut(),
            };
            let rc = unsafe {
                (syms.mlx_map_string_to_array_iterator_next)(
                    &mut key_ptr,
                    &mut value,
                    iter,
                )
            };
            if rc != 0 {
                break;
            }
            if key_ptr.is_null() {
                break;
            }

            let name = unsafe { std::ffi::CStr::from_ptr(key_ptr) }
                .to_str()
                .context("invalid UTF-8 in tensor name")?
                .to_owned();
            names.push(name);
        }

        unsafe {
            (syms.mlx_map_string_to_array_iterator_free)(iter);
        }

        names.sort();
        Ok(names)
    }
}
