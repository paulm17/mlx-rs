use crate::{Array, Error, Result};
use mlx_sys::*;
use std::collections::HashMap;
use std::path::Path;

/// Load tensors from a safetensors file.
pub fn load<P: AsRef<Path>>(path: P) -> Result<HashMap<String, Array>> {
    let path_str = path
        .as_ref()
        .to_str()
        .ok_or_else(|| Error::Message("invalid path".into()))?;
    let c_path = std::ffi::CString::new(path_str)
        .map_err(|_| Error::Message("path contains null byte".into()))?;

    unsafe {
        // mlx-c Load operations only support CPU evaluation
        let s = mlx_default_cpu_stream_new();

        let mut data_map: mlx_map_string_to_array = std::mem::zeroed();
        let mut meta_map: mlx_map_string_to_string = std::mem::zeroed();
        let rc = mlx_load_safetensors(&mut data_map, &mut meta_map, c_path.as_ptr(), s);
        mlx_map_string_to_string_free(meta_map);

        if rc != 0 {
            return Err(Error::SafeTensors(format!("failed to load: {path_str}")));
        }

        let mut result = HashMap::new();
        let it = mlx_map_string_to_array_iterator_new(data_map);

        loop {
            let mut key_ptr: *const std::os::raw::c_char = std::ptr::null();
            let mut val: mlx_array = std::mem::zeroed();
            let has_next = mlx_map_string_to_array_iterator_next(&mut key_ptr, &mut val, it);
            if has_next != 0 {
                break;
            }
            if key_ptr.is_null() {
                break;
            }
            let name = std::ffi::CStr::from_ptr(key_ptr)
                .to_string_lossy()
                .into_owned();

            // iterator_next already returns an owned array handle for the value.
            // Wrapping it directly avoids leaking an extra retained reference.
            let array = Array::from_raw(val);
            array.eval()?;
            result.insert(name, array);
        }

        mlx_map_string_to_array_iterator_free(it);
        mlx_map_string_to_array_free(data_map);

        Ok(result)
    }
}
