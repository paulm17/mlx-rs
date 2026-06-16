use crate::ffi::{MlxDtype, MlxArray};
use crate::loader;

pub struct Array {
    pub(crate) ctx: MlxArray,
}

unsafe impl Send for Array {}

impl Drop for Array {
    fn drop(&mut self) {
        if !self.ctx.ctx.is_null() {
            let syms = loader::symbols().expect("MLX not initialized");
            unsafe {
                (syms.mlx_array_free)(self.ctx);
            }
        }
    }
}

impl Array {
    pub fn from_f32(val: f32) -> anyhow::Result<Self> {
        let syms = loader::symbols()?;
        let ctx = unsafe { (syms.mlx_array_new_float32)(val) };
        if ctx.ctx.is_null() {
            return Err(anyhow::anyhow!("mlx_array_new_float32 returned null"));
        }
        Ok(Self { ctx })
    }

    pub fn from_f64(val: f64) -> anyhow::Result<Self> {
        let syms = loader::symbols()?;
        let ctx = unsafe { (syms.mlx_array_new_float64)(val) };
        if ctx.ctx.is_null() {
            return Err(anyhow::anyhow!("mlx_array_new_float64 returned null"));
        }
        Ok(Self { ctx })
    }

    pub fn from_i32(val: i32) -> anyhow::Result<Self> {
        let syms = loader::symbols()?;
        let ctx = unsafe { (syms.mlx_array_new_int)(val) };
        if ctx.ctx.is_null() {
            return Err(anyhow::anyhow!("mlx_array_new_int returned null"));
        }
        Ok(Self { ctx })
    }

    pub fn from_data_f32(data: &[f32], shape: &[usize]) -> anyhow::Result<Self> {
        let syms = loader::symbols()?;
        let c_shape: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        let ctx = unsafe {
            (syms.mlx_array_new_data)(
                data.as_ptr() as *const std::ffi::c_void,
                c_shape.as_ptr(),
                shape.len() as i32,
                MlxDtype::Float32 as i32,
            )
        };
        if ctx.ctx.is_null() {
            return Err(anyhow::anyhow!("mlx_array_new_data returned null"));
        }
        Ok(Self { ctx })
    }

    pub fn from_data_i32(data: &[i32], shape: &[usize]) -> anyhow::Result<Self> {
        let syms = loader::symbols()?;
        let c_shape: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        let ctx = unsafe {
            (syms.mlx_array_new_data)(
                data.as_ptr() as *const std::ffi::c_void,
                c_shape.as_ptr(),
                shape.len() as i32,
                MlxDtype::Int32 as i32,
            )
        };
        if ctx.ctx.is_null() {
            return Err(anyhow::anyhow!("mlx_array_new_data returned null"));
        }
        Ok(Self { ctx })
    }

    pub fn size(&self) -> usize {
        let syms = loader::symbols().expect("MLX not initialized");
        unsafe { (syms.mlx_array_size)(self.ctx) }
    }

    pub fn nbytes(&self) -> usize {
        let syms = loader::symbols().expect("MLX not initialized");
        unsafe { (syms.mlx_array_nbytes)(self.ctx) }
    }

    pub fn ndim(&self) -> usize {
        let syms = loader::symbols().expect("MLX not initialized");
        unsafe { (syms.mlx_array_ndim)(self.ctx) }
    }

    pub fn shape(&self) -> Vec<usize> {
        let syms = loader::symbols().expect("MLX not initialized");
        let ndim = self.ndim();
        if ndim == 0 {
            return vec![];
        }
        let raw = unsafe { (syms.mlx_array_shape)(self.ctx) };
        if raw.is_null() {
            return vec![];
        }
        (0..ndim)
            .map(|i| unsafe { *raw.add(i) as usize })
            .collect()
    }

    pub fn dim(&self, axis: usize) -> anyhow::Result<usize> {
        let syms = loader::symbols()?;
        let val = unsafe { (syms.mlx_array_dim)(self.ctx, axis as i32) };
        Ok(val as usize)
    }

    pub fn dtype(&self) -> anyhow::Result<MlxDtype> {
        let syms = loader::symbols()?;
        let raw = unsafe { (syms.mlx_array_dtype)(self.ctx) };
        MlxDtype::try_from(raw)
    }

    pub fn item_f64(&self) -> anyhow::Result<f64> {
        let syms = loader::symbols()?;
        let mut val: f64 = 0.0;
        let rc = unsafe { (syms.mlx_array_item_float64)(&mut val, self.ctx) };
        if rc != 0 {
            return Err(anyhow::anyhow!("mlx_array_item_float64 returned error: {rc}"));
        }
        Ok(val)
    }

    pub fn item_i64(&self) -> anyhow::Result<i64> {
        let syms = loader::symbols()?;
        let mut val: i64 = 0;
        let rc = unsafe { (syms.mlx_array_item_int64)(&mut val, self.ctx) };
        if rc != 0 {
            return Err(anyhow::anyhow!("mlx_array_item_int64 returned error: {rc}"));
        }
        Ok(val)
    }

    pub fn data_f32(&self) -> anyhow::Result<&[f32]> {
        let syms = loader::symbols()?;
        let ptr = unsafe { (syms.mlx_array_data_float32)(self.ctx) };
        if ptr.is_null() {
            return Err(anyhow::anyhow!("mlx_array_data_float32 returned null"));
        }
        let len = self.nbytes() / std::mem::size_of::<f32>();
        Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    pub fn data_i32(&self) -> anyhow::Result<&[i32]> {
        let syms = loader::symbols()?;
        let ptr = unsafe { (syms.mlx_array_data_int32)(self.ctx) };
        if ptr.is_null() {
            return Err(anyhow::anyhow!("mlx_array_data_int32 returned null"));
        }
        let len = self.nbytes() / std::mem::size_of::<i32>();
        Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    pub fn eval(&self) -> anyhow::Result<()> {
        let syms = loader::symbols()?;
        let rc = unsafe { (syms.mlx_array_eval)(self.ctx) };
        if rc != 0 {
            return Err(anyhow::anyhow!("mlx_array_eval returned error: {rc}"));
        }
        Ok(())
    }

    pub fn to_string_val(&self) -> anyhow::Result<String> {
        let syms = loader::symbols()?;
        let mut str_ptr = crate::ffi::MlxString { ctx: std::ptr::null_mut() };
        let rc = unsafe { (syms.mlx_array_tostring)(&mut str_ptr, self.ctx) };
        if rc != 0 {
            return Err(anyhow::anyhow!("mlx_array_tostring returned error: {rc}"));
        }
        if str_ptr.ctx.is_null() {
            return Err(anyhow::anyhow!("mlx_array_tostring returned null string"));
        }
        let data = unsafe { (syms.mlx_string_data)(str_ptr) };
        if data.is_null() {
            unsafe { (syms.mlx_string_free)(str_ptr) };
            return Err(anyhow::anyhow!("mlx_string_data returned null"));
        }
        let s = unsafe { std::ffi::CStr::from_ptr(data) }
            .to_str()
            .context("mlx_string_data returned invalid UTF-8")?
            .to_owned();
        unsafe { (syms.mlx_string_free)(str_ptr) };
        Ok(s)
    }

    pub(crate) fn raw(&self) -> MlxArray {
        self.ctx
    }
}

use anyhow::Context;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader;

    fn mlx_available() -> bool {
        loader::check_init().is_ok()
    }

    #[test]
    fn test_from_f32() {
        if !mlx_available() {
            return;
        }
        let arr = Array::from_f32(3.14).unwrap();
        assert_eq!(arr.size(), 1);
        assert_eq!(arr.ndim(), 0);
        assert_eq!(arr.dtype().unwrap(), MlxDtype::Float32);
    }

    #[test]
    fn test_from_i32() {
        if !mlx_available() {
            return;
        }
        let arr = Array::from_i32(42).unwrap();
        assert_eq!(arr.size(), 1);
        assert_eq!(arr.dtype().unwrap(), MlxDtype::Int32);
    }

    #[test]
    fn test_from_data_f32() {
        if !mlx_available() {
            return;
        }
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let arr = Array::from_data_f32(&data, &[2, 3]).unwrap();
        assert_eq!(arr.size(), 6);
        assert_eq!(arr.ndim(), 2);
        assert_eq!(arr.shape(), vec![2, 3]);
        assert_eq!(arr.dim(0).unwrap(), 2);
        assert_eq!(arr.dim(1).unwrap(), 3);
    }

    #[test]
    fn test_from_data_i32() {
        if !mlx_available() {
            return;
        }
        let data = [10i32, 20, 30];
        let arr = Array::from_data_i32(&data, &[3]).unwrap();
        assert_eq!(arr.size(), 3);
        assert_eq!(arr.dtype().unwrap(), MlxDtype::Int32);
    }

    #[test]
    fn test_item_f64() {
        if !mlx_available() {
            return;
        }
        let arr = Array::from_f32(2.5).unwrap();
        arr.eval().unwrap();
        let val = arr.item_f64().unwrap();
        assert!((val - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_item_i64() {
        if !mlx_available() {
            return;
        }
        let arr = Array::from_i32(99).unwrap();
        arr.eval().unwrap();
        let val = arr.item_i64().unwrap();
        assert_eq!(val, 99);
    }

    #[test]
    fn test_data_f32_slice() {
        if !mlx_available() {
            return;
        }
        let data = [1.0f32, 2.0, 3.0];
        let arr = Array::from_data_f32(&data, &[3]).unwrap();
        arr.eval().unwrap();
        let slice = arr.data_f32().unwrap();
        assert_eq!(slice, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_data_i32_slice() {
        if !mlx_available() {
            return;
        }
        let data = [10i32, 20];
        let arr = Array::from_data_i32(&data, &[2]).unwrap();
        arr.eval().unwrap();
        let slice = arr.data_i32().unwrap();
        assert_eq!(slice, &[10, 20]);
    }

    #[test]
    fn test_nbytes() {
        if !mlx_available() {
            return;
        }
        let arr = Array::from_data_f32(&[1.0, 2.0], &[2]).unwrap();
        assert_eq!(arr.nbytes(), 8);
    }

    #[test]
    fn test_to_string() {
        if !mlx_available() {
            return;
        }
        let arr = Array::from_f32(1.0).unwrap();
        let s = arr.to_string_val().unwrap();
        assert!(!s.is_empty());
    }
}
