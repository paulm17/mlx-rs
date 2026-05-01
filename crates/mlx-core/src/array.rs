use crate::{DType, Error, Result, Shape};
use mlx_sys::*;

/// Core array type wrapping an MLX C array handle.
///
/// This is the fundamental data structure for all tensor operations.
/// It wraps the opaque `mlx_array` pointer from the C API and provides
/// safe Rust ergonomics including `Clone`, `Drop`, `Send`, and `Sync`.
pub struct Array {
    pub(crate) inner: mlx_array,
}

// MLX arrays are thread-safe.
unsafe impl Send for Array {}
unsafe impl Sync for Array {}

impl Array {
    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Wrap a raw mlx_array pointer. Ownership is transferred.
    pub(crate) fn from_raw(ptr: mlx_array) -> Self {
        Self { inner: ptr }
    }

    /// Get the default stream for operations (device-based).
    pub(crate) fn default_stream() -> mlx_stream {
        unsafe {
            let mut dev: mlx_device = std::mem::zeroed();
            mlx_get_default_device(&mut dev);
            let mut stream: mlx_stream = std::mem::zeroed();
            mlx_get_default_stream(&mut stream, dev);
            // FIXME: We should not be freeing the device here, since the stream
            // may still reference it. This works because the MLX runtime internally
            // retains the device reference on the stream.
            mlx_device_free(dev);
            stream
        }
    }

    /// Check the return code from an mlx-c function.
    pub(crate) fn check(rc: std::os::raw::c_int, context: &str) -> Result<()> {
        if rc != 0 {
            Err(Error::Mlx(format!("{context} failed (rc={rc})")))
        } else {
            Ok(())
        }
    }

    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Create an array filled with zeros.
    pub fn zeros(shape: &[i32], dtype: DType) -> Result<Self> {
        unsafe {
            let s = Self::default_stream();
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_zeros(&mut out, shape.as_ptr(), shape.len(), dtype.to_mlx(), s),
                "mlx_zeros",
            )?;
            Ok(Self::from_raw(out))
        }
    }

    /// Create an array filled with ones.
    pub fn ones(shape: &[i32], dtype: DType) -> Result<Self> {
        unsafe {
            let s = Self::default_stream();
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_ones(&mut out, shape.as_ptr(), shape.len(), dtype.to_mlx(), s),
                "mlx_ones",
            )?;
            Ok(Self::from_raw(out))
        }
    }

    /// Create a scalar float32 array.
    pub fn from_float(val: f32) -> Result<Self> {
        unsafe {
            let out = mlx_array_new_float(val);
            Ok(Self::from_raw(out))
        }
    }

    /// Create a scalar int32 array.
    pub fn from_int(val: i32) -> Result<Self> {
        unsafe {
            let out = mlx_array_new_int(val);
            Ok(Self::from_raw(out))
        }
    }

    /// Create a 1D array from a slice of f32 values.
    pub fn from_slice_f32(data: &[f32]) -> Result<Self> {
        unsafe {
            let shape = [data.len() as i32];
            let out = mlx_array_new_data(
                data.as_ptr() as *const std::ffi::c_void,
                shape.as_ptr(),
                1, // dim (ndim) as c_int
                mlx_dtype__MLX_FLOAT32,
            );
            Ok(Self::from_raw(out))
        }
    }

    /// Create a 1D array from a slice of i32 values.
    pub fn from_slice_i32(data: &[i32]) -> Result<Self> {
        unsafe {
            let shape = [data.len() as i32];
            let out = mlx_array_new_data(
                data.as_ptr() as *const std::ffi::c_void,
                shape.as_ptr(),
                1,
                mlx_dtype__MLX_INT32,
            );
            Ok(Self::from_raw(out))
        }
    }

    /// Create a 1D range array [start, stop) with given step and dtype.
    pub fn arange(start: f64, stop: f64, step: f64, dtype: DType) -> Result<Self> {
        unsafe {
            let s = Self::default_stream();
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_arange(&mut out, start, stop, step, dtype.to_mlx(), s),
                "mlx_arange",
            )?;
            Ok(Self::from_raw(out))
        }
    }

    /// Create a 1D range array [0, stop) with step 1 and given dtype.
    pub fn arange_int(stop: i32, dtype: DType) -> Result<Self> {
        Self::arange(0.0, stop as f64, 1.0, dtype)
    }

    /// Create an array from raw data with a given shape and dtype.
    pub fn from_data(data: &[u8], shape: &[i32], dtype: DType) -> Result<Self> {
        unsafe {
            let out = mlx_array_new_data(
                data.as_ptr() as *const std::ffi::c_void,
                shape.as_ptr(),
                shape.len() as i32,
                dtype.to_mlx(),
            );
            Ok(Self::from_raw(out))
        }
    }

    // ------------------------------------------------------------------
    // Properties
    // ------------------------------------------------------------------

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        unsafe { mlx_array_ndim(self.inner) }
    }

    /// Total number of bytes.
    pub fn nbytes(&self) -> usize {
        unsafe { mlx_array_nbytes(self.inner) }
    }

    /// The data type.
    pub fn dtype(&self) -> DType {
        let raw = unsafe { mlx_array_dtype(self.inner) };
        DType::from_mlx(raw).unwrap_or(DType::Float32)
    }

    /// The shape as a Vec of i32.
    pub fn shape_raw(&self) -> Vec<i32> {
        unsafe {
            let ndim = self.ndim();
            let mut dims = vec![0i32; ndim];
            for (i, dim) in dims.iter_mut().enumerate() {
                *dim = mlx_array_dim(self.inner, i as i32);
            }
            dims
        }
    }

    /// The shape as a `Shape` type.
    pub fn shape(&self) -> Shape {
        Shape::from_i32(&self.shape_raw())
    }

    /// Total number of elements.
    pub fn elem_count(&self) -> usize {
        self.shape().elem_count()
    }

    /// Size along a specific dimension.
    pub fn dim(&self, axis: i32) -> i32 {
        unsafe { mlx_array_dim(self.inner, axis) }
    }

    // ------------------------------------------------------------------
    // Evaluation
    // ------------------------------------------------------------------

    /// Force evaluation of this array.
    pub fn eval(&self) -> Result<()> {
        unsafe { Self::check(mlx_array_eval(self.inner), "mlx_array_eval") }
    }

    /// Force evaluation of multiple arrays.
    pub fn eval_many(outputs: &[&Array]) -> Result<()> {
        unsafe {
            let vec = mlx_vector_array_new();
            for arr in outputs {
                Self::check(
                    mlx_vector_array_append_value(vec, arr.inner),
                    "mlx_vector_array_append_value",
                )?;
            }
            let rc = mlx_eval(vec);
            mlx_vector_array_free(vec);
            Self::check(rc, "mlx_eval")
        }
    }

    /// Asynchronous evaluation of multiple arrays.
    pub fn async_eval_many(outputs: &[&Array]) -> Result<()> {
        unsafe {
            let vec = mlx_vector_array_new();
            for arr in outputs {
                Self::check(
                    mlx_vector_array_append_value(vec, arr.inner),
                    "mlx_vector_array_append_value",
                )?;
            }
            let rc = mlx_async_eval(vec);
            mlx_vector_array_free(vec);
            Self::check(rc, "mlx_async_eval")
        }
    }

    // ------------------------------------------------------------------
    // Data extraction
    // ------------------------------------------------------------------

    /// Extract data as a Vec of f32 (evaluates if needed).
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        let arr = if self.dtype() != DType::Float32 {
            self.as_type(DType::Float32)?
        } else {
            self.clone()
        };
        let arr = arr.contiguous()?;
        arr.eval()?;
        let n = arr.elem_count();
        let mut out = vec![0f32; n];
        unsafe {
            let ptr = mlx_array_data_float32(arr.inner);
            if ptr.is_null() {
                return Err(Error::NullPointer);
            }
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), n);
        }
        Ok(out)
    }

    /// Extract data as a Vec of i32 (evaluates if needed).
    pub fn to_vec_i32(&self) -> Result<Vec<i32>> {
        self.eval()?;
        let n = self.elem_count();
        let mut out = vec![0i32; n];
        unsafe {
            let ptr = mlx_array_data_int32(self.inner);
            if ptr.is_null() {
                return Err(Error::NullPointer);
            }
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), n);
        }
        Ok(out)
    }

    /// Extract data as a Vec of u32 (evaluates if needed).
    pub fn to_vec_u32(&self) -> Result<Vec<u32>> {
        self.eval()?;
        let n = self.elem_count();
        let mut out = vec![0u32; n];
        unsafe {
            let ptr = mlx_array_data_uint32(self.inner);
            if ptr.is_null() {
                return Err(Error::NullPointer);
            }
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), n);
        }
        Ok(out)
    }

    /// Extract a scalar f32 value.
    pub fn item_f32(&self) -> Result<f32> {
        self.eval()?;
        unsafe {
            let ptr = mlx_array_data_float32(self.inner);
            if ptr.is_null() {
                return Err(Error::NullPointer);
            }
            Ok(*ptr)
        }
    }

    /// Extract a scalar u32 value.
    pub fn item_u32(&self) -> Result<u32> {
        self.eval()?;
        unsafe {
            let ptr = mlx_array_data_uint32(self.inner);
            if ptr.is_null() {
                return Err(Error::NullPointer);
            }
            Ok(*ptr)
        }
    }

    /// Extract a scalar i32 value.
    pub fn item_i32(&self) -> Result<i32> {
        self.eval()?;
        unsafe {
            let ptr = mlx_array_data_int32(self.inner);
            if ptr.is_null() {
                return Err(Error::NullPointer);
            }
            Ok(*ptr)
        }
    }
}

impl Clone for Array {
    fn clone(&self) -> Self {
        unsafe {
            let mut ptr: mlx_array = std::mem::zeroed();
            mlx_array_set(&mut ptr, self.inner);
            Self { inner: ptr }
        }
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        unsafe {
            mlx_array_free(self.inner);
        }
    }
}
