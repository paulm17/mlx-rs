use crate::array::Array;
use crate::ffi::{MlxArray, MlxVectorArray};
use crate::loader;

pub(crate) struct VectorArray {
    ctx: MlxVectorArray,
}

impl VectorArray {
    pub fn new() -> anyhow::Result<Self> {
        let syms = loader::symbols()?;
        let ctx = unsafe { (syms.mlx_vector_array_new)() };
        if ctx.ctx.is_null() {
            return Err(anyhow::anyhow!("mlx_vector_array_new returned null"));
        }
        Ok(Self { ctx })
    }

    pub fn push(&self, arr: &Array) -> anyhow::Result<()> {
        let syms = loader::symbols()?;
        let rc = unsafe { (syms.mlx_vector_array_append_value)(self.ctx, arr.raw()) };
        if rc != 0 {
            return Err(anyhow::anyhow!(
                "mlx_vector_array_append_value returned error: {rc}"
            ));
        }
        Ok(())
    }

    pub fn raw(&self) -> MlxVectorArray {
        self.ctx
    }
}

impl Drop for VectorArray {
    fn drop(&mut self) {
        if !self.ctx.ctx.is_null() {
            let syms = loader::symbols().expect("MLX not initialized");
            unsafe {
                (syms.mlx_vector_array_free)(self.ctx);
            }
        }
    }
}

fn null_stream() -> *const crate::ffi::MlxStream {
    std::ptr::null()
}

pub fn add(a: &Array, b: &Array) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe { (syms.mlx_add)(&mut res, a.raw(), b.raw(), null_stream()) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_add returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn multiply(a: &Array, b: &Array) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe { (syms.mlx_multiply)(&mut res, a.raw(), b.raw(), null_stream()) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_multiply returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn matmul(a: &Array, b: &Array) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe { (syms.mlx_matmul)(&mut res, a.raw(), b.raw(), null_stream()) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_matmul returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn reshape(a: &Array, shape: &[usize]) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let c_shape: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe {
        (syms.mlx_reshape)(
            &mut res,
            a.raw(),
            c_shape.as_ptr(),
            shape.len(),
            null_stream(),
        )
    };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_reshape returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn transpose(a: &Array, axes: &[usize]) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let c_axes: Vec<i32> = axes.iter().map(|&s| s as i32).collect();
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe {
        (syms.mlx_transpose_axes)(
            &mut res,
            a.raw(),
            c_axes.as_ptr(),
            axes.len(),
            null_stream(),
        )
    };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_transpose_axes returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn astype(a: &Array, dtype: crate::ffi::MlxDtype) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe { (syms.mlx_astype)(&mut res, a.raw(), dtype as i32, null_stream()) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_astype returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn concatenate(arrays: &[&Array], axis: i32) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let vec = VectorArray::new()?;
    for arr in arrays {
        vec.push(arr)?;
    }
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe {
        (syms.mlx_concatenate_axis)(&mut res, vec.raw(), axis, null_stream())
    };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_concatenate_axis returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn zeros(shape: &[usize], dtype: crate::ffi::MlxDtype) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let c_shape: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe {
        (syms.mlx_zeros)(
            &mut res,
            c_shape.as_ptr(),
            shape.len(),
            dtype as i32,
            null_stream(),
        )
    };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_zeros returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn sum_axis(a: &Array, axis: usize, keepdims: bool) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe {
        (syms.mlx_sum_axis)(&mut res, a.raw(), axis as i32, keepdims, null_stream())
    };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_sum_axis returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn eval(arrays: &[&Array]) -> anyhow::Result<()> {
    let syms = loader::symbols()?;
    let vec = VectorArray::new()?;
    for arr in arrays {
        vec.push(arr)?;
    }
    let rc = unsafe { (syms.mlx_eval)(vec.raw()) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_eval returned error: {rc}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader;

    fn mlx_available() -> bool {
        loader::check_init().is_ok()
    }

    #[test]
    fn test_add_scalars() {
        if !mlx_available() {
            return;
        }
        let a = Array::from_f32(2.0).unwrap();
        let b = Array::from_f32(3.0).unwrap();
        let c = add(&a, &b).unwrap();
        c.eval().unwrap();
        let val = c.item_f64().unwrap();
        assert!((val - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_multiply_scalars() {
        if !mlx_available() {
            return;
        }
        let a = Array::from_f32(4.0).unwrap();
        let b = Array::from_f32(5.0).unwrap();
        let c = multiply(&a, &b).unwrap();
        c.eval().unwrap();
        let val = c.item_f64().unwrap();
        assert!((val - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_2d() {
        if !mlx_available() {
            return;
        }
        // [1,2] @ [[3],[4]] = [1*3 + 2*4] = [11]
        let a = Array::from_data_f32(&[1.0, 2.0], &[1, 2]).unwrap();
        let b = Array::from_data_f32(&[3.0, 4.0], &[2, 1]).unwrap();
        let c = matmul(&a, &b).unwrap();
        c.eval().unwrap();
        assert_eq!(c.shape(), vec![1, 1]);
        let val = c.item_f64().unwrap();
        assert!((val - 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_reshape() {
        if !mlx_available() {
            return;
        }
        let a = Array::from_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = reshape(&a, &[3, 2]).unwrap();
        b.eval().unwrap();
        assert_eq!(b.shape(), vec![3, 2]);
    }

    #[test]
    fn test_transpose() {
        if !mlx_available() {
            return;
        }
        let a = Array::from_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = transpose(&a, &[1, 0]).unwrap();
        b.eval().unwrap();
        assert_eq!(b.shape(), vec![3, 2]);
    }

    #[test]
    fn test_astype() {
        if !mlx_available() {
            return;
        }
        use crate::ffi::MlxDtype;
        let a = Array::from_f32(7.0).unwrap();
        let b = astype(&a, MlxDtype::Float16).unwrap();
        b.eval().unwrap();
        assert_eq!(b.dtype().unwrap(), MlxDtype::Float16);
    }

    #[test]
    fn test_concatenate() {
        if !mlx_available() {
            return;
        }
        let a = Array::from_data_f32(&[1.0, 2.0], &[1, 2]).unwrap();
        let b = Array::from_data_f32(&[3.0, 4.0], &[1, 2]).unwrap();
        let c = concatenate(&[&a, &b], 0).unwrap();
        c.eval().unwrap();
        assert_eq!(c.shape(), vec![2, 2]);
    }

    #[test]
    fn test_zeros() {
        if !mlx_available() {
            return;
        }
        use crate::ffi::MlxDtype;
        let a = zeros(&[2, 3], MlxDtype::Float32).unwrap();
        a.eval().unwrap();
        assert_eq!(a.shape(), vec![2, 3]);
        let data = a.data_f32().unwrap();
        assert!(data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_sum_axis() {
        if !mlx_available() {
            return;
        }
        let a = Array::from_data_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = sum_axis(&a, 1, false).unwrap();
        b.eval().unwrap();
        assert_eq!(b.shape(), vec![2]);
        let data = b.data_f32().unwrap();
        assert_eq!(data, &[3.0, 7.0]);
    }

    #[test]
    fn test_eval_multiple() {
        if !mlx_available() {
            return;
        }
        let a = Array::from_f32(1.0).unwrap();
        let b = Array::from_f32(2.0).unwrap();
        eval(&[&a, &b]).unwrap();
        let va = a.item_f64().unwrap();
        let vb = b.item_f64().unwrap();
        assert!((va - 1.0).abs() < 1e-6);
        assert!((vb - 2.0).abs() < 1e-6);
    }
}
