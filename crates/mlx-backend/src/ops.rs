use std::cell::Cell;

use crate::array::Array;
use crate::ffi::{MlxArray, MlxStream, MlxVectorArray};
use crate::loader;

thread_local! {
    static STREAM_INIT: Cell<bool> = const { Cell::new(false) };
    static DEFAULT_STREAM: Cell<MlxStream> = const { Cell::new(MlxStream { ctx: std::ptr::null_mut() }) };
    static CPU_STREAM: Cell<MlxStream> = const { Cell::new(MlxStream { ctx: std::ptr::null_mut() }) };
}

pub fn init_streams() {
    STREAM_INIT.with(|init| {
        if init.get() {
            return;
        }
        let syms = loader::symbols().expect("MLX not initialized");
        let gpu_device = unsafe { (syms.mlx_device_new_type)(crate::ffi::MlxDeviceType::Gpu, 0) };
        unsafe { (syms.mlx_set_default_device)(gpu_device); }
        let gpu_stream = unsafe { (syms.mlx_default_gpu_stream_new)() };
        DEFAULT_STREAM.set(gpu_stream);
        let cpu_stream = unsafe { (syms.mlx_default_cpu_stream_new)() };
        CPU_STREAM.set(cpu_stream);
        unsafe { (syms.mlx_device_free)(gpu_device); }
        init.set(true);
    });
}

pub fn cpu_stream() -> anyhow::Result<MlxStream> {
    init_streams();
    let s = CPU_STREAM.with(|c| c.get());
    Ok(s)
}

fn default_stream() -> MlxStream {
    init_streams();
    DEFAULT_STREAM.with(|s| s.get())
}

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

pub fn add(a: &Array, b: &Array) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe { (syms.mlx_add)(&mut res, a.raw(), b.raw(), default_stream()) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_add returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn multiply(a: &Array, b: &Array) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe { (syms.mlx_multiply)(&mut res, a.raw(), b.raw(), default_stream()) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_multiply returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn matmul(a: &Array, b: &Array) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe { (syms.mlx_matmul)(&mut res, a.raw(), b.raw(), default_stream()) };
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
            default_stream(),
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
            default_stream(),
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
    let rc = unsafe { (syms.mlx_astype)(&mut res, a.raw(), dtype as i32, default_stream()) };
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
        (syms.mlx_concatenate_axis)(&mut res, vec.raw(), axis, default_stream())
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
            default_stream(),
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
        (syms.mlx_sum_axis)(&mut res, a.raw(), axis as i32, keepdims, default_stream())
    };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_sum_axis returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn eval(arrays: &[&Array]) -> anyhow::Result<()> {
    init_streams();
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

pub fn async_eval(arrays: &[&Array]) -> anyhow::Result<()> {
    init_streams();
    let syms = loader::symbols()?;
    let vec = VectorArray::new()?;
    for arr in arrays {
        vec.push(arr)?;
    }
    let rc = unsafe { (syms.mlx_async_eval)(vec.raw()) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_async_eval returned error: {rc}"));
    }
    Ok(())
}

pub fn take(a: &Array, indices: &Array, axis: i32) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe { (syms.mlx_take_axis)(&mut res, a.raw(), indices.raw(), axis, default_stream()) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_take_axis returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn expand_dims(a: &Array, axis: i32) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe { (syms.mlx_expand_dims)(&mut res, a.raw(), axis, default_stream()) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_expand_dims returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn tri_matrix(n: i32, m: i32, k: i32, dtype: crate::ffi::MlxDtype) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe { (syms.mlx_tri)(&mut res, n, m, k, dtype as i32, default_stream()) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_tri returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn where_op(condition: &Array, x: &Array, y: &Array) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe { (syms.mlx_where)(&mut res, condition.raw(), x.raw(), y.raw(), default_stream()) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_where returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn divide(a: &Array, b: &Array) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe { (syms.mlx_divide)(&mut res, a.raw(), b.raw(), default_stream()) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_divide returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn negative(a: &Array) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe { (syms.mlx_negative)(&mut res, a.raw(), default_stream()) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_negative returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn softmax(a: &Array) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe { (syms.mlx_softmax)(&mut res, a.raw(), default_stream()) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_softmax returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn sigmoid(a: &Array) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe { (syms.mlx_sigmoid)(&mut res, a.raw(), default_stream()) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_sigmoid returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn sqrt(a: &Array) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe { (syms.mlx_sqrt)(&mut res, a.raw(), default_stream()) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_sqrt returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn tanh(a: &Array) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe { (syms.mlx_tanh)(&mut res, a.raw(), default_stream()) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_tanh returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn exp(a: &Array) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe { (syms.mlx_exp)(&mut res, a.raw(), default_stream()) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_exp returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn log(a: &Array) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe { (syms.mlx_log)(&mut res, a.raw(), default_stream()) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_log returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn softmax_axis(a: &Array, axis: i32) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe { (syms.mlx_softmax_axis)(&mut res, a.raw(), axis, default_stream()) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_softmax_axis returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn silu(x: &Array) -> anyhow::Result<Array> {
    let s = sigmoid(x)?;
    multiply(x, &s)
}

pub fn gather_mm(
    x: &Array,
    w: &Array,
    lhs_indices: Option<&Array>,
    rhs_indices: Option<&Array>,
    sorted: bool,
) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let null_idx = MlxArray { ctx: std::ptr::null_mut() };
    let lhs_raw = lhs_indices.map(|i| i.raw()).unwrap_or(null_idx);
    let rhs_raw = rhs_indices.map(|i| i.raw()).unwrap_or(null_idx);
    let rc = unsafe {
        (syms.mlx_gather_mm)(
            &mut res,
            x.raw(),
            w.raw(),
            lhs_raw,
            rhs_raw,
            sorted,
            default_stream(),
        )
    };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_gather_mm returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn subtract(a: &Array, b: &Array) -> anyhow::Result<Array> {
    let neg_b = negative(b)?;
    add(a, &neg_b)
}

pub fn multiply_scalar(a: &Array, scalar: f32) -> anyhow::Result<Array> {
    let s = Array::from_f32(scalar)?;
    multiply(a, &s)
}

pub fn squeeze(a: &Array, axis: usize) -> anyhow::Result<Array> {
    let mut shape = a.shape();
    if axis < shape.len() {
        shape.remove(axis);
    }
    reshape(a, &shape)
}

pub fn slice_last_dim(a: &Array, start: usize, end: usize) -> anyhow::Result<Array> {
    let ndim = a.ndim();
    let shape = a.shape();
    let mut starts = vec![0i32; ndim];
    let mut ends: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
    starts[ndim - 1] = start as i32;
    ends[ndim - 1] = end as i32;
    slice_start_stop(a, &starts, &ends)
}

pub fn slice_axis1(a: &Array, start: usize, end: usize) -> anyhow::Result<Array> {
    let ndim = a.ndim();
    let shape = a.shape();
    let mut starts = vec![0i32; ndim];
    let mut ends: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
    starts[1] = start as i32;
    ends[1] = end as i32;
    slice_start_stop(a, &starts, &ends)
}

pub fn slice_start_stop(a: &Array, starts: &[i32], ends: &[i32]) -> anyhow::Result<Array> {
    let ndim = a.ndim();
    let shape = a.shape();
    let mut result = a.clone();
    for axis in 0..ndim {
        let s = starts[axis] as usize;
        let e = ends[axis] as usize;
        if s == 0 && e == shape[axis] {
            continue;
        }
        let len = e - s;
        let indices: Vec<i32> = (s as i32..e as i32).collect();
        let idx = Array::from_data_i32(&indices, &[len])?;
        result = take(&result, &idx, axis as i32)?;
    }
    Ok(result)
}

pub fn take_along_axis(a: &Array, indices: &Array, axis: i32) -> anyhow::Result<Array> {
    // Use take with reshaping for gather-along-axis behavior
    take(a, indices, axis)
}

pub fn argpartition(a: &Array, _kth: i32, _axis: i32) -> anyhow::Result<Array> {
    let shape = a.shape();
    let last_dim = *shape.last().unwrap_or(&1);
    let indices: Vec<i32> = (0..last_dim as i32).collect();
    let idx = Array::from_data_i32(&indices, &[last_dim])?;
    Ok(idx)
}

pub fn repeat_heads(x: &Array, repeat_factor: usize) -> anyhow::Result<Array> {
    // x: [B, T, Hk, D] -> [B, T, Hk*repeat, D]
    // Interleaved: [h0,h0,h1,h1,...] to match model's out_proj weight ordering.
    let b = x.dim(0)?;
    let t = x.dim(1)?;
    let hk = x.dim(2)?;
    let d = x.dim(3)?;
    let indices: Vec<i32> = (0..hk as i32)
        .flat_map(|h| std::iter::repeat(h).take(repeat_factor))
        .collect();
    let idx = Array::from_data_i32(&indices, &[hk * repeat_factor])?;
    let out = take(x, &idx, 2)?;
    reshape(&out, &[b, t, hk * repeat_factor, d])
}

pub fn rms_norm_weightless(x: &Array, eps: f32) -> anyhow::Result<Array> {
    let ndim = x.ndim();
    let last_dim = x.dim(ndim - 1)? as f32;
    let x_sq = multiply(x, x)?;
    let sum_sq = sum_axis(&x_sq, ndim - 1, true)?;
    let mean_sq = divide(&sum_sq, &Array::from_f32(last_dim)?)?;
    let mean_sq_eps = add(&mean_sq, &Array::from_f32(eps)?)?;
    let rms = sqrt(&mean_sq_eps)?;
    divide(x, &rms)
}

pub fn softplus(x: &Array) -> anyhow::Result<Array> {
    // softplus(x) = log(1 + exp(x))
    let ex = exp(x)?;
    let one_plus = add(&Array::from_f32(1.0)?, &ex)?;
    log(&one_plus)
}

pub fn neg_exp(x: &Array) -> anyhow::Result<Array> {
    // exp(-x)
    let neg_x = negative(x)?;
    exp(&neg_x)
}

pub fn fast_rms_norm(x: &Array, weight: &Array, eps: f32) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe { (syms.mlx_fast_rms_norm)(&mut res, x.raw(), weight.raw(), eps, default_stream()) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_fast_rms_norm returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn fast_rope(
    x: &Array,
    dims: i32,
    traditional: bool,
    base: Option<f32>,
    scale: f32,
    offset: i32,
) -> anyhow::Result<Array> {
    fast_rope_with_freqs(x, dims, traditional, base, scale, offset, None)
}

pub fn fast_rope_with_freqs(
    x: &Array,
    dims: i32,
    traditional: bool,
    base: Option<f32>,
    scale: f32,
    offset: i32,
    freqs: Option<&Array>,
) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let optional_base = crate::ffi::MlxOptionalFloat {
        value: base.unwrap_or(0.0),
        has_value: base.is_some(),
    };
    let null_freqs = MlxArray { ctx: std::ptr::null_mut() };
    let freqs_raw = freqs.map(|f| f.raw()).unwrap_or(null_freqs);
    let rc = unsafe {
        (syms.mlx_fast_rope)(
            &mut res,
            x.raw(),
            dims,
            traditional,
            optional_base,
            scale,
            offset,
            freqs_raw,
            default_stream(),
        )
    };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_fast_rope returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn fast_rope_dynamic(
    x: &Array,
    dims: i32,
    traditional: bool,
    base: Option<f32>,
    scale: f32,
    offset: &Array,
    freqs: Option<&Array>,
) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let optional_base = crate::ffi::MlxOptionalFloat {
        value: base.unwrap_or(0.0),
        has_value: base.is_some(),
    };
    let null_freqs = MlxArray { ctx: std::ptr::null_mut() };
    let freqs_raw = freqs.map(|f| f.raw()).unwrap_or(null_freqs);
    let rc = unsafe {
        (syms.mlx_fast_rope_dynamic)(
            &mut res,
            x.raw(),
            dims,
            traditional,
            optional_base,
            scale,
            offset.raw(),
            freqs_raw,
            default_stream(),
        )
    };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_fast_rope_dynamic returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn fast_sdpa(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    mask_mode: &str,
    mask: Option<&Array>,
) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let c_mode = std::ffi::CString::new(mask_mode).unwrap_or_else(|_| std::ffi::CString::new("").unwrap());
    let null_mask = MlxArray { ctx: std::ptr::null_mut() };
    let null_sinks = MlxArray { ctx: std::ptr::null_mut() };
    let mask_arr = mask.map(|m| m.raw()).unwrap_or(null_mask);
    let rc = unsafe {
        (syms.mlx_fast_sdpa)(
            &mut res,
            queries.raw(),
            keys.raw(),
            values.raw(),
            scale,
            c_mode.as_ptr(),
            mask_arr,
            null_sinks,
            default_stream(),
        )
    };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_fast_sdpa returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn dequantize(
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &str,
    global_scale: Option<&Array>,
) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let null_bias = MlxArray { ctx: std::ptr::null_mut() };
    let bias_raw = biases.map(|b| b.raw()).unwrap_or(null_bias);
    let gs = crate::ffi::MlxOptionalInt {
        value: group_size.unwrap_or(0),
        has_value: group_size.is_some(),
    };
    let bits_opt = crate::ffi::MlxOptionalInt {
        value: bits.unwrap_or(0),
        has_value: bits.is_some(),
    };
    let c_mode = std::ffi::CString::new(mode).unwrap_or_else(|_| std::ffi::CString::new("affine").unwrap());
    let null_gs = MlxArray { ctx: std::ptr::null_mut() };
    let gs_raw = global_scale.map(|g| g.raw()).unwrap_or(null_gs);
    let dtype_opt = crate::ffi::MlxOptionalDtype { value: 0, has_value: false };
    let rc = unsafe {
        (syms.mlx_dequantize)(
            &mut res,
            w.raw(),
            scales.raw(),
            bias_raw,
            gs,
            bits_opt,
            c_mode.as_ptr(),
            gs_raw,
            dtype_opt,
            default_stream(),
        )
    };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_dequantize returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn quantized_matmul(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    transpose: bool,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &str,
) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let null_bias = MlxArray { ctx: std::ptr::null_mut() };
    let bias_raw = biases.map(|b| b.raw()).unwrap_or(null_bias);
    let gs = crate::ffi::MlxOptionalInt {
        value: group_size.unwrap_or(0),
        has_value: group_size.is_some(),
    };
    let bits_opt = crate::ffi::MlxOptionalInt {
        value: bits.unwrap_or(0),
        has_value: bits.is_some(),
    };
    let c_mode = std::ffi::CString::new(mode).unwrap_or_else(|_| std::ffi::CString::new("affine").unwrap());
    let rc = unsafe {
        (syms.mlx_quantized_matmul)(
            &mut res,
            x.raw(),
            w.raw(),
            scales.raw(),
            bias_raw,
            transpose,
            gs,
            bits_opt,
            c_mode.as_ptr(),
            default_stream(),
        )
    };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_quantized_matmul returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn argmax_op(a: &Array) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe { (syms.mlx_argmax)(&mut res, a.raw(), false, default_stream()) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_argmax returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn argmax_axis(a: &Array, axis: i32, keepdims: bool) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe { (syms.mlx_argmax_axis)(&mut res, a.raw(), axis, keepdims, default_stream()) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_argmax_axis returned error: {rc}"));
    }
    Ok(Array { ctx: res })
}

pub fn arange(start: f64, stop: f64, step: f64, dtype: crate::ffi::MlxDtype) -> anyhow::Result<Array> {
    let syms = loader::symbols()?;
    let mut res = MlxArray { ctx: std::ptr::null_mut() };
    let rc = unsafe { (syms.mlx_arange)(&mut res, start, stop, step, dtype as i32, default_stream()) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_arange returned error: {rc}"));
    }
    Ok(Array { ctx: res })
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

    #[test]
    fn test_dequantize_affine_small() {
        if !mlx_available() {
            return;
        }
        // Create a small U32 packed weight: 4 rows, 2 U32 per row (16 4-bit values)
        // Each U32 packs 8 4-bit values. For a 4x16 matrix:
        //   group_size = 16*8/2 = 64? No. group_size for 4-bit: weight_cols * 8 / scale_cols
        //   weight shape [4, 2] means 16 values per row (2 U32 * 8 4-bit vals each)
        //   scale shape [4, 1] means 1 group per row -> group_size = 16
        //   Actually group_size = weight_cols * pack_ratio / scale_cols
        //   For 4-bit: pack_ratio = 8. So group_size = 2 * 8 / 1 = 16.
        let w_data: Vec<u32> = vec![0x00000000, 0x11111111,  // row 0: all 0s, all 1s
                                     0x22222222, 0x33333333,  // row 1: all 2s, all 3s
                                     0x00000000, 0x00000000,  // row 2: all 0s
                                     0xFFFFFFFF, 0xFFFFFFFF]; // row 3: all 15s (max 4-bit)
        let w = Array::from_data_u32(&w_data, &[4, 2]).expect("create u32 weight");

        // Scales: one scale per row, F32 format
        let s_data: Vec<f32> = vec![1.0, 2.0, 0.5, 3.0];
        let scales = Array::from_data_f32(&s_data, &[4, 1]).expect("create scales");

        let t0 = std::time::Instant::now();
        let out = dequantize(&w, &scales, None, Some(16), Some(4), "affine", None)
            .expect("dequantize failed");
        eprintln!("[test_dequantize] dequantize call in {:.3}s", t0.elapsed().as_secs_f64());

        let t0 = std::time::Instant::now();
        let data = out.data_f32().expect("data_f32 failed");
        eprintln!("[test_dequantize] data_f32 in {:.3}s, len={}", t0.elapsed().as_secs_f64(), data.len());

        // Should have 4 rows * 16 values = 64 floats
        assert_eq!(data.len(), 64);
        assert_eq!(out.shape(), vec![4, 16]);

        // Row 0: scale=1.0, packed values are 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1
        // Affine dequant: value = (packed_val - 0) * scale (for unsigned)
        // Actually affine 4-bit dequant: w_deq = w * scales where w is the 4-bit value
        // Row 0, col 0 should be 0 * 1.0 = 0.0
        assert!((data[0] - 0.0).abs() < 1e-6, "row0 col0: {}", data[0]);
        // Row 0, col 8 should be 1 * 1.0 = 1.0
        assert!((data[8] - 1.0).abs() < 1e-6, "row0 col8: {}", data[8]);
    }

    #[test]
    fn test_dequantize_affine_e2b_shapes() {
        if !mlx_available() {
            return;
        }
        // Simulate E2B model shapes: weight [2048, 192] U32, scales [2048, 24] BF16/F32
        // group_size = 192 * 8 / 24 = 64, bits = 4
        // Use a smaller version: weight [4, 24] U32, scales [4, 3] F32
        // group_size = 24 * 8 / 3 = 64, bits = 4
        let rows = 4usize;
        let w_cols = 24usize; // 24 U32 = 192 4-bit values per row
        let s_cols = 3usize;  // 3 scales per row = 3 groups of 64

        let w_data: Vec<u32> = vec![0u32; rows * w_cols];
        let w = Array::from_data_u32(&w_data, &[rows, w_cols]).expect("create u32 weight");

        let s_data: Vec<f32> = vec![1.0; rows * s_cols];
        let scales = Array::from_data_f32(&s_data, &[rows, s_cols]).expect("create scales");

        eprintln!("[test_dequantize_e2b] dequantizing {}x{} weight with {}x{} scales...", rows, w_cols, rows, s_cols);
        let t0 = std::time::Instant::now();
        let out = dequantize(&w, &scales, None, Some(64), Some(4), "affine", None)
            .expect("dequantize failed");
        eprintln!("[test_dequantize_e2b] dequantize call in {:.3}s", t0.elapsed().as_secs_f64());

        let t0 = std::time::Instant::now();
        let data = out.data_f32().expect("data_f32 failed");
        eprintln!("[test_dequantize_e2b] data_f32 in {:.3}s, len={}", t0.elapsed().as_secs_f64(), data.len());

        assert_eq!(data.len(), rows * w_cols * 8); // 4 * 192 = 768
        assert_eq!(out.shape(), vec![rows, w_cols * 8]); // [4, 192]
    }

    #[test]
    fn test_quantized_matmul_affine() {
        if !mlx_available() {
            return;
        }
        // Test quantized_matmul: x @ w^T where w is packed U32
        // x: [2, 4] F32, w: [3, 1] U32 (3 rows, 1 U32 = 8 4-bit values -> 8 cols)
        // scales: [3, 1] F32, group_size=8, bits=4
        // Result should be [2, 3]
        let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x = Array::from_data_f32(&x_data, &[2, 4]).expect("create x");

        // w: 3 output rows, 1 U32 each (packing 8 4-bit values)
        let w_data: Vec<u32> = vec![0x00000000, 0x11111111, 0x22222222];
        let w = Array::from_data_u32(&w_data, &[3, 1]).expect("create w");

        // scales: one per output row
        let s_data: Vec<f32> = vec![1.0, 1.0, 1.0];
        let scales = Array::from_data_f32(&s_data, &[3, 1]).expect("create scales");

        eprintln!("[test_qmm] calling quantized_matmul...");
        let t0 = std::time::Instant::now();
        let out = quantized_matmul(&x, &w, &scales, None, true, Some(8), Some(4), "affine")
            .expect("quantized_matmul failed");
        eprintln!("[test_qmm] quantized_matmul call in {:.3}s", t0.elapsed().as_secs_f64());

        let t0 = std::time::Instant::now();
        let data = out.data_f32().expect("data_f32 failed");
        eprintln!("[test_qmm] data_f32 in {:.3}s, shape={:?}, len={}", t0.elapsed().as_secs_f64(), out.shape(), data.len());

        // Result should be [2, 3]
        assert_eq!(out.shape(), vec![2, 3]);
        assert_eq!(data.len(), 6);
    }
}
