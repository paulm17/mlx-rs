use crate::{Array, DType, Error, Result};
use mlx_sys::*;
use std::ffi::CString;

impl Array {
    // ------------------------------------------------------------------
    // Arithmetic ops
    // ------------------------------------------------------------------

    pub fn add(&self, other: &Array) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(mlx_add(&mut out, self.inner, other.inner, s), "mlx_add")?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn subtract(&self, other: &Array) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_subtract(&mut out, self.inner, other.inner, s),
                "mlx_subtract",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn multiply(&self, other: &Array) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_multiply(&mut out, self.inner, other.inner, s),
                "mlx_multiply",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn divide(&self, other: &Array) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            mlx_divide(&mut out, self.inner, other.inner, s);
            Ok(Array::from_raw(out))
        }
    }

    pub fn negative(&self) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(mlx_negative(&mut out, self.inner, s), "mlx_negative")?;
            Ok(Array::from_raw(out))
        }
    }

    // ------------------------------------------------------------------
    // Math ops
    // ------------------------------------------------------------------

    pub fn exp(&self) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(mlx_exp(&mut out, self.inner, s), "mlx_exp")?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn log(&self) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(mlx_log(&mut out, self.inner, s), "mlx_log")?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn sqrt(&self) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(mlx_sqrt(&mut out, self.inner, s), "mlx_sqrt")?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn rsqrt(&self) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(mlx_rsqrt(&mut out, self.inner, s), "mlx_rsqrt")?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn abs(&self) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(mlx_abs(&mut out, self.inner, s), "mlx_abs")?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn square(&self) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(mlx_square(&mut out, self.inner, s), "mlx_square")?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn sin(&self) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(mlx_sin(&mut out, self.inner, s), "mlx_sin")?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn cos(&self) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(mlx_cos(&mut out, self.inner, s), "mlx_cos")?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn tanh(&self) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(mlx_tanh(&mut out, self.inner, s), "mlx_tanh")?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn floor(&self) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(mlx_floor(&mut out, self.inner, s), "mlx_floor")?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn ceil(&self) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(mlx_ceil(&mut out, self.inner, s), "mlx_ceil")?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn round(&self, decimals: i32) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(mlx_round(&mut out, self.inner, decimals, s), "mlx_round")?;
            Ok(Array::from_raw(out))
        }
    }

    /// Element-wise maximum.
    pub fn maximum(&self, other: &Array) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(mlx_maximum(&mut out, self.inner, other.inner, s), "mlx_maximum")?;
            Ok(Array::from_raw(out))
        }
    }

    /// Element-wise minimum.
    pub fn minimum(&self, other: &Array) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(mlx_minimum(&mut out, self.inner, other.inner, s), "mlx_minimum")?;
            Ok(Array::from_raw(out))
        }
    }

    /// Clip values to [min, max].
    pub fn clip(&self, min: &Array, max: &Array) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(mlx_clip(&mut out, self.inner, min.inner, max.inner, s), "mlx_clip")?;
            Ok(Array::from_raw(out))
        }
    }

    /// Logical not.
    pub fn logical_not(&self) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(mlx_logical_not(&mut out, self.inner, s), "mlx_logical_not")?;
            Ok(Array::from_raw(out))
        }
    }

    // ------------------------------------------------------------------
    // Activation functions
    // ------------------------------------------------------------------

    pub fn sigmoid(&self) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(mlx_sigmoid(&mut out, self.inner, s), "mlx_sigmoid")?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn softmax(&self, axis: i32) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            mlx_softmax_axis(&mut out, self.inner, axis, false, s);
            Ok(Array::from_raw(out))
        }
    }

    // ------------------------------------------------------------------
    // Reduction ops
    // ------------------------------------------------------------------

    pub fn sum_axis(&self, axis: i32, keepdims: bool) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_sum_axis(&mut out, self.inner, axis, keepdims, s),
                "mlx_sum_axis",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn mean_axis(&self, axis: i32, keepdims: bool) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_mean_axis(&mut out, self.inner, axis, keepdims, s),
                "mlx_mean_axis",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn max(&self, axis: Option<i32>, keepdims: bool) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            let res = match axis {
                Some(ax) => mlx_max_axis(&mut out, self.inner, ax, keepdims, s),
                None => mlx_max(&mut out, self.inner, keepdims, s),
            };
            if res != 0 {
                return Err(Error::Mlx("mlx_max failed".to_string()));
            }
            Ok(Array::from_raw(out))
        }
    }

    pub fn min(&self, axis: Option<i32>, keepdims: bool) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            let res = match axis {
                Some(ax) => mlx_min_axis(&mut out, self.inner, ax, keepdims, s),
                None => mlx_min(&mut out, self.inner, keepdims, s),
            };
            if res != 0 {
                return Err(Error::Mlx("mlx_min failed".to_string()));
            }
            Ok(Array::from_raw(out))
        }
    }

    pub fn argmax(&self, axis: i32) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            mlx_argmax_axis(&mut out, self.inner, axis, false, s);
            Ok(Array::from_raw(out))
        }
    }

    pub fn logsumexp(&self, axis: i32, keepdims: bool) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_logsumexp_axis(&mut out, self.inner, axis, keepdims, s),
                "mlx_logsumexp_axis",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    // ------------------------------------------------------------------
    // Shape manipulation
    // ------------------------------------------------------------------

    pub fn reshape(&self, shape: &[i32]) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            mlx_reshape(&mut out, self.inner, shape.as_ptr(), shape.len(), s);
            Ok(Array::from_raw(out))
        }
    }

    pub fn transpose(&self) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(mlx_transpose(&mut out, self.inner, s), "mlx_transpose")?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn transpose_axes(&self, axes: &[i32]) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_transpose_axes(&mut out, self.inner, axes.as_ptr(), axes.len(), s),
                "mlx_transpose_axes",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn squeeze(&self, axis: i32) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_squeeze_axis(&mut out, self.inner, axis, s),
                "mlx_squeeze_axis",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn expand_dims(&self, axis: i32) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_expand_dims(&mut out, self.inner, axis, s),
                "mlx_expand_dims",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn broadcast_to(&self, shape: &[i32]) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_broadcast_to(&mut out, self.inner, shape.as_ptr(), shape.len(), s),
                "mlx_broadcast_to",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn flatten(&self, start_axis: i32, end_axis: i32) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_flatten(&mut out, self.inner, start_axis, end_axis, s),
                "mlx_flatten",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn contiguous(&self) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_contiguous(&mut out, self.inner, false, s),
                "mlx_contiguous",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    // ------------------------------------------------------------------
    // Indexing / slicing
    // ------------------------------------------------------------------

    pub fn take(&self, indices: &Array, axis: i32) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            mlx_take_axis(&mut out, self.inner, indices.inner, axis, s);
            Ok(Array::from_raw(out))
        }
    }

    pub fn take_along_axis(&self, indices: &Array, axis: i32) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_take_along_axis(&mut out, self.inner, indices.inner, axis, s),
                "mlx_take_along_axis",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn slice(&self, start: &[i32], stop: &[i32]) -> Result<Array> {
        let s = Self::default_stream();
        // Default strides of 1
        let strides: Vec<i32> = vec![1; start.len()];
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_slice(
                    &mut out,
                    self.inner,
                    start.as_ptr(),
                    start.len(),
                    stop.as_ptr(),
                    stop.len(),
                    strides.as_ptr(),
                    strides.len(),
                    s,
                ),
                "mlx_slice",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn slice_update(
        &self,
        update: &Array,
        start: &[i32],
        stop: &[i32],
        strides: &[i32],
    ) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_slice_update(
                    &mut out,
                    self.inner,
                    update.inner,
                    start.as_ptr(),
                    start.len(),
                    stop.as_ptr(),
                    stop.len(),
                    strides.as_ptr(),
                    strides.len(),
                    s,
                ),
                "mlx_slice_update",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn slice_update_dynamic(
        &self,
        update: &Array,
        start: &Array,
        axes: &[i32],
    ) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_slice_update_dynamic(
                    &mut out,
                    self.inner,
                    update.inner,
                    start.inner,
                    axes.as_ptr(),
                    axes.len(),
                    s,
                ),
                "mlx_slice_update_dynamic",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    // ------------------------------------------------------------------
    // Matmul
    // ------------------------------------------------------------------

    pub fn matmul(&self, other: &Array) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_matmul(&mut out, self.inner, other.inner, s),
                "mlx_matmul",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn conv1d(
        &self,
        weight: &Array,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
    ) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_conv1d(
                    &mut out,
                    self.inner,
                    weight.inner,
                    stride,
                    padding,
                    dilation,
                    groups,
                    s,
                ),
                "mlx_conv1d",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    // ------------------------------------------------------------------
    // Type casting
    // ------------------------------------------------------------------

    pub fn as_type(&self, dtype: DType) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_astype(&mut out, self.inner, dtype.to_mlx(), s),
                "mlx_astype",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    // ------------------------------------------------------------------
    // Concatenation (uses mlx_vector_array_new_data like deprecated code)
    // ------------------------------------------------------------------

    pub fn concatenate(arrays: &[&Array], axis: i32) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let inners: Vec<mlx_array> = arrays.iter().map(|a| a.inner).collect();
            let vec_arr = mlx_vector_array_new_data(inners.as_ptr(), inners.len());
            let mut out: mlx_array = std::mem::zeroed();
            let rc = mlx_concatenate_axis(&mut out, vec_arr, axis, s);
            mlx_vector_array_free(vec_arr);
            Self::check(rc, "mlx_concatenate_axis")?;
            Ok(Array::from_raw(out))
        }
    }

    // ------------------------------------------------------------------
    // Quantization (uses c"affine" mode and null-ptr for optional arrays)
    // ------------------------------------------------------------------

    pub fn quantized_matmul(
        &self,
        w: &Array,
        scales: &Array,
        biases: Option<&Array>,
        transpose_w: bool,
        group_size: i32,
        bits: i32,
    ) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            let opt_gs = mlx_optional_int_ {
                value: group_size,
                has_value: true,
            };
            let opt_bits = mlx_optional_int_ {
                value: bits,
                has_value: true,
            };

            // Use null pointer for optional biases (matching deprecated code pattern)
            let b_inner = biases.map_or(
                mlx_array_ {
                    ctx: std::ptr::null_mut(),
                },
                |m| m.inner,
            );

            let mode = CString::new("affine").unwrap();
            let rc = mlx_quantized_matmul(
                &mut out,
                self.inner,
                w.inner,
                scales.inner,
                b_inner,
                transpose_w,
                opt_gs,
                opt_bits,
                mode.as_ptr(),
                s,
            );
            Self::check(rc, "mlx_quantized_matmul")?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn dequantize(
        &self,
        scales: &Array,
        biases: Option<&Array>,
        group_size: i32,
        bits: i32,
    ) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            let opt_gs = mlx_optional_int_ {
                value: group_size,
                has_value: true,
            };
            let opt_bits = mlx_optional_int_ {
                value: bits,
                has_value: true,
            };

            let b_inner = biases.map_or(
                mlx_array_ {
                    ctx: std::ptr::null_mut(),
                },
                |m| m.inner,
            );

            let mode = CString::new("affine").unwrap();
            let dtype = mlx_optional_dtype_ {
                value: 0,
                has_value: false,
            }; // Let MLX infer

            mlx_dequantize(
                &mut out,
                self.inner,
                scales.inner,
                b_inner,
                opt_gs,
                opt_bits,
                mode.as_ptr(),
                dtype,
                s,
            );
            Ok(Array::from_raw(out))
        }
    }

    // ------------------------------------------------------------------
    // Fast ops (MLX-specific hardware-accelerated)
    // ------------------------------------------------------------------

    pub fn fast_rms_norm(&self, weight: &Array, eps: f32) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_fast_rms_norm(&mut out, self.inner, weight.inner, eps, s),
                "mlx_fast_rms_norm",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn fast_rope(
        &self,
        dims: i32,
        traditional: bool,
        base: Option<f32>,
        scale: f32,
        offset: i32,
        freqs: Option<&Array>,
    ) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            let base_opt = mlx_optional_float_ {
                value: base.unwrap_or(0.0),
                has_value: base.is_some(),
            };
            // Use null pointer for optional freqs (matching deprecated code)
            let freqs_val = freqs.map_or(
                mlx_array_ {
                    ctx: std::ptr::null_mut(),
                },
                |m| m.inner,
            );

            mlx_fast_rope(
                &mut out,
                self.inner,
                dims,
                traditional,
                base_opt,
                scale,
                offset,
                freqs_val,
                s,
            );
            Ok(Array::from_raw(out))
        }
    }

    /// Scaled dot-product attention using MLX's fast kernel.
    ///
    /// `mask_mode`: `"causal"` for causal masking, `""` (empty string) for no masking,
    /// or `"array"` to use the `mask` parameter.
    pub fn fast_scaled_dot_product_attention(
        &self,
        keys: &Array,
        values: &Array,
        scale: f32,
        mask_mode: &str,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            let mask_val = mask.map_or(
                mlx_array_ {
                    ctx: std::ptr::null_mut(),
                },
                |m| m.inner,
            );
            let sinks_val = mlx_array_ {
                ctx: std::ptr::null_mut(),
            };
            let c_mask_mode = CString::new(mask_mode)
                .map_err(|_| Error::Message("mask_mode contains null byte".into()))?;

            let rc = mlx_fast_scaled_dot_product_attention(
                &mut out,
                self.inner, // queries
                keys.inner,
                values.inner,
                scale,
                c_mask_mode.as_ptr(), // mask_mode string
                mask_val,             // mask_arr
                sinks_val,            // sinks
                s,                    // stream
            );
            Self::check(rc, "mlx_fast_sdpa")?;
            Ok(Array::from_raw(out))
        }
    }

    // ------------------------------------------------------------------
    // Gather matmul
    // ------------------------------------------------------------------

    pub fn gather_mm(
        &self,
        b: &Array,
        lhs_indices: Option<&Array>,
        rhs_indices: Option<&Array>,
        sorted_indices: bool,
    ) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            let lhs_inner = lhs_indices.map_or(
                mlx_array_ {
                    ctx: std::ptr::null_mut(),
                },
                |v| v.inner,
            );
            let rhs_inner = rhs_indices.map_or(
                mlx_array_ {
                    ctx: std::ptr::null_mut(),
                },
                |v| v.inner,
            );
            let rc = mlx_gather_mm(
                &mut out,
                self.inner,
                b.inner,
                lhs_inner,
                rhs_inner,
                sorted_indices,
                s,
            );
            Self::check(rc, "mlx_gather_mm")?;
            Ok(Array::from_raw(out))
        }
    }

    // ------------------------------------------------------------------
    // Gather quantized matmul
    // ------------------------------------------------------------------

    pub fn gather_qmm(
        &self,
        w: &Array,
        scales: &Array,
        biases: Option<&Array>,
        lhs_indices: Option<&Array>,
        rhs_indices: Option<&Array>,
        transpose: bool,
        group_size: i32,
        bits: i32,
        sorted_indices: bool,
    ) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            let b_inner = biases.map_or(
                mlx_array_ {
                    ctx: std::ptr::null_mut(),
                },
                |v| v.inner,
            );
            let lhs_inner = lhs_indices.map_or(
                mlx_array_ {
                    ctx: std::ptr::null_mut(),
                },
                |v| v.inner,
            );
            let rhs_inner = rhs_indices.map_or(
                mlx_array_ {
                    ctx: std::ptr::null_mut(),
                },
                |v| v.inner,
            );
            let opt_gs = mlx_optional_int_ {
                value: group_size,
                has_value: true,
            };
            let opt_bits = mlx_optional_int_ {
                value: bits,
                has_value: true,
            };
            let mode = CString::new("affine").unwrap();
            let rc = mlx_gather_qmm(
                &mut out,
                self.inner,
                w.inner,
                scales.inner,
                b_inner,
                lhs_inner,
                rhs_inner,
                transpose,
                opt_gs,
                opt_bits,
                mode.as_ptr(),
                sorted_indices,
                s,
            );
            Self::check(rc, "mlx_gather_qmm")?;
            Ok(Array::from_raw(out))
        }
    }

    // ------------------------------------------------------------------
    // Where / conditional
    // ------------------------------------------------------------------

    pub fn where_cond(&self, x: &Array, y: &Array) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_where(&mut out, self.inner, x.inner, y.inner, s),
                "mlx_where",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    // ------------------------------------------------------------------
    // Comparison ops
    // ------------------------------------------------------------------

    pub fn greater(&self, other: &Array) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_greater(&mut out, self.inner, other.inner, s),
                "mlx_greater",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn less(&self, other: &Array) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(mlx_less(&mut out, self.inner, other.inner, s), "mlx_less")?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn equal(&self, other: &Array) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(mlx_equal(&mut out, self.inner, other.inner, s), "mlx_equal")?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn not_equal(&self, other: &Array) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(mlx_not_equal(&mut out, self.inner, other.inner, s), "mlx_not_equal")?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn greater_equal(&self, other: &Array) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_greater_equal(&mut out, self.inner, other.inner, s),
                "mlx_greater_equal",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn less_equal(&self, other: &Array) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_less_equal(&mut out, self.inner, other.inner, s),
                "mlx_less_equal",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    // ------------------------------------------------------------------
    // Cumulative ops
    // ------------------------------------------------------------------

    pub fn cumsum(&self, axis: i32, reverse: bool, inclusive: bool) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_cumsum(&mut out, self.inner, axis, reverse, inclusive, s),
                "mlx_cumsum",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    // ------------------------------------------------------------------
    // Argpartition
    // ------------------------------------------------------------------

    pub fn argpartition(&self, kth: i32, axis: i32) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_argpartition_axis(&mut out, self.inner, kth, axis, s),
                "mlx_argpartition_axis",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    // ------------------------------------------------------------------
    // Power
    // ------------------------------------------------------------------

    pub fn power(&self, other: &Array) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(mlx_power(&mut out, self.inner, other.inner, s), "mlx_power")?;
            Ok(Array::from_raw(out))
        }
    }

    pub fn remainder(&self, other: &Array) -> Result<Array> {
        let s = Self::default_stream();
        unsafe {
            let mut out: mlx_array = std::mem::zeroed();
            Self::check(
                mlx_remainder(&mut out, self.inner, other.inner, s),
                "mlx_remainder",
            )?;
            Ok(Array::from_raw(out))
        }
    }

    // ------------------------------------------------------------------
    // To string
    // ------------------------------------------------------------------

    pub fn to_mlx_string(&self) -> String {
        unsafe {
            let mut s: mlx_string = std::mem::zeroed();
            mlx_array_tostring(&mut s, self.inner);
            let ptr = mlx_string_data(s);
            let result = if ptr.is_null() {
                String::from("<Array>")
            } else {
                std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
            };
            mlx_string_free(s);
            result
        }
    }
}

// ------------------------------------------------------------------
// Operator overloads
// ------------------------------------------------------------------

impl std::ops::Add for &Array {
    type Output = Array;
    fn add(self, rhs: Self) -> Array {
        self.add(rhs).expect("Array::add failed")
    }
}

impl std::ops::Sub for &Array {
    type Output = Array;
    fn sub(self, rhs: Self) -> Array {
        self.subtract(rhs).expect("Array::subtract failed")
    }
}

impl std::ops::Mul for &Array {
    type Output = Array;
    fn mul(self, rhs: Self) -> Array {
        self.multiply(rhs).expect("Array::multiply failed")
    }
}

impl std::ops::Div for &Array {
    type Output = Array;
    fn div(self, rhs: Self) -> Array {
        self.divide(rhs).expect("Array::divide failed")
    }
}

impl std::ops::Neg for &Array {
    type Output = Array;
    fn neg(self) -> Array {
        self.negative().expect("Array::negative failed")
    }
}

impl std::fmt::Display for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_mlx_string())
    }
}

// ------------------------------------------------------------------
// Free functions
// ------------------------------------------------------------------

pub fn eval(outputs: &[&Array]) -> Result<()> {
    Array::eval_many(outputs)
}

pub fn async_eval(outputs: &[&Array]) -> Result<()> {
    Array::async_eval_many(outputs)
}
