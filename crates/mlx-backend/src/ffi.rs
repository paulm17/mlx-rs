use std::ffi::c_int;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MlxString {
    pub ctx: *mut std::ffi::c_void,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MlxDevice {
    pub ctx: *mut std::ffi::c_void,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MlxArray {
    pub ctx: *mut std::ffi::c_void,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MlxStream {
    pub ctx: *mut std::ffi::c_void,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MlxVectorArray {
    pub ctx: *mut std::ffi::c_void,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlxDeviceType {
    Cpu = 0,
    Gpu = 1,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlxDtype {
    Bool = 0,
    Uint8 = 1,
    Uint16 = 2,
    Uint32 = 3,
    Uint64 = 4,
    Int8 = 5,
    Int16 = 6,
    Int32 = 7,
    Int64 = 8,
    Float16 = 9,
    Float32 = 10,
    Float64 = 11,
    Bfloat16 = 12,
    Complex64 = 13,
}

impl TryFrom<c_int> for MlxDeviceType {
    type Error = anyhow::Error;

    fn try_from(value: c_int) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(MlxDeviceType::Cpu),
            1 => Ok(MlxDeviceType::Gpu),
            other => Err(anyhow::anyhow!("unknown device type: {other}")),
        }
    }
}

impl TryFrom<c_int> for MlxDtype {
    type Error = anyhow::Error;

    fn try_from(value: c_int) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(MlxDtype::Bool),
            1 => Ok(MlxDtype::Uint8),
            2 => Ok(MlxDtype::Uint16),
            3 => Ok(MlxDtype::Uint32),
            4 => Ok(MlxDtype::Uint64),
            5 => Ok(MlxDtype::Int8),
            6 => Ok(MlxDtype::Int16),
            7 => Ok(MlxDtype::Int32),
            8 => Ok(MlxDtype::Int64),
            9 => Ok(MlxDtype::Float16),
            10 => Ok(MlxDtype::Float32),
            11 => Ok(MlxDtype::Float64),
            12 => Ok(MlxDtype::Bfloat16),
            13 => Ok(MlxDtype::Complex64),
            other => Err(anyhow::anyhow!("unknown dtype: {other}")),
        }
    }
}

impl MlxDtype {
    pub fn size_bytes(self) -> usize {
        match self {
            MlxDtype::Bool => 1,
            MlxDtype::Uint8 | MlxDtype::Int8 => 1,
            MlxDtype::Uint16 | MlxDtype::Int16 | MlxDtype::Float16 | MlxDtype::Bfloat16 => 2,
            MlxDtype::Uint32 | MlxDtype::Int32 | MlxDtype::Float32 => 4,
            MlxDtype::Uint64 | MlxDtype::Int64 | MlxDtype::Float64 => 8,
            MlxDtype::Complex64 => 8,
        }
    }
}

// String/device (existing)
pub type MlxVersionFn = unsafe extern "C" fn(*mut MlxString) -> c_int;
pub type MlxStringDataFn = unsafe extern "C" fn(MlxString) -> *const std::ffi::c_char;
pub type MlxStringFreeFn = unsafe extern "C" fn(MlxString) -> c_int;
pub type MlxDeviceIsAvailableFn = unsafe extern "C" fn(*mut bool, MlxDevice) -> c_int;
pub type MlxGetDefaultDeviceFn = unsafe extern "C" fn(*mut MlxDevice) -> c_int;
pub type MlxDeviceGetTypeFn = unsafe extern "C" fn(*mut c_int, MlxDevice) -> c_int;
pub type MlxDeviceCountFn = unsafe extern "C" fn(*mut c_int, c_int) -> c_int;
pub type MlxDeviceFreeFn = unsafe extern "C" fn(MlxDevice) -> c_int;

// Array creation
pub type MlxArrayNewFloat32Fn = unsafe extern "C" fn(f32) -> MlxArray;
pub type MlxArrayNewFloat64Fn = unsafe extern "C" fn(f64) -> MlxArray;
pub type MlxArrayNewIntFn = unsafe extern "C" fn(c_int) -> MlxArray;
pub type MlxArrayNewDataFn =
    unsafe extern "C" fn(*const std::ffi::c_void, *const c_int, c_int, c_int) -> MlxArray;
pub type MlxArrayFreeFn = unsafe extern "C" fn(MlxArray) -> c_int;
pub type MlxArraySetFn = unsafe extern "C" fn(*mut MlxArray, MlxArray) -> c_int;

// Array metadata
pub type MlxArraySizeFn = unsafe extern "C" fn(MlxArray) -> usize;
pub type MlxArrayNbytesFn = unsafe extern "C" fn(MlxArray) -> usize;
pub type MlxArrayNdimFn = unsafe extern "C" fn(MlxArray) -> usize;
pub type MlxArrayShapeFn = unsafe extern "C" fn(MlxArray) -> *const c_int;
pub type MlxArrayDimFn = unsafe extern "C" fn(MlxArray, c_int) -> c_int;
pub type MlxArrayDtypeFn = unsafe extern "C" fn(MlxArray) -> c_int;

// Array data access
pub type MlxArrayItemFloat64Fn = unsafe extern "C" fn(*mut f64, MlxArray) -> c_int;
pub type MlxArrayItemInt64Fn = unsafe extern "C" fn(*mut i64, MlxArray) -> c_int;
pub type MlxArrayDataFloat32Fn = unsafe extern "C" fn(MlxArray) -> *const f32;
pub type MlxArrayDataInt32Fn = unsafe extern "C" fn(MlxArray) -> *const i32;

pub type MlxArrayItemInt32Fn = unsafe extern "C" fn(*mut i32, MlxArray) -> c_int;

// Array eval
pub type MlxArrayEvalFn = unsafe extern "C" fn(MlxArray) -> c_int;
pub type MlxEvalFn = unsafe extern "C" fn(MlxVectorArray) -> c_int;
pub type MlxAsyncEvalFn = unsafe extern "C" fn(MlxVectorArray) -> c_int;

// Array string
pub type MlxArrayToStringFn = unsafe extern "C" fn(*mut MlxString, MlxArray) -> c_int;

// Stream & Device
pub type MlxStreamNewFn = unsafe extern "C" fn() -> MlxStream;
pub type MlxStreamNewDeviceFn = unsafe extern "C" fn(MlxDevice) -> MlxStream;
pub type MlxStreamFreeFn = unsafe extern "C" fn(MlxStream) -> c_int;
pub type MlxSetDefaultStreamFn = unsafe extern "C" fn(MlxStream) -> c_int;
pub type MlxDefaultGpuStreamNewFn = unsafe extern "C" fn() -> MlxStream;
pub type MlxDefaultCpuStreamNewFn = unsafe extern "C" fn() -> MlxStream;
pub type MlxDeviceNewTypeFn = unsafe extern "C" fn(MlxDeviceType, c_int) -> MlxDevice;
pub type MlxSetDefaultDeviceFn = unsafe extern "C" fn(MlxDevice) -> c_int;
pub type MlxGetDefaultStreamFn = unsafe extern "C" fn(*mut MlxStream, MlxDevice) -> c_int;

// Vector array
pub type MlxVectorArrayNewFn = unsafe extern "C" fn() -> MlxVectorArray;
pub type MlxVectorArrayFreeFn = unsafe extern "C" fn(MlxVectorArray) -> c_int;
pub type MlxVectorArrayAppendValueFn = unsafe extern "C" fn(MlxVectorArray, MlxArray) -> c_int;

// Ops (stream param passed by value matching mlx_stream C struct)
pub type MlxBinaryOpFn =
    unsafe extern "C" fn(*mut MlxArray, MlxArray, MlxArray, MlxStream) -> c_int;
pub type MlxReshapeFn = unsafe extern "C" fn(
    *mut MlxArray,
    MlxArray,
    *const c_int,
    usize,
    MlxStream,
) -> c_int;
pub type MlxTransposeAxesFn = unsafe extern "C" fn(
    *mut MlxArray,
    MlxArray,
    *const c_int,
    usize,
    MlxStream,
) -> c_int;
pub type MlxAsTypeFn =
    unsafe extern "C" fn(*mut MlxArray, MlxArray, c_int, MlxStream) -> c_int;
pub type MlxConcatenateAxisFn =
    unsafe extern "C" fn(*mut MlxArray, MlxVectorArray, c_int, MlxStream) -> c_int;
pub type MlxZerosFn =
    unsafe extern "C" fn(*mut MlxArray, *const c_int, usize, c_int, MlxStream) -> c_int;
pub type MlxSumAxisFn =
    unsafe extern "C" fn(*mut MlxArray, MlxArray, c_int, bool, MlxStream) -> c_int;
pub type MlxGatherMmFn = unsafe extern "C" fn(
    *mut MlxArray,
    MlxArray,
    MlxArray,
    MlxArray,
    MlxArray,
    bool,
    MlxStream,
) -> c_int;

// Fast ops (from mlx/c/fast.h)
pub type MlxFastRmsNormFn = unsafe extern "C" fn(
    *mut MlxArray,
    MlxArray,
    MlxArray, // weight (nullable)
    f32,      // eps
    MlxStream,
) -> c_int;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MlxOptionalFloat {
    pub value: f32,
    pub has_value: bool,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MlxOptionalInt {
    pub value: c_int,
    pub has_value: bool,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MlxOptionalDtype {
    pub value: c_int,
    pub has_value: bool,
}

pub type MlxFastRopeFn = unsafe extern "C" fn(
    *mut MlxArray,
    MlxArray,
    c_int,             // dims
    bool,              // traditional
    MlxOptionalFloat,  // base
    f32,               // scale
    c_int,             // offset
    MlxArray,          // freqs (nullable)
    MlxStream,
) -> c_int;

pub type MlxFastSdpaFn = unsafe extern "C" fn(
    *mut MlxArray,
    MlxArray,          // queries
    MlxArray,          // keys
    MlxArray,          // values
    f32,               // scale
    *const std::ffi::c_char, // mask_mode
    MlxArray,          // mask (nullable)
    MlxArray,          // sinks (nullable)
    MlxStream,
) -> c_int;

// Additional ops
pub type MlxTakeAxisFn = unsafe extern "C" fn(
    *mut MlxArray,
    MlxArray,
    MlxArray,
    c_int, // axis
    MlxStream,
) -> c_int;

pub type MlxExpandDimsFn = unsafe extern "C" fn(
    *mut MlxArray,
    MlxArray,
    c_int, // axis
    MlxStream,
) -> c_int;

pub type MlxTriFn = unsafe extern "C" fn(
    *mut MlxArray,
    c_int, // n
    c_int, // m
    c_int, // k
    c_int, // dtype
    MlxStream,
) -> c_int;

pub type MlxWhereFn = unsafe extern "C" fn(
    *mut MlxArray,
    MlxArray, // condition
    MlxArray, // x
    MlxArray, // y
    MlxStream,
) -> c_int;

pub type MlxUnaryOpFn = unsafe extern "C" fn(*mut MlxArray, MlxArray, MlxStream) -> c_int;
pub type MlxArgpartitionAxisFn = unsafe extern "C" fn(*mut MlxArray, MlxArray, c_int, c_int, MlxStream) -> c_int;
pub type MlxSoftmaxFn = unsafe extern "C" fn(*mut MlxArray, MlxArray, bool, MlxStream) -> c_int;
pub type MlxSoftmaxAxisFn = unsafe extern "C" fn(*mut MlxArray, MlxArray, c_int, bool, MlxStream) -> c_int;

// Argmax (no-axis variant: reduces entire array to scalar index)
pub type MlxArgmaxFn = unsafe extern "C" fn(
    *mut MlxArray,
    MlxArray,
    bool,              // keepdims
    MlxStream,
) -> c_int;

// Argmax with axis
pub type MlxArgmaxAxisFn = unsafe extern "C" fn(
    *mut MlxArray,
    MlxArray,
    c_int,             // axis
    bool,              // keepdims
    MlxStream,
) -> c_int;

// Arange
pub type MlxArangeFn = unsafe extern "C" fn(
    *mut MlxArray,
    f64,               // start
    f64,               // stop
    f64,               // step
    c_int,             // dtype
    MlxStream,
) -> c_int;

// Quantization ops
pub type MlxDequantizeFn = unsafe extern "C" fn(
    *mut MlxArray,
    MlxArray,           // w
    MlxArray,           // scales
    MlxArray,           // biases (nullable)
    MlxOptionalInt,     // group_size
    MlxOptionalInt,     // bits
    *const std::ffi::c_char, // mode
    MlxArray,           // global_scale (nullable)
    MlxOptionalDtype,   // dtype
    MlxStream,
) -> c_int;

pub type MlxQuantizedMatmulFn = unsafe extern "C" fn(
    *mut MlxArray,
    MlxArray,           // x
    MlxArray,           // w
    MlxArray,           // scales
    MlxArray,           // biases (nullable)
    bool,               // transpose
    MlxOptionalInt,     // group_size
    MlxOptionalInt,     // bits
    *const std::ffi::c_char, // mode
    MlxStream,
) -> c_int;

pub type MlxGatherQmmFn = unsafe extern "C" fn(
    *mut MlxArray,
    MlxArray,           // x
    MlxArray,           // w
    MlxArray,           // scales
    MlxArray,           // biases (nullable)
    MlxArray,           // lhs_indices (nullable)
    MlxArray,           // rhs_indices (nullable)
    bool,               // transpose
    MlxOptionalInt,     // group_size
    MlxOptionalInt,     // bits
    *const std::ffi::c_char, // mode
    bool,               // sorted_indices
    MlxStream,
) -> c_int;

// Map types (for mlx_load_safetensors)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MlxMapStringToArray {
    pub ctx: *mut std::ffi::c_void,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MlxMapStringToArrayIterator {
    pub ctx: *mut std::ffi::c_void,
    pub map_ctx: *mut std::ffi::c_void,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MlxMapStringToString {
    pub ctx: *mut std::ffi::c_void,
}

// Map function types
pub type MlxMapStringToArrayNewFn = unsafe extern "C" fn() -> MlxMapStringToArray;
pub type MlxMapStringToArrayFreeFn = unsafe extern "C" fn(MlxMapStringToArray) -> c_int;
pub type MlxMapStringToArrayGetFn =
    unsafe extern "C" fn(*mut MlxArray, MlxMapStringToArray, *const std::ffi::c_char) -> c_int;
pub type MlxMapStringToArrayIteratorNewFn =
    unsafe extern "C" fn(MlxMapStringToArray) -> MlxMapStringToArrayIterator;
pub type MlxMapStringToArrayIteratorFreeFn =
    unsafe extern "C" fn(MlxMapStringToArrayIterator) -> c_int;
pub type MlxMapStringToArrayIteratorNextFn = unsafe extern "C" fn(
    *mut *const std::ffi::c_char,
    *mut MlxArray,
    MlxMapStringToArrayIterator,
) -> c_int;
pub type MlxMapStringToStringNewFn = unsafe extern "C" fn() -> MlxMapStringToString;
pub type MlxMapStringToStringFreeFn = unsafe extern "C" fn(MlxMapStringToString) -> c_int;
pub type MlxMapStringToStringGetFn = unsafe extern "C" fn(
    *mut *const std::ffi::c_char,
    MlxMapStringToString,
    *const std::ffi::c_char,
) -> c_int;

// IO (safetensors)
pub type MlxLoadSafetensorsFn = unsafe extern "C" fn(
    *mut MlxMapStringToArray,
    *mut MlxMapStringToString,
    *const std::ffi::c_char,
    MlxStream,
) -> c_int;

// Fast rope dynamic (offset is array, not scalar)
pub type MlxFastRopeDynamicFn = unsafe extern "C" fn(
    *mut MlxArray,
    MlxArray,
    c_int,             // dims
    bool,              // traditional
    MlxOptionalFloat,  // base
    f32,               // scale
    MlxArray,          // offset (array)
    MlxArray,          // freqs (nullable)
    MlxStream,
) -> c_int;

// Memory
pub type MlxGetActiveMemoryFn = unsafe extern "C" fn(*mut usize) -> c_int;
pub type MlxGetCacheMemoryFn = unsafe extern "C" fn(*mut usize) -> c_int;
pub type MlxGetPeakMemoryFn = unsafe extern "C" fn(*mut usize) -> c_int;
pub type MlxResetPeakMemoryFn = unsafe extern "C" fn() -> c_int;
pub type MlxClearCacheFn = unsafe extern "C" fn() -> c_int;

pub struct MlxSymbols {
    // String/device
    pub mlx_version: MlxVersionFn,
    pub mlx_string_data: MlxStringDataFn,
    pub mlx_string_free: MlxStringFreeFn,
    pub mlx_device_is_available: MlxDeviceIsAvailableFn,
    pub mlx_get_default_device: MlxGetDefaultDeviceFn,
    pub mlx_device_get_type: MlxDeviceGetTypeFn,
    pub mlx_device_count: MlxDeviceCountFn,
    pub mlx_device_free: MlxDeviceFreeFn,
    // Array
    pub mlx_array_new_float32: MlxArrayNewFloat32Fn,
    pub mlx_array_new_float64: MlxArrayNewFloat64Fn,
    pub mlx_array_new_int: MlxArrayNewIntFn,
    pub mlx_array_new_data: MlxArrayNewDataFn,
    pub mlx_array_free: MlxArrayFreeFn,
    pub mlx_array_set: MlxArraySetFn,
    pub mlx_array_size: MlxArraySizeFn,
    pub mlx_array_nbytes: MlxArrayNbytesFn,
    pub mlx_array_ndim: MlxArrayNdimFn,
    pub mlx_array_shape: MlxArrayShapeFn,
    pub mlx_array_dim: MlxArrayDimFn,
    pub mlx_array_dtype: MlxArrayDtypeFn,
    pub mlx_array_item_float64: MlxArrayItemFloat64Fn,
    pub mlx_array_item_int32: MlxArrayItemInt32Fn,
    pub mlx_array_item_int64: MlxArrayItemInt64Fn,
    pub mlx_array_data_float32: MlxArrayDataFloat32Fn,
    pub mlx_array_data_int32: MlxArrayDataInt32Fn,
    pub mlx_array_eval: MlxArrayEvalFn,
    pub mlx_array_tostring: MlxArrayToStringFn,
    // Eval
    pub mlx_eval: MlxEvalFn,
    pub mlx_async_eval: MlxAsyncEvalFn,
    // Stream
    pub mlx_stream_new: MlxStreamNewFn,
    pub mlx_stream_new_device: MlxStreamNewDeviceFn,
    pub mlx_stream_free: MlxStreamFreeFn,
    pub mlx_set_default_stream: MlxSetDefaultStreamFn,
    pub mlx_default_gpu_stream_new: MlxDefaultGpuStreamNewFn,
    pub mlx_default_cpu_stream_new: MlxDefaultCpuStreamNewFn,
    pub mlx_device_new_type: MlxDeviceNewTypeFn,
    pub mlx_set_default_device: MlxSetDefaultDeviceFn,
    pub mlx_get_default_stream: MlxGetDefaultStreamFn,
    // Vector array
    pub mlx_vector_array_new: MlxVectorArrayNewFn,
    pub mlx_vector_array_free: MlxVectorArrayFreeFn,
    pub mlx_vector_array_append_value: MlxVectorArrayAppendValueFn,
    // Ops
    pub mlx_add: MlxBinaryOpFn,
    pub mlx_multiply: MlxBinaryOpFn,
    pub mlx_matmul: MlxBinaryOpFn,
    pub mlx_reshape: MlxReshapeFn,
    pub mlx_transpose_axes: MlxTransposeAxesFn,
    pub mlx_astype: MlxAsTypeFn,
    pub mlx_concatenate_axis: MlxConcatenateAxisFn,
    pub mlx_zeros: MlxZerosFn,
    pub mlx_sum_axis: MlxSumAxisFn,
    pub mlx_gather_mm: MlxGatherMmFn,
    // Fast ops
    pub mlx_fast_rms_norm: MlxFastRmsNormFn,
    pub mlx_fast_rope: MlxFastRopeFn,
    pub mlx_fast_sdpa: MlxFastSdpaFn,
    // Additional ops
    pub mlx_take_axis: MlxTakeAxisFn,
    pub mlx_take_along_axis: MlxTakeAxisFn,
    pub mlx_argpartition_axis: MlxArgpartitionAxisFn,
    pub mlx_expand_dims: MlxExpandDimsFn,
    pub mlx_tri: MlxTriFn,
    pub mlx_where: MlxWhereFn,
    pub mlx_divide: MlxBinaryOpFn,
    pub mlx_negative: MlxUnaryOpFn,
    pub mlx_softmax: MlxSoftmaxFn,
    pub mlx_sigmoid: MlxUnaryOpFn,
    pub mlx_sqrt: MlxUnaryOpFn,
    pub mlx_tanh: MlxUnaryOpFn,
    pub mlx_exp: MlxUnaryOpFn,
    pub mlx_log: MlxUnaryOpFn,
    pub mlx_softmax_axis: MlxSoftmaxAxisFn,
    // Argmax
    pub mlx_argmax: MlxArgmaxFn,
    pub mlx_argmax_axis: MlxArgmaxAxisFn,
    // Arange
    pub mlx_arange: MlxArangeFn,
    // Quantization ops
    pub mlx_dequantize: MlxDequantizeFn,
    pub mlx_quantized_matmul: MlxQuantizedMatmulFn,
    pub mlx_gather_qmm: MlxGatherQmmFn,
    // Map types (for safetensors loading)
    pub mlx_map_string_to_array_new: MlxMapStringToArrayNewFn,
    pub mlx_map_string_to_array_free: MlxMapStringToArrayFreeFn,
    pub mlx_map_string_to_array_get: MlxMapStringToArrayGetFn,
    pub mlx_map_string_to_array_iterator_new: MlxMapStringToArrayIteratorNewFn,
    pub mlx_map_string_to_array_iterator_free: MlxMapStringToArrayIteratorFreeFn,
    pub mlx_map_string_to_array_iterator_next: MlxMapStringToArrayIteratorNextFn,
    pub mlx_map_string_to_string_new: MlxMapStringToStringNewFn,
    pub mlx_map_string_to_string_free: MlxMapStringToStringFreeFn,
    pub mlx_map_string_to_string_get: MlxMapStringToStringGetFn,
    // IO
    pub mlx_load_safetensors: MlxLoadSafetensorsFn,
    // Fast rope dynamic
    pub mlx_fast_rope_dynamic: MlxFastRopeDynamicFn,
    // Memory
    pub mlx_get_active_memory: MlxGetActiveMemoryFn,
    pub mlx_get_cache_memory: MlxGetCacheMemoryFn,
    pub mlx_get_peak_memory: MlxGetPeakMemoryFn,
    pub mlx_reset_peak_memory: MlxResetPeakMemoryFn,
    pub mlx_clear_cache: MlxClearCacheFn,
}

unsafe impl Send for MlxSymbols {}

macro_rules! load_sym {
    ($lib:expr, $sym:literal, $ty:ty) => {{
        let val: $ty = *$lib
            .get($sym)
            .map_err(|e| anyhow::anyhow!("symbol {} not found: {e}", String::from_utf8_lossy($sym)))?;
        val
    }};
}

impl MlxSymbols {
    pub fn load(lib: &libloading::Library) -> anyhow::Result<Self> {
        unsafe {
            Ok(Self {
                mlx_version: load_sym!(lib, b"mlx_version\0", MlxVersionFn),
                mlx_string_data: load_sym!(lib, b"mlx_string_data\0", MlxStringDataFn),
                mlx_string_free: load_sym!(lib, b"mlx_string_free\0", MlxStringFreeFn),
                mlx_device_is_available: load_sym!(lib, b"mlx_device_is_available\0", MlxDeviceIsAvailableFn),
                mlx_get_default_device: load_sym!(lib, b"mlx_get_default_device\0", MlxGetDefaultDeviceFn),
                mlx_device_get_type: load_sym!(lib, b"mlx_device_get_type\0", MlxDeviceGetTypeFn),
                mlx_device_count: load_sym!(lib, b"mlx_device_count\0", MlxDeviceCountFn),
                mlx_device_free: load_sym!(lib, b"mlx_device_free\0", MlxDeviceFreeFn),
                mlx_array_new_float32: load_sym!(lib, b"mlx_array_new_float32\0", MlxArrayNewFloat32Fn),
                mlx_array_new_float64: load_sym!(lib, b"mlx_array_new_float64\0", MlxArrayNewFloat64Fn),
                mlx_array_new_int: load_sym!(lib, b"mlx_array_new_int\0", MlxArrayNewIntFn),
                mlx_array_new_data: load_sym!(lib, b"mlx_array_new_data\0", MlxArrayNewDataFn),
                mlx_array_free: load_sym!(lib, b"mlx_array_free\0", MlxArrayFreeFn),
                mlx_array_set: load_sym!(lib, b"mlx_array_set\0", MlxArraySetFn),
                mlx_array_size: load_sym!(lib, b"mlx_array_size\0", MlxArraySizeFn),
                mlx_array_nbytes: load_sym!(lib, b"mlx_array_nbytes\0", MlxArrayNbytesFn),
                mlx_array_ndim: load_sym!(lib, b"mlx_array_ndim\0", MlxArrayNdimFn),
                mlx_array_shape: load_sym!(lib, b"mlx_array_shape\0", MlxArrayShapeFn),
                mlx_array_dim: load_sym!(lib, b"mlx_array_dim\0", MlxArrayDimFn),
                mlx_array_dtype: load_sym!(lib, b"mlx_array_dtype\0", MlxArrayDtypeFn),
                mlx_array_item_float64: load_sym!(lib, b"mlx_array_item_float64\0", MlxArrayItemFloat64Fn),
                mlx_array_item_int32: load_sym!(lib, b"mlx_array_item_int32\0", MlxArrayItemInt32Fn),
                mlx_array_item_int64: load_sym!(lib, b"mlx_array_item_int64\0", MlxArrayItemInt64Fn),
                mlx_array_data_float32: load_sym!(lib, b"mlx_array_data_float32\0", MlxArrayDataFloat32Fn),
                mlx_array_data_int32: load_sym!(lib, b"mlx_array_data_int32\0", MlxArrayDataInt32Fn),
                mlx_array_eval: load_sym!(lib, b"mlx_array_eval\0", MlxArrayEvalFn),
                mlx_array_tostring: load_sym!(lib, b"mlx_array_tostring\0", MlxArrayToStringFn),
                mlx_eval: load_sym!(lib, b"mlx_eval\0", MlxEvalFn),
                mlx_async_eval: load_sym!(lib, b"mlx_async_eval\0", MlxAsyncEvalFn),
                mlx_stream_new: load_sym!(lib, b"mlx_stream_new\0", MlxStreamNewFn),
                mlx_stream_new_device: load_sym!(lib, b"mlx_stream_new_device\0", MlxStreamNewDeviceFn),
                mlx_stream_free: load_sym!(lib, b"mlx_stream_free\0", MlxStreamFreeFn),
                mlx_set_default_stream: load_sym!(lib, b"mlx_set_default_stream\0", MlxSetDefaultStreamFn),
                mlx_default_gpu_stream_new: load_sym!(lib, b"mlx_default_gpu_stream_new\0", MlxDefaultGpuStreamNewFn),
                mlx_default_cpu_stream_new: load_sym!(lib, b"mlx_default_cpu_stream_new\0", MlxDefaultCpuStreamNewFn),
                mlx_device_new_type: load_sym!(lib, b"mlx_device_new_type\0", MlxDeviceNewTypeFn),
                mlx_set_default_device: load_sym!(lib, b"mlx_set_default_device\0", MlxSetDefaultDeviceFn),
                mlx_get_default_stream: load_sym!(lib, b"mlx_get_default_stream\0", MlxGetDefaultStreamFn),
                mlx_vector_array_new: load_sym!(lib, b"mlx_vector_array_new\0", MlxVectorArrayNewFn),
                mlx_vector_array_free: load_sym!(lib, b"mlx_vector_array_free\0", MlxVectorArrayFreeFn),
                mlx_vector_array_append_value: load_sym!(lib, b"mlx_vector_array_append_value\0", MlxVectorArrayAppendValueFn),
                mlx_add: load_sym!(lib, b"mlx_add\0", MlxBinaryOpFn),
                mlx_multiply: load_sym!(lib, b"mlx_multiply\0", MlxBinaryOpFn),
                mlx_matmul: load_sym!(lib, b"mlx_matmul\0", MlxBinaryOpFn),
                mlx_reshape: load_sym!(lib, b"mlx_reshape\0", MlxReshapeFn),
                mlx_transpose_axes: load_sym!(lib, b"mlx_transpose_axes\0", MlxTransposeAxesFn),
                mlx_astype: load_sym!(lib, b"mlx_astype\0", MlxAsTypeFn),
                mlx_concatenate_axis: load_sym!(lib, b"mlx_concatenate_axis\0", MlxConcatenateAxisFn),
                mlx_zeros: load_sym!(lib, b"mlx_zeros\0", MlxZerosFn),
                mlx_sum_axis: load_sym!(lib, b"mlx_sum_axis\0", MlxSumAxisFn),
                mlx_gather_mm: load_sym!(lib, b"mlx_gather_mm\0", MlxGatherMmFn),
                // Fast ops
                mlx_fast_rms_norm: load_sym!(lib, b"mlx_fast_rms_norm\0", MlxFastRmsNormFn),
                mlx_fast_rope: load_sym!(lib, b"mlx_fast_rope\0", MlxFastRopeFn),
                mlx_fast_sdpa: load_sym!(lib, b"mlx_fast_scaled_dot_product_attention\0", MlxFastSdpaFn),
                // Additional ops
                mlx_take_axis: load_sym!(lib, b"mlx_take_axis\0", MlxTakeAxisFn),
                mlx_take_along_axis: load_sym!(lib, b"mlx_take_along_axis\0", MlxTakeAxisFn),
                mlx_argpartition_axis: load_sym!(lib, b"mlx_argpartition_axis\0", MlxArgpartitionAxisFn),
                mlx_expand_dims: load_sym!(lib, b"mlx_expand_dims\0", MlxExpandDimsFn),
                mlx_tri: load_sym!(lib, b"mlx_tri\0", MlxTriFn),
                mlx_where: load_sym!(lib, b"mlx_where\0", MlxWhereFn),
                mlx_divide: load_sym!(lib, b"mlx_divide\0", MlxBinaryOpFn),
                mlx_negative: load_sym!(lib, b"mlx_negative\0", MlxUnaryOpFn),
                mlx_softmax: load_sym!(lib, b"mlx_softmax\0", MlxSoftmaxFn),
                mlx_sigmoid: load_sym!(lib, b"mlx_sigmoid\0", MlxUnaryOpFn),
                mlx_sqrt: load_sym!(lib, b"mlx_sqrt\0", MlxUnaryOpFn),
                mlx_tanh: load_sym!(lib, b"mlx_tanh\0", MlxUnaryOpFn),
                mlx_exp: load_sym!(lib, b"mlx_exp\0", MlxUnaryOpFn),
                mlx_log: load_sym!(lib, b"mlx_log\0", MlxUnaryOpFn),
                mlx_softmax_axis: load_sym!(lib, b"mlx_softmax_axis\0", MlxSoftmaxAxisFn),
                // Quantization ops
                mlx_argmax: load_sym!(lib, b"mlx_argmax\0", MlxArgmaxFn),
                mlx_argmax_axis: load_sym!(lib, b"mlx_argmax_axis\0", MlxArgmaxAxisFn),
                // Arange
                mlx_arange: load_sym!(lib, b"mlx_arange\0", MlxArangeFn),
                mlx_dequantize: load_sym!(lib, b"mlx_dequantize\0", MlxDequantizeFn),
                mlx_quantized_matmul: load_sym!(lib, b"mlx_quantized_matmul\0", MlxQuantizedMatmulFn),
                mlx_gather_qmm: load_sym!(lib, b"mlx_gather_qmm\0", MlxGatherQmmFn),
                // Map types
                mlx_map_string_to_array_new: load_sym!(lib, b"mlx_map_string_to_array_new\0", MlxMapStringToArrayNewFn),
                mlx_map_string_to_array_free: load_sym!(lib, b"mlx_map_string_to_array_free\0", MlxMapStringToArrayFreeFn),
                mlx_map_string_to_array_get: load_sym!(lib, b"mlx_map_string_to_array_get\0", MlxMapStringToArrayGetFn),
                mlx_map_string_to_array_iterator_new: load_sym!(lib, b"mlx_map_string_to_array_iterator_new\0", MlxMapStringToArrayIteratorNewFn),
                mlx_map_string_to_array_iterator_free: load_sym!(lib, b"mlx_map_string_to_array_iterator_free\0", MlxMapStringToArrayIteratorFreeFn),
                mlx_map_string_to_array_iterator_next: load_sym!(lib, b"mlx_map_string_to_array_iterator_next\0", MlxMapStringToArrayIteratorNextFn),
                mlx_map_string_to_string_new: load_sym!(lib, b"mlx_map_string_to_string_new\0", MlxMapStringToStringNewFn),
                mlx_map_string_to_string_free: load_sym!(lib, b"mlx_map_string_to_string_free\0", MlxMapStringToStringFreeFn),
                mlx_map_string_to_string_get: load_sym!(lib, b"mlx_map_string_to_string_get\0", MlxMapStringToStringGetFn),
                // IO
                mlx_load_safetensors: load_sym!(lib, b"mlx_load_safetensors\0", MlxLoadSafetensorsFn),
                // Fast rope dynamic
                mlx_fast_rope_dynamic: load_sym!(lib, b"mlx_fast_rope_dynamic\0", MlxFastRopeDynamicFn),
                // Memory
                mlx_get_active_memory: load_sym!(lib, b"mlx_get_active_memory\0", MlxGetActiveMemoryFn),
                mlx_get_cache_memory: load_sym!(lib, b"mlx_get_cache_memory\0", MlxGetCacheMemoryFn),
                mlx_get_peak_memory: load_sym!(lib, b"mlx_get_peak_memory\0", MlxGetPeakMemoryFn),
                mlx_reset_peak_memory: load_sym!(lib, b"mlx_reset_peak_memory\0", MlxResetPeakMemoryFn),
                mlx_clear_cache: load_sym!(lib, b"mlx_clear_cache\0", MlxClearCacheFn),
            })
        }
    }
}
