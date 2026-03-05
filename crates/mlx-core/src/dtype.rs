/// Data types supported by MLX arrays.
///
/// Maps 1:1 to the `mlx_dtype` constants from the C API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    UInt8,
    UInt16,
    UInt32,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    BFloat16,
    Float32,
    Complex64,
}

impl DType {
    /// Convert to the raw mlx-c constant.
    pub fn to_mlx(self) -> u32 {
        use mlx_sys::*;
        match self {
            DType::Bool => mlx_dtype__MLX_BOOL,
            DType::UInt8 => mlx_dtype__MLX_UINT8,
            DType::UInt16 => mlx_dtype__MLX_UINT16,
            DType::UInt32 => mlx_dtype__MLX_UINT32,
            DType::Int8 => mlx_dtype__MLX_INT8,
            DType::Int16 => mlx_dtype__MLX_INT16,
            DType::Int32 => mlx_dtype__MLX_INT32,
            DType::Int64 => mlx_dtype__MLX_INT64,
            DType::Float16 => mlx_dtype__MLX_FLOAT16,
            DType::BFloat16 => mlx_dtype__MLX_BFLOAT16,
            DType::Float32 => mlx_dtype__MLX_FLOAT32,
            DType::Complex64 => mlx_dtype__MLX_COMPLEX64,
        }
    }

    /// Convert from a raw mlx-c constant.
    pub fn from_mlx(raw: u32) -> Option<Self> {
        use mlx_sys::*;
        match raw {
            x if x == mlx_dtype__MLX_BOOL => Some(DType::Bool),
            x if x == mlx_dtype__MLX_UINT8 => Some(DType::UInt8),
            x if x == mlx_dtype__MLX_UINT16 => Some(DType::UInt16),
            x if x == mlx_dtype__MLX_UINT32 => Some(DType::UInt32),
            x if x == mlx_dtype__MLX_INT8 => Some(DType::Int8),
            x if x == mlx_dtype__MLX_INT16 => Some(DType::Int16),
            x if x == mlx_dtype__MLX_INT32 => Some(DType::Int32),
            x if x == mlx_dtype__MLX_INT64 => Some(DType::Int64),
            x if x == mlx_dtype__MLX_FLOAT16 => Some(DType::Float16),
            x if x == mlx_dtype__MLX_BFLOAT16 => Some(DType::BFloat16),
            x if x == mlx_dtype__MLX_FLOAT32 => Some(DType::Float32),
            x if x == mlx_dtype__MLX_COMPLEX64 => Some(DType::Complex64),
            _ => None,
        }
    }

    /// Size of a single element in bytes.
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::Bool | DType::UInt8 | DType::Int8 => 1,
            DType::UInt16 | DType::Int16 | DType::Float16 | DType::BFloat16 => 2,
            DType::UInt32 | DType::Int32 | DType::Float32 => 4,
            DType::Int64 | DType::Complex64 => 8,
        }
    }

    /// Whether this type is a floating-point type.
    pub fn is_float(self) -> bool {
        matches!(
            self,
            DType::Float16 | DType::BFloat16 | DType::Float32 | DType::Complex64
        )
    }

    /// Whether this type is an integer type.
    pub fn is_integer(self) -> bool {
        matches!(
            self,
            DType::UInt8
                | DType::UInt16
                | DType::UInt32
                | DType::Int8
                | DType::Int16
                | DType::Int32
                | DType::Int64
        )
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            DType::Bool => "bool",
            DType::UInt8 => "uint8",
            DType::UInt16 => "uint16",
            DType::UInt32 => "uint32",
            DType::Int8 => "int8",
            DType::Int16 => "int16",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
            DType::Float16 => "float16",
            DType::BFloat16 => "bfloat16",
            DType::Float32 => "float32",
            DType::Complex64 => "complex64",
        };
        write!(f, "{}", s)
    }
}
