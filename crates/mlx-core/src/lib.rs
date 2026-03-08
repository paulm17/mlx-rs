//! mlx-core: Core tensor library for the MLX Rust framework.
//!
//! This crate provides the fundamental building blocks:
//! - [`Array`] — the core tensor type wrapping mlx.c arrays
//! - [`DType`] — data type enum
//! - [`Shape`] — shape type
//! - [`Device`] — CPU/GPU device selection
//! - [`Module`] / [`ModuleT`] — composability traits for neural network layers
//!
//! # Example
//! ```no_run
//! use mlx_core::{Array, DType};
//!
//! let a = Array::zeros(&[2, 3], DType::Float32).unwrap();
//! let b = Array::ones(&[2, 3], DType::Float32).unwrap();
//! let c = a.add(&b).unwrap();
//! ```

pub mod array;
pub mod device;
pub mod dtype;
pub mod error;
pub mod metal;
pub mod ops;
pub mod safetensors;
pub mod shape;
pub mod stream;

// Re-exports for convenience
pub use array::Array;
pub use device::Device;
pub use dtype::DType;
pub use error::{Error, Result};
pub use ops::{async_eval, eval};
pub use shape::{IntoShape, Shape};
pub use stream::{DefaultStreamGuard, Stream};

// ------------------------------------------------------------------
// Module trait — the composability backbone (inspired by candle-core)
// ------------------------------------------------------------------

/// A module with a forward method using a single array argument.
///
/// All neural network layers implement this trait, enabling composability.
///
/// # Example
/// ```no_run
/// use mlx_core::{Array, Module, Result};
///
/// struct MyLayer { /* ... */ }
///
/// impl Module for MyLayer {
///     fn forward(&self, xs: &Array) -> Result<Array> {
///         // ... transform xs ...
///         Ok(xs.clone())
///     }
/// }
/// ```
pub trait Module {
    fn forward(&self, xs: &Array) -> Result<Array>;
}

/// A module with a forward method that also takes a `train` flag.
///
/// This is used to separate training and evaluation behaviors (e.g. dropout).
pub trait ModuleT {
    fn forward_t(&self, xs: &Array, train: bool) -> Result<Array>;
}

// Any `Module` is also a `ModuleT` (ignoring the train flag).
impl<M: Module> ModuleT for M {
    fn forward_t(&self, xs: &Array, _train: bool) -> Result<Array> {
        self.forward(xs)
    }
}

// Allow closures to be used as modules.
impl<F: Fn(&Array) -> Result<Array>> Module for F {
    fn forward(&self, xs: &Array) -> Result<Array> {
        self(xs)
    }
}

// Optional modules pass through if None.
impl<M: Module> Module for Option<&M> {
    fn forward(&self, xs: &Array) -> Result<Array> {
        match self {
            None => Ok(xs.clone()),
            Some(m) => m.forward(xs),
        }
    }
}
