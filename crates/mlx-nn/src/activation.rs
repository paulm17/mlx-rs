use mlx_core::{Array, Module, Result};

/// Activation function enum.
///
/// Each variant implements `Module`, allowing activations to be used
/// composably in model definitions.
#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum Activation {
    #[serde(alias = "gelu")]
    Gelu,
    Relu,
    #[default]
    Silu,
    Sigmoid,
    Tanh,
    #[serde(alias = "swiglu")]
    Swiglu,
}

impl Module for Activation {
    fn forward(&self, xs: &Array) -> Result<Array> {
        match self {
            Activation::Gelu => {
                // GELU ≈ x * sigmoid(1.702 * x)
                let coeff = Array::from_float(1.702)?;
                let scaled = xs.multiply(&coeff)?;
                let gate = scaled.sigmoid()?;
                xs.multiply(&gate)
            }
            Activation::Relu => {
                let zero = Array::from_float(0.0)?;
                let mask = xs.greater(&zero)?;
                mask.where_cond(xs, &zero)
            }
            Activation::Silu => {
                // SiLU = x * sigmoid(x)
                let gate = xs.sigmoid()?;
                xs.multiply(&gate)
            }
            Activation::Sigmoid => xs.sigmoid(),
            Activation::Tanh => xs.tanh(),
            Activation::Swiglu => {
                // Swiglu: split x in half, apply silu to first half, multiply by second
                let ndim = xs.ndim();
                let last_dim = xs.dim(ndim as i32 - 1);
                let half = last_dim / 2;
                let shape = xs.shape_raw();
                let mut start = vec![0i32; ndim];
                let mut stop: Vec<i32> = shape.clone();
                stop[ndim - 1] = half;
                let x1 = xs.slice(&start, &stop)?;

                start[ndim - 1] = half;
                stop[ndim - 1] = last_dim;
                let x2 = xs.slice(&start, &stop)?;

                let gate = x1.sigmoid()?;
                let x1_silu = x1.multiply(&gate)?;
                x1_silu.multiply(&x2)
            }
        }
    }
}

