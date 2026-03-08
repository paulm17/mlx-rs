use crate::Result;
use mlx_sys::*;

/// An MLX compute stream.
pub struct Stream {
    pub(crate) inner: mlx_stream,
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            mlx_stream_free(self.inner);
        }
    }
}

impl Stream {
    /// Create the default GPU stream.
    pub fn new_gpu_default() -> Self {
        unsafe {
            Self {
                inner: mlx_default_gpu_stream_new(),
            }
        }
    }

    /// Create the default CPU stream.
    pub fn new_cpu_default() -> Self {
        unsafe {
            Self {
                inner: mlx_default_cpu_stream_new(),
            }
        }
    }

    /// Synchronize (wait for all pending operations on this stream).
    pub fn synchronize(&self) -> Result<()> {
        unsafe {
            let rc = mlx_synchronize(self.inner);
            if rc != 0 {
                return Err(crate::Error::Mlx("mlx_synchronize failed".into()));
            }
        }
        Ok(())
    }

    /// Set this stream as the default and return a guard that restores the
    /// previous default on drop.
    pub fn set_default(&self) -> Result<DefaultStreamGuard> {
        unsafe {
            let mut dev: mlx_device = std::mem::zeroed();
            mlx_get_default_device(&mut dev);
            let mut prev: mlx_stream = std::mem::zeroed();
            let rc_get = mlx_get_default_stream(&mut prev, dev);
            mlx_device_free(dev);
            if rc_get != 0 {
                return Err(crate::Error::Mlx("mlx_get_default_stream failed".into()));
            }
            let rc_set = mlx_set_default_stream(self.inner);
            if rc_set != 0 {
                return Err(crate::Error::Mlx("mlx_set_default_stream failed".into()));
            }
            Ok(DefaultStreamGuard { previous: prev })
        }
    }
}

/// Restores the previous default stream when dropped.
pub struct DefaultStreamGuard {
    previous: mlx_stream,
}

impl Drop for DefaultStreamGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = mlx_set_default_stream(self.previous);
        }
    }
}
