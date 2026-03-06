use mlx_core::{Array, Error, Result};
use std::sync::atomic::{AtomicUsize, Ordering};

static KV_CACHE_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static KV_CACHE_GROWTHS: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy)]
pub struct KvCacheStats {
    pub allocations: usize,
    pub growths: usize,
}

pub fn reset_kv_cache_stats() {
    KV_CACHE_ALLOCATIONS.store(0, Ordering::Relaxed);
    KV_CACHE_GROWTHS.store(0, Ordering::Relaxed);
}

pub fn kv_cache_stats() -> KvCacheStats {
    KvCacheStats {
        allocations: KV_CACHE_ALLOCATIONS.load(Ordering::Relaxed),
        growths: KV_CACHE_GROWTHS.load(Ordering::Relaxed),
    }
}

/// Key-Value cache for autoregressive decoding.
///
/// Uses stepped preallocation and slice updates to avoid O(n^2) concat growth.
pub struct KvCache {
    k: Option<Array>,
    v: Option<Array>,
    offset: usize,
    step: usize,
}

impl KvCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        let step = std::env::var("MLX_KV_CACHE_STEP")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(512);
        Self {
            k: None,
            v: None,
            offset: 0,
            step,
        }
    }

    fn round_up_steps(&self, n: usize) -> usize {
        let step = self.step.max(1);
        n.div_ceil(step) * step
    }

    fn ensure_layout(name: &str, arr: &Array) -> Result<()> {
        let shape = arr.shape_raw();
        if shape.len() != 4 {
            return Err(Error::Message(format!(
                "{name} must be rank-4 [B, H, T, D], got shape={shape:?}"
            )));
        }
        Ok(())
    }

    fn alloc_like(src: &Array, capacity: usize) -> Result<Array> {
        let mut shape = src.shape_raw();
        shape[2] = capacity as i32;
        Array::zeros(&shape, src.dtype())
    }

    fn copy_prefix(src: &Array, dst: &Array, len_t: usize) -> Result<Array> {
        if len_t == 0 {
            return Ok(dst.clone());
        }
        let shape = src.shape_raw();
        let ndim = shape.len();
        let start = vec![0i32; ndim];
        let mut stop = shape;
        stop[2] = len_t as i32;
        let strides = vec![1i32; ndim];
        let src_prefix = src.slice(&start, &stop)?;
        dst.slice_update(&src_prefix, &start, &stop, &strides)
    }

    fn take_prefix(src: &Array, len_t: usize) -> Result<Array> {
        let shape = src.shape_raw();
        let ndim = shape.len();
        let start = vec![0i32; ndim];
        let mut stop = shape;
        stop[2] = len_t as i32;
        src.slice(&start, &stop)
    }

    fn write_at(dst: &Array, update: &Array, start_t: usize, stop_t: usize) -> Result<Array> {
        let ndim = dst.ndim();
        let mut start = vec![0i32; ndim];
        let mut stop = dst.shape_raw();
        start[2] = start_t as i32;
        stop[2] = stop_t as i32;
        let strides = vec![1i32; ndim];
        dst.slice_update(update, &start, &stop, &strides)
    }

    fn same_bh(k: &Array, v: &Array) -> bool {
        k.dim(0) == v.dim(0) && k.dim(1) == v.dim(1) && k.dim(2) == v.dim(2)
    }

    pub fn update(&mut self, new_k: &Array, new_v: &Array) -> Result<(Array, Array)> {
        Self::ensure_layout("new_k", new_k)?;
        Self::ensure_layout("new_v", new_v)?;
        if !Self::same_bh(new_k, new_v) {
            return Err(Error::Message(format!(
                "new_k/new_v shape mismatch: k={:?}, v={:?}",
                new_k.shape_raw(),
                new_v.shape_raw()
            )));
        }

        let incoming = new_k.dim(2) as usize;
        if incoming == 0 {
            return match (&self.k, &self.v) {
                (Some(k), Some(v)) => Ok((
                    Self::take_prefix(k, self.offset)?,
                    Self::take_prefix(v, self.offset)?,
                )),
                _ => {
                    let empty_k = Array::zeros(
                        &[new_k.dim(0), new_k.dim(1), 0, new_k.dim(3)],
                        new_k.dtype(),
                    )?;
                    let empty_v = Array::zeros(
                        &[new_v.dim(0), new_v.dim(1), 0, new_v.dim(3)],
                        new_v.dtype(),
                    )?;
                    Ok((empty_k, empty_v))
                }
            };
        }

        let needed = self.offset + incoming;
        let current_capacity = self.k.as_ref().map(|k| k.dim(2) as usize).unwrap_or(0);
        if self.k.is_none() || self.v.is_none() {
            let capacity = self.round_up_steps(needed);
            KV_CACHE_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            self.k = Some(Self::alloc_like(new_k, capacity)?);
            self.v = Some(Self::alloc_like(new_v, capacity)?);
        } else if needed > current_capacity {
            let capacity = self.round_up_steps(needed);
            KV_CACHE_GROWTHS.fetch_add(1, Ordering::Relaxed);
            let old_k = self.k.as_ref().unwrap();
            let old_v = self.v.as_ref().unwrap();
            let mut new_k_buf = Self::alloc_like(new_k, capacity)?;
            let mut new_v_buf = Self::alloc_like(new_v, capacity)?;
            new_k_buf = Self::copy_prefix(old_k, &new_k_buf, self.offset)?;
            new_v_buf = Self::copy_prefix(old_v, &new_v_buf, self.offset)?;
            self.k = Some(new_k_buf);
            self.v = Some(new_v_buf);
        }

        let start = self.offset;
        self.offset = needed;
        let k_buf = Self::write_at(self.k.as_ref().unwrap(), new_k, start, self.offset)?;
        let v_buf = Self::write_at(self.v.as_ref().unwrap(), new_v, start, self.offset)?;

        self.k = Some(k_buf);
        self.v = Some(v_buf);

        let k = Self::take_prefix(self.k.as_ref().unwrap(), self.offset)?;
        let v = Self::take_prefix(self.v.as_ref().unwrap(), self.offset)?;
        Ok((k, v))
    }

    /// Get the current sequence length (number of cached tokens).
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Reset the cache (e.g. for a new conversation).
    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
        self.offset = 0;
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.offset == 0
    }
}

impl Default for KvCache {
    fn default() -> Self {
        Self::new()
    }
}
