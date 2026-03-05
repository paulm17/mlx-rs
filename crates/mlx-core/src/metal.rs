use mlx_sys::*;

/// Set the Metal wired memory limit (in bytes).
pub fn set_wired_limit(limit: usize) {
    unsafe {
        let mut res: usize = 0;
        mlx_set_wired_limit(&mut res, limit);
    }
}

/// Set the MLX allocator cache limit (in bytes).
pub fn set_cache_limit(limit: usize) {
    unsafe {
        let mut res: usize = 0;
        mlx_set_cache_limit(&mut res, limit);
    }
}

/// Get the MLX allocator memory limit (in bytes).
pub fn memory_limit() -> usize {
    unsafe {
        let mut res: usize = 0;
        mlx_get_memory_limit(&mut res);
        res
    }
}

/// Clear the MLX allocator cache.
pub fn clear_cache() {
    unsafe {
        mlx_clear_cache();
    }
}

/// Clear MLX compiled-kernel cache.
pub fn clear_compile_cache() {
    unsafe {
        mlx_detail_compile_clear_cache();
    }
}

/// Memory info from the Metal allocator.
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub active_memory: usize,
    pub peak_memory: usize,
    pub cache_memory: usize,
}

/// Get current Metal memory info.
pub fn memory_info() -> MemoryInfo {
    unsafe {
        let mut active: usize = 0;
        let mut peak: usize = 0;
        let mut cache: usize = 0;
        mlx_get_active_memory(&mut active);
        mlx_get_peak_memory(&mut peak);
        mlx_get_cache_memory(&mut cache);
        MemoryInfo {
            active_memory: active,
            peak_memory: peak,
            cache_memory: cache,
        }
    }
}

/// Reset Metal peak memory counter.
pub fn reset_peak_memory() {
    unsafe {
        mlx_reset_peak_memory();
    }
}
