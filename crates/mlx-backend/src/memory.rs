use crate::loader;

pub fn active_memory() -> anyhow::Result<usize> {
    let syms = loader::symbols()?;
    let mut val: usize = 0;
    let rc = unsafe { (syms.mlx_get_active_memory)(&mut val) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_get_active_memory returned error: {rc}"));
    }
    Ok(val)
}

pub fn cache_memory() -> anyhow::Result<usize> {
    let syms = loader::symbols()?;
    let mut val: usize = 0;
    let rc = unsafe { (syms.mlx_get_cache_memory)(&mut val) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_get_cache_memory returned error: {rc}"));
    }
    Ok(val)
}

pub fn peak_memory() -> anyhow::Result<usize> {
    let syms = loader::symbols()?;
    let mut val: usize = 0;
    let rc = unsafe { (syms.mlx_get_peak_memory)(&mut val) };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_get_peak_memory returned error: {rc}"));
    }
    Ok(val)
}

pub fn reset_peak_memory() -> anyhow::Result<()> {
    let syms = loader::symbols()?;
    let rc = unsafe { (syms.mlx_reset_peak_memory)() };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_reset_peak_memory returned error: {rc}"));
    }
    Ok(())
}

pub fn clear_cache() -> anyhow::Result<()> {
    let syms = loader::symbols()?;
    let rc = unsafe { (syms.mlx_clear_cache)() };
    if rc != 0 {
        return Err(anyhow::anyhow!("mlx_clear_cache returned error: {rc}"));
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
    fn test_active_memory() {
        if !mlx_available() {
            return;
        }
        let val = active_memory().unwrap();
        let _ = val; // just check it doesn't error
    }

    #[test]
    fn test_cache_memory() {
        if !mlx_available() {
            return;
        }
        let val = cache_memory().unwrap();
        let _ = val;
    }

    #[test]
    fn test_peak_memory() {
        if !mlx_available() {
            return;
        }
        let val = peak_memory().unwrap();
        let _ = val;
    }

    #[test]
    fn test_reset_peak_memory() {
        if !mlx_available() {
            return;
        }
        reset_peak_memory().unwrap();
    }

    #[test]
    fn test_clear_cache() {
        if !mlx_available() {
            return;
        }
        clear_cache().unwrap();
    }
}
