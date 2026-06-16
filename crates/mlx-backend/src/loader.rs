use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::Context;
use glob::glob;

use crate::ffi::{MlxDeviceType, MlxSymbols};

struct MlxState {
    #[allow(dead_code)] // held to keep the library loaded
    lib: libloading::Library,
    path: PathBuf,
    symbols: MlxSymbols,
}

static MLX: OnceLock<Result<MlxState, String>> = OnceLock::new();

fn lib_names() -> Vec<&'static str> {
    if cfg!(target_os = "macos") {
        vec!["libmlxc.dylib"]
    } else if cfg!(target_os = "linux") {
        vec!["libmlxc.so"]
    } else {
        vec!["libmlxc.dll", "mlxc.dll"]
    }
}

fn candidate_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    if let Ok(exe) = std::env::current_exe() {
        if let Ok(eval) = exe.canonicalize() {
            let exe_dir = eval.parent().unwrap_or(Path::new("."));
            if cfg!(target_os = "macos") {
                dirs.push(exe_dir.join("lib").join("ollama"));
                dirs.push(exe_dir.join("..").join("lib").join("ollama"));
                dirs.push(exe_dir.to_path_buf());
            } else if cfg!(target_os = "linux") {
                dirs.push(exe_dir.join("..").join("lib").join("ollama"));
            } else {
                dirs.push(exe_dir.join("lib").join("ollama"));
                dirs.push(exe_dir.join("..").join("lib").join("ollama"));
            }
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        dirs.push(cwd.join("build").join("lib").join("ollama"));

        if let Ok(pattern) = glob(&cwd.join("build").join("*").join("lib").join("ollama").to_string_lossy()) {
            let mut matches: Vec<PathBuf> = pattern.filter_map(Result::ok).collect();
            matches.sort_by(|a, b| b.cmp(a));
            dirs.extend(matches);
        }

        dirs.push(cwd.join("dist").join(format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)).join("lib").join("ollama"));
    }

    dirs
}

pub(crate) fn init() -> anyhow::Result<()> {
    let result = MLX.get_or_init(|| {
        let forced = std::env::var("MLX_RS_MLX_LIBRARY")
            .ok()
            .or_else(|| {
                let v = std::env::var("OLLAMA_LLM_LIBRARY").ok()?;
                if v.starts_with("mlx_") { Some(v) } else { None }
            });

        find_and_load(forced.as_deref())
            .map_err(|e| format!("{e:#}"))
    });

    match result {
        Ok(_) => Ok(()),
        Err(msg) => Err(anyhow::anyhow!("{msg}")),
    }
}

fn find_and_load(forced_variant: Option<&str>) -> anyhow::Result<MlxState> {
    for root in candidate_dirs() {
        if let Some(variant) = forced_variant {
            if let Some(state) = try_load_from_dir(&root.join(variant)) {
                prepend_library_path(state.path.parent().unwrap_or(Path::new(".")));
                return Ok(state);
            }
        } else {
            if let Some(state) = try_load_from_mlx_subdirs(&root) {
                prepend_library_path(state.path.parent().unwrap_or(Path::new(".")));
                return Ok(state);
            }
            if let Some(state) = try_load_from_dir(&root) {
                prepend_library_path(state.path.parent().unwrap_or(Path::new(".")));
                return Ok(state);
            }
        }
    }

    let searched = candidate_dirs();
    Err(anyhow::anyhow!(
        "MLX-C runtime not found (searched {} directories)",
        searched.len()
    ))
}

fn try_load_from_mlx_subdirs(dir: &Path) -> Option<MlxState> {
    let pattern = dir.join("mlx_*").to_string_lossy().to_string();
    let mut matches: Vec<PathBuf> = glob(&pattern)
        .ok()?
        .filter_map(Result::ok)
        .collect();
    matches.sort_by(|a, b| b.cmp(a));

    for m in matches {
        if let Some(state) = try_load_from_dir(&m) {
            return Some(state);
        }
    }
    None
}

fn try_load_from_dir(dir: &Path) -> Option<MlxState> {
    for name in lib_names() {
        let path = dir.join(name);
        if !path.exists() {
            continue;
        }
        match load_library(&path) {
            Ok(state) => return Some(state),
            Err(_) => continue,
        }
    }
    None
}

fn load_library(path: &Path) -> anyhow::Result<MlxState> {
    let lib = unsafe {
        libloading::Library::new(path)
            .with_context(|| format!("failed to load MLX library: {}", path.display()))?
    };

    let symbols = MlxSymbols::load(&lib)
        .with_context(|| format!("failed to load MLX symbols from: {}", path.display()))?;

    Ok(MlxState {
        lib,
        path: path.to_path_buf(),
        symbols,
    })
}

#[cfg(unix)]
fn prepend_library_path(dir: &Path) {
    let env_var = if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    };

    let dir_str = dir.to_string_lossy();
    match std::env::var(env_var) {
        Ok(existing) if !existing.is_empty() => {
            std::env::set_var(env_var, format!("{dir_str}:{existing}"));
        }
        _ => {
            std::env::set_var(env_var, dir_str.as_ref());
        }
    }
}

#[cfg(not(unix))]
fn prepend_library_path(_dir: &Path) {}

pub fn check_init() -> anyhow::Result<()> {
    init()
}

pub fn loaded_library_path() -> anyhow::Result<PathBuf> {
    let state = MLX.get().ok_or_else(|| anyhow::anyhow!("MLX not initialized"))?;
    match state {
        Ok(s) => Ok(s.path.clone()),
        Err(msg) => Err(anyhow::anyhow!("{msg}")),
    }
}

pub fn version() -> anyhow::Result<String> {
    init()?;
    let state = MLX.get().unwrap().as_ref().unwrap();
    let symbols = &state.symbols;

    unsafe {
        let mut str_ptr = std::mem::MaybeUninit::uninit();
        let rc = (symbols.mlx_version)(str_ptr.as_mut_ptr());
        if rc != 0 {
            return Err(anyhow::anyhow!("mlx_version returned error: {rc}"));
        }
        let str_ptr = str_ptr.assume_init();
        if str_ptr.ctx.is_null() {
            return Err(anyhow::anyhow!("mlx_version returned null string"));
        }

        let data = (symbols.mlx_string_data)(str_ptr);
        if data.is_null() {
            (symbols.mlx_string_free)(str_ptr);
            return Err(anyhow::anyhow!("mlx_version returned null data"));
        }

        let version = std::ffi::CStr::from_ptr(data)
            .to_str()
            .context("mlx_version returned invalid UTF-8")?
            .to_owned();

        (symbols.mlx_string_free)(str_ptr);
        Ok(version)
    }
}

pub fn default_device_available() -> anyhow::Result<bool> {
    init()?;
    let state = MLX.get().unwrap().as_ref().unwrap();
    let symbols = &state.symbols;

    unsafe {
        let mut dev = std::mem::MaybeUninit::uninit();
        let rc = (symbols.mlx_get_default_device)(dev.as_mut_ptr());
        if rc != 0 {
            return Err(anyhow::anyhow!("mlx_get_default_device returned error: {rc}"));
        }
        let dev = dev.assume_init();

        let mut available = false;
        let rc = (symbols.mlx_device_is_available)(&mut available, dev);
        (symbols.mlx_device_free)(dev);

        if rc != 0 {
            return Err(anyhow::anyhow!("mlx_device_is_available returned error: {rc}"));
        }
        Ok(available)
    }
}

pub fn symbols() -> anyhow::Result<&'static MlxSymbols> {
    init()?;
    let state = MLX.get().unwrap().as_ref().unwrap();
    Ok(&state.symbols)
}

pub fn device_count(device_type: MlxDeviceType) -> anyhow::Result<c_int> {
    init()?;
    let state = MLX.get().unwrap().as_ref().unwrap();
    let symbols = &state.symbols;

    unsafe {
        let mut count: c_int = 0;
        let rc = (symbols.mlx_device_count)(&mut count, device_type as c_int);
        if rc != 0 {
            return Err(anyhow::anyhow!("mlx_device_count returned error: {rc}"));
        }
        Ok(count)
    }
}

use std::ffi::c_int;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_without_library() {
        std::env::remove_var("MLX_RS_MLX_LIBRARY");
        std::env::remove_var("OLLAMA_LLM_LIBRARY");

        let result = init();
        if cfg!(target_os = "macos") || cfg!(target_os = "linux") {
            if result.is_ok() {
                let ver = version();
                assert!(ver.is_ok(), "version should work after successful init");
                let ver = ver.unwrap();
                assert!(!ver.is_empty(), "version should not be empty");
            }
        }
    }

    #[test]
    fn test_loaded_library_path_before_init() {
        MLX.get_or_init(|| Err("not initialized".to_string()));
        let result = loaded_library_path();
        assert!(result.is_err());
    }

    #[test]
    fn test_candidate_dirs_returns_values() {
        let dirs = candidate_dirs();
        assert!(dirs.iter().all(|d| d.is_absolute()));
    }

    #[test]
    fn test_lib_names_platform() {
        let names = lib_names();
        assert!(!names.is_empty());
        if cfg!(target_os = "macos") {
            assert!(names.contains(&"libmlxc.dylib"));
        } else if cfg!(target_os = "linux") {
            assert!(names.contains(&"libmlxc.so"));
        }
    }

    #[test]
    fn test_forced_variant_env() {
        std::env::set_var("MLX_RS_MLX_LIBRARY", "mlx_metal_v4");
        let forced = std::env::var("MLX_RS_MLX_LIBRARY").ok();
        assert_eq!(forced.as_deref(), Some("mlx_metal_v4"));
        std::env::remove_var("MLX_RS_MLX_LIBRARY");
    }

    #[test]
    fn test_ollama_compat_env() {
        std::env::set_var("OLLAMA_LLM_LIBRARY", "mlx_metal_v3");
        let v = std::env::var("OLLAMA_LLM_LIBRARY").ok();
        assert!(v.as_deref().unwrap().starts_with("mlx_"));
        std::env::remove_var("OLLAMA_LLM_LIBRARY");

        std::env::set_var("OLLAMA_LLM_LIBRARY", "cpu");
        let v = std::env::var("OLLAMA_LLM_LIBRARY").ok();
        assert!(!v.as_deref().unwrap().starts_with("mlx_"));
        std::env::remove_var("OLLAMA_LLM_LIBRARY");
    }
}
