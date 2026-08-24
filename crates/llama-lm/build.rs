use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let source = env::var_os("MLX_RS_LLAMA_CPP_SYS_SOURCE")
        .map(PathBuf::from)
        .or_else(|| {
            let cargo_home = env::var_os("CARGO_HOME")
                .map(PathBuf::from)
                .or_else(|| dirs_fallback())?;
            let registry = cargo_home.join("registry/src");
            find_source(&registry)
        })
        .expect("could not locate llama-cpp-sys-2 source; set MLX_RS_LLAMA_CPP_SYS_SOURCE");
    let llama = source.join("llama.cpp");

    println!("cargo:rerun-if-changed=llama_compat.cpp");
    println!("cargo:rerun-if-env-changed=MLX_RS_LLAMA_CPP_SYS_SOURCE");

    cc::Build::new()
        .cpp(true)
        .file("llama_compat.cpp")
        .include(&source)
        .include(&llama)
        .include(llama.join("common"))
        .include(llama.join("include"))
        .include(llama.join("ggml/include"))
        .include(llama.join("vendor"))
        .flag_if_supported("-std=c++17")
        .compile("mlx_rs_llama_compat");
}

fn dirs_fallback() -> Option<PathBuf> {
    #[cfg(unix)]
    {
        std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cargo"))
    }
    #[cfg(not(unix))]
    {
        None
    }
}

fn find_source(registry: &PathBuf) -> Option<PathBuf> {
    for hash_dir in fs::read_dir(registry).ok()?.flatten() {
        let hash_path = hash_dir.path();
        if !hash_path.is_dir() {
            continue;
        }
        for package in fs::read_dir(hash_path).ok()?.flatten() {
            let path = package.path();
            if path.file_name().and_then(|name| name.to_str()) == Some("llama-cpp-sys-2-0.1.154")
                && path.join("llama.cpp/common/chat.h").exists()
            {
                return Some(path);
            }
        }
    }
    None
}
