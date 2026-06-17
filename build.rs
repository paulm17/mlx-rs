//! Build script for llama-rs.
//!
//! On macOS ARM64, if `LLAMA_RS_BUILD_MLX=1` is set and `libmlxc.dylib` is not
//! already available, this script clones and builds the MLX-C runtime following
//! Ollama's conventions:
//!
//!   - Pinned commits from MLX_VERSION / MLX_C_VERSION
//!   - Output placed in build/lib/ollama/ (found by the mlx-backend loader)
//!
//! Set `LLAMA_RS_BUILD_MLX=1` to enable automatic build.
//! Set `LLAMA_RS_MLX_SOURCE` / `LLAMA_RS_MLX_C_SOURCE` to use local source dirs.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    // Only build MLX-C on macOS ARM64 with opt-in
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    if target_os != "macos" || target_arch != "aarch64" {
        return;
    }

    let build_mlx = std::env::var("LLAMA_RS_BUILD_MLX")
        .map(|v| v == "1")
        .unwrap_or(false);
    if !build_mlx {
        return;
    }

    // Check if already built
    let lib_dir = manifest_dir.join("build").join("lib").join("ollama");
    let dylib_path = lib_dir.join("libmlxc.dylib");
    if dylib_path.exists() {
        println!("cargo:warning=MLX-C already built at {}", dylib_path.display());
        return;
    }

    // Check prerequisites
    check_prerequisite("cmake", &["--version"]);
    check_prerequisite("git", &["--version"]);

    // Read pinned versions (following Ollama convention)
    let mlx_version = read_version_file(&manifest_dir.join("MLX_VERSION"), "MLX_VERSION");
    let mlx_c_version = read_version_file(&manifest_dir.join("MLX_C_VERSION"), "MLX_C_VERSION");

    // Source directories
    let mlx_src = env_or(
        "LLAMA_RS_MLX_SOURCE",
        out_dir.join("mlx-src"),
    );
    let mlx_c_src = env_or(
        "LLAMA_RS_MLX_C_SOURCE",
        out_dir.join("mlx-c-src"),
    );

    // Clone repos if not already present
    clone_if_needed(
        "https://github.com/ml-explore/mlx.git",
        &mlx_version,
        &mlx_src,
    );
    clone_if_needed(
        "https://github.com/ml-explore/mlx-c.git",
        &mlx_c_version,
        &mlx_c_src,
    );

    // Build directory
    let build_dir = out_dir.join("mlx-c-build");
    let _ = std::fs::create_dir_all(&build_dir);

    // Configure
    println!("cargo:warning=Configuring MLX-C build...");
    let mut cmake = Command::new("cmake");
    cmake.arg("-S").arg(&mlx_c_src);
    cmake.arg("-B").arg(&build_dir);
    cmake.arg("-DCMAKE_BUILD_TYPE=Release");
    cmake.arg("-DBUILD_SHARED_LIBS=ON");
    cmake.arg("-DMLX_BUILD_GGUF=OFF");
    cmake.arg("-DMLX_BUILD_SAFETENSORS=ON");
    cmake.arg("-DMLX_BUILD_METAL=ON");
    cmake.arg("-DMLX_BUILD_CUDA=OFF");
    cmake.arg(format!("-DFETCHCONTENT_SOURCE_DIR_MLX={}", mlx_src.display()));
    cmake.arg(format!("-DFETCHCONTENT_SOURCE_DIR_MLX-C={}", mlx_c_src.display()));
    cmake.arg(format!(
        "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}",
        lib_dir.display()
    ));
    cmake.arg(format!(
        "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={}",
        lib_dir.display()
    ));
    cmake.arg(format!(
        "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={}",
        lib_dir.display()
    ));

    let status = cmake.status().expect("failed to run cmake configure");
    if !status.success() {
        panic!("cmake configure failed");
    }

    // Build
    println!("cargo:warning=Building MLX-C...");
    let num_jobs = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "4".to_string());
    let status = Command::new("cmake")
        .arg("--build").arg(&build_dir)
        .arg("--target").arg("mlxc")
        .arg("--target").arg("mlx")
        .arg("-j").arg(&num_jobs)
        .status()
        .expect("failed to run cmake build");
    if !status.success() {
        panic!("cmake build failed");
    }

    // Verify output
    if !dylib_path.exists() {
        panic!(
            "MLX-C build completed but libmlxc.dylib not found at {}",
            dylib_path.display()
        );
    }

    // Copy metallib if built
    for candidate in &[
        build_dir.join("_deps").join("mlx-build").join("mlx").join("backend").join("metal").join("kernels").join("mlx.metallib"),
        build_dir.join("mlx").join("backend").join("metal").join("kernels").join("mlx.metallib"),
    ] {
        if candidate.exists() {
            let _ = std::fs::copy(candidate, lib_dir.join("mlx.metallib"));
            break;
        }
    }

    println!("cargo:warning=MLX-C built successfully at {}", dylib_path.display());

    // Emit rerun hint so changes to version files trigger rebuild
    println!("cargo:rerun-if-changed=MLX_VERSION");
    println!("cargo:rerun-if-changed=MLX_C_VERSION");
    println!("cargo:rerun-if-env-changed=LLAMA_RS_BUILD_MLX");
    println!("cargo:rerun-if-env-changed=LLAMA_RS_MLX_SOURCE");
    println!("cargo:rerun-if-env-changed=LLAMA_RS_MLX_C_SOURCE");
}

fn check_prerequisite(cmd: &str, args: &[&str]) {
    let status = Command::new(cmd)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    match status {
        Ok(s) if s.success() => {}
        _ => panic!("required tool '{}' not found. Install it to build MLX-C.", cmd),
    }
}

fn read_version_file(path: &PathBuf, name: &str) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("missing {name} file at {}", path.display()))
        .trim()
        .to_string()
}

fn clone_if_needed(repo_url: &str, commit: &str, dest: &PathBuf) {
    if dest.join(".git").exists() {
        println!("cargo:warning=Using existing source at {}", dest.display());
        return;
    }

    println!("cargo:warning=Cloning {}...", repo_url);
    let status = Command::new("git")
        .arg("clone")
        .arg("--depth").arg("1")
        .arg(repo_url)
        .arg(dest)
        .status()
        .expect("failed to run git clone");

    if !status.success() {
        panic!("git clone failed for {}", repo_url);
    }

    // Fetch and checkout the exact pinned commit
    let status = Command::new("git")
        .arg("-C").arg(dest)
        .arg("fetch")
        .arg("--depth").arg("1")
        .arg("origin")
        .arg(commit)
        .status()
        .expect("failed to run git fetch");
    if !status.success() {
        panic!("git fetch failed for commit {}", commit);
    }

    let status = Command::new("git")
        .arg("-C").arg(dest)
        .arg("checkout")
        .arg("FETCH_HEAD")
        .status()
        .expect("failed to run git checkout");
    if !status.success() {
        panic!("git checkout failed for commit {}", commit);
    }

    // Sync submodules if any
    let status = Command::new("git")
        .arg("-C").arg(dest)
        .arg("submodule")
        .arg("update")
        .arg("--init")
        .arg("--depth").arg("1")
        .status()
        .expect("failed to run git submodule update");
    if !status.success() {
        println!("cargo:warning=git submodule update had issues (may be ok)");
    }
}

fn env_or(name: &str, default: PathBuf) -> PathBuf {
    std::env::var(name)
        .ok()
        .map(PathBuf::from)
        .unwrap_or(default)
}
