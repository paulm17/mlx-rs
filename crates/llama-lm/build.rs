use std::path::PathBuf;

fn main() {
    // Must match the workspace's patched sys crate, including for FFI headers.
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vendor/llama-cpp-sys-2");
    let llama = source.join("llama.cpp");
    for file in ["llama_compat.cpp", "clef_compat.cpp", "decision_compat.cpp"] {
        println!("cargo:rerun-if-changed={file}");
    }
    println!(
        "cargo:rerun-if-changed={}",
        llama.join("src/llama-ext.h").display()
    );
    cc::Build::new()
        .cpp(true)
        .file("llama_compat.cpp")
        .file("clef_compat.cpp")
        .file("decision_compat.cpp")
        .include(&source)
        .include(&llama)
        .include(llama.join("common"))
        .include(llama.join("include"))
        .include(llama.join("ggml/include"))
        .include(llama.join("vendor"))
        .flag_if_supported("-std=c++17")
        .compile("mlx_rs_llama_compat");
}
