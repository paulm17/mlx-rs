use std::env;
use std::path::PathBuf;

fn main() {
    // Link against the MLX C library
    println!("cargo:rustc-link-lib=mlxc");
    println!("cargo:rustc-link-search=/opt/homebrew/lib");

    // Re-run if the header changes
    println!("cargo:rerun-if-changed=/opt/homebrew/include/mlx/c/mlx.h");

    let bindings = bindgen::Builder::default()
        .header("/opt/homebrew/include/mlx/c/mlx.h")
        .clang_arg("-I/opt/homebrew/include")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings for mlx.c");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
