use mlx_rs::Array;
use ndarray::ArrayD;
use numpy::PyArrayDyn;
use pyo3::{Python, PyObject};
use std::fs;

fn load_npy(path: &str) -> Array {
    let bytes = fs::read(path).expect("Failed to read npy");
    let arr: ArrayD<f32> = ndarray_npy::read_npy(&bytes[..]).expect("Failed to parse npy");
    let shape: Vec<i32> = arr.shape().iter().map(|&d| d as i32).collect();
    let data = arr.as_standard_layout().as_slice().unwrap().to_vec();
    Array::from_slice(&data, &shape)
}

fn mse(a: &Array, b: &Array) -> f64 {
    let diff = a.subtract(b);
    let sq = diff.multiply(&diff);
    let mean = sq.mean(None, None);
    mean.item::<f64>()
}

fn main() {
    println!("Loading Python intermediates...");
    let pixel_values = load_npy("debug_pixel_values.npy");
    let patch_embed = load_npy("debug_patch_embed.npy");
    let pos_embed = load_npy("debug_pos_embed.npy");
    let after_pos = load_npy("debug_after_pos.npy");
    let encoder_out = load_npy("debug_encoder_out.npy");
    let pooled = load_npy("debug_pooled.npy");
    let projected = load_npy("debug_projected.npy");
    let full_vision = load_npy("debug_full_vision.npy");

    println!("pixel_values shape: {:?}", pixel_values.shape());
    println!("patch_embed shape: {:?}", patch_embed.shape());
    println!("pos_embed shape: {:?}", pos_embed.shape());
    println!("encoder_out shape: {:?}", encoder_out.shape());
    println!("pooled shape: {:?}", pooled.shape());
    println!("projected shape: {:?}", projected.shape());
    println!("full_vision shape: {:?}", full_vision.shape());

    // We don't have Rust equivalents yet, but this at least loads them
    println!("\nSuccessfully loaded all Python intermediates.");
    println!("Now we need to compare with Rust layer outputs.");
}
