use anyhow::Result;

fn main() -> Result<()> {
    let model_dir = mlx_lm::resolve_model_dir("unsloth/gemma-4-E2B-it-UD-MLX-4bit")?;
    let mut vlm = mlx_vlm::load_gemma4_vlm(&model_dir)?;

    // Load Python pre-saved pixel_values and input_ids
    let pv_weights = mlx_core::safetensors::load("/tmp/python_pixel_values.safetensors")?;
    let pixel_values = pv_weights.get("pixel_values").unwrap().clone();
    let ids_weights = mlx_core::safetensors::load("/tmp/python_input_ids.safetensors")?;
    let input_ids = ids_weights.get("input_ids").unwrap().clone().as_type(mlx_core::DType::Int32)?;

    println!("=== Rust prefill logits (first token) ===");
    let logits = vlm.model.forward_logits(&input_ids, Some(&pixel_values))?;
    let lshape = logits.shape_raw();
    let last_logits = logits.slice(&[0, lshape[1] - 1, 0], &[1, lshape[1], lshape[2]])?;
    let flat: Vec<f32> = last_logits.reshape(&[-1])?.to_vec_f32()?;

    let mut indexed: Vec<(usize, f32)> = flat.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Top-10 tokens:");
    for (i, &(tok, val)) in indexed.iter().take(10).enumerate() {
        println!("  {}: token_id={} logit={:.6}", i + 1, tok, val);
    }
    println!("Argmax: token_id={} logit={:.6}", indexed[0].0, indexed[0].1);

    // Now load Python logits for comparison
    let py_logits = mlx_core::safetensors::load("/tmp/python_final_logits.safetensors")?;
    let py_arr = py_logits.get("final_logits").unwrap();
    let py_lshape = py_arr.shape_raw();
    let py_last = py_arr.slice(&[0, py_lshape[1] - 1, 0], &[1, py_lshape[1], py_lshape[2]])?;
    let py_flat: Vec<f32> = py_last.reshape(&[-1])?.to_vec_f32()?;

    let mut py_indexed: Vec<(usize, f32)> = py_flat.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    py_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\n=== Python prefill logits (first token) ===");
    for (i, &(tok, val)) in py_indexed.iter().take(10).enumerate() {
        let rust_val = flat[tok];
        println!("  {}: token_id={} py_logit={:.6} rust_logit={:.6} diff={:.6}", 
            i + 1, tok, val, rust_val, val - rust_val);
    }
    println!("Python argmax: token_id={} logit={:.6}", py_indexed[0].0, py_indexed[0].1);

    // Check if Rust's argmax token appears in Python's top-N
    let rust_argmax = indexed[0].0;
    let py_rank = py_indexed.iter().position(|&(tok, _)| tok == rust_argmax);
    println!("\nRust argmax token {} rank in Python top: {:?}", rust_argmax, py_rank.map(|r| r + 1));
    let py_argmax = py_indexed[0].0;
    let rust_rank = indexed.iter().position(|&(tok, _)| tok == py_argmax);
    println!("Python argmax token {} rank in Rust top: {:?}", py_argmax, rust_rank.map(|r| r + 1));

    Ok(())
}
