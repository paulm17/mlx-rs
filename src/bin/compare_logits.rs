use anyhow::Result;
fn main() -> Result<()> {
    let model_id = "unsloth/gemma-4-E2B-it-UD-MLX-4bit";
    println!("Loading model: {}", model_id);
    let model_dir = mlx_lm::resolve_model_dir(model_id)?;
    let mut vlm = mlx_vlm::load_gemma4_vlm(&model_dir)?;

    // Load Python input_ids and pixel_values
    let ids_weights = mlx_core::safetensors::load("/tmp/python_input_ids.safetensors")?;
    let input_ids = ids_weights.get("input_ids").unwrap().clone();
    let pv_weights = mlx_core::safetensors::load("/tmp/python_pixel_values.safetensors")?;
    let pixel_values = pv_weights.get("pixel_values").unwrap().clone();

    println!("input_ids shape: {:?}", input_ids.shape_raw());
    println!("pixel_values shape: {:?}", pixel_values.shape_raw());
    
    // Check pixel_values stats
    let pv_flat = pixel_values.reshape(&[-1])?.to_vec_f32()?;
    let pv_min = pv_flat.iter().cloned().fold(f32::INFINITY, f32::min);
    let pv_max = pv_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let pv_mean = pv_flat.iter().sum::<f32>() / pv_flat.len() as f32;
    println!("Rust pixel_values min/max/mean: {} / {} / {}", pv_min, pv_max, pv_mean);

    // Compare vision embeddings
    println!("\n--- Comparing vision embeddings ---");
    let rust_vision = vlm.model.encode_image(&pixel_values)?;
    let py_vision_weights = mlx_core::safetensors::load("/tmp/python_vision_out_f32.safetensors")?;
    let py_vision = py_vision_weights.get("vision_out").unwrap();
    println!("Rust vision shape: {:?} dtype: {:?}", rust_vision.shape_raw(), rust_vision.dtype());
    println!("Python vision shape: {:?} dtype: {:?}", py_vision.shape_raw(), py_vision.dtype());
    
    let rust_flat = rust_vision.reshape(&[-1])?.to_vec_f32()?;
    let py_flat = py_vision.reshape(&[-1])?.to_vec_f32()?;
    let rust_nans = rust_flat.iter().filter(|&&x| x.is_nan()).count();
    let py_nans = py_flat.iter().filter(|&&x| x.is_nan()).count();
    println!("Rust vision NaNs: {} / {}", rust_nans, rust_flat.len());
    println!("Python vision NaNs: {} / {}", py_nans, py_flat.len());
    
    let mut mse = 0.0f64;
    let mut max_diff = 0.0f32;
    let mut max_idx = 0;
    for i in 0..rust_flat.len() {
        let diff = (rust_flat[i] - py_flat[i]).abs();
        if diff > max_diff { max_diff = diff; max_idx = i; }
        mse += (rust_flat[i] - py_flat[i]) as f64 * (rust_flat[i] - py_flat[i]) as f64;
    }
    mse /= rust_flat.len() as f64;
    println!("Vision MSE: {}", mse);
    println!("Vision max diff: {} at index {}", max_diff, max_idx);

    // Run forward_logits
    println!("\n--- Comparing final logits ---");
    let logits = vlm.model.forward_logits(&input_ids, Some(&pixel_values))?;
    println!("logits shape: {:?}", logits.shape_raw());

    // Extract last token logits
    let seq_len = logits.dim(1) as i32;
    let vocab_size = logits.dim(2) as i32;
    let last_logits = logits.slice(&[0i32, seq_len - 1, 0i32], &[1i32, seq_len, vocab_size])?
        .squeeze(0)?.squeeze(0)?;
    println!("last_logits shape: {:?}", last_logits.shape_raw());

    // Compute top 10 manually since no argsort
    let flat = last_logits.reshape(&[-1])?;
    let vec = flat.to_vec_f32()?;
    let mut indexed: Vec<(usize, f32)> = vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top10: Vec<i32> = indexed.iter().take(10).map(|(i, _)| *i as i32).collect();
    let top10_vals: Vec<f32> = indexed.iter().take(10).map(|(_, v)| *v).collect();

    println!("Rust top 10 tokens: {:?}", top10);
    println!("Rust top 10 logits: {:?}", top10_vals);

    // Load Python logits for direct comparison
    let py_weights = mlx_core::safetensors::load("/tmp/python_logits.safetensors")?;
    let py_last = py_weights.get("last").unwrap();
    let py_flat = py_last.reshape(&[-1])?;
    let py_vec = py_flat.to_vec_f32()?;
    let mut py_indexed: Vec<(usize, f32)> = py_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    py_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let py_top10: Vec<i32> = py_indexed.iter().take(10).map(|(i, _)| *i as i32).collect();
    let py_top10_vals: Vec<f32> = py_indexed.iter().take(10).map(|(_, v)| *v).collect();

    println!("Python top 10 tokens: {:?}", py_top10);
    println!("Python top 10 logits: {:?}", py_top10_vals);

    // Check if they match
    if top10 == py_top10 {
        println!("TOP 10 MATCH!");
    } else {
        println!("TOP 10 MISMATCH!");
        println!("Matching tokens: {}", top10.iter().zip(py_top10.iter()).filter(|(a,b)| a==b).count());
    }

    // Compute MSE
    let mut mse = 0.0f64;
    for i in 0..vec.len() {
        let diff = (vec[i] - py_vec[i]) as f64;
        mse += diff * diff;
    }
    mse /= vec.len() as f64;
    println!("MSE between Rust and Python logits: {}", mse);

    Ok(())
}
