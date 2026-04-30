use anyhow::Result;
use mlx_core::{Array, Module};

fn compare(name: &str, rust: &Array, py_path: &str, py_key: &str) -> Result<()> {
    let py_weights = mlx_core::safetensors::load(py_path)?;
    let py = py_weights.get(py_key).unwrap().clone();
    
    println!("\n=== {} ===", name);
    println!("Rust shape: {:?} dtype: {:?}", rust.shape_raw(), rust.dtype());
    println!("Python shape: {:?} dtype: {:?}", py.shape_raw(), py.dtype());
    
    let rust_f = rust.reshape(&[-1])?.to_vec_f32()?;
    let py_f = py.reshape(&[-1])?.to_vec_f32()?;
    
    if rust_f.len() != py_f.len() {
        println!("LENGTH MISMATCH: {} vs {}", rust_f.len(), py_f.len());
        return Ok(());
    }
    
    let mut mse = 0.0f64;
    let mut max_diff = 0.0f32;
    let mut max_idx = 0;
    let mut matching = 0;
    
    for i in 0..rust_f.len() {
        let diff = (rust_f[i] - py_f[i]).abs();
        if diff > max_diff { max_diff = diff; max_idx = i; }
        mse += (diff as f64).powi(2);
        if diff < 0.001 { matching += 1; }
    }
    mse /= rust_f.len() as f64;
    
    println!("MSE: {:.6}", mse);
    println!("Max diff: {} at index {} (rust={} py={})", max_diff, max_idx, rust_f[max_idx], py_f[max_idx]);
    println!("Matching (<0.001): {} / {} ({:.1}%)", matching, rust_f.len(), 100.0 * matching as f64 / rust_f.len() as f64);
    
    Ok(())
}

fn main() -> Result<()> {
    let model_id = "unsloth/gemma-4-E2B-it-UD-MLX-4bit";
    println!("Loading model: {}", model_id);
    let model_dir = mlx_lm::resolve_model_dir(model_id)?;
    let mut vlm = mlx_vlm::load_gemma4_vlm(&model_dir)?;
    
    // Load Python pixel_values and input_ids
    let pv_weights = mlx_core::safetensors::load("/tmp/python_pixel_values.safetensors")?;
    let pixel_values = pv_weights.get("pixel_values").unwrap().clone();
    let ids_weights = mlx_core::safetensors::load("/tmp/python_input_ids.safetensors")?;
    let input_ids = ids_weights.get("input_ids").unwrap().clone().as_type(mlx_core::DType::Int32)?;
    
    println!("input_ids shape: {:?}", input_ids.shape_raw());
    println!("pixel_values shape: {:?}", pixel_values.shape_raw());
    
    // 1. text_embeddings (pre-scatter)
    let text_emb = {
        let raw = vlm.model.language_model.model.embed_tokens.forward(&input_ids)?;
        raw.multiply(&Array::from_float(vlm.model.language_model.model.embed_scale)?)?
    };
    compare("text_embeddings", &text_emb, "/tmp/python_text_embeddings.safetensors", "text_embeddings")?;
    
    // 2. vision_tower_output
    let vision_out = vlm.model.vision_tower.forward(&pixel_values)?;
    compare("vision_tower_output", &vision_out, "/tmp/python_vision_tower_output.safetensors", "vision_tower_output")?;
    
    // 3. vision_embeddings (after projector)
    let vision_emb = vlm.model.embed_vision.forward(&vision_out)?;
    compare("vision_embeddings", &vision_emb, "/tmp/python_vision_embeddings.safetensors", "vision_embeddings")?;
    
    // 4. combined_embeddings (after scatter)
    let image_token_id = vlm.model.config().image_token_id as i32;
    let image_mask = input_ids.equal(&Array::from_int(image_token_id)?)?;
    let image_mask_expanded = image_mask.expand_dims(-1)?.broadcast_to(&text_emb.shape_raw())?;
    let combined = mlx_models::gemma4::Gemma4::masked_scatter(&text_emb, &image_mask_expanded, &vision_emb)?;
    compare("combined_embeddings", &combined, "/tmp/python_combined_embeddings.safetensors", "combined_embeddings")?;
    
    // 5. final_logits
    let logits = vlm.model.forward_logits(&input_ids, Some(&pixel_values))?;
    compare("final_logits", &logits, "/tmp/python_final_logits.safetensors", "final_logits")?;
    
    Ok(())
}
