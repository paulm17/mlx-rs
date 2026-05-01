use anyhow::Result;
use mlx_core::Module;
use std::collections::HashMap;
use std::path::Path;

fn compare_tensor(
    name: &str,
    rust_arr: &mlx_core::Array,
    py_dict: &HashMap<String, mlx_core::Array>,
) -> Result<()> {
    let py_arr = match py_dict.get(name) {
        Some(a) => a,
        None => {
            println!("{name}: PYTHON TENSOR MISSING");
            return Ok(());
        }
    };
    let rust_f32 = rust_arr.to_vec_f32()?;
    let py_f32 = py_arr.to_vec_f32()?;

    if rust_f32.len() != py_f32.len() {
        println!(
            "{name}: SHAPE MISMATCH rust={} py={}",
            rust_f32.len(),
            py_f32.len()
        );
        return Ok(());
    }

    let n = rust_f32.len();
    let mut mse = 0.0f64;
    let mut max_diff = 0.0f64;
    let mut max_idx = 0;
    for j in 0..n {
        let d = (rust_f32[j] as f64 - py_f32[j] as f64).abs();
        mse += d * d;
        if d > max_diff {
            max_diff = d;
            max_idx = j;
        }
    }
    mse /= n as f64;

    let matching = rust_f32
        .iter()
        .zip(py_f32.iter())
        .filter(|(r, p)| (**r - **p).abs() < 0.001)
        .count();

    let pct = 100.0 * matching as f64 / n as f64;
    // Short-format: one line per layer
    println!(
        "  {name:<25} MSE={mse:>12.6}  max={max_diff:>10.4}  match={pct:>5.1}%",
    );
    Ok(())
}

fn main() -> Result<()> {
    let model_dir = "/Volumes/Data/Users/paul/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it-UD-MLX-4bit/snapshots/3236b6b700bae91f3045cf0f4f0c12595530f182";
    let ref_dir = std::path::Path::new("/tmp");

    // Load model
    let vlm = mlx_vlm::load_gemma4_vlm(Path::new(model_dir))?;
    let mut model = vlm.model;

    // Load Python layer outputs
    let py_layers = mlx_core::safetensors::load(Path::new("/tmp/python_llm_layers.safetensors"))?;

    // Load input tensors (safetensors stores them as float32, cast to int32)
    let input_ids = mlx_core::safetensors::load(&ref_dir.join("python_input_ids.safetensors"))?
        .remove("input_ids")
        .unwrap()
        .as_type(mlx_core::DType::Int32)?;
    let pixel_values =
        mlx_core::safetensors::load(&ref_dir.join("python_pixel_values.safetensors"))?
            .remove("pixel_values")
            .unwrap();

    // Compute embeddings matching compute_embeddings
    eprintln!("embed_tokens...");
    let mut h = model
        .language_model
        .model
        .embed_tokens
        .forward(&input_ids)?;
    eprintln!("embed_scale multiply...");
    h = h.multiply(&mlx_core::Array::from_float(
        model.language_model.model.embed_scale,
    )?)?;

    // Vision
    eprintln!("vision_tower...");
    let vision_features = model.vision_tower.forward(&pixel_values)?;
    eprintln!("embed_vision...");
    let vision_emb = model.embed_vision.forward(&vision_features)?;
    eprintln!("image_token_id_arr build...");
    let image_token_id_arr = mlx_core::Array::from_int(model.config().image_token_id as i32)?;
    eprintln!("input_ids.equal...");
    let image_mask = input_ids.equal(&image_token_id_arr)?;
    eprintln!("expand/broadcast...");
    let image_mask_expanded = image_mask
        .expand_dims(-1)?
        .broadcast_to(&h.shape_raw())?;
    eprintln!("masked_scatter...");
    h = mlx_models::gemma4::Gemma4::masked_scatter(
        &h,
        &image_mask_expanded,
        &vision_emb,
    )?;
    eprintln!("masked_scatter done");

    // Per-layer inputs (matching Python's Gemma4TextModel.__call__)
    let per_layer_inputs =
        if model.language_model.model.embed_tokens_per_layer.is_some() {
            eprintln!("computing per_layer_inputs...");
            let pli = model
                .language_model
                .model
                .compute_per_layer_inputs(&input_ids, &h)?;
            Some(pli)
        } else {
            None
        };

    // Compare layer_input (before any LLM layers)
    compare_tensor("layer_input", &h, &py_layers)?;
    println!("Layer input compared OK. Running layers...");

    // Run each layer and compare output
    let seq_len = h.shape_raw()[1] as usize;
    let batch = h.shape_raw()[0] as usize;
    let sliding_window = model.language_model.model.sliding_window;

    for (i, layer) in model.language_model.model.layers.iter_mut().enumerate() {
        eprintln!("  layer {i}...");
        let cache_idx = model.language_model.model.layer_idx_to_cache_idx[i];
        let cache = &mut model.language_model.model.caches[cache_idx];

        let mask = if layer.is_sliding {
            eprintln!("    computing sliding window mask (offset={})", cache.offset());
            let offset = cache.offset();
            Some(mlx_models::gemma4::sliding_window_mask(
                batch,
                seq_len,
                offset,
                sliding_window,
                h.dtype(),
            )?)
        } else {
            eprintln!("    global attention, no mask");
            None
        };

        let layer_emb = per_layer_inputs
            .as_ref()
            .map(|pli| {
                let start = vec![0i32, 0, i as i32, 0];
                let mut stop = pli.shape_raw();
                stop[2] = (i + 1) as i32;
                pli.slice(&start, &stop).unwrap().squeeze(2).unwrap()
            });
        h = layer.forward(&h, mask.as_ref(), cache, layer_emb.as_ref())?;
        compare_tensor(&format!("layer_{i}_output"), &h, &py_layers)?;
    }

    // Final norm
    eprintln!("  final norm...");
    let h_norm = model.language_model.model.norm.forward(&h)?;
    compare_tensor("final_norm_output", &h_norm, &py_layers)?;

    println!("\nDone.");
    Ok(())
}
