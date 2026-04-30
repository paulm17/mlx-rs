use anyhow::Result;
use std::path::Path;

fn main() -> Result<()> {
    let model_id = "unsloth/gemma-4-E2B-it-UD-MLX-4bit";
    let image_path = Path::new("./red_square.png");

    println!("Loading model: {}", model_id);
    let model_dir = mlx_lm::resolve_model_dir(model_id)?;
    let mut vlm = mlx_vlm::load_gemma4_vlm(&model_dir)?;

    println!("Processing image {:?}...", image_path);
    let processed = vlm.processor.process(image_path)?;
    println!("Processed image: {}x{}", processed.width, processed.height);
    println!("num_soft_tokens: {}", processed.num_soft_tokens);

    let pixel_values = vlm.processor.to_array(&processed)?;
    println!("pixel_values shape: {:?}", pixel_values.shape_raw());

    // Run encode_image (vision tower + projection)
    let encoded = vlm.model.encode_image(&pixel_values)?;
    println!("encoded shape: {:?}", encoded.shape_raw());

    let flat = encoded.reshape(&[-1])?;
    let vec = flat.to_vec_f32()?;
    let min_v = vec.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_v = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean_v = vec.iter().sum::<f32>() / vec.len() as f32;
    println!("encoded min/max/mean: {} / {} / {}", min_v, max_v, mean_v);

    Ok(())
}
