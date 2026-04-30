use anyhow::Result;
use std::path::Path;

fn main() -> Result<()> {
    let model_id = "unsloth/gemma-4-E2B-it-UD-MLX-4bit";
    let image_path = Path::new("./red_square.png");

    println!("Loading model: {}", model_id);
    let model_dir = mlx_lm::resolve_model_dir(model_id)?;
    let vlm = mlx_vlm::load_gemma4_vlm(&model_dir)?;
    let tokenizer = vlm.tokenizer.clone();

    // Process image
    let processed = vlm.processor.process(image_path)?;
    let pixel_values = vlm.processor.to_array(&processed)?;

    // Get input_ids matching Python exactly
    let template_options = mlx_lm::ChatTemplateOptions {
        add_generation_prompt: true,
        continue_final_message: false,
        enable_thinking: false,
    };
    let messages = vec![
        mlx_lm::Message::user("<|image|>Describe this image."),
    ];
    let prompt = mlx_lm::ChatTemplate::from_model_dir(&model_dir)?
        .apply(&messages, &template_options)?;

    let encoding = tokenizer.encode(prompt.clone(), false)
        .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
    let token_ids = encoding.get_ids().to_vec();

    // Expand image token
    let image_token_id = vlm.config.image_token_id as u32;
    let image_open_token_id = tokenizer.token_to_id("<|image>").unwrap_or(image_token_id);
    let image_close_token_id = tokenizer.token_to_id("<image|>").unwrap_or(image_token_id);
    let num_soft_tokens = processed.num_soft_tokens;
    let mut expanded = Vec::new();
    let mut found_image = false;
    for &tid in &token_ids {
        if tid == image_token_id && !found_image {
            expanded.push(image_open_token_id);
            for _ in 0..num_soft_tokens {
                expanded.push(image_token_id);
            }
            expanded.push(image_close_token_id);
            found_image = true;
        } else {
            expanded.push(tid);
        }
    }

    // Print token sequence for comparison
    println!("Prompt text: {:?}", prompt);
    println!("Token count: {}", expanded.len());
    println!("First 10 tokens: {:?}", &expanded[..10.min(expanded.len())]);
    println!("Tokens around image: {:?}", &expanded[3..12.min(expanded.len())]);
    println!("Last 10 tokens: {:?}", &expanded[expanded.len().saturating_sub(10)..]);

    // Print vision embeddings shape
    let mut model = vlm.model;
    let vision_emb = model.encode_image(&pixel_values)?;
    println!("Vision embeddings shape: {:?}", vision_emb.shape_raw());

    // Print first few vision embedding values
    let flat = vision_emb.reshape(&[-1])?;
    let vec = flat.to_vec_f32()?;
    println!("First 20 vision embedding values: {:?}", &vec[..20.min(vec.len())]);

    // Run forward_logits and print top predictions for first new token
    let prompt_i32: Vec<i32> = expanded.iter().map(|&x| x as i32).collect();
    let input_ids = mlx_core::Array::from_slice_i32(&prompt_i32)?
        .reshape(&[1, prompt_i32.len() as i32])?;

    let logits = model.forward_logits(&input_ids, Some(&pixel_values))?;
    let last_logits = logits.slice(&[0i32, -1i32, 0i32], &[1i32, logits.dim(1) as i32, logits.dim(2) as i32])?
        .squeeze(1)?;
    
    // Get top 10 tokens
    let sorted = last_logits.argsort(false)?;
    let top_indices = sorted.slice(&[0i32], &[1i32])?.squeeze(0)?;
    let top_10 = top_indices.slice(&[0i32], &[10i32])?.to_vec_i32()?;
    println!("Top 10 predicted token IDs: {:?}", top_10);
    
    // Print their logits values
    for &tid in &top_10 {
        let val = last_logits.slice(&[tid], &[tid + 1])?.to_vec_f32()?;
        println!("  Token {} -> logit = {}", tid, val[0]);
    }

    Ok(())
}
