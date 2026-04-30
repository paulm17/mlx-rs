use anyhow::Result;
use std::path::Path;

fn test_generation(model_dir: &str, image_path: Option<&Path>, label: &str) -> Result<()> {
    println!("\n=== {} ===", label);
    
    let vlm = mlx_vlm::load_gemma4_vlm(Path::new(model_dir))?;
    let tokenizer = vlm.tokenizer.clone();
    let mut pipeline = mlx_vlm::VlmGenerationPipeline::new(vlm.model, tokenizer.clone());

    let config = vlm.config;
    let num_image_tokens = config.vision_soft_tokens_per_image;
    let image_placeholder = "<|image|>".repeat(num_image_tokens);
    let user_message = format!("{}Describe this image in one sentence.", image_placeholder);

    let template_options = mlx_lm::ChatTemplateOptions {
        add_generation_prompt: true,
        continue_final_message: false,
        enable_thinking: false,
    };

    let prompt_text = mlx_lm::ChatTemplate::from_model_dir(Path::new(model_dir))?
        .apply(&[mlx_lm::Message::user(&user_message)], &template_options)?;

    let encoding = tokenizer.encode(prompt_text.clone(), false)
        .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
    let token_ids = encoding.get_ids().to_vec();

    let image_token_count = token_ids.iter().filter(|&&t| t == config.image_token_id).count();
    println!("Image token count: {}", image_token_count);

    let prompt_i32: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();
    let input_ids = mlx_core::Array::from_slice_i32(&prompt_i32)?
        .reshape(&[1, prompt_i32.len() as i32])?;

    let pixel_values = if let Some(path) = image_path {
        println!("Processing image...");
        let processed = vlm.processor.process(path)?;
        Some(vlm.processor.to_array(&processed)?)
    } else {
        None
    };

    let opts = mlx_vlm::VlmGenerateOptions {
        max_tokens: 32,
        temperature: 0.0,
    };

    let tokens = pipeline.generate_tokens(&input_ids, pixel_values.as_ref(), &opts)?;
    let text = tokenizer.decode(&tokens, true)
        .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;

    println!("OUTPUT: {}", text);
    Ok(())
}

fn main() -> Result<()> {
    let model_dir = "/Volumes/Data/Users/paul/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it-UD-MLX-4bit/snapshots/3236b6b700bae91f3045cf0f4f0c12595530f182";
    let image_path = Path::new("./red_sqaure.png");

    // Test 1: With image
    test_generation(model_dir, Some(image_path), "WITH IMAGE")?;

    // Test 2: Without image (same prompt, no pixel values)
    test_generation(model_dir, None, "WITHOUT IMAGE")?;

    Ok(())
}
