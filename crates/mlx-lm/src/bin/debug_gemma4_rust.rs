use mlx_core::Array;
use mlx_lm::loader::load_model;

fn main() {
    let model_dir = "/Volumes/Data/Users/paul/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it-UD-MLX-4bit/snapshots/3236b6b700bae91f3045cf0f4f0c12595530f182";
    
    println!("Loading model from {}...", model_dir);
    let (mut model, _tokenizer) = load_model(std::path::Path::new(model_dir)).unwrap();
    println!("Model loaded");
    
    // Same input IDs as Python: [2, 105, 2364, 107, 818, 3823, 8864, 37423, 106, 107, 105, 4368, 107]
    let input_ids_vec = vec![2i32, 105, 2364, 107, 818, 3823, 8864, 37423, 106, 107, 105, 4368, 107];
    let input_ids = Array::from_slice_i32(&input_ids_vec).unwrap().reshape(&[1, 13]).unwrap();
    
    println!("Running forward with input_ids shape {:?}", input_ids.shape_raw());
    
    // Use CausalLM trait for last token
    use mlx_lm::generate::CausalLM;
    let last_logits = model.forward_last_token_logits(&input_ids).unwrap();
    
    println!("Last logits shape: {:?}", last_logits.shape_raw());
    
    // Save last logits for comparison
    let flat = last_logits.reshape(&[262144i32]).unwrap();
    let _ = flat.eval();
    let vec = flat.to_vec_f32().unwrap();
    let mean = vec.iter().sum::<f32>() / vec.len() as f32;
    let variance = vec.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / vec.len() as f32;
    println!("Last logits mean: {:.6}, std: {:.6}", mean, variance.sqrt());
    
    // Find top 10 manually
    let mut indexed: Vec<(usize, f32)> = vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Top-10 tokens: {:?}", indexed[..10].iter().map(|(i, _)| *i).collect::<Vec<_>>());
    println!("Top-10 logits: {:?}", indexed[..10].iter().map(|(_, v)| *v).collect::<Vec<_>>());
    
    println!("Done");
}
