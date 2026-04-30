use anyhow::Result;

fn main() -> Result<()> {
    let model_dir = "/Volumes/Data/Users/paul/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it-UD-MLX-4bit/snapshots/3236b6b700bae91f3045cf0f4f0c12595530f182";
    let tokenizer = tokenizers::Tokenizer::from_file(model_dir.to_string() + "/tokenizer.json")
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
    
    for s in &["<|image|>", "<|image>", "<image|>", "<|boi|>", "<|eoi|>", "<|turn|>", "<turn|>", "<bos>"] {
        let encoding = tokenizer.encode(s.to_string(), false)
            .map_err(|e| anyhow::anyhow!("encode failed: {e}"))?;
        let ids = encoding.get_ids();
        println!("{:?} -> {:?}", s, ids);
    }
    
    Ok(())
}
