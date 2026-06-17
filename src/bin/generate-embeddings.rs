use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use llama_lm::config::LlamaCppConfig;
use llama_lm::GenerationPipeline;

#[allow(unused_imports)]
use mlx_backend;

#[derive(Parser, Debug)]
#[command(name = "generate-embeddings", about = "Generate embeddings with encoder models")]
struct Args {
    #[arg(long)]
    model: String,

    #[arg(long, default_value = "config.toml")]
    config: PathBuf,

    #[arg(long, default_value = "Hello, how are you?")]
    prompt: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let model_path = &args.model;

    let llamacpp_config = if args.config.exists() {
        let toml_content = std::fs::read_to_string(&args.config)?;
        let server_config = llama_lm::ServerConfig::from_toml_str(&toml_content)?;
        server_config.to_llamacpp_config()
    } else {
        LlamaCppConfig {
            n_ctx: Some(512),
            ..Default::default()
        }
    };

    eprintln!("Loading model...");
    let mut pipeline = GenerationPipeline::new(model_path, llamacpp_config)?;
    eprintln!("Model loaded.");

    let backend = pipeline.backend_mut();
    if !backend.embeddings_enabled() {
        anyhow::bail!("Model does not support embeddings");
    }

    eprintln!("Generating embeddings for: {:?}", args.prompt);
    let start = std::time::Instant::now();
    let output = backend.embed(&args.prompt)?;
    let elapsed = start.elapsed();

    let embedding_dim = output.data[0].embedding.len();
    eprintln!("Embedding dimension: {}", embedding_dim);
    eprintln!("Time: {:.3}s", elapsed.as_secs_f64());
    eprintln!("Tokens: {}", output.usage.prompt_tokens);

    let preview_len = 10.min(embedding_dim);
    let preview: Vec<String> = output.data[0].embedding[..preview_len]
        .iter()
        .map(|v| format!("{:.6}", v))
        .collect();
    println!("Embedding (first {} values): [{}]", preview_len, preview.join(", "));

    Ok(())
}
