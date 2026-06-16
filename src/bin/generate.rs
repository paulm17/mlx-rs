use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

/// MLX-RS text generation CLI (llama.cpp backend - not yet implemented).
#[derive(Parser, Debug)]
#[command(name = "generate", about = "Generate text with MLX models")]
struct Args {
    /// Model identifier: local path or Hugging Face repo ID
    #[arg(long)]
    model: String,

    /// Path to TOML config file
    #[arg(long, default_value = "config.toml")]
    config: PathBuf,

    /// The prompt to generate from
    #[arg(long, default_value = "Hello, how are you?")]
    prompt: String,

    /// Maximum number of tokens to generate
    #[arg(long)]
    max_tokens: Option<usize>,

    /// Sampling temperature (0 = greedy)
    #[arg(long, default_value_t = 0.6)]
    temperature: f32,

    /// Top-p (nucleus) sampling threshold
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,

    /// Use chat template
    #[arg(long, default_value_t = false)]
    chat: bool,

    /// System prompt for chat mode
    #[arg(long, default_value = "You are a helpful assistant.")]
    system_prompt: String,
}

fn main() -> Result<()> {
    let _args = Args::parse();
    eprintln!("mlx-rs is being rewritten around llama.cpp/GGUF. Generation is not yet implemented.");
    std::process::exit(1);
}
