use anyhow::Result;
use clap::Parser;

/// MLX-RS local chat server (llama.cpp backend - not yet implemented).
#[derive(Parser, Debug)]
#[command(name = "mlx-server", about = "Start local MLX chat server")]
struct Args {
    /// Path to TOML config file
    #[arg(long, default_value = "config.toml")]
    config: std::path::PathBuf,

    /// Override bind address
    #[arg(long)]
    bind: Option<String>,

    /// Override port
    #[arg(long)]
    port: Option<u16>,

    /// Override startup model path
    #[arg(long)]
    model: Option<String>,

    /// Optional API key
    #[arg(long)]
    api_key: Option<String>,

    /// Optional rate limit (requests per minute)
    #[arg(long)]
    rate_limit_rpm: Option<u32>,

    /// Enable/disable thinking mode
    #[arg(long)]
    thinking: Option<bool>,
}

fn main() -> Result<()> {
    let _args = Args::parse();
    eprintln!("mlx-rs is being rewritten around llama.cpp/GGUF. Server is not yet implemented.");
    std::process::exit(1);
}
