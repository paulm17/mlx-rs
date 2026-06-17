use anyhow::Result;
use clap::Parser;

// Ensure mlx-backend is linked and its ctor registers the safetensors factory
#[allow(unused_imports)]
use mlx_backend;

/// Local chat server powered by llama.cpp and GGUF models.
#[derive(Parser, Debug)]
#[command(name = "llama-server", about = "Start local llama.cpp/GGUF chat server")]
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
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();

    let mut config = llama_lm::ServerConfig::from_toml_path(&args.config)?;

    if let Some(bind) = args.bind {
        config.bind = Some(bind);
    }
    if let Some(port) = args.port {
        config.port = Some(port);
    }
    if let Some(model) = args.model {
        config.model_path = Some(model);
    }
    if let Some(api_key) = args.api_key {
        config.api_key = Some(api_key);
    }
    if let Some(rpm) = args.rate_limit_rpm {
        config.rate_limit_rpm = Some(rpm);
    }

    llama_lm::run_server(config).await
}
