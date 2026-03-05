use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "mlx-server", about = "Start local MLX chat server")]
struct Args {
    /// Path to TOML config file
    #[arg(long, default_value = "config.toml")]
    config: std::path::PathBuf,

    /// Override bind address (e.g. 0.0.0.0:3000)
    #[arg(long)]
    bind: Option<String>,

    /// Override port (used when --bind is not set)
    #[arg(long)]
    port: Option<u16>,

    /// Override startup model path
    #[arg(long)]
    model: Option<String>,

    /// Optional API key (x-api-key or Authorization: Bearer)
    #[arg(long)]
    api_key: Option<String>,

    /// Optional global chat rate limit (requests per minute), 0 disables
    #[arg(long)]
    rate_limit_rpm: Option<u32>,

    /// Enable/disable thinking mode in chat templates
    #[arg(long)]
    thinking: Option<bool>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut cfg = mlx_lm::ServerConfig::from_toml_path(&args.config)?;

    if let Some(v) = args.bind {
        cfg.bind = Some(v);
    }
    if let Some(v) = args.port {
        cfg.port = Some(v);
    }
    if let Some(v) = args.model {
        cfg.model_path = Some(v);
        cfg.model = None;
    }
    if let Some(v) = args.api_key {
        cfg.api_key = Some(v);
    }
    if let Some(v) = args.rate_limit_rpm {
        cfg.rate_limit_rpm = Some(v);
    }
    if let Some(v) = args.thinking {
        cfg.thinking = Some(v);
    }

    mlx_lm::run_server(cfg)
}
