use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use mlx_lm::config::LlamaCppConfig;
use mlx_lm::{resolve_model_path, GenerationOptions, GenerationPipeline};

#[derive(Parser, Debug)]
#[command(name = "generate", about = "Generate text with GGUF models via llama.cpp")]
struct Args {
    #[arg(long)]
    model: String,

    #[arg(long, default_value = "config.toml")]
    config: PathBuf,

    #[arg(long, default_value = "Hello, how are you?")]
    prompt: String,

    #[arg(long)]
    max_tokens: Option<usize>,

    #[arg(long, default_value_t = 0.6)]
    temperature: f32,

    #[arg(long, default_value_t = 0.9)]
    top_p: f32,

    #[arg(long, default_value_t = false)]
    chat: bool,

    #[arg(long, default_value = "You are a helpful assistant.")]
    system_prompt: String,
}

fn build_chat_prompt(system: &str, user: &str) -> String {
    let mut s = String::new();
    s.push_str("[INST] <<SYS>>\n");
    s.push_str(system);
    s.push_str("\n<</SYS>>\n\n");
    s.push_str(user);
    s.push_str(" [/INST]");
    s
}

fn main() -> Result<()> {
    let args = Args::parse();

    let model_path = resolve_model_path(&args.model)?;

    let llamacpp_config = if args.config.exists() {
        let toml_content = std::fs::read_to_string(&args.config)?;
        let server_config = mlx_lm::ServerConfig::from_toml_str(&toml_content)?;
        server_config.to_llamacpp_config()
    } else {
        LlamaCppConfig {
            n_ctx: Some(4096),
            ..Default::default()
        }
    };

    let mut pipeline = GenerationPipeline::new(
        model_path.to_str().unwrap(),
        llamacpp_config,
    )?;

    let prompt = if args.chat {
        build_chat_prompt(&args.system_prompt, &args.prompt)
    } else {
        args.prompt
    };

    let options = GenerationOptions {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        ..Default::default()
    };

    let output = pipeline.generate(&prompt, &options)?;

    println!("{}", output.text);
    eprintln!("\n--- Metrics ---");
    eprintln!("Prompt tokens: {}", output.metrics.prompt_tokens);
    eprintln!("Generated tokens: {}", output.metrics.generated_tokens);
    eprintln!("Total tokens: {}", output.metrics.total_tokens);
    if let Some(ttft) = output.metrics.ttft_s {
        eprintln!("Time to first token: {:.3}s", ttft);
    }
    if let Some(total) = output.metrics.total_s {
        eprintln!("Total time: {:.3}s", total);
    }
    if let Some(tps) = output.metrics.tokens_per_s {
        eprintln!("Tokens/sec: {:.2}", tps);
    }
    eprintln!("Stop reason: {:?}", output.stop_reason);

    Ok(())
}
