use anyhow::Result;
use clap::Parser;
use serde_json::json;
use std::io::Write;
use std::path::PathBuf;

fn sampler_for_model(model_dir: &std::path::Path, temperature: f32, top_p: f32) -> mlx_lm::Sampler {
    let mut sampler = mlx_lm::Sampler::new(temperature, top_p);
    let config_path = model_dir.join("config.json");
    if let Ok(config_str) = std::fs::read_to_string(&config_path) {
        if let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_str) {
            if let Ok(arch) = mlx_lm::loader::detect_architecture(&config) {
                if matches!(
                    arch,
                    mlx_lm::loader::ModelArch::QwenMoe
                        | mlx_lm::loader::ModelArch::QwenMoePythonPort
                ) {
                    sampler = sampler
                        .with_greedy_tie_break(0.05)
                        .with_greedy_tie_break_after(180);
                }
            }
        }
    }
    sampler
}

/// MLX-RS text generation CLI.
#[derive(Parser, Debug)]
#[command(name = "generate", about = "Generate text with MLX models")]
struct Args {
    /// Path to the model directory
    #[arg(long)]
    model_dir: PathBuf,

    /// The prompt to generate from
    #[arg(long, default_value = "Hello")]
    prompt: String,

    /// Maximum number of tokens to generate (unset = no cap, stop on EOS)
    #[arg(long)]
    max_tokens: Option<usize>,

    /// Sampling temperature (0 = greedy)
    #[arg(long, default_value_t = 0.6)]
    temperature: f32,

    /// Top-p (nucleus) sampling threshold
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,

    /// Use chat template from tokenizer_config.json
    #[arg(long, default_value_t = false)]
    chat: bool,

    /// System prompt for chat mode
    #[arg(long, default_value = "You are a helpful assistant.")]
    system_prompt: String,

    /// Enable/disable thinking mode in chat templates
    #[arg(long, value_parser = clap::value_parser!(bool))]
    thinking: Option<bool>,

    /// Optional JSON dump path for prompt/tokens/output diagnostics
    #[arg(long)]
    dump_json_out: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    eprintln!("Loading model from {:?}...", args.model_dir);
    let (model, tokenizer) = mlx_lm::load_model(&args.model_dir)?;
    let dump_tokenizer = tokenizer.clone();

    let sampler = sampler_for_model(&args.model_dir, args.temperature, args.top_p);
    let mut pipeline = mlx_lm::GenerationPipeline::new(model, tokenizer, sampler);
    let template_options = mlx_lm::ChatTemplateOptions {
        add_generation_prompt: true,
        continue_final_message: false,
        enable_thinking: args.thinking.unwrap_or(false),
    };

    // Build the prompt — either raw or via chat template
    let prompt = if args.chat {
        let messages = vec![
            mlx_lm::Message::system(&args.system_prompt),
            mlx_lm::Message::user(&args.prompt),
        ];
        match mlx_lm::ChatTemplate::from_model_dir(&args.model_dir) {
            Ok(template) => match template.apply(&messages, &template_options) {
                Ok(p) => p,
                Err(_) => mlx_lm::ChatTemplate::chatml()
                    .apply(&messages, &template_options)
                    .or_else(|_| mlx_lm::ChatTemplate::qwen35().apply(&messages, &template_options))
                    .map_err(|e| anyhow::anyhow!("Failed to apply fallback chat template: {e}"))?,
            },
            Err(_) => mlx_lm::ChatTemplate::chatml()
                .apply(&messages, &template_options)
                .or_else(|_| mlx_lm::ChatTemplate::qwen35().apply(&messages, &template_options))
                .map_err(|e| anyhow::anyhow!("Failed to apply fallback chat template: {e}"))?,
        }
    } else {
        // Auto-apply chat template when present; fall back to ChatML for incompatible Jinja templates.
        let messages = vec![mlx_lm::Message::user(&args.prompt)];
        match mlx_lm::ChatTemplate::from_model_dir(&args.model_dir) {
            Ok(template) => match template.apply(&messages, &template_options) {
                Ok(p) => p,
                Err(_) => mlx_lm::ChatTemplate::chatml()
                    .apply(&messages, &template_options)
                    .or_else(|_| mlx_lm::ChatTemplate::qwen35().apply(&messages, &template_options))
                    .map_err(|e| anyhow::anyhow!("Failed to apply fallback chat template: {e}"))?,
            },
            Err(_) => args.prompt.clone(),
        }
    };

    eprintln!("Generating...");
    let start = std::time::Instant::now();
    let mut stdout = std::io::stdout();
    let (output, metrics) =
        pipeline.generate_with_metrics(&prompt, args.max_tokens, |_token, piece| {
            let _ = stdout.write_all(piece.as_bytes());
            let _ = stdout.flush();
        })?;
    let elapsed = start.elapsed();

    if let Some(path) = &args.dump_json_out {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let prompt_ids = dump_tokenizer.encode(&prompt)?;
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&json!({
                "prompt": prompt,
                "prompt_ids": prompt_ids,
                "generated_token_ids": metrics.generated_token_ids,
                "last_token_id": metrics.last_token_id,
                "stop_token_ids": metrics.stop_token_ids,
                "tail_token_ids": metrics.tail_token_ids,
                "stop_reason": metrics.stop_reason,
                "ttft_s": metrics.ttft_s,
                "total_s": metrics.total_s,
                "output": output,
            }))?,
        )?;
    }

    if !output.ends_with('\n') {
        println!();
    }
    eprintln!(
        "\n--- Generated {} chars in {:.2}s ---",
        output.len(),
        elapsed.as_secs_f64()
    );

    Ok(())
}
