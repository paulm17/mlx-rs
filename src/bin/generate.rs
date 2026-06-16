use anyhow::Result;
use clap::Parser;
use serde_json::json;

use std::io::{IsTerminal, Write};
#[cfg(unix)]
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn detect_arch(model_dir: &std::path::Path) -> Option<mlx_lm::loader::ModelArch> {
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path).ok()?;
    let config: serde_json::Value = serde_json::from_str(&config_str).ok()?;
    mlx_lm::loader::detect_architecture(&config).ok()
}

fn sampler_for_model(model_dir: &std::path::Path, temperature: f32, top_p: f32) -> mlx_lm::Sampler {
    let mut sampler = mlx_lm::Sampler::new(temperature, top_p);
    if let Some(arch) = detect_arch(model_dir) {
        if matches!(
            arch,
            mlx_lm::loader::ModelArch::QwenMoe | mlx_lm::loader::ModelArch::QwenMoePythonPort
        ) {
            sampler = sampler
                .with_greedy_tie_break(0.05)
                .with_greedy_tie_break_after(180);
        }
    }
    sampler
}

fn fallback_template_for_arch(arch: &Option<mlx_lm::loader::ModelArch>) -> mlx_lm::ChatTemplate {
    match arch {
        Some(mlx_lm::loader::ModelArch::Gemma4)
        | Some(mlx_lm::loader::ModelArch::Gemma4Diffusion) => mlx_lm::ChatTemplate::gemma4(),
        _ => mlx_lm::ChatTemplate::chatml(),
    }
}

struct CanvasRedrawer {
    active: bool,
    rows: usize,
}

impl CanvasRedrawer {
    fn new() -> Self {
        Self {
            active: std::io::stdout().is_terminal(),
            rows: 0,
        }
    }

    fn terminal_cols() -> usize {
        Self::ioctl_terminal_cols()
            .or_else(Self::stty_terminal_cols)
            .or_else(|| {
                std::env::var("COLUMNS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .filter(|&v| v > 1)
            })
            .unwrap_or(120usize)
            .saturating_sub(1)
            .max(20)
    }

    #[cfg(unix)]
    fn ioctl_terminal_cols() -> Option<usize> {
        #[repr(C)]
        struct Winsize {
            ws_row: u16,
            ws_col: u16,
            ws_xpixel: u16,
            ws_ypixel: u16,
        }

        #[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd"))]
        const TIOCGWINSZ: u64 = 0x40087468;
        #[cfg(target_os = "linux")]
        const TIOCGWINSZ: u64 = 0x5413;

        unsafe extern "C" {
            fn ioctl(fd: i32, request: u64, ...) -> i32;
        }

        let mut winsize = Winsize {
            ws_row: 0,
            ws_col: 0,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        let rc = unsafe { ioctl(std::io::stdout().as_raw_fd(), TIOCGWINSZ, &mut winsize) };
        (rc == 0 && winsize.ws_col > 1).then_some(winsize.ws_col as usize)
    }

    #[cfg(not(unix))]
    fn ioctl_terminal_cols() -> Option<usize> {
        None
    }

    fn stty_terminal_cols() -> Option<usize> {
        std::process::Command::new("stty")
            .arg("size")
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|v| {
                v.split_whitespace()
                    .nth(1)
                    .and_then(|cols| cols.parse().ok())
            })
            .filter(|&v| v > 1)
    }

    fn wrap_text(text: &str, width: usize) -> String {
        let mut wrapped = Vec::new();
        for raw_line in text.replace('\r', "\\r").split('\n') {
            let mut line = String::new();
            let mut line_width = 0usize;
            for word in raw_line.split(' ') {
                let word_width = word.chars().count();
                let separator = usize::from(!line.is_empty());
                if line_width + separator + word_width <= width {
                    if separator == 1 {
                        line.push(' ');
                    }
                    line.push_str(word);
                    line_width += separator + word_width;
                    continue;
                }
                if !line.is_empty() {
                    wrapped.push(line);
                    line = String::new();
                }
                let mut rest = word;
                while rest.chars().count() > width {
                    let split_at = rest
                        .char_indices()
                        .nth(width)
                        .map(|(idx, _)| idx)
                        .unwrap_or(rest.len());
                    let (head, tail) = rest.split_at(split_at);
                    wrapped.push(head.to_string());
                    rest = tail;
                }
                line.push_str(rest);
                line_width = rest.chars().count();
            }
            wrapped.push(line);
        }
        wrapped.join("\n")
    }

    fn draw(&mut self, text: &str) {
        if !self.active {
            return;
        }
        let mut stdout = std::io::stdout();
        let canvas = Self::wrap_text(text, Self::terminal_cols());
        let lines: Vec<&str> = canvas.split('\n').collect();
        if self.rows > 0 {
            let _ = write!(stdout, "\r");
            for _ in 1..self.rows {
                let _ = write!(stdout, "\x1b[1A");
            }
        }
        for (idx, line) in lines.iter().enumerate() {
            if idx > 0 {
                let _ = write!(stdout, "\n");
            }
            let _ = write!(stdout, "\x1b[2K{line}");
        }
        let _ = write!(stdout, "\x1b[0J");
        let _ = stdout.flush();
        self.rows = lines.len().max(1);
    }

    fn finish(&mut self) {
        if !self.active || self.rows == 0 {
            return;
        }
        let mut stdout = std::io::stdout();
        let _ = write!(stdout, "\r");
        for _ in 1..self.rows {
            let _ = write!(stdout, "\x1b[1A");
        }
        let _ = write!(stdout, "\x1b[0J");
        let _ = stdout.flush();
        self.rows = 0;
    }
}

fn canvas_draft_text(
    tokenizer: &mlx_lm::Tokenizer,
    canvas: &[u32],
    accepted_mask: &[bool],
) -> String {
    let mut pending = Vec::new();
    let mut pieces = Vec::new();

    for (&token, &accepted) in canvas.iter().zip(accepted_mask.iter()) {
        if accepted {
            pending.push(token);
            continue;
        }
        if !pending.is_empty() {
            pieces.push(tokenizer.decode(&pending).unwrap_or_default());
            pending.clear();
        }
        pieces.push("[Mask]".to_string());
    }

    if !pending.is_empty() {
        pieces.push(tokenizer.decode(&pending).unwrap_or_default());
    }

    pieces
        .into_iter()
        .filter(|piece| !piece.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .replace('\n', "\\n")
        .trim()
        .to_string()
}

fn diffusion_draft_text(
    tokenizer: &mlx_lm::Tokenizer,
    finalized: &[u32],
    canvas: &[u32],
    accepted_mask: &[bool],
) -> String {
    let finalized_text = if finalized.is_empty() {
        String::new()
    } else {
        tokenizer.decode(finalized).unwrap_or_default()
    };
    let canvas_text = canvas_draft_text(tokenizer, canvas, accepted_mask);
    format!("{finalized_text}{canvas_text}")
}

/// MLX-RS text generation CLI.
#[derive(Parser, Debug)]
#[command(name = "generate", about = "Generate text with MLX models")]
struct Args {
    /// Model identifier: local path or Hugging Face repo ID (e.g. mlx-community/Qwen3.5-0.8B-MLX-4bit)
    #[arg(long)]
    model: String,

    /// Path to TOML config file
    #[arg(long, default_value = "config.toml")]
    config: PathBuf,

    /// The prompt to generate from
    #[arg(long, default_value = "Hello, how are you?")]
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
    dotenvy::dotenv().ok();
    let args = Args::parse();

    let model_dir = mlx_lm::resolve_model_dir(&args.model)?;
    let arch = detect_arch(&model_dir);
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
        match mlx_lm::ChatTemplate::from_model_dir(&model_dir) {
            Ok(template) => match template.apply(&messages, &template_options) {
                Ok(p) => p,
                Err(_) => fallback_template_for_arch(&arch)
                    .apply(&messages, &template_options)
                    .or_else(|_| mlx_lm::ChatTemplate::qwen35().apply(&messages, &template_options))
                    .map_err(|e| anyhow::anyhow!("Failed to apply fallback chat template: {e}"))?,
            },
            Err(_) => fallback_template_for_arch(&arch)
                .apply(&messages, &template_options)
                .or_else(|_| mlx_lm::ChatTemplate::qwen35().apply(&messages, &template_options))
                .map_err(|e| anyhow::anyhow!("Failed to apply fallback chat template: {e}"))?,
        }
    } else {
        // Auto-apply chat template for instruction-tuned models
        let messages = vec![mlx_lm::Message::user(&args.prompt)];
        match mlx_lm::ChatTemplate::from_model_dir(&model_dir) {
            Ok(template) => match template.apply(&messages, &template_options) {
                Ok(p) => p,
                Err(_) => fallback_template_for_arch(&arch)
                    .apply(&messages, &template_options)
                    .map_err(|e| anyhow::anyhow!("Failed to apply fallback chat template: {e}"))?,
            },
            Err(_) => fallback_template_for_arch(&arch)
                .apply(&messages, &template_options)
                .map_err(|e| anyhow::anyhow!("Failed to apply fallback chat template: {e}"))?,
        }
    };

    if matches!(arch, Some(mlx_lm::loader::ModelArch::Gemma4Diffusion)) {
        eprintln!("Loading model from {:?}...", model_dir);
        let (mut model, tokenizer) = mlx_lm::load_gemma4_diffusion_model(&model_dir)?;
        eprintln!("Generating...");
        let start = std::time::Instant::now();
        let prompt_ids = tokenizer.encode(&prompt)?;
        let prompt_i32: Vec<i32> = prompt_ids.iter().map(|&x| x as i32).collect();
        let input =
            mlx_core::Array::from_slice_i32(&prompt_i32)?.reshape(&[1, prompt_i32.len() as i32])?;
        let mut redrawer = CanvasRedrawer::new();
        let generated_token_ids = model
            .generate_block_diffusion_token_ids_with_drafts_and_temperature(
                &input,
                args.max_tokens,
                args.temperature,
                |draft| {
                    let canvas_text = diffusion_draft_text(
                        &tokenizer,
                        &draft.finalized_token_ids,
                        &draft.canvas_token_ids,
                        &draft.accepted_mask,
                    );
                    redrawer.draw(&canvas_text);
                    Ok(())
                },
            )?;
        redrawer.finish(); // clear last render

        let mut stdout = std::io::stdout();
        let mut accepted = Vec::new();
        let mut output = String::new();
        let mut last_len = 0usize;
        let mut stop_reason = "length";
        let mut last_token_id = None;
        for token in generated_token_ids {
            last_token_id = Some(token);
            if tokenizer.is_stop_token(token) {
                stop_reason = "stop";
                break;
            }
            accepted.push(token);
            let decoded = tokenizer.decode(&accepted)?;
            if decoded.len() > last_len {
                let piece = &decoded[last_len..];
                stdout.write_all(piece.as_bytes())?;
                stdout.flush()?;
                last_len = decoded.len();
                output = decoded;
            }
        }
        let elapsed = start.elapsed();

        if let Some(path) = &args.dump_json_out {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(
                path,
                serde_json::to_vec_pretty(&json!({
                    "prompt": prompt,
                    "prompt_ids": prompt_ids,
                    "generated_token_ids": accepted,
                    "last_token_id": last_token_id,
                    "stop_token_ids": tokenizer.stop_token_ids(),
                    "stop_reason": stop_reason,
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
        return Ok(());
    }

    eprintln!("Loading model from {:?}...", model_dir);
    let (model, tokenizer) = mlx_lm::load_model(&model_dir)?;
    let dump_tokenizer = tokenizer.clone();

    let sampler = sampler_for_model(&model_dir, args.temperature, args.top_p);
    let stop_signal = Arc::new(AtomicBool::new(false));
    let signal_clone = Arc::clone(&stop_signal);
    ctrlc::set_handler(move || {
        eprintln!("\nStopping generation...");
        signal_clone.store(true, Ordering::Relaxed);
    })?;
    let mut pipeline =
        mlx_lm::GenerationPipeline::new(model, tokenizer, sampler).with_stop_signal(stop_signal);

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
