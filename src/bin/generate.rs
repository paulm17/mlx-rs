use anyhow::Result;
use clap::Parser;
use std::io::{self, Write};
use std::path::PathBuf;

use llama_lm::config::LlamaCppConfig;
use llama_lm::{
    AppliedChatTemplate, ChatMessage, ChatTemplateOptions, GenerationOptions, GenerationPipeline,
};

// Ensure mlx-backend is linked and its ctor registers the safetensors factory
#[allow(unused_imports)]
use mlx_backend;

#[derive(Parser, Debug)]
#[command(
    name = "generate",
    about = "Generate text with GGUF models via llama.cpp"
)]
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

    #[arg(long, default_value_t = 64)]
    top_k: usize,

    #[arg(long, default_value_t = false)]
    chat: bool,

    #[arg(long, default_value_t = false)]
    raw: bool,

    #[arg(long, default_value_t = false)]
    think: bool,

    #[arg(long, default_value = "You are a helpful assistant.")]
    system_prompt: String,

    #[arg(long, default_value_t = false)]
    stream: bool,
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
            n_ctx: Some(4096),
            ..Default::default()
        }
    };

    eprintln!("Loading model...");
    let mut pipeline = GenerationPipeline::new(model_path, llamacpp_config)?;
    eprintln!("Model loaded.");

    let messages = vec![
        ChatMessage::system(&args.system_prompt),
        ChatMessage::user(&args.prompt),
    ];
    let chat_template = if pipeline.backend().supports_chat_template() {
        pipeline
            .apply_chat_template_with_options(
                &messages,
                &ChatTemplateOptions {
                    enable_thinking: args.think,
                    ..Default::default()
                },
            )
            .ok()
    } else {
        None
    };

    let use_chat_prompt = !args.raw && (args.chat || looks_like_instruction_model(model_path));

    let mut applied_template = chat_template;

    let prompt = if use_chat_prompt {
        if let Some(template) = &applied_template {
            template.prompt.clone()
        } else {
            let prompt = pipeline.apply_chat_template(&messages)?;
            applied_template = Some(AppliedChatTemplate {
                prompt: prompt.clone(),
                additional_stops: Vec::new(),
                parser: None,
                generation_prompt: String::new(),
                chat_format: 0,
                parse_tool_calls: false,
            });
            prompt
        }
    } else {
        args.prompt
    };

    let stop = if use_chat_prompt {
        applied_template
            .as_ref()
            .map(|template| template.additional_stops.clone())
            .filter(|stops| !stops.is_empty())
    } else {
        None
    };

    let options = GenerationOptions {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        stop,
        ..Default::default()
    };

    if args.stream {
        let metrics = pipeline.generate_stream(&prompt, &options, |piece| {
            print!("{}", piece);
            io::stdout().flush().ok();
            true
        })?;
        eprintln!("\n--- Metrics ---");
        eprintln!("Prompt tokens: {}", metrics.prompt_tokens);
        eprintln!("Generated tokens: {}", metrics.generated_tokens);
        if let Some(ttft) = metrics.ttft_s {
            eprintln!("Time to first token: {:.3}s", ttft);
        }
        if let Some(total) = metrics.total_s {
            eprintln!("Total time: {:.3}s", total);
        }
        if let Some(tps) = metrics.tokens_per_s {
            eprintln!("Tokens/sec: {:.2}", tps);
        }
    } else {
        let mut output = pipeline.generate(&prompt, &options)?;
        if let Some(template) = &applied_template {
            output.text = pipeline.parse_chat_response(template, &output.text, false)?;
        }
        if use_chat_prompt {
            output.text = normalize_chat_output(&output.text);
        }
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
    }

    Ok(())
}

fn looks_like_instruction_model(model: &str) -> bool {
    let normalized = model.to_ascii_lowercase();
    normalized.contains("-it")
        || normalized.contains("_it")
        || normalized.contains("instruct")
        || normalized.contains("chat")
}

fn normalize_chat_output(text: &str) -> String {
    parse_gemma_channel_output(text).unwrap_or_else(|| text.to_string())
}

fn parse_gemma_channel_output(text: &str) -> Option<String> {
    let mut rest = text.trim_start();
    let mut parsed_channel = false;

    if let Some(after_marker) = rest.strip_prefix("<|channel>thought") {
        parsed_channel = true;
        rest = strip_one_line_break(after_marker)?;
        if let Some(separator) = rest.find("<channel|>") {
            rest = &rest[separator + "<channel|>".len()..];
        } else {
            return None;
        }
    }

    if let Some(after_marker) = rest.trim_start().strip_prefix("<|channel>final") {
        parsed_channel = true;
        rest = strip_one_line_break(after_marker)?;
    }

    if !parsed_channel {
        None
    } else {
        Some(rest.trim_start_matches(['\r', '\n']).to_string())
    }
}

fn strip_one_line_break(text: &str) -> Option<&str> {
    text.strip_prefix("\r\n")
        .or_else(|| text.strip_prefix('\n'))
        .or_else(|| text.strip_prefix('\r'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_instruction_model_names() {
        assert!(looks_like_instruction_model("unsloth/gemma-4-12b-it-GGUF"));
        assert!(looks_like_instruction_model("Qwen-Instruct.gguf"));
        assert!(looks_like_instruction_model("local/chat-model.gguf"));
        assert!(!looks_like_instruction_model("models/base-model.gguf"));
    }

    #[test]
    fn normalizes_gemma_12b_empty_thought_channel() {
        let text = "<|channel>thought\n<channel|>Hello! How can I help you today?";
        assert_eq!(
            normalize_chat_output(text),
            "Hello! How can I help you today?"
        );
    }

    #[test]
    fn normalizes_gemma_12b_thought_and_final_channels() {
        let text = "<|channel>thought\r\ninternal\r\n<channel|><|channel>final\r\nAnswer";
        assert_eq!(normalize_chat_output(text), "Answer");
    }
}
