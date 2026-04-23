use anyhow::Result;
use serde::Deserialize;
use std::collections::HashSet;
use std::time::{Duration, Instant};

#[derive(Debug, Deserialize, Default)]
struct HarnessConfig {
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default = "default_system_prompt")]
    system_prompt: String,
    #[serde(default = "default_expected_answer")]
    expected_answer: String,
    #[serde(default = "default_run_tokenizer_roundtrip")]
    run_tokenizer_roundtrip: bool,
    #[serde(default = "default_run_reasoning_check")]
    run_reasoning_check: bool,
    #[serde(default = "default_run_moe_performance_check")]
    run_moe_performance_check: bool,
    #[serde(default = "default_moe_min_tokens_per_sec")]
    moe_min_tokens_per_sec: f64,
}

#[derive(Debug, Deserialize, Default)]
struct ModelsConfig {
    #[serde(default)]
    models: Vec<String>,
}

#[derive(Debug, Deserialize, Default)]
struct Config {
    #[serde(default)]
    harness: HarnessConfig,
    #[serde(default)]
    models: ModelsConfig,
}

fn default_temperature() -> f32 {
    0.6
}
fn default_top_p() -> f32 {
    0.9
}
fn default_system_prompt() -> String {
    "You are a helpful assistant.".to_string()
}
fn default_expected_answer() -> String {
    "4".to_string()
}
fn default_run_tokenizer_roundtrip() -> bool {
    true
}
fn default_run_reasoning_check() -> bool {
    true
}
fn default_run_moe_performance_check() -> bool {
    true
}
fn default_moe_min_tokens_per_sec() -> f64 {
    1.0
}

#[derive(Debug, Clone)]
enum TestStatus {
    Passed,
    Failed(String),
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct QualityMetrics {
    has_repetition: bool,
    has_unicode_replacement: bool,
    hit_max_tokens: bool,
    thinking_leaked: bool,
    unique_token_ratio: f32,
    response_length: usize,
    expected_answer_found: bool,
    tokenizer_roundtrip_ok: bool,
    tokens_per_sec: f64,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct TestResult {
    model: String,
    status: TestStatus,
    duration: Duration,
    tokens_generated: usize,
    output_preview: String,
    full_output: String,
    prompt: String,
    quality: QualityMetrics,
    warnings: Vec<String>,
    stop_reason: String,
}

impl TestResult {
    fn pass(
        model: String,
        duration: Duration,
        tokens: usize,
        output: String,
        prompt: String,
        quality: QualityMetrics,
        warnings: Vec<String>,
        stop_reason: String,
    ) -> Self {
        let preview = output.trim().replace('\n', " ").replace('\r', "");
        let preview: String = preview.chars().take(60).collect();
        let preview = if output.trim().chars().count() > 60 {
            format!("{}...", preview)
        } else {
            preview
        };
        Self {
            model,
            status: TestStatus::Passed,
            duration,
            tokens_generated: tokens,
            output_preview: preview,
            full_output: output,
            prompt,
            quality,
            warnings,
            stop_reason,
        }
    }

    fn fail(model: String, duration: Duration, error: String) -> Self {
        Self {
            model,
            status: TestStatus::Failed(error),
            duration,
            tokens_generated: 0,
            output_preview: String::new(),
            full_output: String::new(),
            prompt: String::new(),
            quality: QualityMetrics {
                has_repetition: false,
                has_unicode_replacement: false,
                hit_max_tokens: false,
                thinking_leaked: false,
                unique_token_ratio: 0.0,
                response_length: 0,
                expected_answer_found: false,
                tokenizer_roundtrip_ok: false,
                tokens_per_sec: 0.0,
            },
            warnings: Vec::new(),
            stop_reason: String::new(),
        }
    }

}

fn detect_arch(model_dir: &std::path::Path) -> Option<mlx_lm::loader::ModelArch> {
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path).ok()?;
    let config: serde_json::Value = serde_json::from_str(&config_str).ok()?;
    mlx_lm::loader::detect_architecture(&config).ok()
}

fn is_reasoning_model(model_id: &str, _arch: Option<&mlx_lm::loader::ModelArch>) -> bool {
    let lower = model_id.to_lowercase();
    lower.contains("deepseek-r1")
        || lower.contains("qwen3")
        || lower.contains("bonsai")
}

fn is_moe_model(arch: Option<&mlx_lm::loader::ModelArch>) -> bool {
    matches!(
        arch,
        Some(mlx_lm::loader::ModelArch::QwenMoe)
            | Some(mlx_lm::loader::ModelArch::QwenMoePythonPort)
    )
}

fn is_base_text_model(model_id: &str) -> bool {
    let lower = model_id.to_lowercase();
    // Google's "text" models are base (not instruction-tuned)
    lower.contains("gemma-3-text-")
        || lower.contains("gemma-2-text-")
        || lower.contains("gemma-1-text-")
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

fn fallback_template_for_arch(arch: Option<mlx_lm::loader::ModelArch>) -> mlx_lm::ChatTemplate {
    match arch {
        Some(mlx_lm::loader::ModelArch::Gemma4) => mlx_lm::ChatTemplate::gemma4(),
        Some(mlx_lm::loader::ModelArch::Gemma3) => mlx_lm::ChatTemplate::gemma(),
        Some(mlx_lm::loader::ModelArch::Llama) => mlx_lm::ChatTemplate::llama3(),
        Some(mlx_lm::loader::ModelArch::Qwen35) => mlx_lm::ChatTemplate::qwen35(),
        Some(
            mlx_lm::loader::ModelArch::Qwen2
            | mlx_lm::loader::ModelArch::Qwen3
            | mlx_lm::loader::ModelArch::QwenMoe
            | mlx_lm::loader::ModelArch::QwenMoePythonPort,
        ) => mlx_lm::ChatTemplate::chatml(),
        _ => mlx_lm::ChatTemplate::chatml(),
    }
}

fn build_prompt(
    model_dir: &std::path::Path,
    prompt: &str,
    system_prompt: &str,
    template_options: &mlx_lm::ChatTemplateOptions,
) -> Result<String> {
    let arch = detect_arch(model_dir);
    let messages = vec![
        mlx_lm::Message::system(system_prompt),
        mlx_lm::Message::user(prompt),
    ];

    let result = match mlx_lm::ChatTemplate::from_model_dir(model_dir) {
        Ok(template) => match template.apply(&messages, template_options) {
            Ok(p) => Ok(p),
            Err(_) => fallback_template_for_arch(arch)
                .apply(&messages, template_options)
                .or_else(|_| mlx_lm::ChatTemplate::qwen35().apply(&messages, template_options))
                .map_err(|e| anyhow::anyhow!("Failed to apply fallback chat template: {e}")),
        },
        Err(_) => fallback_template_for_arch(arch)
            .apply(&messages, template_options)
            .or_else(|_| mlx_lm::ChatTemplate::qwen35().apply(&messages, template_options))
            .map_err(|e| anyhow::anyhow!("Failed to apply fallback chat template: {e}")),
    };
    result
}

fn verify_tokenizer_roundtrip(tokenizer: &mlx_lm::Tokenizer, test_str: &str) -> Result<()> {
    let encoded = tokenizer.encode(test_str)?;
    let decoded = tokenizer.decode(&encoded)?;
    // Not exact equality (BOS/EOS), but the core text should survive
    let prefix = &test_str[..test_str.len().min(20)];
    if !decoded.contains(prefix) {
        anyhow::bail!(
            "Tokenizer roundtrip failed: expected to find '{}' in {:?}",
            prefix,
            decoded
        );
    }
    Ok(())
}

fn analyse_output(
    output: &str,
    tokens_generated: usize,
    max_tokens: Option<usize>,
    expected_answer: &str,
    tokenizer_roundtrip_ok: bool,
    duration_secs: f64,
) -> QualityMetrics {
    // Check for runaway repetition via sliding window on words
    let words: Vec<&str> = output.split_whitespace().collect();
    let has_repetition = if words.len() >= 20 {
        let ngrams: Vec<String> = words.windows(5).map(|w| w.join(" ")).collect();
        let total = ngrams.len();
        let unique = ngrams.iter().collect::<HashSet<_>>().len();
        (unique as f32 / total.max(1) as f32) < 0.5
    } else {
        false
    };

    let unique_token_ratio = {
        let chars: Vec<char> = output.chars().collect();
        let unique = chars.iter().collect::<HashSet<_>>().len();
        unique as f32 / chars.len().max(1) as f32
    };

    let expected_answer_found = output.trim().contains(expected_answer);

    let tokens_per_sec = if duration_secs > 0.0 {
        tokens_generated as f64 / duration_secs
    } else {
        0.0
    };

    QualityMetrics {
        has_repetition,
        has_unicode_replacement: output.contains('\u{FFFD}'),
        hit_max_tokens: max_tokens.map(|m| tokens_generated >= m).unwrap_or(false),
        thinking_leaked: output.contains("<think>") || output.contains("</think>"),
        unique_token_ratio,
        response_length: output.len(),
        expected_answer_found,
        tokenizer_roundtrip_ok,
        tokens_per_sec,
    }
}

fn run_model_test(model_id: &str, config: &Config) -> TestResult {
    let start = Instant::now();

    let result = (|| -> Result<(String, String, usize, String, QualityMetrics, Vec<String>)> {
        let model_dir = mlx_lm::resolve_model_dir(model_id)?;
        eprintln!("  Loading model from {:?}...", model_dir);
        let (model, tokenizer) = mlx_lm::load_model(&model_dir)?;

        let arch = detect_arch(&model_dir);

        // Tokenizer roundtrip check
        let tokenizer_roundtrip_ok = if config.harness.run_tokenizer_roundtrip {
            let roundtrip_str = "What is 2 + 2? Answer with just the number.";
            match verify_tokenizer_roundtrip(&tokenizer, roundtrip_str) {
                Ok(()) => {
                    eprintln!("  Tokenizer roundtrip: OK");
                    true
                }
                Err(e) => {
                    eprintln!("  Tokenizer roundtrip: FAILED ({e})");
                    false
                }
            }
        } else {
            true
        };

        let template_options = mlx_lm::ChatTemplateOptions {
            add_generation_prompt: true,
            continue_final_message: false,
            enable_thinking: false,
        };

        let sampler = sampler_for_model(&model_dir, config.harness.temperature, config.harness.top_p);
        let mut pipeline = mlx_lm::GenerationPipeline::new(model, tokenizer, sampler)
            .with_strip_thinking(!template_options.enable_thinking)
            .with_stop_strings(vec![
                "<|im_start|>".to_string(),
                "<|im_end|>".to_string(),
                "<|eot_id|>".to_string(),
                "<|end_of_text|>".to_string(),
                "<|endoftext|>".to_string(),
            ]);

        let prompt = build_prompt(
            &model_dir,
            &config.harness.prompt,
            &config.harness.system_prompt,
            &template_options,
        )?;

        eprintln!("  Generating...");
        let (output, metrics) =
            pipeline.generate_with_metrics(&prompt, config.harness.max_tokens, |_token, _piece| {})?;

        let duration_secs = start.elapsed().as_secs_f64();

        let quality = analyse_output(
            &output,
            metrics.tokens,
            config.harness.max_tokens,
            &config.harness.expected_answer,
            tokenizer_roundtrip_ok,
            duration_secs,
        );

        let mut warnings = Vec::new();

        if quality.has_repetition {
            warnings.push("repetition detected".to_string());
        }
        if quality.has_unicode_replacement {
            warnings.push("unicode replacement chars".to_string());
        }
        if quality.hit_max_tokens {
            warnings.push("hit max_tokens".to_string());
        }
        if quality.thinking_leaked {
            warnings.push("thinking tokens leaked".to_string());
        }
        if quality.unique_token_ratio < 0.1 && !output.trim().is_empty() {
            warnings.push(format!(
                "degenerate output (unique ratio {:.2})",
                quality.unique_token_ratio
            ));
        }
        if !quality.expected_answer_found && !config.harness.expected_answer.is_empty() {
            warnings.push(format!(
                "expected answer '{}' not found",
                config.harness.expected_answer
            ));
        }
        if !tokenizer_roundtrip_ok {
            warnings.push("tokenizer roundtrip failed".to_string());
        }

        // Reasoning model check: <think> blocks should be absent when enable_thinking=false
        if config.harness.run_reasoning_check && is_reasoning_model(model_id, arch.as_ref()) {
            if quality.thinking_leaked {
                warnings.push(
                    "REASONING: <think> blocks present with enable_thinking=false".to_string(),
                );
            }
        }

        // Base model check: models with "text" in their name are not instruction-tuned
        if is_base_text_model(model_id) {
            warnings.push(
                "BASE MODEL: not instruction-tuned, chat template results are unreliable"
                    .to_string(),
            );
        }

        // MoE performance check
        if config.harness.run_moe_performance_check && is_moe_model(arch.as_ref()) {
            if quality.tokens_per_sec < config.harness.moe_min_tokens_per_sec {
                warnings.push(format!(
                    "MOE PERFORMANCE: {:.2} tok/s below threshold {:.2}",
                    quality.tokens_per_sec, config.harness.moe_min_tokens_per_sec
                ));
            }
        }

        Ok((output, prompt, metrics.tokens, metrics.stop_reason.to_string(), quality, warnings))
    })();

    let duration = start.elapsed();

    match result {
        Ok((output, prompt, tokens, stop_reason, quality, warnings)) => {
            TestResult::pass(
                model_id.to_string(),
                duration,
                tokens,
                output,
                prompt,
                quality,
                warnings,
                stop_reason,
            )
        }
        Err(e) => TestResult::fail(model_id.to_string(), duration, format!("{e:?}")),
    }
}

fn print_banner() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║              MLX-RS Model Test Harness                               ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();
}

fn print_summary_table(results: &[TestResult]) {
    let total = results.len();
    let passed = results
        .iter()
        .filter(|r| matches!(r.status, TestStatus::Passed))
        .count();
    let failed = results
        .iter()
        .filter(|r| matches!(r.status, TestStatus::Failed(_)))
        .count();
    let model_width = results
        .iter()
        .map(|r| r.model.len())
        .max()
        .unwrap_or(20)
        .max(20);

    println!();
    println!(
        "┌{}┬──────────┬──────────┬─────────────────────────────────────────────┐",
        "─".repeat(model_width + 2)
    );
    println!(
        "│ {:<model_width$} │ {:<8} │ {:<8} │ {:<43} │",
        "Model",
        "Status",
        "Time",
        "Details",
        model_width = model_width
    );
    println!(
        "├{}┼──────────┼──────────┼─────────────────────────────────────────────┤",
        "─".repeat(model_width + 2)
    );

    for result in results {
        let (status_str, details) = match &result.status {
            TestStatus::Passed => {
                let mut details = format!(
                    "{} tokens, preview: {}",
                    result.tokens_generated, result.output_preview
                );
                if !result.warnings.is_empty() {
                    details = format!("{} [WARN: {}]", details, result.warnings.join(", "));
                }
                let details = if details.chars().count() > 43 {
                    let truncated: String = details.chars().take(40).collect();
                    format!("{}...", truncated)
                } else {
                    details
                };
                ("PASS".to_string(), details)
            }
            TestStatus::Failed(err) => {
                let err = err.replace('\n', " ").replace('\r', "");
                let err = if err.chars().count() > 43 {
                    let truncated: String = err.chars().take(40).collect();
                    format!("{}...", truncated)
                } else {
                    err
                };
                ("FAIL".to_string(), err)
            }

        };

        println!(
            "│ {:<model_width$} │ {:<8} │ {:>6.2}s │ {:<43} │",
            result.model,
            status_str,
            result.duration.as_secs_f64(),
            details,
            model_width = model_width
        );
    }

    println!(
        "└{}┴──────────┴──────────┴─────────────────────────────────────────────┘",
        "─".repeat(model_width + 2)
    );

    println!();
    println!(
        "Summary: {} passed, {} failed out of {}",
        passed, failed, total
    );
    if failed == 0 {
        println!("All models loaded and generated successfully!");
    }
    println!();

    // Print detailed warnings for each model
    let models_with_warnings: Vec<_> = results
        .iter()
        .filter(|r| !r.warnings.is_empty())
        .collect();
    if !models_with_warnings.is_empty() {
        println!("Detailed Warnings:");
        for result in models_with_warnings {
            println!("  {}:", result.model);
            for warning in &result.warnings {
                println!("    - {}", warning);
            }
        }
        println!();
    }

    // Print full outputs for all models
    println!("Full Outputs:");
    for result in results {
        println!("  ──────────────────────────────────────────────────────────────────────────────");
        println!("  Model: {}", result.model);
        println!("  Tokens: {} | Stop reason: {} | Duration: {:.2}s", result.tokens_generated, result.stop_reason, result.duration.as_secs_f64());
        if !result.warnings.is_empty() {
            println!("  Warnings: {}", result.warnings.join(", "));
        }
        println!("  Prompt:");
        for line in result.prompt.lines() {
            println!("    {}", line);
        }
        println!("  Output:");
        for line in result.full_output.lines() {
            println!("    {}", line);
        }
        if result.full_output.is_empty() {
            println!("    <empty>");
        }
    }
    println!("  ──────────────────────────────────────────────────────────────────────────────");
}

fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "config.toml".to_string());

    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("Failed to read config '{}': {}", config_path, e))?;
    let config: Config = toml::from_str(&config_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse config '{}': {}", config_path, e))?;

    if config.models.models.is_empty() {
        anyhow::bail!("No models configured in [models] section of config.toml");
    }

    print_banner();
    println!("Configuration: {}", config_path);
    println!("Models to test: {}", config.models.models.len());
    println!("Prompt: {}", config.harness.prompt);
    if let Some(max) = config.harness.max_tokens {
        println!("Max tokens: {}", max);
    } else {
        println!("Max tokens: unlimited");
    }
    println!("Temperature: {}", config.harness.temperature);
    println!("Top-p: {}", config.harness.top_p);
    println!(
        "Expected answer: '{}'",
        config.harness.expected_answer
    );
    println!(
        "Tokenizer roundtrip: {}",
        if config.harness.run_tokenizer_roundtrip {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!(
        "Reasoning check: {}",
        if config.harness.run_reasoning_check {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!(
        "MoE performance check: {}",
        if config.harness.run_moe_performance_check {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!();

    let mut results = Vec::new();

    for (i, model_id) in config.models.models.iter().enumerate() {
        println!(
            "[{}/{}] Testing: {}",
            i + 1,
            config.models.models.len(),
            model_id
        );
        let result = run_model_test(model_id, &config);
        results.push(result.clone());

        // Print immediate status
        match &result.status {
            TestStatus::Passed => {
                let mut msg = format!(
                    "  -> PASS ({} tokens in {:.2}s, stop={})",
                    result.tokens_generated,
                    result.duration.as_secs_f64(),
                    result.stop_reason
                );
                if !result.warnings.is_empty() {
                    msg.push_str(&format!(" [WARN: {}]", result.warnings.join(", ")));
                }
                println!("{}", msg);
            }
            TestStatus::Failed(err) => {
                println!(
                    "  -> FAIL ({:.2}s): {}",
                    result.duration.as_secs_f64(),
                    err
                );
            }

        }
        println!();
    }

    print_summary_table(&results);

    let failed = results
        .iter()
        .filter(|r| matches!(r.status, TestStatus::Failed(_)))
        .count();
    if failed > 0 {
        std::process::exit(1);
    }

    Ok(())
}
