use anyhow::Result;
use serde::Deserialize;
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

#[derive(Debug, Clone)]
enum TestStatus {
    Passed,
    Failed(String),
}

#[derive(Debug, Clone)]
struct TestResult {
    model: String,
    status: TestStatus,
    duration: Duration,
    tokens_generated: usize,
    output_preview: String,
}

impl TestResult {
    fn pass(model: String, duration: Duration, tokens: usize, output: String) -> Self {
        Self {
            model,
            status: TestStatus::Passed,
            duration,
            tokens_generated: tokens,
            output_preview: output,
        }
    }

    fn fail(model: String, duration: Duration, error: String) -> Self {
        Self {
            model,
            status: TestStatus::Failed(error),
            duration,
            tokens_generated: 0,
            output_preview: String::new(),
        }
    }
}

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

fn fallback_template_for_arch(arch: Option<mlx_lm::loader::ModelArch>) -> mlx_lm::ChatTemplate {
    match arch {
        Some(mlx_lm::loader::ModelArch::Gemma4) => mlx_lm::ChatTemplate::gemma4(),
        _ => mlx_lm::ChatTemplate::chatml(),
    }
}

fn build_prompt(
    model_dir: &std::path::Path,
    prompt: &str,
    system_prompt: &str,
) -> Result<String> {
    let arch = detect_arch(model_dir);
    let messages = vec![
        mlx_lm::Message::system(system_prompt),
        mlx_lm::Message::user(prompt),
    ];
    let template_options = mlx_lm::ChatTemplateOptions {
        add_generation_prompt: true,
        continue_final_message: false,
        enable_thinking: false,
    };

    let result = match mlx_lm::ChatTemplate::from_model_dir(model_dir) {
        Ok(template) => match template.apply(&messages, &template_options) {
            Ok(p) => Ok(p),
            Err(_) => fallback_template_for_arch(arch)
                .apply(&messages, &template_options)
                .or_else(|_| mlx_lm::ChatTemplate::qwen35().apply(&messages, &template_options))
                .map_err(|e| anyhow::anyhow!("Failed to apply fallback chat template: {e}")),
        },
        Err(_) => fallback_template_for_arch(arch)
            .apply(&messages, &template_options)
            .or_else(|_| mlx_lm::ChatTemplate::qwen35().apply(&messages, &template_options))
            .map_err(|e| anyhow::anyhow!("Failed to apply fallback chat template: {e}")),
    };
    result
}

fn run_model_test(model_id: &str, config: &Config) -> TestResult {
    let start = Instant::now();

    let result = (|| -> Result<(String, usize)> {
        let model_dir = mlx_lm::resolve_model_dir(model_id)?;
        eprintln!("  Loading model from {:?}...", model_dir);
        let (model, tokenizer) = mlx_lm::load_model(&model_dir)?;

        let sampler = sampler_for_model(&model_dir, config.harness.temperature, config.harness.top_p);
        let mut pipeline = mlx_lm::GenerationPipeline::new(model, tokenizer, sampler);

        let prompt = build_prompt(
            &model_dir,
            &config.harness.prompt,
            &config.harness.system_prompt,
        )?;

        eprintln!("  Generating...");
        let (output, metrics) =
            pipeline.generate_with_metrics(&prompt, config.harness.max_tokens, |_token, _piece| {})?;

        Ok((output, metrics.tokens))
    })();

    let duration = start.elapsed();

    match result {
        Ok((output, tokens)) => {
            let preview = output.trim().replace('\n', " ").replace('\r', "");
            let preview = if preview.len() > 60 {
                format!("{}...", &preview[..60])
            } else {
                preview
            };
            TestResult::pass(model_id.to_string(), duration, tokens, preview)
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
    let passed = results.iter().filter(|r| matches!(r.status, TestStatus::Passed)).count();
    let failed = results.iter().filter(|r| matches!(r.status, TestStatus::Failed(_))).count();

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
                let details = format!(
                    "{} tokens, preview: {}",
                    result.tokens_generated, result.output_preview
                );
                let details = if details.len() > 43 {
                    format!("{}...", &details[..40])
                } else {
                    details
                };
                ("PASS".to_string(), details)
            }
            TestStatus::Failed(err) => {
                let err = err.replace('\n', " ").replace('\r', "");
                let err = if err.len() > 43 {
                    format!("{}...", &err[..40])
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
    println!("Summary: {} passed, {} failed out of {}", passed, failed, total);
    if failed == 0 {
        println!("All models loaded and generated successfully!");
    }
    println!();
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
                println!(
                    "  -> PASS ({} tokens in {:.2}s)",
                    result.tokens_generated,
                    result.duration.as_secs_f64()
                );
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

    let failed = results.iter().filter(|r| matches!(r.status, TestStatus::Failed(_))).count();
    if failed > 0 {
        std::process::exit(1);
    }

    Ok(())
}
