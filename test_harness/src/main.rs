use anyhow::Result;
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;
use std::time::{Duration, Instant};

// ------------------------------------------------------------------
// Config
// ------------------------------------------------------------------

#[derive(Debug, Deserialize, Default)]
struct HarnessConfig {
    #[serde(default)]
    prompt: String,
    #[serde(default = "default_vision_prompt")]
    vision_prompt: String,
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
    #[serde(default = "default_vision_expected_answer")]
    vision_expected_answer: String,
    #[serde(default = "default_run_tokenizer_roundtrip")]
    run_tokenizer_roundtrip: bool,
    #[serde(default = "default_run_reasoning_check")]
    run_reasoning_check: bool,
    #[serde(default = "default_run_moe_performance_check")]
    run_moe_performance_check: bool,
    #[serde(default = "default_moe_min_tokens_per_sec")]
    moe_min_tokens_per_sec: f64,
    #[serde(default = "default_vision_image")]
    vision_image: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ModelEntry {
    Plain(String),
    Typed {
        id: String,
        #[serde(default = "default_text_type")]
        r#type: String,
        #[serde(default)]
        image: Option<String>,
    },
}

impl ModelEntry {
    fn id(&self) -> &str {
        match self {
            ModelEntry::Plain(s) => s,
            ModelEntry::Typed { id, .. } => id,
        }
    }

    fn model_type(&self) -> ModelType {
        match self {
            ModelEntry::Plain(_) => ModelType::Text,
            ModelEntry::Typed { r#type, .. } => match r#type.as_str() {
                "vision" => ModelType::Vision,
                _ => ModelType::Text,
            },
        }
    }

    fn image_path(&self) -> Option<&str> {
        match self {
            ModelEntry::Plain(_) => None,
            ModelEntry::Typed { image, .. } => image.as_deref(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelType {
    Text,
    Vision,
}

#[derive(Debug, Deserialize, Default)]
struct ModelsConfig {
    #[serde(default)]
    models: Vec<ModelEntry>,
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
fn default_vision_expected_answer() -> String {
    String::new()
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
fn default_vision_prompt() -> String {
    "Describe this image in one sentence.".to_string()
}
fn default_vision_image() -> String {
    "test_image.png".to_string()
}
fn default_text_type() -> String {
    "text".to_string()
}

// ------------------------------------------------------------------
// Test result types
// ------------------------------------------------------------------

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
    model_type: ModelType,
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
        model_type: ModelType,
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
            model_type,
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

    fn fail(model: String, model_type: ModelType, duration: Duration, error: String) -> Self {
        Self {
            model,
            model_type,
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

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

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

fn gather_warnings(
    model_id: &str,
    arch: Option<&mlx_lm::loader::ModelArch>,
    quality: &QualityMetrics,
    config: &Config,
    tokenizer_roundtrip_ok: bool,
) -> Vec<String> {
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
    if quality.unique_token_ratio < 0.1 && !quality.response_length == 0 {
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

    if config.harness.run_reasoning_check && is_reasoning_model(model_id, arch) {
        if quality.thinking_leaked {
            warnings.push(
                "REASONING: <think> blocks present with enable_thinking=false".to_string(),
            );
        }
    }

    if is_base_text_model(model_id) {
        warnings.push(
            "BASE MODEL: not instruction-tuned, chat template results are unreliable"
                .to_string(),
        );
    }

    if config.harness.run_moe_performance_check && is_moe_model(arch) {
        if quality.tokens_per_sec < config.harness.moe_min_tokens_per_sec {
            warnings.push(format!(
                "MOE PERFORMANCE: {:.2} tok/s below threshold {:.2}",
                quality.tokens_per_sec, config.harness.moe_min_tokens_per_sec
            ));
        }
    }

    warnings
}

// ------------------------------------------------------------------
// Text model test
// ------------------------------------------------------------------

fn run_text_model_test(model_id: &str, config: &Config) -> TestResult {
    let start = Instant::now();

    let result = (|| -> Result<(String, String, usize, String, QualityMetrics, Vec<String>)> {
        let model_dir = mlx_lm::resolve_model_dir(model_id)?;
        eprintln!("  Loading model from {:?}...", model_dir);
        let (model, tokenizer) = mlx_lm::load_model(&model_dir)?;

        let arch = detect_arch(&model_dir);

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

        let warnings = gather_warnings(model_id, arch.as_ref(), &quality, config, tokenizer_roundtrip_ok);

        Ok((output, prompt, metrics.tokens, metrics.stop_reason.to_string(), quality, warnings))
    })();

    let duration = start.elapsed();

    match result {
        Ok((output, prompt, tokens, stop_reason, quality, warnings)) => {
            TestResult::pass(
                model_id.to_string(),
                ModelType::Text,
                duration,
                tokens,
                output,
                prompt,
                quality,
                warnings,
                stop_reason,
            )
        }
        Err(e) => TestResult::fail(model_id.to_string(), ModelType::Text, duration, format!("{e:?}")),
    }
}

// ------------------------------------------------------------------
// Vision model test
// ------------------------------------------------------------------

fn run_vision_model_test(model_id: &str, image_path: &Path, config: &Config) -> TestResult {
    let start = Instant::now();

    let result = (|| -> Result<(String, String, usize, String, QualityMetrics, Vec<String>)> {
        let model_dir = mlx_lm::resolve_model_dir(model_id)?;
        eprintln!("  Loading vision model from {:?}...", model_dir);

        let vlm = mlx_vlm::load_gemma4_vlm(&model_dir)?;
        let mut pipeline = mlx_vlm::VlmGenerationPipeline::new(vlm.model, vlm.tokenizer.clone());

        let template_options = mlx_lm::ChatTemplateOptions {
            add_generation_prompt: true,
            continue_final_message: false,
            enable_thinking: false,
        };

        // Build prompt with image token placeholder for vision models.
        // The chat template will place it in the correct position.
        let vision_prompt_with_placeholder = format!("<|image|>{}", config.harness.vision_prompt);
        let prompt_text = build_prompt(
            &model_dir,
            &vision_prompt_with_placeholder,
            &config.harness.system_prompt,
            &template_options,
        )?;

        // Tokenize base prompt
        let encoding = vlm.tokenizer.inner()
            .encode(prompt_text.clone(), false)
            .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
        let token_ids = encoding.get_ids().to_vec();

        // Process image to determine actual num_soft_tokens
        eprintln!("  Processing image {:?}...", image_path);
        let processed = vlm.processor.process(image_path)?;
        let pixel_values = vlm.processor.to_array(&processed)?;
        let num_soft_tokens = processed.num_soft_tokens;

        // Expand single image token to match Python Gemma4Processor behavior:
        // <|image> + <|image|> * num_soft_tokens + <image|>
        let image_token_id = vlm.config.image_token_id as u32;
        let image_open_token_id = vlm.tokenizer.inner().token_to_id("<|image>").unwrap_or(image_token_id);
        let image_close_token_id = vlm.tokenizer.inner().token_to_id("<image|>").unwrap_or(image_token_id);
        let mut expanded = Vec::new();
        let mut found_image = false;
        for &tid in &token_ids {
            if tid == image_token_id && !found_image {
                expanded.push(image_open_token_id);
                for _ in 0..num_soft_tokens {
                    expanded.push(image_token_id);
                }
                expanded.push(image_close_token_id);
                found_image = true;
            } else {
                expanded.push(tid);
            }
        }

        let prompt_i32: Vec<i32> = expanded.iter().map(|&x| x as i32).collect();
        let input_ids = mlx_core::Array::from_slice_i32(&prompt_i32)?
            .reshape(&[1, prompt_i32.len() as i32])?;

        eprintln!("  Generating with image...");
        let opts = mlx_vlm::VlmGenerateOptions {
            max_tokens: config.harness.max_tokens.unwrap_or(128),
            temperature: config.harness.temperature,
        };

        let tokens = pipeline.generate_tokens(&input_ids, Some(&pixel_values), &opts)?;
        let output = vlm.tokenizer
            .decode(&tokens)
            .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;

        let token_count = tokens.len();

        let eos_token_id = vlm.eos_token_id;
        let stop_reason = if tokens.last() == Some(&eos_token_id) {
            "stop"
        } else {
            "length"
        }
        .to_string();

        let duration_secs = start.elapsed().as_secs_f64();

        let quality = analyse_output(
            &output,
            token_count,
            config.harness.max_tokens,
            &config.harness.vision_expected_answer,
            true,
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
        if !quality.expected_answer_found && !config.harness.vision_expected_answer.is_empty() {
            warnings.push(format!(
                "expected answer '{}' not found",
                config.harness.vision_expected_answer
            ));
        }

        Ok((output, prompt_text, token_count, stop_reason, quality, warnings))
    })();

    let duration = start.elapsed();

    match result {
        Ok((output, prompt, tokens, stop_reason, quality, warnings)) => {
            TestResult::pass(
                model_id.to_string(),
                ModelType::Vision,
                duration,
                tokens,
                output,
                prompt,
                quality,
                warnings,
                stop_reason,
            )
        }
        Err(e) => TestResult::fail(
            model_id.to_string(),
            ModelType::Vision,
            duration,
            format!("{e:?}"),
        ),
    }
}

// ------------------------------------------------------------------
// UI
// ------------------------------------------------------------------

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
        "┌{}┬──────────┬──────────┬──────────┬─────────────────────────────────────────────┐",
        "─".repeat(model_width + 2)
    );
    println!(
        "│ {:<model_width$} │ {:<8} │ {:<8} │ {:<8} │ {:<43} │",
        "Model",
        "Type",
        "Status",
        "Time",
        "Details",
        model_width = model_width
    );
    println!(
        "├{}┼──────────┼──────────┼──────────┼─────────────────────────────────────────────┤",
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

        let type_str = match result.model_type {
            ModelType::Text => "text",
            ModelType::Vision => "vision",
        };

        println!(
            "│ {:<model_width$} │ {:<8} │ {:<8} │ {:>6.2}s │ {:<43} │",
            result.model,
            type_str,
            status_str,
            result.duration.as_secs_f64(),
            details,
            model_width = model_width
        );
    }

    println!(
        "└{}┴──────────┴──────────┴──────────┴─────────────────────────────────────────────┘",
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

    println!("Full Outputs:");
    for result in results {
        println!("  ──────────────────────────────────────────────────────────────────────────────");
        println!("  Model: {}", result.model);
        let type_str = match result.model_type {
            ModelType::Text => "text",
            ModelType::Vision => "vision",
        };
        println!("  Type: {}", type_str);
        println!(
            "  Tokens: {} | Stop reason: {} | Duration: {:.2}s",
            result.tokens_generated,
            result.stop_reason,
            result.duration.as_secs_f64()
        );
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

// ------------------------------------------------------------------
// Main
// ------------------------------------------------------------------

fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let args: Vec<String> = std::env::args().collect();

    // Parse args: config path and optional --type filter
    let mut config_path = "config.toml".to_string();
    let mut type_filter: Option<ModelType> = None;

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--type" || args[i] == "-t" {
            i += 1;
            if i < args.len() {
                type_filter = match args[i].as_str() {
                    "text" => Some(ModelType::Text),
                    "vision" => Some(ModelType::Vision),
                    _ => {
                        eprintln!("Warning: unknown type filter '{}', ignoring", args[i]);
                        None
                    }
                };
            }
        } else if !args[i].starts_with('-') {
            config_path = args[i].clone();
        }
        i += 1;
    }

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
    if let Some(filter) = type_filter {
        let filter_str = match filter {
            ModelType::Text => "text",
            ModelType::Vision => "vision",
        };
        println!("Type filter: {}", filter_str);
    }
    // Determine which types will actually be tested
    let will_test_text = type_filter.map_or(true, |f| f == ModelType::Text);
    let will_test_vision = type_filter.map_or(true, |f| f == ModelType::Vision);

    if will_test_text {
        println!("Prompt (text): {}", config.harness.prompt);
    }
    if will_test_vision {
        println!("Prompt (vision): {}", config.harness.vision_prompt);
    }
    if let Some(max) = config.harness.max_tokens {
        println!("Max tokens: {}", max);
    } else {
        println!("Max tokens: unlimited");
    }
    println!("Temperature: {}", config.harness.temperature);
    println!("Top-p: {}", config.harness.top_p);
    if will_test_text && !config.harness.expected_answer.is_empty() {
        println!(
            "Expected answer (text): '{}'",
            config.harness.expected_answer
        );
    }
    if will_test_vision && !config.harness.vision_expected_answer.is_empty() {
        println!(
            "Expected answer (vision): '{}'",
            config.harness.vision_expected_answer
        );
    }
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

    for (i, entry) in config.models.models.iter().enumerate() {
        let model_id = entry.id();
        let model_type = entry.model_type();

        // Apply type filter
        if let Some(filter) = type_filter {
            if model_type != filter {
                continue;
            }
        }

        let type_str = match model_type {
            ModelType::Text => "text",
            ModelType::Vision => "vision",
        };

        println!(
            "[{}/{}] Testing: {} ({})",
            i + 1,
            config.models.models.len(),
            model_id,
            type_str
        );

        let result = match model_type {
            ModelType::Text => run_text_model_test(model_id, &config),
            ModelType::Vision => {
                let image_path = entry
                    .image_path()
                    .map(Path::new)
                    .unwrap_or_else(|| Path::new(&config.harness.vision_image));
                run_vision_model_test(model_id, image_path, &config)
            }
        };
        results.push(result.clone());

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
