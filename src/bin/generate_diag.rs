use anyhow::Result;
use clap::Parser;
use serde_json::json;
use std::path::PathBuf;

fn use_qwen_tie_break(model_dir: &std::path::Path) -> bool {
    let config_path = model_dir.join("config.json");
    let Ok(config_str) = std::fs::read_to_string(&config_path) else {
        return false;
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_str) else {
        return false;
    };
    matches!(
        mlx_lm::loader::detect_architecture(&config),
        Ok(mlx_lm::loader::ModelArch::QwenMoe | mlx_lm::loader::ModelArch::QwenMoePythonPort)
    )
}

#[derive(Parser, Debug)]
#[command(name = "generate_diag", about = "Generate text with per-step top-k diagnostics")]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long)]
    prompt: String,
    #[arg(long)]
    max_tokens: Option<usize>,
    #[arg(long, default_value_t = false)]
    chat: bool,
    #[arg(long, default_value = "You are a helpful assistant.")]
    system_prompt: String,
    #[arg(long, value_parser = clap::value_parser!(bool))]
    thinking: Option<bool>,
    #[arg(long)]
    dump_json_out: PathBuf,
    #[arg(long)]
    trace_step: Option<usize>,
    #[arg(long, default_value_t = 4)]
    trace_window: usize,
    #[arg(long, default_value_t = 8)]
    topk: usize,
}

fn squeeze_all_singletons(mut arr: mlx_core::Array) -> Result<mlx_core::Array> {
    loop {
        let shape = arr.shape_raw();
        let mut squeezed = false;
        for axis in (0..shape.len()).rev() {
            if shape[axis] == 1 {
                arr = arr.squeeze(axis as i32)?;
                squeezed = true;
                break;
            }
        }
        if !squeezed {
            return Ok(arr);
        }
    }
}

fn topk_from_logits(logits: &mlx_core::Array, topk: usize) -> Result<Vec<(u32, f32)>> {
    let arr = squeeze_all_singletons(logits.clone())?.contiguous()?;
    let vals = arr.to_vec_f32()?;
    let mut idxs: Vec<usize> = (0..vals.len()).collect();
    idxs.sort_by(|&a, &b| vals[b].partial_cmp(&vals[a]).unwrap_or(std::cmp::Ordering::Equal));
    Ok(idxs
        .into_iter()
        .take(topk)
        .map(|i| (i as u32, vals[i]))
        .collect())
}

fn main() -> Result<()> {
    let args = Args::parse();

    let (mut model, tokenizer) = mlx_lm::load_model(&args.model_dir)?;
    let template_options = mlx_lm::ChatTemplateOptions {
        add_generation_prompt: true,
        continue_final_message: false,
        enable_thinking: args.thinking.unwrap_or(false),
    };

    let prompt = if args.chat {
        let messages = vec![
            mlx_lm::Message::system(&args.system_prompt),
            mlx_lm::Message::user(&args.prompt),
        ];
        match mlx_lm::ChatTemplate::from_model_dir(&args.model_dir) {
            Ok(template) => template.apply(&messages, &template_options)?,
            Err(_) => args.prompt.clone(),
        }
    } else {
        let messages = vec![mlx_lm::Message::user(&args.prompt)];
        match mlx_lm::ChatTemplate::from_model_dir(&args.model_dir) {
            Ok(template) => match template.apply(&messages, &template_options) {
                Ok(p) => p,
                Err(_) => mlx_lm::ChatTemplate::chatml()
                    .apply(&messages, &template_options)
                    .or_else(|_| mlx_lm::ChatTemplate::qwen35().apply(&messages, &template_options))?,
            },
            Err(_) => args.prompt.clone(),
        }
    };

    let prompt_ids = tokenizer.encode(&prompt)?;
    let prompt_i32: Vec<i32> = prompt_ids.iter().map(|&x| x as i32).collect();
    let input = mlx_core::Array::from_slice_i32(&prompt_i32)?.reshape(&[1, prompt_i32.len() as i32])?;

    model.clear_cache();
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut output = String::new();
    let mut step_topk = Vec::new();
    let t0 = std::time::Instant::now();
    let mut ttft_s: Option<f64> = None;
    let max_tokens = args.max_tokens.unwrap_or(256);

    let mut logits = model.forward_last_token_logits(&input)?;
    let mut step = 0usize;
    let mut last_token_id: Option<u32>;
    let stop_tokens = tokenizer.stop_token_ids().to_vec();
    let stop_reason: &'static str;

    loop {
        if let Some(target) = args.trace_step {
            if step.abs_diff(target) <= args.trace_window {
                let topk = topk_from_logits(&logits, args.topk)?;
                step_topk.push(json!({
                    "step": step,
                    "topk": topk.into_iter().map(|(token_id, logit)| json!({"token_id": token_id, "logit": logit})).collect::<Vec<_>>(),
                }));
            }
        }

        let token = if use_qwen_tie_break(&args.model_dir) {
            let sampler = mlx_lm::Sampler::new(0.0, 1.0).with_greedy_tie_break(0.05);
            sampler.sample_raw_last_token_logits(&logits, &[])?
        } else {
            let token_arr = logits.argmax(logits.ndim().saturating_sub(1) as i32)?;
            let token_arr = squeeze_all_singletons(token_arr)?;
            match token_arr.item_u32() {
                Ok(v) => v,
                Err(_) => token_arr.item_i32()? as u32,
            }
        };
        last_token_id = Some(token);

        if tokenizer.is_stop_token(token) {
            stop_reason = "stop";
            break;
        }

        generated_ids.push(token);
        let piece = tokenizer.decode(&[token])?;
        output.push_str(&piece);
        if ttft_s.is_none() {
            ttft_s = Some(t0.elapsed().as_secs_f64());
        }
        step += 1;
        if step >= max_tokens {
            stop_reason = "length";
            break;
        }
        let next = mlx_core::Array::from_int(token as i32)?.reshape(&[1, 1])?;
        logits = model.forward_last_token_logits(&next)?;
    }

    if let Some(parent) = args.dump_json_out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(
        &args.dump_json_out,
        serde_json::to_vec_pretty(&json!({
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "generated_token_ids": generated_ids,
            "last_token_id": last_token_id,
            "stop_token_ids": stop_tokens,
            "stop_reason": stop_reason,
            "ttft_s": ttft_s,
            "total_s": t0.elapsed().as_secs_f64(),
            "step_topk": step_topk,
            "output": output,
        }))?,
    )?;

    Ok(())
}
