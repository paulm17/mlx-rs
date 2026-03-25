use anyhow::Result;
use clap::Parser;
use mlx_core::Array;
use mlx_lm::{EmbeddingModel, EmbeddingPooling, ModelRuntime};
use serde::Serialize;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "embed_bench",
    about = "Benchmark embedding extraction directly without the HTTP server"
)]
struct Args {
    /// Path to the model directory
    #[arg(long)]
    model_dir: PathBuf,

    /// Inline input text. Repeat to benchmark multiple texts.
    #[arg(long)]
    input: Vec<String>,

    /// File containing one input per line.
    #[arg(long)]
    input_file: Option<PathBuf>,

    /// Number of warmup passes before timing.
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Number of timed passes over the full input set.
    #[arg(long, default_value_t = 5)]
    iterations: usize,

    /// Mirror the current server path by clearing model state before each input.
    #[arg(long, default_value_t = true)]
    clear_cache_per_input: bool,

    /// Optional JSON output path.
    #[arg(long)]
    json_out: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
struct StageTotals {
    encode_s: f64,
    forward_s: f64,
    pool_s: f64,
    normalize_s: f64,
    total_s: f64,
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkSummary {
    model_dir: String,
    input_count: usize,
    warmup: usize,
    iterations: usize,
    clear_cache_per_input: bool,
    total_embeddings: usize,
    total_prompt_tokens: usize,
    avg_prompt_tokens_per_embedding: f64,
    embeddings_per_s: f64,
    prompt_tokens_per_s: f64,
    avg_ms_per_embedding: f64,
    stage_totals: StageTotals,
}

fn normalize_embedding(mut embedding: Vec<f32>) -> Result<Vec<f32>> {
    let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm == 0.0 {
        anyhow::bail!("embedding vector has zero norm");
    }
    for value in &mut embedding {
        *value /= norm;
    }
    Ok(embedding)
}

fn pool_embedding(
    hidden_states: &Array,
    pooling: EmbeddingPooling,
    attention_mask: &[u32],
) -> Result<Vec<f32>> {
    let hidden_shape = hidden_states.shape_raw();
    if hidden_shape.len() < 3 {
        anyhow::bail!("unexpected hidden state rank {}", hidden_states.ndim());
    }
    let batch = hidden_shape[0];
    let seq_len = hidden_shape[hidden_shape.len() - 2] as usize;
    let hidden_size = hidden_shape[hidden_shape.len() - 1] as usize;
    if batch != 1 {
        anyhow::bail!("expected batch size 1 for embedding extraction, got {batch}");
    }
    if seq_len == 0 || hidden_size == 0 {
        anyhow::bail!("hidden states did not contain any sequence positions");
    }

    match pooling {
        EmbeddingPooling::LastToken => {
            let mut start = vec![0i32; hidden_shape.len()];
            let mut stop = hidden_shape.clone();
            let seq_axis = hidden_shape.len() - 2;
            start[seq_axis] = seq_len as i32 - 1;
            stop[seq_axis] = seq_len as i32;
            Ok(hidden_states.slice(&start, &stop)?.to_vec_f32()?)
        }
        EmbeddingPooling::Mean => {
            if attention_mask.len() != seq_len {
                anyhow::bail!(
                    "attention mask length {} did not match sequence length {seq_len}",
                    attention_mask.len()
                );
            }
            let values = hidden_states.to_vec_f32()?;
            let mut pooled = vec![0.0f32; hidden_size];
            let mut count = 0.0f32;
            for (token_idx, &mask) in attention_mask.iter().enumerate() {
                if mask == 0 {
                    continue;
                }
                count += 1.0;
                let offset = token_idx * hidden_size;
                for dim in 0..hidden_size {
                    pooled[dim] += values[offset + dim];
                }
            }
            if count == 0.0 {
                anyhow::bail!("attention mask excluded every token");
            }
            for value in &mut pooled {
                *value /= count;
            }
            Ok(pooled)
        }
    }
}

fn load_inputs(args: &Args) -> Result<Vec<String>> {
    let mut inputs = args.input.clone();
    if let Some(path) = &args.input_file {
        let raw = fs::read_to_string(path)?;
        inputs.extend(
            raw.lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(ToOwned::to_owned),
        );
    }
    if inputs.is_empty() {
        anyhow::bail!("provide at least one --input or an --input-file");
    }
    Ok(inputs)
}

fn run_pass(
    model: &mut dyn ModelRuntime,
    tokenizer: &mlx_lm::Tokenizer,
    inputs: &[String],
    clear_cache_per_input: bool,
) -> Result<(StageTotals, usize)> {
    let mut totals = StageTotals {
        encode_s: 0.0,
        forward_s: 0.0,
        pool_s: 0.0,
        normalize_s: 0.0,
        total_s: 0.0,
    };
    let mut prompt_tokens = 0usize;
    let total_start = Instant::now();

    for input in inputs {
        if clear_cache_per_input {
            model.clear_cache();
        }

        let t0 = Instant::now();
        let encoded = tokenizer.encode_for_embeddings(input)?;
        totals.encode_s += t0.elapsed().as_secs_f64();

        if encoded.ids.is_empty() {
            anyhow::bail!("input produced no tokens");
        }
        prompt_tokens += encoded.ids.len();

        let input_ids = encoded.ids.iter().map(|v| *v as i32).collect::<Vec<_>>();
        let input = Array::from_slice_i32(&input_ids)?.reshape(&[1, input_ids.len() as i32])?;

        let t1 = Instant::now();
        let hidden_states = model.forward_hidden_states(&input)?;
        hidden_states.eval()?;
        totals.forward_s += t1.elapsed().as_secs_f64();

        let t2 = Instant::now();
        let embedding = pool_embedding(
            &hidden_states,
            model.embedding_pooling(),
            &encoded.attention_mask,
        )?;
        totals.pool_s += t2.elapsed().as_secs_f64();

        let t3 = Instant::now();
        let _embedding = normalize_embedding(embedding)?;
        totals.normalize_s += t3.elapsed().as_secs_f64();
    }

    model.clear_cache();
    totals.total_s = total_start.elapsed().as_secs_f64();
    Ok((totals, prompt_tokens))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let inputs = load_inputs(&args)?;

    eprintln!("Loading model from {:?}...", args.model_dir);
    let (mut model, tokenizer) = mlx_lm::load_model(&args.model_dir)?;

    for pass in 0..args.warmup {
        eprintln!("Warmup pass {}/{}...", pass + 1, args.warmup);
        let _ = run_pass(&mut *model, &tokenizer, &inputs, args.clear_cache_per_input)?;
    }

    let mut totals = StageTotals {
        encode_s: 0.0,
        forward_s: 0.0,
        pool_s: 0.0,
        normalize_s: 0.0,
        total_s: 0.0,
    };
    let mut total_prompt_tokens = 0usize;

    for pass in 0..args.iterations {
        eprintln!("Timed pass {}/{}...", pass + 1, args.iterations);
        let (pass_totals, pass_tokens) =
            run_pass(&mut *model, &tokenizer, &inputs, args.clear_cache_per_input)?;
        totals.encode_s += pass_totals.encode_s;
        totals.forward_s += pass_totals.forward_s;
        totals.pool_s += pass_totals.pool_s;
        totals.normalize_s += pass_totals.normalize_s;
        totals.total_s += pass_totals.total_s;
        total_prompt_tokens += pass_tokens;
    }

    let total_embeddings = inputs.len() * args.iterations;
    let total_s = totals.total_s.max(1e-9);
    let summary = BenchmarkSummary {
        model_dir: args.model_dir.display().to_string(),
        input_count: inputs.len(),
        warmup: args.warmup,
        iterations: args.iterations,
        clear_cache_per_input: args.clear_cache_per_input,
        total_embeddings,
        total_prompt_tokens,
        avg_prompt_tokens_per_embedding: total_prompt_tokens as f64 / total_embeddings as f64,
        embeddings_per_s: total_embeddings as f64 / total_s,
        prompt_tokens_per_s: total_prompt_tokens as f64 / total_s,
        avg_ms_per_embedding: (total_s * 1000.0) / total_embeddings as f64,
        stage_totals: totals.clone(),
    };

    println!("{}", serde_json::to_string_pretty(&summary)?);

    if let Some(path) = &args.json_out {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(&summary)?)?;
    }

    Ok(())
}
