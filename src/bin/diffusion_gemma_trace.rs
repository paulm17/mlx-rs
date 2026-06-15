use anyhow::Result;
use clap::Parser;
use mlx_core::{Array, DType};
use serde::Serialize;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "diffusion_gemma_trace",
    about = "Dump Rust DiffusionGemma parity tensors"
)]
struct Args {
    #[arg(long)]
    model: String,

    #[arg(long)]
    trace_inputs: PathBuf,

    #[arg(long, default_value = "/tmp/diffusion_gemma_trace/rust")]
    out_dir: PathBuf,

    #[arg(long, default_value_t = 48)]
    cur_step: usize,
}

#[derive(Serialize)]
struct TensorEntry {
    file: String,
    shape: Vec<i32>,
    dtype: String,
    elements: usize,
}

#[derive(Serialize)]
struct Manifest {
    tensors: BTreeMap<String, TensorEntry>,
}

fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '.' || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn write_f32_tensor(out_dir: &Path, name: &str, array: &Array) -> Result<TensorEntry> {
    let shape = array.shape_raw();
    let values = array.as_type(DType::Float32)?.to_vec_f32()?;
    let file = format!("{}.f32", sanitize_name(name));
    let path = out_dir.join(&file);
    let mut writer = std::fs::File::create(&path)?;
    for value in &values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(TensorEntry {
        file,
        shape,
        dtype: "float32".to_string(),
        elements: values.len(),
    })
}

fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let args = Args::parse();
    std::fs::create_dir_all(&args.out_dir)?;

    let model_dir = mlx_lm::resolve_model_dir(&args.model)?;
    eprintln!("Loading model from {:?}...", model_dir);
    let (mut model, _tokenizer) = mlx_lm::load_gemma4_diffusion_model(&model_dir)?;

    let mut inputs = mlx_core::safetensors::load(&args.trace_inputs)?;
    let input_ids = inputs
        .remove("input_ids")
        .ok_or_else(|| anyhow::anyhow!("trace inputs missing input_ids"))?
        .as_type(DType::Int32)?;
    let canvas_ids = inputs
        .remove("canvas_ids")
        .ok_or_else(|| anyhow::anyhow!("trace inputs missing canvas_ids"))?
        .as_type(DType::Int32)?;

    let traces = model.trace_one_diffusion_step(&input_ids, &canvas_ids, args.cur_step)?;
    let mut manifest = Manifest {
        tensors: BTreeMap::new(),
    };
    let mut names = traces.keys().cloned().collect::<Vec<_>>();
    names.sort();
    for name in names {
        let entry = write_f32_tensor(&args.out_dir, &name, traces.get(&name).unwrap())?;
        manifest.tensors.insert(name, entry);
    }

    std::fs::write(
        args.out_dir.join("rust_trace_manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    eprintln!("Wrote Rust trace to {:?}", args.out_dir);
    Ok(())
}
