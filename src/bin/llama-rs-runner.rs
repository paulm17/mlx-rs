use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use serde::{Deserialize, Serialize};

use backend_trait::Backend;
use mlx_backend::MlxBackend;

#[derive(Parser)]
#[command(name = "llama-rs-runner", about = "MLX runner subprocess")]
struct Args {
    #[arg(long)]
    mlx_engine: bool,
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "127.0.0.1")]
    bind: String,
    #[arg(long)]
    port: u16,
}

#[derive(Clone)]
struct RunnerState {
    backend: Arc<Mutex<Option<MlxBackend>>>,
}

#[derive(Serialize)]
struct StatusResponse {
    status: String,
    progress: f32,
    context_length: usize,
    memory: u64,
}

#[derive(Deserialize)]
struct CompletionRequest {
    prompt: String,
    #[serde(default)]
    options: CompletionOptions,
}

#[derive(Deserialize, Default)]
struct CompletionOptions {
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    stop: Option<Vec<String>>,
}

#[derive(Serialize)]
struct CompletionResponse {
    content: String,
    done: bool,
    done_reason: i32,
    prompt_eval_count: usize,
    eval_count: usize,
    error: Option<String>,
}

#[derive(Deserialize)]
struct TokenizeRequest {
    text: String,
    #[serde(default = "default_true")]
    add_bos: bool,
}

fn default_true() -> bool {
    true
}

async fn status_handler(State(state): State<RunnerState>) -> impl IntoResponse {
    let backend = state.backend.lock().unwrap();
    if let Some(b) = backend.as_ref() {
        let info = b.model_info();
        Json(StatusResponse {
            status: "ok".to_string(),
            progress: 1.0,
            context_length: info.context_length.unwrap_or(0),
            memory: 0,
        })
    } else {
        Json(StatusResponse {
            status: "loading".to_string(),
            progress: 0.0,
            context_length: 0,
            memory: 0,
        })
    }
}

async fn completions_handler(
    State(state): State<RunnerState>,
    Json(req): Json<CompletionRequest>,
) -> impl IntoResponse {
    let mut backend = state.backend.lock().unwrap();
    let backend = match backend.as_mut() {
        Some(b) => b,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(CompletionResponse {
                    content: String::new(),
                    done: true,
                    done_reason: 0,
                    prompt_eval_count: 0,
                    eval_count: 0,
                    error: Some("model not loaded".to_string()),
                }),
            )
                .into_response();
        }
    };

    let options = backend_trait::GenerationOptions {
        max_tokens: req.options.max_tokens,
        temperature: req.options.temperature.unwrap_or(0.6),
        top_p: req.options.top_p.unwrap_or(0.9),
        top_k: req.options.top_k.unwrap_or(40),
        min_p: req.options.min_p.unwrap_or(0.05),
        stop: req.options.stop,
    };

    match backend.generate(&req.prompt, &options) {
        Ok(output) => {
            let prompt_tokens = output.metrics.prompt_tokens;
            let eval_tokens = output.metrics.generated_tokens;
            (
                StatusCode::OK,
                Json(CompletionResponse {
                    content: output.text,
                    done: true,
                    done_reason: 0,
                    prompt_eval_count: prompt_tokens,
                    eval_count: eval_tokens,
                    error: None,
                }),
            )
                .into_response()
        }
        Err(_e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(CompletionResponse {
                content: String::new(),
                done: true,
                done_reason: 0,
                prompt_eval_count: 0,
                eval_count: 0,
                error: Some(_e.to_string()),
            }),
        )
            .into_response(),
    }
}

async fn tokenize_handler(
    State(state): State<RunnerState>,
    Json(req): Json<TokenizeRequest>,
) -> impl IntoResponse {
    let backend = state.backend.lock().unwrap();
    let backend = match backend.as_ref() {
        Some(b) => b,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(Vec::<i32>::new()),
            )
                .into_response();
        }
    };

    match backend.tokenize(&req.text, req.add_bos) {
        Ok(tokens) => (StatusCode::OK, Json(tokens)).into_response(),
        Err(_e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(Vec::<i32>::new()),
        )
            .into_response(),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    if !args.mlx_engine {
        anyhow::bail!("--mlx-engine flag is required");
    }

    let model_path = args.model;
    if !model_path.exists() {
        anyhow::bail!("model path does not exist: {}", model_path.display());
    }

    let backend = MlxBackend::load(&model_path)?;

    let state = RunnerState {
        backend: Arc::new(Mutex::new(Some(backend))),
    };

    let app = Router::new()
        .route("/v1/status", get(status_handler))
        .route("/v1/completions", post(completions_handler))
        .route("/v1/tokenize", post(tokenize_handler))
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", args.bind, args.port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
