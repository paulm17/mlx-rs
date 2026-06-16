use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use crate::config::LlamaCppConfig;
use crate::loader::resolve_model_path;
use crate::runtime::Runtime;
use crate::types::{ChatMessage, GenerationOptions, StopReason};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ServerConfig {
    pub bind: Option<String>,
    pub port: Option<u16>,
    pub model_path: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub rate_limit_rpm: Option<u32>,
    pub thinking: Option<bool>,
    pub embeddings_batch_size: Option<usize>,
    // llama.cpp runtime config
    pub n_ctx: Option<u32>,
    pub n_batch: Option<u32>,
    pub n_ubatch: Option<u32>,
    pub n_gpu_layers: Option<u32>,
    pub n_threads: Option<u32>,
    pub n_threads_batch: Option<u32>,
    pub embedding: Option<bool>,
    pub pooling: Option<String>,
    pub use_mmap: Option<bool>,
    pub use_mlock: Option<bool>,
    pub flash_attn: Option<bool>,
}

impl ServerConfig {
    pub fn from_toml_path(path: &Path) -> Result<Self> {
        let mut cfg = if !path.exists() {
            Self::default()
        } else {
            let content = std::fs::read_to_string(path)?;
            Self::from_toml_str(&content)?
        };
        if cfg.api_key.is_none() {
            let env_path = path.parent().unwrap_or(Path::new(".")).join(".env");
            if let Ok(env_content) = std::fs::read_to_string(&env_path) {
                for line in env_content.lines() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') {
                        continue;
                    }
                    if let Some(val) = line.strip_prefix("HF_API_KEY=") {
                        let val = val.trim().trim_matches('"').trim_matches('\'');
                        if !val.is_empty() {
                            cfg.api_key = Some(val.to_string());
                        }
                    }
                }
            }
        }
        Ok(cfg)
    }

    pub fn from_toml_str(content: &str) -> Result<Self> {
        let parsed: serde_json::Value = basic_toml_to_json(content)
            .map_err(|e| anyhow::anyhow!("Failed to parse config: {e}"))?;
        let server = parsed
            .get("server")
            .cloned()
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
        let cfg: Self = serde_json::from_value(server)
            .map_err(|e| anyhow::anyhow!("Failed to parse server config: {e}"))?;
        Ok(cfg)
    }

    pub fn to_llamacpp_config(&self) -> LlamaCppConfig {
        LlamaCppConfig {
            n_ctx: self.n_ctx,
            n_batch: self.n_batch,
            n_ubatch: self.n_ubatch,
            n_gpu_layers: self.n_gpu_layers,
            n_threads: self.n_threads,
            n_threads_batch: self.n_threads_batch,
            embedding: self.embedding,
            pooling: self.pooling.clone(),
            use_mmap: self.use_mmap,
            use_mlock: self.use_mlock,
            flash_attn: self.flash_attn,
        }
    }
}

fn basic_toml_to_json(input: &str) -> Result<serde_json::Value> {
    let mut root = serde_json::Map::new();
    let mut current_section: Option<String> = None;

    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            let name = line[1..line.len() - 1].trim().to_string();
            current_section = Some(name);
            continue;
        }
        if let Some(eq_pos) = line.find('=') {
            let key = line[..eq_pos].trim();
            let value_str = line[eq_pos + 1..].trim();
            let value = parse_toml_value(value_str);

            let section = current_section.as_deref().unwrap_or("");
            if section.is_empty() {
                root.insert(key.to_string(), value);
            } else {
                let section_map = root
                    .entry(section.to_string())
                    .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
                if let serde_json::Value::Object(ref mut m) = section_map {
                    m.insert(key.to_string(), value);
                }
            }
        }
    }

    Ok(serde_json::Value::Object(root))
}

fn parse_toml_value(s: &str) -> serde_json::Value {
    let s = s.trim();
    if s == "true" {
        serde_json::Value::Bool(true)
    } else if s == "false" {
        serde_json::Value::Bool(false)
    } else if let Ok(n) = s.parse::<u64>() {
        serde_json::Value::Number(n.into())
    } else if let Ok(n) = s.parse::<i64>() {
        serde_json::Value::Number(n.into())
    } else if let Some(inner) = s.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
        serde_json::Value::String(inner.to_string())
    } else {
        serde_json::Value::String(s.to_string())
    }
}

// --- Server state and types ---

struct RateLimiter {
    window_start: Mutex<Instant>,
    count: Mutex<u32>,
    max_rpm: u32,
}

impl RateLimiter {
    fn new(max_rpm: u32) -> Self {
        Self {
            window_start: Mutex::new(Instant::now()),
            count: Mutex::new(0),
            max_rpm,
        }
    }

    fn check(&self) -> bool {
        let mut start = self.window_start.lock().unwrap();
        let mut count = self.count.lock().unwrap();

        if start.elapsed() >= Duration::from_secs(60) {
            *start = Instant::now();
            *count = 0;
        }

        if *count >= self.max_rpm {
            return false;
        }

        *count += 1;
        true
    }
}

struct ServerState {
    runtime: Mutex<Option<Runtime>>,
    config: ServerConfig,
    api_key: Option<String>,
    rate_limiter: Option<RateLimiter>,
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stop: Option<Vec<String>>,
}

fn default_temperature() -> f32 {
    0.6
}

fn default_top_p() -> f32 {
    0.9
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: ChatUsage,
}

#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessage,
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct ChatUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<ChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<ChatUsage>,
}

#[derive(Serialize)]
struct ChatChunkChoice {
    index: usize,
    delta: ChatDelta,
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct ChatDelta {
    role: Option<String>,
    content: Option<String>,
}

#[derive(Serialize)]
struct ModelListResponse {
    object: String,
    data: Vec<ModelObject>,
}

#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: String,
    created: i64,
    owned_by: String,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    model_loaded: bool,
}

#[derive(Deserialize)]
struct LoadRequest {
    model_path: String,
}

#[derive(Serialize)]
struct LoadResponse {
    status: String,
    model_path: String,
    context_length: Option<usize>,
    embedding_dimension: Option<usize>,
    vocab_size: Option<usize>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Deserialize)]
struct EmbeddingRequest {
    input: EmbeddingInput,
    model: String,
}

#[derive(Serialize)]
struct EmbeddingObject {
    object: String,
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Serialize)]
struct EmbeddingResponse {
    object: String,
    data: Vec<EmbeddingObject>,
    model: String,
    usage: EmbeddingUsageResponse,
}

#[derive(Serialize)]
struct EmbeddingUsageResponse {
    prompt_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

// --- Handlers ---

async fn health_handler(State(state): State<Arc<ServerState>>) -> Json<HealthResponse> {
    let runtime = state.runtime.lock().unwrap();
    Json(HealthResponse {
        status: "ok".to_string(),
        model_loaded: runtime.is_some(),
    })
}

async fn models_handler(State(state): State<Arc<ServerState>>) -> Json<ModelListResponse> {
    let runtime = state.runtime.lock().unwrap();
    let data = if let Some(ref rt) = *runtime {
        let info = rt.model_info();
        vec![ModelObject {
            id: info.model_path.clone(),
            object: "model".to_string(),
            created: 0,
            owned_by: "local".to_string(),
        }]
    } else {
        vec![]
    };
    Json(ModelListResponse {
        object: "list".to_string(),
        data,
    })
}

async fn load_handler(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<LoadRequest>,
) -> Result<Json<LoadResponse>, (StatusCode, Json<ErrorResponse>)> {
    let model_path = resolve_model_path(&req.model_path).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let llamacpp_config = state.config.to_llamacpp_config();

    let rt = Runtime::new(model_path.to_str().unwrap(), llamacpp_config).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let info = rt.model_info();

    let mut runtime = state.runtime.lock().unwrap();
    *runtime = Some(rt);

    Ok(Json(LoadResponse {
        status: "ok".to_string(),
        model_path: info.model_path,
        context_length: info.context_length,
        embedding_dimension: info.embedding_dimension,
        vocab_size: info.vocab_size,
    }))
}

async fn chat_completions_handler(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    // Auth check
    if !check_auth(&headers, &state.api_key) {
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "Invalid API key".to_string(),
            }),
        ));
    }

    // Rate limit check
    if let Some(ref rl) = state.rate_limiter {
        if !rl.check() {
            return Err((
                StatusCode::TOO_MANY_REQUESTS,
                Json(ErrorResponse {
                    error: "Rate limit exceeded".to_string(),
                }),
            ));
        }
    }

    let options = GenerationOptions {
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stop: req.stop,
        ..Default::default()
    };

    if req.stream {
        let (tx, rx) = mpsc::channel::<Result<String, String>>(64);
        let state = state.clone();
        let model = req.model.clone();
        let messages = req.messages.clone();
        let options = options.clone();

        tokio::task::spawn_blocking(move || {
            let mut runtime = state.runtime.lock().unwrap();
            let rt = match runtime.as_mut() {
                Some(rt) => rt,
                None => {
                    let _ = tx.blocking_send(Err("No model loaded".to_string()));
                    return;
                }
            };

            let prompt = match rt.apply_chat_template(&messages) {
                Ok(p) => p,
                Err(e) => {
                    let _ = tx.blocking_send(Err(format!("Template error: {}", e)));
                    return;
                }
            };

            let chunk_id = format!("chatcmpl-{}", uuid_simple());
            let created = now_secs();

            // Send role delta
            let role_chunk = ChatCompletionChunk {
                id: chunk_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.clone(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        role: Some("assistant".to_string()),
                        content: None,
                    },
                    finish_reason: None,
                }],
                usage: None,
            };
            if send_sse_data(&tx, &role_chunk).is_err() {
                return;
            }

            let result = rt.generate_with_callback_output(&prompt, &options, |piece| {
                let chunk = ChatCompletionChunk {
                    id: chunk_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: ChatDelta {
                            role: None,
                            content: Some(piece.to_string()),
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                };
                send_sse_data(&tx, &chunk).is_ok()
            });

            // Drop the runtime lock before sending final messages
            drop(runtime);

            match result {
                Ok(output) => {
                    let final_chunk = ChatCompletionChunk {
                        id: chunk_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model.clone(),
                        choices: vec![ChatChunkChoice {
                            index: 0,
                            delta: ChatDelta {
                                role: None,
                                content: None,
                            },
                            finish_reason: Some(
                                finish_reason_for_stop(&output.stop_reason).to_string(),
                            ),
                        }],
                        usage: Some(ChatUsage {
                            prompt_tokens: output.metrics.prompt_tokens,
                            completion_tokens: output.metrics.generated_tokens,
                            total_tokens: output.metrics.total_tokens,
                        }),
                    };
                    let _ = send_sse_data(&tx, &final_chunk);
                    let _ = tx.blocking_send(Ok("[DONE]".to_string()));
                }
                Err(e) => {
                    let _ = tx.blocking_send(Err(e.to_string()));
                }
            }
        });

        let stream = ReceiverStream::new(rx);
        Ok(Sse::new(stream.map(|item| match item {
            Ok(data) => Ok(axum::response::sse::Event::default().data(data)),
            Err(e) => Err(axum::Error::new(e)),
        }))
        .into_response())
    } else {
        let mut runtime = state.runtime.lock().unwrap();
        let rt = match runtime.as_mut() {
            Some(rt) => rt,
            None => {
                return Err((
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(ErrorResponse {
                        error: "No model loaded".to_string(),
                    }),
                ));
            }
        };

        let prompt = rt.apply_chat_template(&req.messages).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Template error: {}", e),
                }),
            )
        })?;

        let output = rt.generate(&prompt, &options).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;

        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{}", uuid_simple()),
            object: "chat.completion".to_string(),
            created: now_secs(),
            model: req.model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: output.text,
                },
                finish_reason: Some(finish_reason_for_stop(&output.stop_reason).to_string()),
            }],
            usage: ChatUsage {
                prompt_tokens: output.metrics.prompt_tokens,
                completion_tokens: output.metrics.generated_tokens,
                total_tokens: output.metrics.total_tokens,
            },
        };

        Ok(Json(response).into_response())
    }
}

fn send_sse_data<T: Serialize>(
    tx: &mpsc::Sender<Result<String, String>>,
    payload: &T,
) -> Result<(), mpsc::error::SendError<Result<String, String>>> {
    tx.blocking_send(Ok(serde_json::to_string(payload).unwrap()))
}

fn finish_reason_for_stop(stop_reason: &StopReason) -> &'static str {
    match stop_reason {
        StopReason::Eos | StopReason::Cancelled => "stop",
        StopReason::MaxTokens => "length",
    }
}

async fn embeddings_handler(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Auth check
    if !check_auth(&headers, &state.api_key) {
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "Invalid API key".to_string(),
            }),
        ));
    }

    // Rate limit check
    if let Some(ref rl) = state.rate_limiter {
        if !rl.check() {
            return Err((
                StatusCode::TOO_MANY_REQUESTS,
                Json(ErrorResponse {
                    error: "Rate limit exceeded".to_string(),
                }),
            ));
        }
    }

    let inputs = match req.input {
        EmbeddingInput::Single(s) => vec![s],
        EmbeddingInput::Multiple(v) => v,
    };

    let mut runtime = state.runtime.lock().unwrap();
    let rt = match runtime.as_mut() {
        Some(rt) => rt,
        None => {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse {
                    error: "No model loaded".to_string(),
                }),
            ));
        }
    };

    if !rt.embeddings_enabled() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Model does not support embeddings".to_string(),
            }),
        ));
    }

    let mut data = Vec::new();
    let mut total_tokens = 0usize;

    for (i, input) in inputs.iter().enumerate() {
        let tokens = rt.tokenize(input, true).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Tokenization failed: {}", e),
                }),
            )
        })?;
        total_tokens += tokens.len();

        let embedding = rt.embed(input).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Embedding failed: {}", e),
                }),
            )
        })?;

        data.push(EmbeddingObject {
            object: "embedding".to_string(),
            index: i,
            embedding,
        });
    }

    Ok(Json(EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: req.model,
        usage: EmbeddingUsageResponse {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    }))
}

// --- Helpers ---

#[cfg(test)]
fn build_prompt_from_messages(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str(&format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n", msg.content));
            }
            "user" => {
                if prompt.is_empty() {
                    prompt.push_str(&format!("[INST] {} [/INST]", msg.content));
                } else {
                    prompt.push_str(&format!("{} [/INST]", msg.content));
                }
            }
            "assistant" => {
                prompt.push_str(&msg.content);
                prompt.push_str(" ");
            }
            _ => {}
        }
    }
    prompt
}

fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:x}", t)
}

fn now_secs() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

// --- Auth middleware ---

fn check_auth(headers: &HeaderMap, api_key: &Option<String>) -> bool {
    let required = match api_key {
        Some(key) => key,
        None => return true,
    };

    // Check x-api-key header
    if let Some(header_key) = headers.get("x-api-key") {
        if let Ok(val) = header_key.to_str() {
            if val == required {
                return true;
            }
        }
    }

    // Check Authorization: Bearer
    if let Some(auth) = headers.get("authorization") {
        if let Ok(val) = auth.to_str() {
            if let Some(token) = val.strip_prefix("Bearer ") {
                if token == required {
                    return true;
                }
            }
        }
    }

    false
}

// --- Run server ---

pub async fn run_server(config: ServerConfig) -> Result<()> {
    let state = Arc::new(ServerState {
        runtime: Mutex::new(None),
        api_key: config.api_key.clone(),
        rate_limiter: config.rate_limit_rpm.map(RateLimiter::new),
        config: config.clone(),
    });

    // Preload model if configured
    if let Some(ref model_path) = config.model_path {
        let resolved = resolve_model_path(model_path)?;
        let llamacpp_config = config.to_llamacpp_config();
        match Runtime::new(resolved.to_str().unwrap(), llamacpp_config) {
            Ok(rt) => {
                eprintln!("Preloaded model: {}", rt.model_path());
                *state.runtime.lock().unwrap() = Some(rt);
            }
            Err(e) => {
                eprintln!("Warning: failed to preload model {}: {}", model_path, e);
            }
        }
    } else if let Some(ref model) = config.model {
        let resolved = resolve_model_path(model)?;
        let llamacpp_config = config.to_llamacpp_config();
        match Runtime::new(resolved.to_str().unwrap(), llamacpp_config) {
            Ok(rt) => {
                eprintln!("Preloaded model: {}", rt.model_path());
                *state.runtime.lock().unwrap() = Some(rt);
            }
            Err(e) => {
                eprintln!("Warning: failed to preload model {}: {}", model, e);
            }
        }
    }

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/v1/models", get(models_handler))
        .route("/llm/load", post(load_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/embeddings", post(embeddings_handler))
        .with_state(state);

    let bind = config.bind.as_deref().unwrap_or("127.0.0.1");
    let port = config.port.unwrap_or(8080);
    let addr = format!("{}:{}", bind, port);

    eprintln!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

pub async fn run_server_from_toml_path(path: impl AsRef<Path>) -> Result<()> {
    let cfg = ServerConfig::from_toml_path(path.as_ref())?;
    run_server(cfg).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_toml(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[test]
    fn test_parse_empty_config() {
        let cfg = ServerConfig::from_toml_path(Path::new("/nonexistent")).unwrap();
        assert!(cfg.bind.is_none());
        assert!(cfg.port.is_none());
        assert!(cfg.model_path.is_none());
    }

    #[test]
    fn test_parse_basic_server_config() {
        let toml = r#"
[server]
bind = "0.0.0.0:3000"
port = 8080
model_path = "/models/llama.gguf"
api_key = "secret"
rate_limit_rpm = 120
thinking = true
embeddings_batch_size = 32
"#;
        let f = write_temp_toml(toml);
        let cfg = ServerConfig::from_toml_path(f.path()).unwrap();
        assert_eq!(cfg.bind.as_deref(), Some("0.0.0.0:3000"));
        assert_eq!(cfg.port, Some(8080));
        assert_eq!(cfg.model_path.as_deref(), Some("/models/llama.gguf"));
        assert_eq!(cfg.api_key.as_deref(), Some("secret"));
        assert_eq!(cfg.rate_limit_rpm, Some(120));
        assert_eq!(cfg.thinking, Some(true));
        assert_eq!(cfg.embeddings_batch_size, Some(32));
    }

    #[test]
    fn test_parse_llamacpp_config_keys() {
        let toml = r#"
[server]
n_ctx = 4096
n_batch = 512
n_ubatch = 256
n_gpu_layers = 99
n_threads = 8
n_threads_batch = 8
embedding = true
pooling = "mean"
use_mmap = true
use_mlock = false
flash_attn = true
"#;
        let f = write_temp_toml(toml);
        let cfg = ServerConfig::from_toml_path(f.path()).unwrap();
        assert_eq!(cfg.n_ctx, Some(4096));
        assert_eq!(cfg.n_batch, Some(512));
        assert_eq!(cfg.n_ubatch, Some(256));
        assert_eq!(cfg.n_gpu_layers, Some(99));
        assert_eq!(cfg.n_threads, Some(8));
        assert_eq!(cfg.n_threads_batch, Some(8));
        assert_eq!(cfg.embedding, Some(true));
        assert_eq!(cfg.pooling.as_deref(), Some("mean"));
        assert_eq!(cfg.use_mmap, Some(true));
        assert_eq!(cfg.use_mlock, Some(false));
        assert_eq!(cfg.flash_attn, Some(true));
    }

    #[test]
    fn test_parse_comments_and_blanks() {
        let toml = r#"
# This is a comment

[server]
bind = "127.0.0.1:3000"
# model_path = "/ignored"
port = 3001
"#;
        let f = write_temp_toml(toml);
        let cfg = ServerConfig::from_toml_path(f.path()).unwrap();
        assert_eq!(cfg.bind.as_deref(), Some("127.0.0.1:3000"));
        assert_eq!(cfg.port, Some(3001));
        assert!(cfg.model_path.is_none());
    }

    #[test]
    fn test_parse_partial_config() {
        let toml = r#"
[server]
model = "llama-3.2-1b"
n_gpu_layers = 99
"#;
        let f = write_temp_toml(toml);
        let cfg = ServerConfig::from_toml_path(f.path()).unwrap();
        assert_eq!(cfg.model.as_deref(), Some("llama-3.2-1b"));
        assert_eq!(cfg.n_gpu_layers, Some(99));
        assert!(cfg.bind.is_none());
        assert!(cfg.n_ctx.is_none());
    }

    #[test]
    fn test_to_llamacpp_config() {
        let cfg = ServerConfig {
            n_ctx: Some(2048),
            n_gpu_layers: Some(32),
            flash_attn: Some(true),
            ..Default::default()
        };
        let llamacpp = cfg.to_llamacpp_config();
        assert_eq!(llamacpp.n_ctx, Some(2048));
        assert_eq!(llamacpp.n_gpu_layers, Some(32));
        assert_eq!(llamacpp.flash_attn, Some(true));
        assert!(llamacpp.n_batch.is_none());
    }

    #[test]
    fn test_toml_value_types() {
        assert_eq!(parse_toml_value("true"), serde_json::Value::Bool(true));
        assert_eq!(parse_toml_value("false"), serde_json::Value::Bool(false));
        assert_eq!(
            parse_toml_value("42"),
            serde_json::Value::Number(42u64.into())
        );
        assert_eq!(
            parse_toml_value("\"hello\""),
            serde_json::Value::String("hello".to_string())
        );
        assert_eq!(
            parse_toml_value("mean"),
            serde_json::Value::String("mean".to_string())
        );
    }

    #[test]
    fn test_rate_limiter_allows() {
        let rl = RateLimiter::new(10);
        assert!(rl.check());
    }

    #[test]
    fn test_rate_limiter_blocks() {
        let rl = RateLimiter::new(2);
        assert!(rl.check());
        assert!(rl.check());
        assert!(!rl.check());
    }

    #[test]
    fn test_check_auth_no_key() {
        let headers = HeaderMap::new();
        assert!(check_auth(&headers, &None));
    }

    #[test]
    fn test_check_auth_x_api_key() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "secret".parse().unwrap());
        assert!(check_auth(&headers, &Some("secret".to_string())));
        assert!(!check_auth(&headers, &Some("other".to_string())));
    }

    #[test]
    fn test_check_auth_bearer() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer token123".parse().unwrap());
        assert!(check_auth(&headers, &Some("token123".to_string())));
        assert!(!check_auth(&headers, &Some("wrong".to_string())));
    }

    #[test]
    fn test_check_auth_missing() {
        let headers = HeaderMap::new();
        assert!(!check_auth(&headers, &Some("required".to_string())));
    }

    #[tokio::test]
    async fn test_health_handler_without_model() {
        let state = Arc::new(ServerState {
            runtime: Mutex::new(None),
            config: ServerConfig::default(),
            api_key: None,
            rate_limiter: None,
        });

        let Json(resp) = health_handler(State(state)).await;
        assert_eq!(resp.status, "ok");
        assert!(!resp.model_loaded);
    }

    #[tokio::test]
    async fn test_models_handler_empty_before_load() {
        let state = Arc::new(ServerState {
            runtime: Mutex::new(None),
            config: ServerConfig::default(),
            api_key: None,
            rate_limiter: None,
        });

        let Json(resp) = models_handler(State(state)).await;
        assert_eq!(resp.object, "list");
        assert!(resp.data.is_empty());
    }

    #[tokio::test]
    async fn test_load_invalid_model_path() {
        let state = Arc::new(ServerState {
            runtime: Mutex::new(None),
            config: ServerConfig::default(),
            api_key: None,
            rate_limiter: None,
        });

        let result = load_handler(
            State(state),
            Json(LoadRequest {
                model_path: "/definitely/missing/model.gguf".to_string(),
            }),
        )
        .await;

        let err = match result {
            Ok(_) => panic!("missing model path should fail"),
            Err(err) => err,
        };
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert!(err.1 .0.error.contains("does not exist"));
    }

    #[tokio::test]
    async fn test_chat_auth_failure_handler() {
        let state = Arc::new(ServerState {
            runtime: Mutex::new(None),
            config: ServerConfig::default(),
            api_key: Some("secret".to_string()),
            rate_limiter: None,
        });

        let err = chat_completions_handler(
            State(state),
            HeaderMap::new(),
            Json(ChatCompletionRequest {
                model: "local".to_string(),
                messages: vec![ChatMessage::user("hello")],
                max_tokens: Some(1),
                temperature: 0.0,
                top_p: 1.0,
                stream: false,
                stop: None,
            }),
        )
        .await
        .unwrap_err();

        assert_eq!(err.0, StatusCode::UNAUTHORIZED);
        assert_eq!(err.1 .0.error, "Invalid API key");
    }

    #[tokio::test]
    async fn test_models_handler_after_env_gated_load() {
        let model_path = match std::env::var("MLX_RS_TEST_GGUF") {
            Ok(path) => path,
            Err(_) => {
                eprintln!("Skipping model load handler test: MLX_RS_TEST_GGUF not set");
                return;
            }
        };

        let state = Arc::new(ServerState {
            runtime: Mutex::new(None),
            config: ServerConfig {
                n_ctx: Some(512),
                n_gpu_layers: Some(0),
                ..Default::default()
            },
            api_key: None,
            rate_limiter: None,
        });

        let result = load_handler(
            State(state.clone()),
            Json(LoadRequest {
                model_path: model_path.clone(),
            }),
        )
        .await;
        if let Err((status, Json(error))) = result {
            panic!(
                "env-gated model should load, got {}: {}",
                status, error.error
            );
        }

        let Json(resp) = models_handler(State(state)).await;
        assert_eq!(resp.object, "list");
        assert_eq!(resp.data.len(), 1);
        assert_eq!(resp.data[0].object, "model");
    }

    #[test]
    fn test_build_prompt_from_messages() {
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello"),
        ];
        let prompt = build_prompt_from_messages(&messages);
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains("Hello"));
        assert!(prompt.contains("[INST]"));
    }

    #[test]
    fn test_chat_finish_reason_mapping() {
        assert_eq!(finish_reason_for_stop(&StopReason::Eos), "stop");
        assert_eq!(finish_reason_for_stop(&StopReason::Cancelled), "stop");
        assert_eq!(finish_reason_for_stop(&StopReason::MaxTokens), "length");
    }

    #[test]
    fn test_chat_completion_response_shape() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion".to_string(),
            created: 123,
            model: "local".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage::assistant("hello"),
                finish_reason: Some(finish_reason_for_stop(&StopReason::MaxTokens).to_string()),
            }],
            usage: ChatUsage {
                prompt_tokens: 2,
                completion_tokens: 3,
                total_tokens: 5,
            },
        };

        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["choices"][0]["message"]["role"], "assistant");
        assert_eq!(json["choices"][0]["message"]["content"], "hello");
        assert_eq!(json["choices"][0]["finish_reason"], "length");
        assert_eq!(json["usage"]["prompt_tokens"], 2);
        assert_eq!(json["usage"]["completion_tokens"], 3);
        assert_eq!(json["usage"]["total_tokens"], 5);
    }

    #[test]
    fn test_streaming_chunk_shapes() {
        let first = ChatCompletionChunk {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 123,
            model: "local".to_string(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };
        let token = ChatCompletionChunk {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 123,
            model: "local".to_string(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: None,
                    content: Some("hello".to_string()),
                },
                finish_reason: None,
            }],
            usage: None,
        };
        let final_chunk = ChatCompletionChunk {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 123,
            model: "local".to_string(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: None,
                    content: None,
                },
                finish_reason: Some("length".to_string()),
            }],
            usage: Some(ChatUsage {
                prompt_tokens: 2,
                completion_tokens: 3,
                total_tokens: 5,
            }),
        };

        let chunks = vec![
            serde_json::to_value(&first).unwrap(),
            serde_json::to_value(&token).unwrap(),
            serde_json::to_value(&final_chunk).unwrap(),
        ];

        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(chunks[1]["choices"][0]["delta"]["content"], "hello");
        assert_eq!(chunks[2]["choices"][0]["finish_reason"], "length");
        assert_eq!(chunks[2]["usage"]["total_tokens"], 5);
        assert!(chunks[0].get("usage").is_none());
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn test_streaming_closed_channel_cancels_send() {
        let (tx, rx) = mpsc::channel::<Result<String, String>>(1);
        drop(rx);

        let chunk = ChatCompletionChunk {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 123,
            model: "local".to_string(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: None,
                    content: Some("hello".to_string()),
                },
                finish_reason: None,
            }],
            usage: None,
        };

        assert!(send_sse_data(&tx, &chunk).is_err());
    }

    #[test]
    fn test_streaming_send_uses_raw_event_data() {
        let (tx, mut rx) = mpsc::channel::<Result<String, String>>(1);
        let chunk = ChatCompletionChunk {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 123,
            model: "local".to_string(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: None,
                    content: Some("hello".to_string()),
                },
                finish_reason: None,
            }],
            usage: None,
        };

        send_sse_data(&tx, &chunk).unwrap();
        let data = rx.blocking_recv().unwrap().unwrap();
        assert!(data.starts_with('{'));
        assert!(!data.starts_with("data:"));
        assert!(data.contains("\"chat.completion.chunk\""));
    }

    #[test]
    fn test_embedding_request_single_input() {
        let json = r#"{"input": "hello world", "model": "test-model"}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        match req.input {
            EmbeddingInput::Single(s) => assert_eq!(s, "hello world"),
            _ => panic!("Expected Single"),
        }
    }

    #[test]
    fn test_embedding_request_array_input() {
        let json = r#"{"input": ["hello", "world"], "model": "test-model"}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        match req.input {
            EmbeddingInput::Multiple(v) => assert_eq!(v, vec!["hello", "world"]),
            _ => panic!("Expected Multiple"),
        }
    }

    #[test]
    fn test_embedding_rejects_token_array_input() {
        let json = r#"{"input": [1, 2, 3], "model": "test-model"}"#;
        let err = match serde_json::from_str::<EmbeddingRequest>(json) {
            Ok(_) => panic!("token arrays should not deserialize as embedding requests"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("data did not match any variant"));
    }

    #[tokio::test]
    async fn test_embedding_non_embedding_model_returns_unsupported() {
        let model_path = match std::env::var("MLX_RS_TEST_GGUF") {
            Ok(path) => path,
            Err(_) => {
                eprintln!("Skipping non-embedding model handler test: MLX_RS_TEST_GGUF not set");
                return;
            }
        };

        let runtime = Runtime::new(
            &model_path,
            LlamaCppConfig {
                n_ctx: Some(512),
                embedding: Some(false),
                ..Default::default()
            },
        )
        .expect("Failed to load non-embedding model");

        let state = Arc::new(ServerState {
            runtime: Mutex::new(Some(runtime)),
            config: ServerConfig::default(),
            api_key: None,
            rate_limiter: None,
        });

        let result = embeddings_handler(
            State(state),
            HeaderMap::new(),
            Json(EmbeddingRequest {
                input: EmbeddingInput::Single("hello".to_string()),
                model: "local".to_string(),
            }),
        )
        .await;

        let err = match result {
            Ok(_) => panic!("non-embedding runtime should reject embeddings"),
            Err(err) => err,
        };
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert_eq!(err.1 .0.error, "Model does not support embeddings");
    }

    #[test]
    fn test_embedding_response_shape() {
        let resp = EmbeddingResponse {
            object: "list".to_string(),
            data: vec![EmbeddingObject {
                object: "embedding".to_string(),
                index: 0,
                embedding: vec![0.1, 0.2, 0.3],
            }],
            model: "test".to_string(),
            usage: EmbeddingUsageResponse {
                prompt_tokens: 5,
                total_tokens: 5,
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"object\":\"list\""));
        assert!(json.contains("\"embedding\":[0.1,0.2,0.3]"));
        assert!(json.contains("\"prompt_tokens\":5"));
    }
}
