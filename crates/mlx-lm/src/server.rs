use crate::{
    load_model, CausalLM, ChatTemplate, ChatTemplateOptions, GenerationPipeline,
    Message as LmMessage, Sampler, Tokenizer,
};
use anyhow::Result;
use serde::Deserialize;
use serde_json::json;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::Instant;

struct LoadedModel {
    model_dir: PathBuf,
    model: Box<dyn CausalLM>,
    tokenizer: Tokenizer,
}

fn release_loaded_model(loaded: &mut Option<LoadedModel>) {
    let _ = mlx_core::Stream::new_gpu_default().synchronize();
    let _ = mlx_core::Stream::new_cpu_default().synchronize();
    if let Some(mut model) = loaded.take() {
        model.model.clear_cache();
        drop(model);
    }
    mlx_core::metal::set_cache_limit(0);
    mlx_core::metal::clear_cache();
    mlx_core::metal::clear_compile_cache();
    let _ = mlx_core::Stream::new_gpu_default().synchronize();
    let _ = mlx_core::Stream::new_cpu_default().synchronize();
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    pub bind: Option<String>,
    pub port: Option<u16>,
    pub model_path: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub rate_limit_rpm: Option<u32>,
    pub thinking: Option<bool>,
}

impl ServerConfig {
    pub fn from_toml_path(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        Self::from_toml_str(&content)
    }

    pub fn from_toml_str(content: &str) -> Result<Self> {
        let mut in_server = false;
        let mut cfg = ServerConfig {
            bind: None,
            port: None,
            model_path: None,
            model: None,
            api_key: None,
            rate_limit_rpm: None,
            thinking: None,
        };

        for raw_line in content.lines() {
            let line = raw_line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with('[') && line.ends_with(']') {
                in_server = line == "[server]";
                continue;
            }
            if !in_server {
                continue;
            }
            let (k, v) = match line.split_once('=') {
                Some((k, v)) => (k.trim(), v.trim()),
                None => continue,
            };

            let unquote = |s: &str| -> String {
                let s = s.trim();
                if s.len() >= 2 && s.starts_with('\"') && s.ends_with('\"') {
                    s[1..s.len() - 1].to_string()
                } else {
                    s.to_string()
                }
            };

            match k {
                "bind" => cfg.bind = Some(unquote(v)),
                "port" => cfg.port = unquote(v).parse::<u16>().ok(),
                "model_path" => cfg.model_path = Some(unquote(v)),
                "model" => cfg.model = Some(unquote(v)),
                "api_key" => cfg.api_key = Some(unquote(v)),
                "rate_limit_rpm" => cfg.rate_limit_rpm = unquote(v).parse::<u32>().ok(),
                "thinking" => {
                    let vv = unquote(v).to_ascii_lowercase();
                    cfg.thinking = match vv.as_str() {
                        "true" | "1" | "yes" | "on" => Some(true),
                        "false" | "0" | "no" | "off" => Some(false),
                        _ => None,
                    };
                }
                _ => {}
            }
        }

        Ok(cfg)
    }

    fn addr(&self) -> String {
        if let Some(bind) = &self.bind {
            return bind.clone();
        }
        match self.port {
            Some(port) => format!("0.0.0.0:{port}"),
            None => "127.0.0.1:3000".to_string(),
        }
    }

    fn startup_model_path(&self) -> Option<PathBuf> {
        self.model_path
            .as_ref()
            .or(self.model.as_ref())
            .map(PathBuf::from)
    }

    fn rate_limit(&self) -> Option<u32> {
        match self.rate_limit_rpm {
            Some(0) | None => None,
            Some(v) => Some(v),
        }
    }

    fn thinking_enabled(&self) -> bool {
        self.thinking.unwrap_or(false)
    }
}

#[derive(Debug, Deserialize)]
struct LoadRequest {
    model_path: String,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stream: Option<bool>,
}

struct HttpRequest {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

struct FixedWindowRateLimiter {
    rpm: u32,
    window_start: Instant,
    count: u32,
}

impl FixedWindowRateLimiter {
    fn new(rpm: u32) -> Self {
        Self {
            rpm,
            window_start: Instant::now(),
            count: 0,
        }
    }

    fn try_acquire(&mut self) -> bool {
        if self.window_start.elapsed().as_secs() >= 60 {
            self.window_start = Instant::now();
            self.count = 0;
        }
        if self.count >= self.rpm {
            return false;
        }
        self.count += 1;
        true
    }
}

fn read_http_request(stream: &mut TcpStream) -> Result<HttpRequest> {
    stream.set_read_timeout(Some(std::time::Duration::from_secs(30)))?;
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];

    let header_end;
    loop {
        let n = stream.read(&mut tmp)?;
        if n == 0 {
            return Err(anyhow::anyhow!("connection closed"));
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            header_end = pos + 4;
            break;
        }
        if buf.len() > 1024 * 1024 {
            return Err(anyhow::anyhow!("request headers too large"));
        }
    }

    let header_bytes = &buf[..header_end];
    let header_text = String::from_utf8_lossy(header_bytes);
    let mut lines = header_text.lines();
    let request_line = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing request line"))?;
    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing method"))?
        .to_string();
    let path = parts
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing path"))?
        .to_string();

    let mut content_length = 0usize;
    let mut headers = HashMap::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            let key = name.trim().to_ascii_lowercase();
            let val = value.trim().to_string();
            if key == "content-length" {
                content_length = val.parse::<usize>().unwrap_or(0);
            }
            headers.insert(key, val);
        }
    }

    let mut body = buf[header_end..].to_vec();
    while body.len() < content_length {
        let n = stream.read(&mut tmp)?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
    }
    body.truncate(content_length);

    Ok(HttpRequest {
        method,
        path,
        headers,
        body,
    })
}

fn write_json(stream: &mut TcpStream, status: u16, value: serde_json::Value) -> Result<()> {
    let body = serde_json::to_vec(&value)?;
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        _ => "OK",
    };
    let headers = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status,
        status_text,
        body.len()
    );
    stream.write_all(headers.as_bytes())?;
    stream.write_all(&body)?;
    stream.flush()?;
    Ok(())
}

fn write_chunk(stream: &mut TcpStream, bytes: &[u8]) -> Result<()> {
    let header = format!("{:X}\r\n", bytes.len());
    stream.write_all(header.as_bytes())?;
    stream.write_all(bytes)?;
    stream.write_all(b"\r\n")?;
    Ok(())
}

fn write_sse_headers(stream: &mut TcpStream) -> Result<()> {
    let headers = concat!(
        "HTTP/1.1 200 OK\r\n",
        "Content-Type: text/event-stream\r\n",
        "Cache-Control: no-cache\r\n",
        "Connection: close\r\n",
        "Transfer-Encoding: chunked\r\n",
        "\r\n"
    );
    stream.write_all(headers.as_bytes())?;
    stream.flush()?;
    Ok(())
}

fn write_sse_event(stream: &mut TcpStream, value: serde_json::Value) -> Result<()> {
    let payload = format!("data: {}\n\n", serde_json::to_string(&value)?);
    write_chunk(stream, payload.as_bytes())?;
    stream.flush()?;
    Ok(())
}

fn write_sse_done(stream: &mut TcpStream) -> Result<()> {
    write_chunk(stream, b"data: [DONE]\n\n")?;
    stream.write_all(b"0\r\n\r\n")?;
    stream.flush()?;
    Ok(())
}

fn error_json(code_name: &str, message: impl ToString) -> serde_json::Value {
    json!({
        "error": {
            "type": "invalid_request_error",
            "code": code_name,
            "message": message.to_string()
        }
    })
}

fn generation_debug_json(
    metrics: &crate::GenerationMetrics,
    prompt_token_count: usize,
    prompt_render_s: f64,
    prompt_tokenize_s: f64,
    stream_write_s: f64,
) -> Value {
    let mut debug = Map::new();
    debug.insert("ttft_s".into(), json!(metrics.ttft_s));
    debug.insert("total_s".into(), json!(metrics.total_s));
    debug.insert("tokens".into(), json!(metrics.tokens));
    debug.insert(
        "tokens_per_s".into(),
        json!(if metrics.total_s > 0.0 {
            metrics.tokens as f64 / metrics.total_s
        } else {
            0.0
        }),
    );
    debug.insert("stop_reason".into(), json!(metrics.stop_reason));
    debug.insert("last_token_id".into(), json!(metrics.last_token_id));

    if let Some(profile) = &metrics.profile {
        debug.insert(
            "profile".into(),
            json!({
                "prompt_render_s": prompt_render_s,
                "prompt_tokenize_s": prompt_tokenize_s,
                "stream_write_s": stream_write_s,
                "clear_cache_s": profile.clear_cache_s,
                "tokenize_s": profile.tokenize_s,
                "prefill_forward_s": profile.prefill_forward_s,
                "prefill_eval_s": profile.prefill_eval_s,
                "prefill_normalize_s": profile.prefill_normalize_s,
                "first_sample_s": profile.first_sample_s,
                "first_decode_s": profile.first_decode_s,
                "decode_forward_s": profile.decode_forward_s,
                "decode_eval_s": profile.decode_eval_s,
                "decode_normalize_s": profile.decode_normalize_s,
                "sample_s": profile.sample_s,
                "decode_text_s": profile.decode_text_s,
                "eval_calls": profile.eval_calls,
                "sample_calls": profile.sample_calls,
                "cpu_logits_extractions": profile.cpu_logits_extractions,
                "decoded_pieces": profile.decoded_pieces,
                "kv_cache_allocations": profile.kv_cache_allocations,
                "kv_cache_growths": profile.kv_cache_growths,
                "moe_router_host_s": profile.moe_router_host_s,
                "moe_routing_build_s": profile.moe_routing_build_s,
                "moe_expert_forward_s": profile.moe_expert_forward_s,
                "moe_shared_expert_s": profile.moe_shared_expert_s,
                "moe_single_token_fast_path_hits": profile.moe_single_token_fast_path_hits,
                "prompt_tokens": prompt_token_count
            }),
        );
    }

    Value::Object(debug)
}

fn apply_chat_template(model_dir: &PathBuf, req: &ChatRequest, thinking: bool) -> Result<String> {
    let mut msgs = Vec::with_capacity(req.messages.len());
    for m in &req.messages {
        match m.role.as_str() {
            "system" => msgs.push(LmMessage::system(&m.content)),
            "assistant" => msgs.push(LmMessage::assistant(&m.content)),
            _ => msgs.push(LmMessage::user(&m.content)),
        }
    }
    let options = ChatTemplateOptions {
        add_generation_prompt: true,
        continue_final_message: false,
        enable_thinking: thinking,
    };

    if let Ok(template) = ChatTemplate::from_model_dir(model_dir) {
        if let Ok(rendered) = template.apply(&msgs, &options) {
            return Ok(rendered);
        }
        // Incompatible Jinja template: fall back to ChatML, then Qwen3.5-style.
        return ChatTemplate::chatml()
            .apply(&msgs, &options)
            .or_else(|_| ChatTemplate::qwen35().apply(&msgs, &options))
            .map_err(|e| anyhow::anyhow!("chat template apply failed (fallback): {e}"));
    }

    let mut out = String::new();
    for m in &req.messages {
        out.push_str(&m.role);
        out.push_str(": ");
        out.push_str(&m.content);
        out.push('\n');
    }
    Ok(out)
}

fn check_api_key(req: &HttpRequest, api_key: &str) -> bool {
    if let Some(v) = req.headers.get("x-api-key") {
        if v == api_key {
            return true;
        }
    }
    if let Some(v) = req.headers.get("authorization") {
        if let Some(token) = v.strip_prefix("Bearer ") {
            return token.trim() == api_key;
        }
    }
    false
}

fn handle_request(
    req: HttpRequest,
    stream: &mut TcpStream,
    loaded: &mut Option<LoadedModel>,
    api_key: Option<&str>,
    limiter: &mut Option<FixedWindowRateLimiter>,
    thinking: bool,
) -> Option<(u16, serde_json::Value)> {
    if req.path != "/health" {
        if let Some(key) = api_key {
            if !check_api_key(&req, key) {
                return Some((
                    401,
                    error_json("unauthorized", "missing or invalid API key"),
                ));
            }
        }
    }

    if req.path == "/v1/chat/completions" {
        if let Some(limit) = limiter.as_mut() {
            if !limit.try_acquire() {
                return Some((
                    429,
                    error_json("rate_limit_exceeded", "rate limit exceeded"),
                ));
            }
        }
    }

    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => Some((200, json!({"status": "ok"}))),
        ("GET", "/v1/models") => {
            let data = match loaded.as_ref() {
                Some(m) => vec![json!({"id": m.model_dir.display().to_string(), "object": "model"})],
                None => vec![],
            };
            Some((200, json!({"data": data})))
        }
        ("POST", "/llm/load") => {
            let parsed: LoadRequest = match serde_json::from_slice(&req.body) {
                Ok(v) => v,
                Err(e) => return Some((400, error_json("invalid_json", format!("bad JSON: {e}")))),
            };
            let model_path = PathBuf::from(parsed.model_path);
            match load_model(&model_path) {
                Ok((model, tokenizer)) => {
                    let next = LoadedModel {
                        model_dir: model_path,
                        model,
                        tokenizer,
                    };
                    release_loaded_model(loaded);
                    *loaded = Some(next);
                    Some((200, json!({"ok": true})))
                }
                Err(e) => Some((
                    400,
                    error_json("model_load_error", format!("Model load error: {e}")),
                )),
            }
        }
        ("POST", "/llm/unload") => {
            release_loaded_model(loaded);
            let mem = mlx_core::metal::memory_info();
            Some((
                200,
                json!({
                    "ok": true,
                    "memory": {
                        "active_bytes": mem.active_memory,
                        "cache_bytes": mem.cache_memory,
                        "peak_bytes": mem.peak_memory
                    }
                }),
            ))
        }
        ("POST", "/v1/chat/completions") => {
            let parsed: ChatRequest = match serde_json::from_slice(&req.body) {
                Ok(v) => v,
                Err(e) => return Some((400, error_json("invalid_json", format!("bad JSON: {e}")))),
            };
            let lm = match loaded.as_mut() {
                Some(v) => v,
                None => {
                    return Some((
                        400,
                        error_json("model_not_loaded", "No model loaded. Call /llm/load first."),
                    ))
                }
            };
            let prompt_render_t0 = Instant::now();
            let prompt = match apply_chat_template(&lm.model_dir, &parsed, thinking) {
                Ok(p) => p,
                Err(e) => return Some((400, error_json("template_error", e.to_string()))),
            };
            let prompt_render_s = prompt_render_t0.elapsed().as_secs_f64();
            let max_tokens = parsed.max_tokens;
            let temperature = parsed.temperature.unwrap_or(0.0);
            let top_p = parsed.top_p.unwrap_or(1.0);
            let sampler = Sampler::new(temperature, top_p);
            let prompt_tokenize_t0 = Instant::now();
            let prompt_token_count = lm.tokenizer.encode(&prompt).map(|v| v.len()).unwrap_or(0);
            let prompt_tokenize_s = prompt_tokenize_t0.elapsed().as_secs_f64();

            let mut pipeline = GenerationPipeline::new(&mut lm.model, lm.tokenizer.clone(), sampler);
            if parsed.stream.unwrap_or(false) {
                if let Err(e) = write_sse_headers(stream) {
                    return Some((500, error_json("stream_write_error", e.to_string())));
                }
                let mut stream_error: Option<anyhow::Error> = None;
                let created = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                let id = "chatcmpl-local";
                let mut stream_write_s = 0.0f64;

                match pipeline.generate_with_metrics(&prompt, max_tokens, |_token, piece| {
                    if stream_error.is_some() {
                        return;
                    }
                    let event = json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": piece},
                            "finish_reason": serde_json::Value::Null
                        }]
                    });
                    let write_t0 = Instant::now();
                    if let Err(e) = write_sse_event(stream, event) {
                        stream_error = Some(anyhow::anyhow!(e.to_string()));
                    } else {
                        stream_write_s += write_t0.elapsed().as_secs_f64();
                    }
                }) {
                    Ok((_text, metrics)) => {
                        if let Some(e) = stream_error {
                            return Some((500, error_json("stream_write_error", e.to_string())));
                        }
                        let finish_reason = if metrics.stop_reason == "length" {
                            "length"
                        } else {
                            "stop"
                        };
                        let final_event = json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish_reason
                            }],
                            "usage": {
                                "prompt_tokens": prompt_token_count,
                                "completion_tokens": metrics.tokens,
                                "total_tokens": prompt_token_count + metrics.tokens
                            },
                            "debug": generation_debug_json(&metrics, prompt_token_count, prompt_render_s, prompt_tokenize_s, stream_write_s)
                        });
                        if let Err(e) = write_sse_event(stream, final_event) {
                            return Some((500, error_json("stream_write_error", e.to_string())));
                        }
                        if let Err(e) = write_sse_done(stream) {
                            return Some((500, error_json("stream_write_error", e.to_string())));
                        }
                        return None;
                    }
                    Err(e) => {
                        let event = json!({
                            "error": {
                                "type": "server_error",
                                "code": "generation_error",
                                "message": e.to_string()
                            }
                        });
                        let _ = write_sse_event(stream, event);
                        let _ = write_sse_done(stream);
                        return None;
                    }
                }
            }

            match pipeline.generate_with_metrics(&prompt, max_tokens, |_token, _piece| {}) {
                Ok((text, metrics)) => {
                    let finish_reason = if metrics.stop_reason == "length" {
                        "length"
                    } else {
                        "stop"
                    };
                    Some((
                        200,
                        json!({
                            "id": "chatcmpl-local",
                            "object": "chat.completion",
                            "choices": [{
                                "index": 0,
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": finish_reason
                            }],
                            "usage": {
                                "prompt_tokens": prompt_token_count,
                                "completion_tokens": metrics.tokens,
                                "total_tokens": prompt_token_count + metrics.tokens
                            },
                            "debug": generation_debug_json(&metrics, prompt_token_count, prompt_render_s, prompt_tokenize_s, 0.0)
                        }),
                    ))
                }
                Err(e) => Some((500, error_json("generation_error", e.to_string()))),
            }
        }
        _ => Some((404, error_json("not_found", "endpoint not found"))),
    }
}

pub fn run_server(config: ServerConfig) -> Result<()> {
    let addr = config.addr();
    let listener = TcpListener::bind(&addr)?;
    eprintln!("mlx-server listening on {addr}");

    let mut loaded: Option<LoadedModel> = None;
    if let Some(model_path) = config.startup_model_path() {
        eprintln!("Loading startup model from {:?}...", model_path);
        let (model, tokenizer) = load_model(&model_path)?;
        loaded = Some(LoadedModel {
            model_dir: model_path,
            model,
            tokenizer,
        });
    }

    let mut limiter = config.rate_limit().map(FixedWindowRateLimiter::new);
    let api_key = config.api_key.clone();
    let thinking = config.thinking_enabled();

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        let req = match read_http_request(&mut stream) {
            Ok(r) => r,
            Err(e) => {
                let _ = write_json(&mut stream, 400, error_json("bad_request", e.to_string()));
                continue;
            }
        };
        if let Some((status, body)) = handle_request(
            req,
            &mut stream,
            &mut loaded,
            api_key.as_deref(),
            &mut limiter,
            thinking,
        ) {
            let _ = write_json(&mut stream, status, body);
        }
    }

    Ok(())
}

pub fn run_server_from_toml_path(path: impl AsRef<Path>) -> Result<()> {
    let cfg = ServerConfig::from_toml_path(path)?;
    run_server(cfg)
}
