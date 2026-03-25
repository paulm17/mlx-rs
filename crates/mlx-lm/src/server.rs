use crate::{
    load_model, resolve_model_dir, ChatTemplate, ChatTemplateOptions, EmbeddingPooling,
    GenerationPipeline, HuggingFaceOptions, Message as LmMessage, ModelRuntime, Sampler, Tokenizer,
};
use anyhow::Result;
use mlx_core::Array;
use serde::Deserialize;
use serde_json::json;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use std::time::Instant;

struct LoadedModel {
    model_dir: PathBuf,
    model: Box<dyn ModelRuntime>,
    tokenizer: Tokenizer,
}

fn sampler_for_model(model_dir: &Path, temperature: f32, top_p: f32) -> Sampler {
    let mut sampler = Sampler::new(temperature, top_p);
    let config_path = model_dir.join("config.json");
    if let Ok(config_str) = std::fs::read_to_string(&config_path) {
        if let Ok(config) = serde_json::from_str::<Value>(&config_str) {
            if let Ok(arch) = crate::loader::detect_architecture(&config) {
                if matches!(
                    arch,
                    crate::loader::ModelArch::QwenMoe | crate::loader::ModelArch::QwenMoePythonPort
                ) {
                    sampler = sampler
                        .with_greedy_tie_break(0.05)
                        .with_greedy_tie_break_after(180);
                }
            }
        }
    }
    sampler
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
    pub huggingface: HuggingFaceOptions,
    pub api_key: Option<String>,
    pub rate_limit_rpm: Option<u32>,
    pub thinking: Option<bool>,
    pub embeddings_batch_size: Option<usize>,
}

impl ServerConfig {
    pub fn from_toml_path(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        Self::from_toml_str(&content)
    }

    pub fn from_toml_str(content: &str) -> Result<Self> {
        let mut in_server = false;
        let mut in_huggingface = false;
        let mut cfg = ServerConfig {
            bind: None,
            port: None,
            model_path: None,
            model: None,
            huggingface: HuggingFaceOptions::default(),
            api_key: None,
            rate_limit_rpm: None,
            thinking: None,
            embeddings_batch_size: None,
        };

        for raw_line in content.lines() {
            let line = raw_line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with('[') && line.ends_with(']') {
                in_server = line == "[server]";
                in_huggingface = line == "[huggingface]";
                continue;
            }
            if !in_server && !in_huggingface {
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

            if in_server {
                match k {
                    "bind" => cfg.bind = Some(unquote(v)),
                    "port" => cfg.port = unquote(v).parse::<u16>().ok(),
                    "model_path" => cfg.model_path = Some(unquote(v)),
                    "model" => cfg.model = Some(unquote(v)),
                    "api_key" => cfg.api_key = Some(unquote(v)),
                    "rate_limit_rpm" => cfg.rate_limit_rpm = unquote(v).parse::<u32>().ok(),
                    "embeddings_batch_size" => {
                        cfg.embeddings_batch_size = unquote(v).parse::<usize>().ok()
                    }
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
            } else if in_huggingface && k == "hf_token" {
                cfg.huggingface.hf_token = Some(unquote(v));
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

    fn embeddings_batch_size(&self) -> usize {
        self.embeddings_batch_size.unwrap_or(32).max(1)
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

#[derive(Debug, Clone, Deserialize)]
struct EmbeddingsRequest {
    input: Value,
    model: Option<String>,
    encoding_format: Option<String>,
}

fn parse_embeddings_input(input: Value) -> Result<Vec<String>> {
    match input {
        Value::String(s) => Ok(vec![s]),
        Value::Array(values) => values
            .into_iter()
            .map(|value| match value {
                Value::String(s) => Ok(s),
                Value::Array(_) => anyhow::bail!("token array inputs are not supported"),
                _ => anyhow::bail!("input array entries must be strings"),
            })
            .collect(),
        Value::Null => anyhow::bail!("\"input\" must be provided"),
        _ => anyhow::bail!("\"input\" must be a string or an array of strings"),
    }
}

fn normalize_embedding(mut embedding: Vec<f32>) -> Result<Vec<f32>> {
    let norm = embedding
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 {
        anyhow::bail!("embedding vector has zero norm");
    }
    for value in &mut embedding {
        *value = (*value as f64 / norm) as f32;
    }
    Ok(embedding)
}

fn embedding_response(
    model_name: String,
    embeddings: Vec<Vec<f32>>,
    prompt_tokens: usize,
) -> Value {
    let data = embeddings
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| {
            json!({
                "object": "embedding",
                "index": index,
                "embedding": embedding,
            })
        })
        .collect::<Vec<_>>();

    json!({
        "object": "list",
        "model": model_name,
        "data": data,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        }
    })
}

fn pool_embedding_values(
    values: &[f32],
    seq_len: usize,
    hidden_size: usize,
    pooling: EmbeddingPooling,
    attention_mask: &[u32],
) -> Result<Vec<f32>> {
    match pooling {
        EmbeddingPooling::LastToken => {
            if seq_len == 0 {
                anyhow::bail!("hidden states did not contain any sequence positions");
            }
            let offset = (seq_len - 1) * hidden_size;
            Ok(values[offset..offset + hidden_size].to_vec())
        }
        EmbeddingPooling::Mean => {
            if attention_mask.len() != seq_len {
                anyhow::bail!(
                    "attention mask length {} did not match sequence length {seq_len}",
                    attention_mask.len()
                );
            }
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

fn mean_pool_embeddings_gpu(hidden_states: &Array, attention_masks: &[Vec<u32>]) -> Result<Array> {
    let shape = hidden_states.shape_raw();
    if shape.len() != 3 {
        anyhow::bail!(
            "expected rank-3 hidden states for mean pooling, got rank {}",
            shape.len()
        );
    }
    let batch = shape[0] as usize;
    let seq_len = shape[1] as usize;
    if attention_masks.len() != batch {
        anyhow::bail!(
            "attention mask batch length {} did not match hidden batch {batch}",
            attention_masks.len()
        );
    }
    let mut flat_mask = Vec::with_capacity(batch * seq_len);
    for mask in attention_masks {
        if mask.len() != seq_len {
            anyhow::bail!(
                "attention mask length {} did not match sequence length {seq_len}",
                mask.len()
            );
        }
        flat_mask.extend(mask.iter().map(|&v| v as f32));
    }
    let mask = Array::from_slice_f32(&flat_mask)?
        .reshape(&[batch as i32, seq_len as i32, 1])?
        .as_type(hidden_states.dtype())?;
    let summed = hidden_states.multiply(&mask)?.sum_axis(1, false)?;
    let counts = mask.sum_axis(1, false)?;
    if counts.to_vec_f32()?.into_iter().any(|count| count <= 0.0) {
        anyhow::bail!("attention mask excluded every token");
    }
    Ok(summed.divide(&counts)?)
}

fn normalize_embeddings_gpu(embeddings: &Array) -> Result<Array> {
    let squared = embeddings.square()?;
    let norms = squared.sum_axis(-1, true)?.sqrt()?;
    Ok(embeddings.divide(&norms)?)
}

fn embeddings_from_array(embeddings: &Array) -> Result<Vec<Vec<f32>>> {
    let shape = embeddings.shape_raw();
    if shape.len() != 2 {
        anyhow::bail!("expected rank-2 embeddings array, got rank {}", shape.len());
    }
    let batch = shape[0] as usize;
    let hidden_size = shape[1] as usize;
    let values = embeddings.to_vec_f32()?;
    if values.len() != batch * hidden_size {
        anyhow::bail!(
            "embedding values length {} did not match expected {}",
            values.len(),
            batch * hidden_size
        );
    }
    Ok(values
        .chunks(hidden_size)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>())
}

fn compute_embeddings(
    loaded: &mut LoadedModel,
    inputs: &[String],
    response_model_name: Option<String>,
    embeddings_batch_size: usize,
) -> Result<Value> {
    #[derive(Debug, Clone)]
    struct EncodedEmbeddingInput {
        original_index: usize,
        ids: Vec<u32>,
        attention_mask: Vec<u32>,
    }

    fn pool_embedding_batch_values(
        values: &[f32],
        batch: usize,
        seq_len: usize,
        hidden_size: usize,
        pooling: EmbeddingPooling,
        attention_masks: &[Vec<u32>],
    ) -> Result<Vec<Vec<f32>>> {
        if attention_masks.len() != batch {
            anyhow::bail!(
                "attention mask batch length {} did not match hidden batch {batch}",
                attention_masks.len()
            );
        }
        let expected = batch
            .checked_mul(seq_len)
            .and_then(|v| v.checked_mul(hidden_size))
            .ok_or_else(|| anyhow::anyhow!("hidden state size overflow"))?;
        if values.len() != expected {
            anyhow::bail!(
                "hidden state values length {} did not match expected {}",
                values.len(),
                expected
            );
        }

        let mut embeddings = Vec::with_capacity(batch);
        for (batch_idx, attention_mask) in attention_masks.iter().enumerate() {
            let base = batch_idx * seq_len * hidden_size;
            let embedding = match pooling {
                EmbeddingPooling::LastToken => {
                    let offset = base + (seq_len - 1) * hidden_size;
                    values[offset..offset + hidden_size].to_vec()
                }
                EmbeddingPooling::Mean => pool_embedding_values(
                    &values[base..base + seq_len * hidden_size],
                    seq_len,
                    hidden_size,
                    pooling,
                    attention_mask,
                )?,
            };
            embeddings.push(normalize_embedding(embedding)?);
        }
        Ok(embeddings)
    }

    let mut prompt_tokens = 0usize;
    let mut encoded_inputs = Vec::with_capacity(inputs.len());
    for (original_index, input) in inputs.iter().enumerate() {
        let encoded = loaded.tokenizer.encode_for_embeddings(input)?;
        if encoded.ids.is_empty() {
            anyhow::bail!("input produced no tokens");
        }
        prompt_tokens += encoded.ids.len();
        encoded_inputs.push(EncodedEmbeddingInput {
            original_index,
            ids: encoded.ids,
            attention_mask: encoded.attention_mask,
        });
    }

    if loaded.model.supports_padded_embedding_batching() {
        let pooling = loaded.model.embedding_pooling();
        if pooling == EmbeddingPooling::Mean {
            let mut embeddings: Vec<Option<Vec<f32>>> = vec![None; inputs.len()];
            for chunk in encoded_inputs.chunks(embeddings_batch_size.max(1)) {
                loaded.model.clear_cache();
                let batch = chunk.len();
                let max_seq_len = chunk
                    .iter()
                    .map(|encoded| encoded.ids.len())
                    .max()
                    .unwrap_or(0);
                let mut flat_input_ids = Vec::with_capacity(batch * max_seq_len);
                let mut flat_attention_mask = Vec::with_capacity(batch * max_seq_len);
                let mut attention_masks = Vec::with_capacity(batch);
                for encoded in chunk {
                    let mut ids = encoded.ids.iter().map(|v| *v as i32).collect::<Vec<_>>();
                    ids.resize(max_seq_len, 0);
                    flat_input_ids.extend(ids);

                    let mut attention_mask = encoded.attention_mask.clone();
                    attention_mask.resize(max_seq_len, 0);
                    flat_attention_mask.extend(attention_mask.iter().map(|&v| v as f32));
                    attention_masks.push(attention_mask);
                }
                let input = Array::from_slice_i32(&flat_input_ids)?
                    .reshape(&[batch as i32, max_seq_len as i32])?;
                let attention_mask_arr = Array::from_slice_f32(&flat_attention_mask)?
                    .reshape(&[batch as i32, max_seq_len as i32])?;
                let hidden_states = loaded
                    .model
                    .forward_hidden_states_masked(&input, Some(&attention_mask_arr))
                    .map_err(|e| anyhow::anyhow!("embedding inference failed: {e}"))?;
                let pooled = mean_pool_embeddings_gpu(&hidden_states, &attention_masks)?;
                let normalized = normalize_embeddings_gpu(&pooled)?;
                let batch_embeddings = embeddings_from_array(&normalized)?;
                for (encoded, embedding) in chunk.iter().zip(batch_embeddings.into_iter()) {
                    embeddings[encoded.original_index] = Some(embedding);
                }
            }

            loaded.model.clear_cache();
            return Ok(embedding_response(
                response_model_name.unwrap_or_else(|| loaded.model_dir.display().to_string()),
                embeddings
                    .into_iter()
                    .map(|embedding| {
                        embedding.ok_or_else(|| {
                            anyhow::anyhow!("missing embedding result after batching")
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                prompt_tokens,
            ));
        }
    }

    let pooling = loaded.model.embedding_pooling();
    let mut embeddings: Vec<Option<Vec<f32>>> = vec![None; inputs.len()];
    let mut groups: HashMap<usize, Vec<EncodedEmbeddingInput>> = HashMap::new();
    for encoded in encoded_inputs {
        groups.entry(encoded.ids.len()).or_default().push(encoded);
    }

    let mut lengths = groups.keys().copied().collect::<Vec<_>>();
    lengths.sort_unstable();
    for seq_len in lengths {
        let group = groups
            .remove(&seq_len)
            .ok_or_else(|| anyhow::anyhow!("missing encoded group for length {seq_len}"))?;
        for chunk in group.chunks(embeddings_batch_size.max(1)) {
            loaded.model.clear_cache();
            let batch = chunk.len();
            let mut flat_input_ids = Vec::with_capacity(batch * seq_len);
            let mut attention_masks = Vec::with_capacity(batch);
            for encoded in chunk {
                flat_input_ids.extend(encoded.ids.iter().map(|v| *v as i32));
                attention_masks.push(encoded.attention_mask.clone());
            }
            let input =
                Array::from_slice_i32(&flat_input_ids)?.reshape(&[batch as i32, seq_len as i32])?;
            let hidden_states = loaded
                .model
                .forward_hidden_states(&input)
                .map_err(|e| anyhow::anyhow!("embedding inference failed: {e}"))?;
            let hidden_shape = hidden_states.shape_raw();
            if hidden_shape.len() < 3 {
                anyhow::bail!("unexpected hidden state rank {}", hidden_states.ndim());
            }
            let hidden_batch = hidden_shape[0] as usize;
            let hidden_seq_len = hidden_shape[hidden_shape.len() - 2] as usize;
            let hidden_size = hidden_shape[hidden_shape.len() - 1] as usize;
            if hidden_batch != batch {
                anyhow::bail!(
                    "hidden state batch {} did not match requested batch {batch}",
                    hidden_batch
                );
            }
            if hidden_seq_len != seq_len {
                anyhow::bail!(
                    "hidden state sequence length {} did not match requested length {seq_len}",
                    hidden_seq_len
                );
            }
            let values = hidden_states.to_vec_f32()?;
            let batch_embeddings = pool_embedding_batch_values(
                &values,
                batch,
                seq_len,
                hidden_size,
                pooling,
                &attention_masks,
            )?;
            for (encoded, embedding) in chunk.iter().zip(batch_embeddings.into_iter()) {
                embeddings[encoded.original_index] = Some(embedding);
            }
        }
    }

    loaded.model.clear_cache();
    Ok(embedding_response(
        response_model_name.unwrap_or_else(|| loaded.model_dir.display().to_string()),
        embeddings
            .into_iter()
            .map(|embedding| {
                embedding.ok_or_else(|| anyhow::anyhow!("missing embedding result after batching"))
            })
            .collect::<Result<Vec<_>>>()?,
        prompt_tokens,
    ))
}

fn handle_embeddings_request(
    parsed: EmbeddingsRequest,
    loaded: &mut Option<LoadedModel>,
    embeddings_batch_size: usize,
) -> (u16, Value) {
    if let Some(format) = parsed.encoding_format.as_deref() {
        if format != "float" {
            return (
                400,
                error_json(
                    "invalid_encoding_format",
                    "encoding_format must be \"float\"",
                ),
            );
        }
    }
    let inputs = match parse_embeddings_input(parsed.input) {
        Ok(v) => v,
        Err(e) => return (400, error_json("invalid_request", e.to_string())),
    };
    let lm = match loaded.as_mut() {
        Some(v) => v,
        None => {
            return (
                400,
                error_json("model_not_loaded", "No model loaded. Call /llm/load first."),
            )
        }
    };
    match compute_embeddings(lm, &inputs, parsed.model, embeddings_batch_size) {
        Ok(body) => (200, body),
        Err(e) => {
            let message = e.to_string();
            let code = if message.contains("embeddings are not supported") {
                "embeddings_not_supported"
            } else {
                "embedding_error"
            };
            (400, error_json(code, message))
        }
    }
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

fn write_processing_status(stream: &mut TcpStream) -> Result<()> {
    stream.write_all(b"HTTP/1.1 102 Processing\r\n\r\n")?;
    stream.flush()?;
    Ok(())
}

fn spawn_load_heartbeat(stream: &TcpStream) -> Result<(Arc<AtomicBool>, thread::JoinHandle<()>)> {
    let mut heartbeat_stream = stream.try_clone()?;
    let running = Arc::new(AtomicBool::new(true));
    let keep_running = Arc::clone(&running);
    let handle = thread::spawn(move || {
        while keep_running.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_secs(2));
            if !keep_running.load(Ordering::Relaxed) {
                break;
            }
            if write_processing_status(&mut heartbeat_stream).is_err() {
                break;
            }
        }
    });
    Ok((running, handle))
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
    debug.insert(
        "generated_token_ids".into(),
        json!(metrics.generated_token_ids),
    );
    debug.insert("tail_token_ids".into(), json!(metrics.tail_token_ids));
    debug.insert("stop_token_ids".into(), json!(metrics.stop_token_ids));

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
                "moe_device_router_shadow_checks": profile.moe_device_router_shadow_checks,
                "moe_device_router_shadow_mismatches": profile.moe_device_router_shadow_mismatches,
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
    huggingface: &HuggingFaceOptions,
    api_key: Option<&str>,
    limiter: &mut Option<FixedWindowRateLimiter>,
    thinking: bool,
    embeddings_batch_size: usize,
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

    if req.path == "/v1/chat/completions" || req.path == "/v1/embeddings" {
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
                Some(m) => {
                    vec![json!({"id": m.model_dir.display().to_string(), "object": "model"})]
                }
                None => vec![],
            };
            Some((200, json!({"data": data})))
        }
        ("POST", "/llm/load") => {
            let parsed: LoadRequest = match serde_json::from_slice(&req.body) {
                Ok(v) => v,
                Err(e) => return Some((400, error_json("invalid_json", format!("bad JSON: {e}")))),
            };
            if let Err(e) = write_processing_status(stream) {
                return Some((
                    500,
                    error_json(
                        "load_heartbeat_error",
                        format!("failed to write initial heartbeat: {e}"),
                    ),
                ));
            }
            let heartbeat = match spawn_load_heartbeat(stream) {
                Ok(v) => Some(v),
                Err(e) => {
                    return Some((
                        500,
                        error_json(
                            "load_heartbeat_error",
                            format!("failed to start heartbeat: {e}"),
                        ),
                    ))
                }
            };
            let model_path = match resolve_model_dir(&parsed.model_path, Some(huggingface)) {
                Ok(path) => path,
                Err(e) => {
                    if let Some((running, handle)) = heartbeat {
                        running.store(false, Ordering::Relaxed);
                        let _ = handle.join();
                    }
                    return Some((
                        400,
                        error_json("model_resolve_error", format!("Model resolve error: {e}")),
                    ));
                }
            };
            let response = match load_model(&model_path) {
                Ok((model, tokenizer)) => {
                    let next = LoadedModel {
                        model_dir: model_path,
                        model,
                        tokenizer,
                    };
                    release_loaded_model(loaded);
                    *loaded = Some(next);
                    (200, json!({"ok": true}))
                }
                Err(e) => (
                    400,
                    error_json("model_load_error", format!("Model load error: {e}")),
                ),
            };
            if let Some((running, handle)) = heartbeat {
                running.store(false, Ordering::Relaxed);
                let _ = handle.join();
            }
            Some(response)
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
            let sampler = sampler_for_model(&lm.model_dir, temperature, top_p);
            let prompt_tokenize_t0 = Instant::now();
            let prompt_token_count = lm.tokenizer.encode(&prompt).map(|v| v.len()).unwrap_or(0);
            let prompt_tokenize_s = prompt_tokenize_t0.elapsed().as_secs_f64();

            let mut pipeline =
                GenerationPipeline::new(&mut lm.model, lm.tokenizer.clone(), sampler);
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
        ("POST", "/v1/embeddings") => {
            let parsed: EmbeddingsRequest = match serde_json::from_slice(&req.body) {
                Ok(v) => v,
                Err(e) => return Some((400, error_json("invalid_json", format!("bad JSON: {e}")))),
            };
            Some(handle_embeddings_request(
                parsed,
                loaded,
                embeddings_batch_size,
            ))
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
        let model_path =
            resolve_model_dir(&model_path.display().to_string(), Some(&config.huggingface))?;
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
    let embeddings_batch_size = config.embeddings_batch_size();
    let huggingface = config.huggingface.clone();

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
            &huggingface,
            api_key.as_deref(),
            &mut limiter,
            thinking,
            embeddings_batch_size,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_embeddings_input_accepts_string_and_arrays() {
        assert_eq!(
            parse_embeddings_input(json!("hello")).expect("single input"),
            vec!["hello".to_string()]
        );
        assert_eq!(
            parse_embeddings_input(json!(["hello", "world"])).expect("batch input"),
            vec!["hello".to_string(), "world".to_string()]
        );
    }

    #[test]
    fn parse_embeddings_input_rejects_token_arrays() {
        let err = parse_embeddings_input(json!([[1, 2, 3]])).expect_err("token array rejected");
        assert!(err.to_string().contains("token array"));
    }

    #[test]
    fn normalize_embedding_returns_unit_vector() {
        let embedding = normalize_embedding(vec![3.0, 4.0]).expect("normalized");
        let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mean_pooling_uses_attention_mask() {
        let pooled = pool_embedding_values(
            &[1.0, 2.0, 3.0, 4.0, 100.0, 200.0],
            3,
            2,
            EmbeddingPooling::Mean,
            &[1, 1, 0],
        )
        .expect("pooled");
        assert_eq!(pooled, vec![2.0, 3.0]);
    }

    #[test]
    fn last_token_pooling_returns_final_position() {
        let pooled = pool_embedding_values(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            3,
            2,
            EmbeddingPooling::LastToken,
            &[1, 1, 1],
        )
        .expect("pooled");
        assert_eq!(pooled, vec![5.0, 6.0]);
    }

    #[test]
    fn embeddings_route_requires_loaded_model() {
        let req: EmbeddingsRequest =
            serde_json::from_value(json!({"input": "hello"})).expect("request");
        let mut loaded = None;
        let res = handle_embeddings_request(req, &mut loaded, 32);
        assert_eq!(res.0, 400);
        assert_eq!(res.1["error"]["code"], "model_not_loaded");
    }

    #[test]
    fn embeddings_route_rejects_invalid_encoding_format() {
        let req: EmbeddingsRequest =
            serde_json::from_value(json!({"input": "hello", "encoding_format": "base64"}))
                .expect("request");
        let mut loaded = None;

        let res = handle_embeddings_request(req, &mut loaded, 32);
        assert_eq!(res.0, 400);
        assert_eq!(res.1["error"]["code"], "invalid_encoding_format");
    }

    #[test]
    fn embeddings_route_rejects_token_arrays() {
        let req: EmbeddingsRequest =
            serde_json::from_value(json!({"input": [[1, 2, 3]]})).expect("request");
        let mut loaded = None;

        let res = handle_embeddings_request(req, &mut loaded, 32);
        assert_eq!(res.0, 400);
        assert_eq!(res.1["error"]["code"], "invalid_request");
    }

    #[test]
    fn embedding_response_uses_openai_shape() {
        let body = embedding_response("test-model".into(), vec![vec![0.6, 0.8], vec![1.0, 0.0]], 4);
        assert_eq!(body["object"], "list");
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["data"].as_array().expect("data").len(), 2);
        assert_eq!(body["usage"]["prompt_tokens"], 4);
        assert_eq!(body["usage"]["total_tokens"], 4);

        let embedding = body["data"][0]["embedding"]
            .as_array()
            .expect("embedding array")
            .iter()
            .map(|v| v.as_f64().expect("f64"))
            .collect::<Vec<_>>();
        assert!((embedding[0] - 0.6).abs() < 1e-6);
        assert!((embedding[1] - 0.8).abs() < 1e-6);
    }
}
