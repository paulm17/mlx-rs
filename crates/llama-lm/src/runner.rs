use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::backend::Backend;
use crate::types::{
    ChatMessage, EmbeddingOutput, GenerateOutput, GenerationMetrics, GenerationOptions,
    LoadedModelInfo, StopReason,
};

// --- Protocol types ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    pub status: i32,
    pub progress: i32,
    pub context_length: usize,
    pub memory: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadRequest {
    pub model_path: String,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadResponse {
    pub status: String,
    pub model_path: String,
    pub context_length: Option<usize>,
    pub embedding_dimension: Option<usize>,
    pub vocab_size: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub prompt: String,
    #[serde(default)]
    pub options: CompletionOptions,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompletionOptions {
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub top_k: usize,
    #[serde(default)]
    pub min_p: f32,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
}

fn default_temperature() -> f32 {
    0.6
}

fn default_top_p() -> f32 {
    0.9
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub content: String,
    pub done: bool,
    #[serde(default)]
    pub done_reason: i32,
    #[serde(default)]
    pub prompt_eval_count: usize,
    #[serde(default)]
    pub eval_count: usize,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizeRequest {
    pub text: String,
    #[serde(default = "default_true")]
    pub add_bos: bool,
}

fn default_true() -> bool {
    true
}

// --- RunnerClient ---

pub struct RunnerClient {
    base_url: String,
    model_info: LoadedModelInfo,
}

impl RunnerClient {
    pub fn new(base_url: &str, model_info: LoadedModelInfo) -> Self {
        Self {
            base_url: base_url.to_string(),
            model_info,
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    fn http_post(&self, path: &str, json_body: &str) -> Result<String> {
        let url = self.url(path);
        let host = url
            .strip_prefix("http://")
            .unwrap_or(&url)
            .split('/')
            .next()
            .unwrap_or("127.0.0.1:8081");
        let path_part = url
            .strip_prefix("http://")
            .unwrap_or(&url)
            .strip_prefix(host)
            .unwrap_or("/");

        let mut stream = TcpStream::connect(host).context("connecting to runner")?;

        let request = format!(
            "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            path_part,
            host,
            json_body.len(),
            json_body
        );
        stream.write_all(request.as_bytes())?;
        stream.flush()?;

        let mut reader = BufReader::new(stream.try_clone()?);
        let mut headers = String::new();
        loop {
            let mut line = String::new();
            reader.read_line(&mut line)?;
            if line.trim().is_empty() {
                break;
            }
            headers.push_str(&line);
        }

        let status_line = headers.lines().next().unwrap_or("");
        if !status_line.contains("200") {
            let mut body = String::new();
            reader.read_line(&mut body)?;
            anyhow::bail!("runner returned {}: {}", status_line.trim(), body.trim());
        }

        let mut body = String::new();
        reader.read_to_string(&mut body)?;
        Ok(body)
    }

    fn http_post_stream(
        &self,
        path: &str,
        json_body: &str,
        on_line: &mut dyn FnMut(&str) -> Result<bool>,
    ) -> Result<()> {
        let url = self.url(path);
        let host = url
            .strip_prefix("http://")
            .unwrap_or(&url)
            .split('/')
            .next()
            .unwrap_or("127.0.0.1:8081");
        let path_part = url
            .strip_prefix("http://")
            .unwrap_or(&url)
            .strip_prefix(host)
            .unwrap_or("/");

        let mut stream = TcpStream::connect(host).context("connecting to runner")?;

        let request = format!(
            "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            path_part,
            host,
            json_body.len(),
            json_body
        );
        stream.write_all(request.as_bytes())?;
        stream.flush()?;

        let mut reader = BufReader::new(stream.try_clone()?);
        let mut headers = String::new();
        loop {
            let mut line = String::new();
            reader.read_line(&mut line)?;
            if line.trim().is_empty() {
                break;
            }
            headers.push_str(&line);
        }

        let status_line = headers.lines().next().unwrap_or("");
        if !status_line.contains("200") {
            let mut body = String::new();
            reader.read_line(&mut body)?;
            anyhow::bail!("runner returned {}: {}", status_line.trim(), body.trim());
        }

        loop {
            let mut line = String::new();
            let n = reader.read_line(&mut line)?;
            if n == 0 {
                break;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if !on_line(trimmed)? {
                break;
            }
        }

        Ok(())
    }
}

impl Backend for RunnerClient {
    fn model_info(&self) -> LoadedModelInfo {
        self.model_info.clone()
    }

    fn tokenize(&self, text: &str, add_bos: bool) -> Result<Vec<i32>> {
        let req = TokenizeRequest {
            text: text.to_string(),
            add_bos,
        };
        let body = serde_json::to_string(&req)?;
        let resp = self.http_post("/v1/tokenize", &body)?;
        let tokens: Vec<i32> = serde_json::from_str(&resp)?;
        Ok(tokens)
    }

    fn detokenize(&self, _tokens: &[i32]) -> Result<String> {
        anyhow::bail!("detokenize not supported via runner protocol")
    }

    fn detokenize_piece(&self, _token_id: i32) -> Result<String> {
        anyhow::bail!("detokenize_piece not supported via runner protocol")
    }

    fn is_eog(&self, _token_id: i32) -> bool {
        false
    }

    fn token_eos(&self) -> i32 {
        2
    }

    fn embeddings_enabled(&self) -> bool {
        false
    }

    fn apply_chat_template(&self, _messages: &[ChatMessage]) -> Result<String> {
        anyhow::bail!("apply_chat_template not supported via runner protocol")
    }

    fn generate(&mut self, prompt: &str, options: &GenerationOptions) -> Result<GenerateOutput> {
        let req = CompletionRequest {
            prompt: prompt.to_string(),
            options: CompletionOptions {
                max_tokens: options.max_tokens,
                temperature: options.temperature,
                top_p: options.top_p,
                top_k: options.top_k,
                min_p: options.min_p,
                stop: options.stop.clone(),
            },
        };
        let body = serde_json::to_string(&req)?;

        let mut content = String::new();
        let mut prompt_tokens = 0usize;
        let mut generated_tokens = 0usize;
        let mut stop_reason = StopReason::Eos;

        self.http_post_stream("/v1/completions", &body, &mut |line| {
            let resp: CompletionResponse = serde_json::from_str(line)?;
            if let Some(err) = resp.error {
                anyhow::bail!("runner error: {}", err);
            }
            content.push_str(&resp.content);
            if resp.prompt_eval_count > 0 {
                prompt_tokens = resp.prompt_eval_count;
            }
            generated_tokens += resp.eval_count;
            if resp.done {
                stop_reason = match resp.done_reason {
                    1 => StopReason::MaxTokens,
                    _ => StopReason::Eos,
                };
                return Ok(false);
            }
            Ok(true)
        })?;

        Ok(GenerateOutput {
            text: content,
            stop_reason,
            metrics: GenerationMetrics {
                prompt_tokens,
                generated_tokens,
                total_tokens: prompt_tokens + generated_tokens,
                ttft_s: None,
                total_s: None,
                tokens_per_s: None,
            },
        })
    }

    fn generate_stream(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
        mut on_token: Box<dyn FnMut(&str) -> bool + Send>,
    ) -> Result<GenerationMetrics> {
        let req = CompletionRequest {
            prompt: prompt.to_string(),
            options: CompletionOptions {
                max_tokens: options.max_tokens,
                temperature: options.temperature,
                top_p: options.top_p,
                top_k: options.top_k,
                min_p: options.min_p,
                stop: options.stop.clone(),
            },
        };
        let body = serde_json::to_string(&req)?;

        let mut prompt_tokens = 0usize;
        let mut generated_tokens = 0usize;

        self.http_post_stream("/v1/completions", &body, &mut |line| {
            let resp: CompletionResponse = serde_json::from_str(line)?;
            if let Some(err) = resp.error {
                anyhow::bail!("runner error: {}", err);
            }
            if resp.prompt_eval_count > 0 {
                prompt_tokens = resp.prompt_eval_count;
            }
            if !resp.content.is_empty() {
                let cont = on_token(&resp.content);
                generated_tokens += resp.eval_count;
                if !cont {
                    return Ok(false);
                }
            }
            if resp.done {
                return Ok(false);
            }
            Ok(true)
        })?;

        Ok(GenerationMetrics {
            prompt_tokens,
            generated_tokens,
            total_tokens: prompt_tokens + generated_tokens,
            ttft_s: None,
            total_s: None,
            tokens_per_s: None,
        })
    }

    fn generate_stream_output(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
        mut on_token: Box<dyn FnMut(&str) -> bool + Send>,
    ) -> Result<GenerateOutput> {
        let req = CompletionRequest {
            prompt: prompt.to_string(),
            options: CompletionOptions {
                max_tokens: options.max_tokens,
                temperature: options.temperature,
                top_p: options.top_p,
                top_k: options.top_k,
                min_p: options.min_p,
                stop: options.stop.clone(),
            },
        };
        let body = serde_json::to_string(&req)?;

        let mut content = String::new();
        let mut prompt_tokens = 0usize;
        let mut generated_tokens = 0usize;
        let mut stop_reason = StopReason::Eos;

        self.http_post_stream("/v1/completions", &body, &mut |line| {
            let resp: CompletionResponse = serde_json::from_str(line)?;
            if let Some(err) = resp.error {
                anyhow::bail!("runner error: {}", err);
            }
            if resp.prompt_eval_count > 0 {
                prompt_tokens = resp.prompt_eval_count;
            }
            if !resp.content.is_empty() {
                let cont = on_token(&resp.content);
                content.push_str(&resp.content);
                generated_tokens += resp.eval_count;
                if !cont {
                    return Ok(false);
                }
            }
            if resp.done {
                stop_reason = match resp.done_reason {
                    1 => StopReason::MaxTokens,
                    _ => StopReason::Eos,
                };
                return Ok(false);
            }
            Ok(true)
        })?;

        Ok(GenerateOutput {
            text: content,
            stop_reason,
            metrics: GenerationMetrics {
                prompt_tokens,
                generated_tokens,
                total_tokens: prompt_tokens + generated_tokens,
                ttft_s: None,
                total_s: None,
                tokens_per_s: None,
            },
        })
    }

    fn embed(&mut self, _text: &str) -> Result<EmbeddingOutput> {
        anyhow::bail!("embed not supported via runner protocol")
    }
}

// --- Mock runner server for testing ---

pub struct MockRunnerServer {
    pub port: u16,
    pub model_path: String,
    pub context_length: usize,
    pub tokenize_response: Vec<i32>,
    pub completion_responses: Vec<CompletionResponse>,
    pub handle: Option<std::thread::JoinHandle<()>>,
    pub shutdown: Option<std::sync::mpsc::Sender<()>>,
}

impl MockRunnerServer {
    pub fn new(port: u16) -> Self {
        Self {
            port,
            model_path: "/mock/model.gguf".to_string(),
            context_length: 2048,
            tokenize_response: vec![1, 2, 3],
            completion_responses: vec![
                CompletionResponse {
                    content: "Hello".to_string(),
                    done: false,
                    done_reason: 0,
                    prompt_eval_count: 5,
                    eval_count: 1,
                    error: None,
                },
                CompletionResponse {
                    content: " world".to_string(),
                    done: true,
                    done_reason: 0,
                    prompt_eval_count: 0,
                    eval_count: 2,
                    error: None,
                },
            ],
            handle: None,
            shutdown: None,
        }
    }

    pub fn start(&mut self) -> Result<()> {
        let addr = format!("127.0.0.1:{}", self.port);
        let listener = std::net::TcpListener::bind(&addr)
            .with_context(|| format!("failed to bind mock runner on {}", addr))?;
        listener.set_nonblocking(true)?;

        let model_path = self.model_path.clone();
        let context_length = self.context_length;
        let tokenize_response = self.tokenize_response.clone();
        let completion_responses = self.completion_responses.clone();

        let (shutdown_tx, shutdown_rx) = std::sync::mpsc::channel();
        self.shutdown = Some(shutdown_tx);

        let handle = std::thread::spawn(move || {
            loop {
                if shutdown_rx.try_recv().is_ok() {
                    break;
                }

                let (stream, _addr) = match listener.accept() {
                    Ok(s) => s,
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(std::time::Duration::from_millis(10));
                        continue;
                    }
                    Err(e) => {
                        eprintln!("mock runner accept error: {}", e);
                        break;
                    }
                };

                let model_path = model_path.clone();
                let tokenize_response = tokenize_response.clone();
                let completion_responses = completion_responses.clone();

                std::thread::spawn(move || {
                    handle_connection(
                        stream,
                        &model_path,
                        context_length,
                        &tokenize_response,
                        &completion_responses,
                    );
                });
            }
        });

        self.handle = Some(handle);

        // Wait for server to be ready
        std::thread::sleep(std::time::Duration::from_millis(50));

        Ok(())
    }
}

impl Drop for MockRunnerServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn handle_connection(
    mut stream: TcpStream,
    model_path: &str,
    context_length: usize,
    tokenize_response: &[i32],
    completion_responses: &[CompletionResponse],
) {
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .ok();
    stream
        .set_write_timeout(Some(std::time::Duration::from_secs(5)))
        .ok();

    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }

    let parts: Vec<&str> = request_line.trim().split_whitespace().collect();
    if parts.len() < 2 {
        return;
    }
    let method = parts[0];
    let path = parts[1];

    // Read headers
    let mut content_length: usize = 0;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).is_err() || header.trim().is_empty() {
            break;
        }
        if header.to_lowercase().starts_with("content-length:") {
            if let Some(val) = header.split(':').nth(1) {
                content_length = val.trim().parse().unwrap_or(0);
            }
        }
    }

    // Read body
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        let _ = reader.read_exact(&mut body);
    }

    match (method, path) {
        ("GET", "/v1/status") => {
            let resp = StatusResponse {
                status: 0,
                progress: 100,
                context_length,
                memory: 0,
            };
            let json = serde_json::to_string(&resp).unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                json.len(),
                json
            );
            let _ = stream.write_all(response.as_bytes());
        }
        ("POST", "/v1/load") => {
            let resp = LoadResponse {
                status: "ok".to_string(),
                model_path: model_path.to_string(),
                context_length: Some(context_length),
                embedding_dimension: Some(768),
                vocab_size: Some(32000),
            };
            let json = serde_json::to_string(&resp).unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                json.len(),
                json
            );
            let _ = stream.write_all(response.as_bytes());
        }
        ("POST", "/v1/completions") => {
            let mut body = String::new();
            for resp in completion_responses {
                let json = serde_json::to_string(resp).unwrap();
                body.push_str(&json);
                body.push('\n');
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/jsonl\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
        }
        ("POST", "/v1/tokenize") => {
            let json = serde_json::to_string(tokenize_response).unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                json.len(),
                json
            );
            let _ = stream.write_all(response.as_bytes());
        }
        _ => {
            let response = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            let _ = stream.write_all(response.as_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_runner_status() {
        let port = 18901;
        let mut server = MockRunnerServer::new(port);
        server.start().unwrap();

        let resp = reqwest::blocking::get(format!("http://127.0.0.1:{}/v1/status", port)).unwrap();
        assert_eq!(resp.status(), 200);

        let status: StatusResponse = resp.json().unwrap();
        assert_eq!(status.status, 0);
        assert_eq!(status.progress, 100);
        assert_eq!(status.context_length, 2048);
    }

    #[test]
    fn test_mock_runner_tokenize() {
        let port = 18902;
        let mut server = MockRunnerServer::new(port);
        server.start().unwrap();

        let client = reqwest::blocking::Client::new();
        let resp = client
            .post(format!("http://127.0.0.1:{}/v1/tokenize", port))
            .json(&TokenizeRequest {
                text: "hello".to_string(),
                add_bos: true,
            })
            .send()
            .unwrap();

        assert_eq!(resp.status(), 200);
        let tokens: Vec<i32> = resp.json().unwrap();
        assert_eq!(tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_mock_runner_completions() {
        let port = 18903;
        let mut server = MockRunnerServer::new(port);
        server.start().unwrap();

        let client = reqwest::blocking::Client::new();
        let resp = client
            .post(format!("http://127.0.0.1:{}/v1/completions", port))
            .json(&CompletionRequest {
                prompt: "test".to_string(),
                options: CompletionOptions::default(),
            })
            .send()
            .unwrap();

        assert_eq!(resp.status(), 200);
        let body = resp.text().unwrap();
        let lines: Vec<&str> = body.lines().filter(|l| !l.trim().is_empty()).collect();
        assert_eq!(lines.len(), 2);

        let first: CompletionResponse = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first.content, "Hello");
        assert!(!first.done);

        let second: CompletionResponse = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(second.content, " world");
        assert!(second.done);
    }

    #[test]
    fn test_runner_client_tokenize() {
        let port = 18904;
        let mut server = MockRunnerServer::new(port);
        server.start().unwrap();

        let client = RunnerClient::new(
            &format!("http://127.0.0.1:{}", port),
            LoadedModelInfo {
                model_path: "/mock/model.gguf".to_string(),
                context_length: Some(2048),
                embedding_dimension: None,
                vocab_size: None,
            },
        );

        let tokens = client.tokenize("hello", true).unwrap();
        assert_eq!(tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_runner_client_generate() {
        let port = 18905;
        let mut server = MockRunnerServer::new(port);
        server.start().unwrap();

        let mut client = RunnerClient::new(
            &format!("http://127.0.0.1:{}", port),
            LoadedModelInfo {
                model_path: "/mock/model.gguf".to_string(),
                context_length: Some(2048),
                embedding_dimension: None,
                vocab_size: None,
            },
        );

        let output = client.generate("test", &GenerationOptions::default()).unwrap();
        assert_eq!(output.text, "Hello world");
        assert_eq!(output.metrics.prompt_tokens, 5);
        assert_eq!(output.metrics.generated_tokens, 3);
    }

    #[test]
    fn test_runner_client_generate_stream() {
        use std::sync::{Arc, Mutex};

        let port = 18906;
        let mut server = MockRunnerServer::new(port);
        server.start().unwrap();

        let mut client = RunnerClient::new(
            &format!("http://127.0.0.1:{}", port),
            LoadedModelInfo {
                model_path: "/mock/model.gguf".to_string(),
                context_length: Some(2048),
                embedding_dimension: None,
                vocab_size: None,
            },
        );

        let tokens = Arc::new(Mutex::new(Vec::new()));
        let tokens_clone = tokens.clone();
        let metrics = client
            .generate_stream(
                "test",
                &GenerationOptions::default(),
                Box::new(move |t| {
                    tokens_clone.lock().unwrap().push(t.to_string());
                    true
                }),
            )
            .unwrap();

        let tokens = tokens.lock().unwrap();
        assert_eq!(*tokens, vec!["Hello", " world"]);
        assert_eq!(metrics.prompt_tokens, 5);
        assert_eq!(metrics.generated_tokens, 3);
    }

    #[test]
    fn test_runner_client_generate_stream_output() {
        use std::sync::{Arc, Mutex};

        let port = 18907;
        let mut server = MockRunnerServer::new(port);
        server.start().unwrap();

        let mut client = RunnerClient::new(
            &format!("http://127.0.0.1:{}", port),
            LoadedModelInfo {
                model_path: "/mock/model.gguf".to_string(),
                context_length: Some(2048),
                embedding_dimension: None,
                vocab_size: None,
            },
        );

        let streamed = Arc::new(Mutex::new(Vec::new()));
        let streamed_clone = streamed.clone();
        let output = client
            .generate_stream_output(
                "test",
                &GenerationOptions::default(),
                Box::new(move |t| {
                    streamed_clone.lock().unwrap().push(t.to_string());
                    true
                }),
            )
            .unwrap();

        let streamed = streamed.lock().unwrap();
        assert_eq!(output.text, "Hello world");
        assert_eq!(*streamed, vec!["Hello", " world"]);
        assert_eq!(output.metrics.prompt_tokens, 5);
    }

    #[test]
    fn test_runner_client_model_info() {
        let port = 18908;
        let mut server = MockRunnerServer::new(port);
        server.start().unwrap();

        let client = RunnerClient::new(
            &format!("http://127.0.0.1:{}", port),
            LoadedModelInfo {
                model_path: "/mock/model.gguf".to_string(),
                context_length: Some(2048),
                embedding_dimension: Some(768),
                vocab_size: Some(32000),
            },
        );

        let info = client.model_info();
        assert_eq!(info.model_path, "/mock/model.gguf");
        assert_eq!(info.context_length, Some(2048));
    }
}
