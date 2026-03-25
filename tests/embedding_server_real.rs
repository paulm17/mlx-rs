use serde_json::{json, Value};
use std::fs;
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const DEFAULT_MODEL_PATH: &str = "/Volumes/Data/Users/paul/.cache/huggingface/hub/models--mlx-community--mxbai-embed-large-v1/snapshots/3df9bfd9b9e80b101460ff7302e45476664e9588";

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind ephemeral port")
        .local_addr()
        .expect("local addr")
        .port()
}

fn wait_for_health(port: u16, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    loop {
        match http_request("GET", port, "/health", None) {
            Ok((200, body)) if body["status"] == "ok" => return,
            _ if Instant::now() < deadline => thread::sleep(Duration::from_millis(200)),
            other => panic!("server did not become healthy in time: {:?}", other),
        }
    }
}

fn temp_server_config() -> PathBuf {
    let path = std::env::temp_dir().join(format!("mlx-server-empty-{}.toml", std::process::id()));
    fs::write(&path, "[server]\nthinking = false\n").expect("write temp config");
    path
}

fn start_server(port: u16, config_path: &PathBuf) -> Child {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_mlx-server"));
    cmd.arg("--config")
        .arg(config_path)
        .arg("--bind")
        .arg(format!("127.0.0.1:{port}"))
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let child = cmd.spawn().expect("spawn mlx-server");
    wait_for_health(port, Duration::from_secs(30));
    child
}

fn read_response(mut stream: TcpStream) -> (u16, Vec<u8>) {
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).expect("read response");
    let mut cursor = 0usize;
    loop {
        let header_rel_end = buf[cursor..]
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .expect("header terminator");
        let header_end = cursor + header_rel_end + 4;
        let header_text = String::from_utf8_lossy(&buf[cursor..header_end]);
        let status = header_text
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|status| status.parse::<u16>().ok())
            .expect("status code");
        if (100..200).contains(&status) {
            cursor = header_end;
            continue;
        }

        let body = if header_text
            .to_ascii_lowercase()
            .contains("transfer-encoding: chunked")
        {
            decode_chunked_body(&buf[header_end..])
        } else {
            buf[header_end..].to_vec()
        };
        return (status, body);
    }
}

fn decode_chunked_body(data: &[u8]) -> Vec<u8> {
    let mut cursor = 0usize;
    let mut out = Vec::new();
    while cursor < data.len() {
        let size_end = data[cursor..]
            .windows(2)
            .position(|w| w == b"\r\n")
            .expect("chunk size terminator")
            + cursor;
        let size = usize::from_str_radix(
            std::str::from_utf8(&data[cursor..size_end])
                .expect("chunk size utf8")
                .trim(),
            16,
        )
        .expect("chunk size hex");
        cursor = size_end + 2;
        if size == 0 {
            break;
        }
        out.extend_from_slice(&data[cursor..cursor + size]);
        cursor += size + 2;
    }
    out
}

fn http_request(
    method: &str,
    port: u16,
    path: &str,
    body: Option<&Value>,
) -> anyhow::Result<(u16, Value)> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    let body_bytes = match body {
        Some(value) => serde_json::to_vec(value)?,
        None => Vec::new(),
    };
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body_bytes.len()
    );
    stream.write_all(request.as_bytes())?;
    if !body_bytes.is_empty() {
        stream.write_all(&body_bytes)?;
    }
    stream.flush()?;
    stream.shutdown(Shutdown::Write)?;

    let (status, raw_body) = read_response(stream);
    let parsed = if raw_body.is_empty() {
        json!(null)
    } else {
        serde_json::from_slice(&raw_body)?
    };
    Ok((status, parsed))
}

#[test]
#[ignore = "requires local socket binding and a real MLX embedding model"]
fn embedding_server_real_model_lifecycle() {
    let model_path =
        std::env::var("MLX_TEST_EMBED_MODEL").unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string());
    assert!(
        std::path::Path::new(&model_path).exists(),
        "model path does not exist: {model_path}"
    );

    let port = free_port();
    let config_path = temp_server_config();
    let mut child = start_server(port, &config_path);

    let test_result = (|| -> anyhow::Result<()> {
        let (status, models) = http_request("GET", port, "/v1/models", None)?;
        assert_eq!(status, 200);
        assert_eq!(models["data"], json!([]));

        let (status, load_body) = http_request(
            "POST",
            port,
            "/llm/load",
            Some(&json!({ "model_path": model_path })),
        )?;
        assert_eq!(status, 200, "load response: {load_body}");
        assert_eq!(load_body["ok"], true);

        let (status, embed_body) = http_request(
            "POST",
            port,
            "/v1/embeddings",
            Some(&json!({
                "model": "mxbai-embed-large-v1",
                "input": [
                    "In the beginning God created the heavens and the earth.",
                    "And the earth was without form, and void."
                ],
                "encoding_format": "float"
            })),
        )?;
        assert_eq!(status, 200, "embedding response: {embed_body}");
        let data = embed_body["data"].as_array().expect("embedding data array");
        assert_eq!(data.len(), 2);
        let first = data[0]["embedding"].as_array().expect("first embedding");
        assert!(!first.is_empty(), "embedding vector should not be empty");
        assert_eq!(first.len(), 1024, "expected mxbai embedding width");
        assert!(embed_body["usage"]["prompt_tokens"].as_u64().unwrap_or(0) > 0);
        assert_eq!(
            embed_body["usage"]["prompt_tokens"],
            embed_body["usage"]["total_tokens"]
        );

        let norm = first
            .iter()
            .map(|value| {
                let v = value.as_f64().expect("embedding scalar");
                v * v
            })
            .sum::<f64>()
            .sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "expected normalized embedding, got norm={norm}"
        );

        let (status, unload_body) = http_request("POST", port, "/llm/unload", Some(&json!({})))?;
        assert_eq!(status, 200, "unload response: {unload_body}");
        assert_eq!(unload_body["ok"], true);

        let (status, models_after) = http_request("GET", port, "/v1/models", None)?;
        assert_eq!(status, 200);
        assert_eq!(models_after["data"], json!([]));

        Ok(())
    })();

    let _ = child.kill();
    let _ = child.wait();
    let _ = fs::remove_file(&config_path);

    test_result.expect("real embedding lifecycle test");
}
