use std::net::TcpListener;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::runner::RunnerClient;
use crate::types::LoadedModelInfo;

pub struct RunnerSubprocess {
    child: Child,
    port: u16,
}

impl RunnerSubprocess {
    pub fn spawn(model_path: &Path) -> Result<Self> {
        let timeout_secs = std::env::var("MLX_RS_RUNNER_TIMEOUT")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(120);
        Self::spawn_with_timeout(model_path, Duration::from_secs(timeout_secs))
    }

    pub fn spawn_with_timeout(model_path: &Path, timeout: Duration) -> Result<Self> {
        let port = find_free_port()?;
        let exe = std::env::current_exe()
            .context("failed to get current executable path")?;

        let mut cmd = Command::new(exe);
        cmd.arg("--mlx-engine")
            .arg("--model")
            .arg(model_path)
            .arg("--port")
            .arg(port.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let child = cmd.spawn()
            .context("failed to spawn llama-rs-runner subprocess")?;

        let subprocess = Self { child, port };
        subprocess.wait_until_ready(timeout)?;
        Ok(subprocess)
    }

    fn wait_until_ready(&self, timeout: Duration) -> Result<()> {
        let client = reqwest::blocking::Client::new();
        let url = format!("http://127.0.0.1:{}/v1/status", self.port);
        let start = Instant::now();

        loop {
            if start.elapsed() > timeout {
                anyhow::bail!("runner subprocess did not become ready within {}s", timeout.as_secs());
            }

            match client.get(&url).send() {
                Ok(resp) if resp.status().is_success() => {
                    return Ok(());
                }
                _ => {
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        }
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn client(&self, model_info: LoadedModelInfo) -> RunnerClient {
        let base_url = format!("http://127.0.0.1:{}", self.port);
        RunnerClient::new(&base_url, model_info)
    }
}

impl Drop for RunnerSubprocess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn find_free_port() -> Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .context("failed to bind to free port")?;
    let port = listener.local_addr()
        .context("failed to get local address")?
        .port();
    Ok(port)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_free_port() {
        let port = find_free_port().unwrap();
        assert!(port > 0);
    }
}
