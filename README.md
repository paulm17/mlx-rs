# mlx-rs

Local LLM inference server and CLI powered by **llama.cpp** and **GGUF** models.

## Features

- OpenAI-compatible API server (`mlx-server`)
- Direct text generation CLI (`generate`)
- Streaming and non-streaming chat completions
- Native GGUF chat templates (falls back to Llama2-style)
- Embeddings endpoint (`/v1/embeddings`)
- Auth via API key and rate limiting
- Metal/GPU layer offloading

## Requirements

- Rust 1.75+
- CMake (for building llama.cpp from source)
- A GGUF model file

## Quick Start

```bash
# Build
cargo build --release

# Generate text
cargo run --release --bin generate -- --model /path/to/model.gguf --prompt "Hello!"

# Chat mode with streaming
cargo run --release --bin generate -- \
  --model /path/to/model.gguf \
  --chat \
  --stream \
  --prompt "What is Rust?"

# Start server
cargo run --release --bin mlx-server -- --model /path/to/model.gguf
```

## Server API

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/v1/models` | List loaded model |
| `POST` | `/llm/load` | Load a model |
| `POST` | `/v1/chat/completions` | Chat completions (streaming + non-streaming) |
| `POST` | `/v1/embeddings` | Text embeddings |

### Chat Completions

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [
      {"role": "system", "content": "You are helpful."},
      {"role": "user", "content": "Hello!"}
    ],
    "stream": true
  }'
```

### Embeddings

```bash
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "model": "local"}'
```

Requires `embedding = true` in config.

## Configuration

Create a `config.toml`:

```toml
[server]
bind = "127.0.0.1"
port = 8080
model_path = "/path/to/model.gguf"
# api_key = "secret"
# rate_limit_rpm = 120

# llama.cpp engine
n_ctx = 4096
n_gpu_layers = 99       # Metal/GPU offloading
n_threads = 4
# embedding = true      # for embeddings endpoint
# pooling = "mean"      # pooling type for embeddings
```

### Metal / GPU

Set `n_gpu_layers` to offload transformer layers to GPU. On macOS with Metal:

```toml
n_gpu_layers = 99
```

## Generate CLI

```
Usage: generate [OPTIONS] --model <MODEL>

Options:
  --model <MODEL>          Path to GGUF model or directory
  --config <CONFIG>        Config file [default: config.toml]
  --prompt <PROMPT>        Input prompt [default: "Hello, how are you?"]
  --max-tokens <N>         Max tokens to generate
  --temperature <FLOAT>    Sampling temperature [default: 0.6]
  --top-p <FLOAT>          Top-p sampling [default: 0.9]
  --chat                   Use chat template mode
  --system-prompt <TEXT>   System prompt for chat mode
  --stream                 Stream tokens as they arrive
```

## Non-Goals (Phase 1)

- Safetensors model support
- MLX tensor API
- VLM / vision models
- Diffusion generation

## License

MIT
