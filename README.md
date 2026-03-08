# mlx-rs

`mlx-rs` is a Rust-first MLX runtime for local LLM inference and embedding extraction on Apple Silicon.

It is structured as layered crates that mirror the same separation of concerns you would expect from candle-style design:
- FFI bindings and raw MLX C ABI
- tensor core ops
- neural network primitives
- model architecture implementations
- LLM runtime features (loader, tokenizer, sampling, generation, server)

The project currently supports dense and MoE decoder models plus encoder-style embedding models through the new `/crates` implementation (non-deprecated path).

## Current Status

The current runtime is at or near Python `mlx_lm` parity on the benchmark set in `tests/benchmark_models.json`.

Latest aligned streaming benchmark snapshot (`logs/python_vs_rust_benchmark_20260306_201924.json`):
- `Llama-3.2-1B-Instruct-4bit`: `94.05%` of Python decode throughput
- `Qwen3-1.7B-MLX-4bit`: `114.54%`
- `Qwen1.5-MoE-A2.7B-4bit`: `98.24%`, with stop-parity fixed
- `LFM2-24B-A2B-MLX-4bit`: `101.26%`

The two MoE implementations reached parity after moving the stable runtime to the same execution shape used by Python MLX:
- device-side router `softmax`
- device-side `argpartition`
- device-side `take_along_axis`
- `SwitchGLU`-style expert execution
- on-device weighted reduction

## Project Structure

### Workspace Layout

- `crates/mlx-sys`
- `crates/mlx-core`
- `crates/mlx-nn`
- `crates/mlx-models`
- `crates/mlx-lm`
- `src/bin/generate.rs`
- `src/bin/mlx-server.rs`
- `scripts/run_enclosed_benchmarks.sh`
- `scripts/benchmark_python_vs_rust_servers.py`
- `tests/benchmark_models.json`
- `tests/benchmark_cases.json`
- `config.toml`

### Crate Responsibilities

#### `mlx-sys`
Low-level FFI bindings to `mlx.c`.

What it does:
- Exposes generated Rust bindings to C functions in MLX.
- Contains no model logic.

Typical use:
- Never imported directly by app code.
- Used by `mlx-core`.

#### `mlx-core`
Tensor runtime abstraction over MLX arrays and operators.

What it does:
- Owns `Array`, `DType`, `Shape`, stream/device helpers.
- Implements tensor ops used by all higher layers.
- Includes safetensors loading and convenience utilities.
- Exposes attention kernels, quantized matmul, gather matmul, `conv1d`, etc.

Key idea:
- This is the “numerical substrate” for everything above it.

#### `mlx-nn`
Neural network building blocks.

What it does:
- `Linear` with quantized and dense paths.
- `Embedding` including quantized lookup path.
- `RmsNorm`, `RoPE`, `KvCache`.
- `VarBuilder` for scoped tensor lookup via prefix push (`pp()`), mirroring candle-style var scoping.

Key idea:
- Reusable layers that are architecture-agnostic.

#### `mlx-models`
Architecture-specific model implementations.

What it does:
- Implements decoder and encoder model families with clear config-layer-block-model composition.
- Handles per-family details such as naming, norms, attention variants, embedding inputs, and MoE routing.

Current modules:
- `bert.rs`
- `llama.rs`
- `qwen3.rs`
- `qwen3_moe.rs` (used for Qwen1.5/Qwen2 MoE style)
- `lfm2_moe.rs` (hybrid conv/full-attention + switch-MoE)

Key idea:
- “How to execute one forward pass” per architecture.

#### `mlx-lm`
Runtime orchestration for text generation and serving.

What it does:
- Architecture detection and model loading (`loader.rs`).
- Tokenization and stop-token handling (`tokenizer.rs`).
- Chat template rendering (`chat_template.rs`) with fallback support for `chat_template.jinja`.
- Sampling and repetition handling (`sampler.rs`).
- Generation pipeline with metrics (TTFT/total/tokens) (`generate.rs`).
- Embedded HTTP server runtime (`server.rs`) with optional API key, embeddings route, and RPM limiter.

Key idea:
- “How inference is run in practice,” not just model math.

## Binary Entrypoints

### `generate`
Single-shot local generation CLI.

Typical usage:

```bash
cargo run --bin generate -- \
  --model-dir /path/to/model/snapshot \
  --temperature 0 \
  --prompt "What is Rust programming language?"
```

Notes:
- If `--max-tokens` is omitted, decode is uncapped and stops on stop tokens.
- Chat templates are auto-applied when available.
- Template resolution supports:
1. `tokenizer_config.json` `chat_template`
2. fallback `chat_template.jinja`

### `mlx-server`
OpenAI-compatible-ish local chat server with model preload from config.

Typical usage:

```bash
cargo run --bin mlx-server -- --config config.toml
```

Optional overrides:

```bash
cargo run --bin mlx-server -- \
  --config config.toml \
  --port 3001 \
  --model /path/to/model \
  --api-key secret \
  --rate-limit-rpm 120 \
  --thinking false
```

## End-to-End Flow

1. Client sends prompt to either `generate` CLI or `mlx-server`.
2. `mlx-lm::loader` reads `config.json` and detects architecture.
3. `VarBuilder` loads all `.safetensors` tensors from model snapshot.
4. `mlx-models` instantiates the model struct for that architecture.
5. Prompt is prepared:
1. chat template is applied when available
2. tokenizer converts text to token IDs
6. Prefill pass runs on full prompt tokens.
7. Decode loop runs one token at a time with KV cache updates.
8. `Sampler` chooses next token from logits.
9. Loop stops on stop token or optional cap.
10. Text and metrics (TTFT, total time, token count) are returned.

## Server Runtime Flow

1. Start `mlx-server`.
2. Read `[server]` settings from `config.toml`.
3. Bind listener address from `bind` or `port`.
4. If `model_path`/`model` is configured, preload model at startup.
5. For each request:
1. `/health` is always open.
2. if `api_key` is set, validate `x-api-key` or `Authorization: Bearer`.
3. if `rate_limit_rpm` is set, enforce fixed-window RPM on `/v1/chat/completions`.
4. route request and return JSON response.

## Supported Models (Current New Runtime)

### Working families

- Bert encoder / embedding models
  - Example tested: `mlx-community/mxbai-embed-large-v1`
- Llama
  - Example tested: `mlx-community/Llama-3.2-1B-Instruct-4bit`
- Qwen3 dense
  - Example tested: `Qwen/Qwen3-1.7B-MLX-4bit`
- Qwen3.5 dense
  - Example tested: `mlx-community/Qwen3.5-0.8B-MLX-4bit`
- Qwen MoE (`qwen1.5_moe` / `qwen2_moe`)
  - Example tested: `mlx-community/Qwen1.5-MoE-A2.7B-4bit`
- LFM2 MoE (`lfm2_moe`)
  - Example tested: `LiquidAI/LFM2-24B-A2B-MLX-4bit`

### Detection logic

Architecture is inferred from `config.json` using:
- `model_type`
- fallback `architectures[]`

Mapped runtime enums:
- `Bert`
- `Llama`
- `Qwen3`
- `Qwen35`
- `QwenMoe`
- `Lfm2Moe`

## Model-Specific Notes

### Bert
- Encoder-style architecture used for embedding models.
- The server embedding path uses tokenizer-provided embedding inputs and model hidden states rather than text generation.
- Pooling behavior for embeddings is model/runtime driven and the `/v1/embeddings` response is L2-normalized.

### Llama
- Standard decoder-only attention + MLP path.
- Uses KV cache and RoPE.

### Qwen3 (dense)
- Optional Q/K RMS norm path where checkpoint provides it.
- Dense FFN.

### Qwen MoE
- Uses Python-style MoE execution in the stable path:
  - device router softmax/top-k selection
  - `SwitchGLU` expert execution
  - weighted reduce on device
- Greedy decode includes a narrow late-step tie-break path for near-equal top-2 logits in Qwen MoE, which keeps stop parity aligned with Python on the benchmarked prompt.

### LFM2 MoE
- Hybrid per-layer operator:
  - `conv` layers use short convolution block.
  - `full_attention` layers use attention block.
- MoE feed-forward uses the stable Python-style `switch_mlp` path:
  - device router softmax/top-k selection
  - `SwitchGLU` expert execution
  - weighted reduce on device
- Uses `embedding_norm` head path where checkpoint uses tied embeddings.
- Supports checkpoints where chat template is only in `chat_template.jinja`.

## `config.toml` Server Configuration

Current root config format:

```toml
[server]
bind = "0.0.0.0:3000"
model_path = "/absolute/path/to/model/snapshot"
# model = "/absolute/path/to/model/snapshot"  # alias for model_path
# api_key = "secret"                           # optional
# rate_limit_rpm = 120                          # optional, 0 disables
# thinking = false                              # optional, include thinking block in prompt template
```

### Config semantics

- `bind`: full bind address (`host:port`).
- `port`: optional alternative if `bind` is not set.
- `model_path` or `model`: preload this model at startup.
- `api_key`: if set, all endpoints except `/health` require key.
- `rate_limit_rpm`: fixed-window limiter for `/v1/chat/completions` and `/v1/embeddings`.
- `thinking`: if `true`, enables chat-template thinking mode; if `false`, disables it.

## Server API Surface

### `GET /health`
Returns health JSON.

### `GET /v1/models`
Returns current loaded model metadata.

### `POST /llm/load`
Loads a model on demand.

Body:

```json
{
  "model_path": "/absolute/path/to/model/snapshot"
}
```

### `POST /llm/unload`
Unloads current model from memory.

### `POST /v1/chat/completions`
Generates completion.

Body (example):

```json
{
  "messages": [
    {"role": "user", "content": "What is Rust?"}
  ],
  "max_tokens": 256,
  "temperature": 0.0,
  "top_p": 1.0,
  "stream": false
}
```

When `stream: false`, returns OpenAI-style `choices` plus a `debug` block containing:
- `ttft_s`
- `total_s`
- `tokens`
- `tokens_per_s`

When `stream: true`, returns `text/event-stream` with OpenAI-style chunk objects:
- per token: `choices[0].delta.content`
- terminal chunk: `choices[0].finish_reason = "stop"` with `usage` and `debug`
- completion terminator: `data: [DONE]`

### `POST /v1/embeddings`
Returns OpenAI-style embedding vectors produced from model activations.

Body (example):

```json
{
  "model": "optional-response-model-name",
  "input": [
    "In the beginning God created the heavens and the earth.",
    "And the earth was without form, and void."
  ],
  "encoding_format": "float"
}
```

Request notes:
- `input` must be a string or an array of strings.
- token-array inputs are not supported.
- `encoding_format` currently supports only `"float"`.

Response shape:

```json
{
  "object": "list",
  "model": "optional-response-model-name",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.1, 0.2, 0.3]
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "total_tokens": 12
  }
}
```

Behavior notes:
- embeddings are produced from final hidden states, not generated text.
- pooled responses are L2-normalized before they are returned.
- the route is always exposed, but requests fail if no model is loaded or the loaded model does not support hidden-state extraction.

## Auth and Rate Limiting

### API key
If `api_key` is configured, send one of:
- Header `x-api-key: <key>`
- Header `Authorization: Bearer <key>`

### Rate limiting
If `rate_limit_rpm` is set to a positive integer:
- Applied to `/v1/chat/completions` and `/v1/embeddings`
- Fixed 60-second window
- Exceeded requests return HTTP `429`

## Direct Embedding in Another Rust App

You can embed the server directly without the binary.

### 1. Add dependency

From another Cargo project:

```toml
[dependencies]
mlx-lm = { path = "/absolute/path/to/mlx-rs/crates/mlx-lm" }
anyhow = "1"
```

### 2. Minimal embedded server (explicit config)

```rust
use anyhow::Result;
use mlx_lm::{run_server, ServerConfig};

fn main() -> Result<()> {
    let cfg = ServerConfig {
        bind: Some("0.0.0.0:4000".to_string()),
        port: None,
        model_path: Some("/absolute/path/to/model/snapshot".to_string()),
        model: None,
        api_key: Some("secret".to_string()),
        rate_limit_rpm: Some(120),
        thinking: Some(false),
    };
    run_server(cfg)
}
```

### 3. Embedded server from TOML

```rust
use anyhow::Result;
use mlx_lm::run_server_from_toml_path;

fn main() -> Result<()> {
    run_server_from_toml_path("config.toml")
}
```

## Build and Run

### Build

```bash
cargo build
```

### Run generator

```bash
cargo run --bin generate -- \
  --model-dir /absolute/path/to/model/snapshot \
  --temperature 0 \
  --prompt "What is Rust?"
```

### Run server

```bash
cargo run --bin mlx-server -- --config config.toml
```

## Testing

### Fast crate tests

```bash
cargo test -p mlx-lm
```

### Real embedding lifecycle test

This repo includes an ignored integration test that exercises the full server lifecycle with a real embedding checkpoint:
- start `mlx-server` with no preloaded model
- verify `/v1/models` is empty
- load a model through `/llm/load`
- request vectors through `/v1/embeddings`
- unload through `/llm/unload`

Run it with the default `mxbai-embed-large-v1` snapshot:

```bash
cargo test --test embedding_server_real -- --ignored --nocapture
```

To override the model path:

```bash
MLX_TEST_EMBED_MODEL=/absolute/path/to/model/snapshot \
cargo test --test embedding_server_real -- --ignored --nocapture
```

## Benchmarking

Scripts:
- `scripts/run_enclosed_benchmarks.sh`
- `scripts/benchmark_python_vs_rust_servers.py`
- `scripts/benchmark_embeddings_concurrency.py`
- `cargo run --bin embed_bench -- --model-dir /path/to/model --input "hello world"`

Benchmark inputs:
- Models list in `tests/benchmark_models.json`
- Prompt/case in `tests/benchmark_cases.json`
- Embedding-specific models list in `tests/benchmark_models_embeddings.json`
- Embedding-specific request case in `tests/benchmark_case_embeddings.json`

Output:
- JSON logs under `logs/`
- terminal table with TTFT, tokens/sec, decode TPS, tokens, total time, and comparability flags

Aligned benchmark notes:
- Python and Rust are both exercised through the streaming path.
- The harness records stop reason and comparability metadata, so token-count or stop-condition mismatches are explicit in the output.
- Embedding benchmarks use `/v1/embeddings`, record prompt-token throughput plus embeddings/sec, and annotate Python/Rust vector cosine similarity.
- `scripts/run_enclosed_benchmarks.sh` now runs both the default generation suite and the hardcoded `mxbai-embed-large-v1` embedding comparison.
- `embed_bench` bypasses HTTP and reports encode, forward, pooling, normalization, plus overall embeddings/sec for server-path debugging.
- The Rust server now batches same-length embedding inputs within a request; tune with `embeddings_batch_size` in `[server]` or `mlx-server --embeddings-batch-size`.
- `scripts/benchmark_embeddings_concurrency.py` drives concurrent `/v1/embeddings` traffic against a running server and reports request throughput plus latency percentiles.
- For model-specific debugging, use:
  - `src/bin/generate.rs` with `--dump-json-out`
  - `src/bin/generate_diag.rs`
  - `scripts/generate_python.py`
  - `scripts/compare_generation_dumps.py`

## Engineering Principles in This Repo

- Candle-style crate layering and model composition.
- Runtime behavior defaults that favor production ergonomics.
- Model-specific chat template handling done behind the scenes.
- Configurable server deployment path:
  - binary usage for local ops
  - direct embedding for Rust integrations

## Current Limits

- Server is synchronous (single-process, simple TCP loop).
- Advanced scheduling/concurrency controls are not yet implemented.

## Deprecated Paths

Directories ending in `_deprecated` are historical references and are not the active runtime implementation.

Use the active `/crates` and `/src/bin` paths for all new development.
