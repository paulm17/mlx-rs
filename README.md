# mlx-rs

`mlx-rs` is being rewritten around **llama.cpp and GGUF models**.

The previous MLX-native Rust runtime (hand-written tensor ops, safetensors, custom model implementations) has been removed. The new implementation targets:

- `generate` CLI
- `mlx-server` CLI
- `mlx-lm` high-level Rust facade
- OpenAI-compatible `/v1/chat/completions`, `/v1/embeddings`, `/v1/models`
- Config loading and server options

All backed by llama.cpp for GGUF model inference.

## Status

**Phase 1 in progress.** See `PLAN.md` for the milestone plan.

Current state: workspace skeleton only. No inference, no server, no generation.

## Project Structure

```
Cargo.toml               # workspace root
crates/mlx-lm/            # public facade crate
src/bin/generate.rs       # generation CLI (not yet implemented)
src/bin/mlx-server.rs     # server CLI (not yet implemented)
config.toml               # server configuration
PLAN.md                   # detailed milestone plan
AGENTS.md                 # development rules
```

## Build

```bash
cargo check --workspace
```

## Configuration

See `config.toml` for server configuration format.

## Non-Goals (Phase 1)

- Safetensors model support
- MLX tensor API
- VLM / vision models
- Diffusion generation
