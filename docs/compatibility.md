# Compatibility Matrix

Last updated: 2026-06-17

## Backend Support

| Feature | llama.cpp / GGUF | MLX / safetensors | Subprocess runner |
|---------|:-----------------:|:-----------------:|:------------------:|
| Text generation | Yes | Yes | Yes |
| Chat completions | Yes | Yes | Yes |
| Streaming generation | Yes | Yes | Yes |
| Native chat templates | Yes | Partial | No |
| Embeddings | Yes (opt-in) | No | No |
| Stop sequences | Yes | No | No |
| Tool / function calling | Yes | No | No |
| Vision / multimodal | No | No | No |
| KV cache reuse | N/A (llama.cpp manages) | Yes (prefix cache) | Yes (delegated) |
| Memory info | No | Yes | No |
| Tokenize | Yes | Yes | Yes |
| Detokenize | Yes | Yes | No |
| Auth / rate limiting | Yes (server) | Yes (server) | Yes (server) |

## Model Architectures

### llama.cpp / GGUF

Any model that llama.cpp supports in GGUF format. The server does not restrict architectures at this level.

### MLX / safetensors

| Architecture | Model Family | Text Generation | Notes |
|-------------|-------------|:---------------:|-------|
| `LlamaForCausalLM` | Llama, LLaMA, Mistral | Yes | Dense transformer |
| `LlamaForSequenceClassification` | Llama classifier | No | Not a generation model |
| `Qwen3ForCausalLM` | Qwen 3 | Yes | Q/K RMSNorm, RoPE |
| `Gemma3ForCausalLM` | Gemma 3 (text) | Yes | 4 norms per layer, GELU MLP |
| `Gemma3ForConditionalGeneration` | Gemma 3 (VLM arch) | Yes | Text-only; no vision encoder |
| `Gemma4ForCausalLM` | Gemma 4 (text) | Yes | Sliding window attention |
| `Gemma4ForConditionalGeneration` | Gemma 4 (VLM arch) | Yes | Text-only; no vision encoder |

## Configuration

### llama.cpp options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_ctx` | int | 32768 | Context window size |
| `n_batch` | int | 512 | Batch size for prompt processing |
| `n_ubatch` | int | 512 | Micro-batch size |
| `n_gpu_layers` | int | 99 | GPU layers to offload |
| `n_threads` | int | 4 | Thread count for generation |
| `n_threads_batch` | int | 4 | Thread count for batch processing |
| `embedding` | bool | false | Enable embedding mode |
| `pooling` | string | "mean" | Pooling type for embeddings |
| `use_mmap` | bool | true | Use memory-mapped files |
| `use_mlock` | bool | false | Lock model in memory |
| `flash_attn` | bool | auto | Flash attention (true/false/auto) |

### MLX options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mlx_cache_limit` | int | 64 | Prefix cache capacity (number of entries) |
| `mlx_compile` | bool | false | Enable MLX compile (reserved; not yet active) |
| `mlx_prefill_chunk_size` | int | 2048 | Tokens per prefill chunk |

## Server Endpoints

| Method | Path | GGUF | safetensors | Notes |
|--------|------|:----:|:-----------:|-------|
| `GET` | `/health` | Yes | Yes | Returns status and model_loaded |
| `GET` | `/models` | Yes | Yes | llama.cpp-compatible model discovery; lists the loaded model |
| `GET` | `/v1/models` | Yes | Yes | OpenAI-compatible model list; lists the loaded model |
| `POST` | `/llm/load` | Yes | Yes | Auto-detects format |
| `POST` | `/v1/chat/completions` | Yes | Yes | Streaming and non-streaming; llama.cpp supports text parts, reasoning, and tools |
| `POST` | `/v1/embeddings` | Yes | No | Requires `embedding = true` |

## Known Limitations

- **Embeddings** on MLX backend are not yet supported. Use llama.cpp/GGUF with `embedding = true`.
- **Stop sequences** are not enforced by the MLX backend during generation; only EOG tokens and `max_tokens` halt generation.
- **Vision models** (`Gemma3ForConditionalGeneration`, `Gemma4ForConditionalGeneration`) register their architecture names but implement only the text/language forward pass. No image encoder, pixel processing, or vision inference is available.
- **Tool / function calling** is supported for llama.cpp/GGUF through the
  OpenAI-compatible request fields and llama.cpp chat-template parser. The
  supported `tool_choice` modes are `auto`, `none`, and `required`;
  function-specific choice objects are rejected. MLX remains text-only for
  tool calls.
- **Reasoning** for llama.cpp/GGUF is returned as `reasoning_content` and can
  be streamed. OhMyPi-compatible reasoning efforts are `low`, `medium`, and
  `xhigh`; `high` is rejected by OhMyPi's local model metadata before it sends
  a request.
- **Message content** accepts a string or an array of text parts. Image and
  audio content parts are rejected because this server's chat path is text-only.
- **MoE models** are not yet supported by the MLX backend.
- **Subprocess runner** (`llama-rs-runner`) uses default MLX config; per-request MLX performance knobs are not forwarded through the runner protocol yet.
- **Model discovery** lists only the configured or dynamically loaded model. This server does not scan a directory or implement llama.cpp router-mode model management.
