# mlx-rs Clean Break Plan: llama.cpp First, Ollama-Style MLX Later

Date: 2026-06-16

This file is the durable working plan for replacing the current `mlx-rs` implementation. It is intentionally detailed because future sessions may have little context. Re-read this file before doing any implementation work.

## Goal

Make a clean break from the current hand-written Rust MLX stack.

Phase 1 replaces the project with a llama.cpp-backed runtime that preserves the useful public API shape:

- `generate` CLI
- `mlx-server` CLI
- `mlx_lm` high-level Rust facade
- OpenAI-style `/v1/chat/completions`, `/v1/embeddings`, `/v1/models`
- config loading and server options

Phase 2 adds an Ollama-style MLX backend after the llama.cpp rewrite is stable. This must be an optional second backend behind the same runner/runtime interface, not a return to the current monolithic MLX model stack.

## Important Decision

The first implementation target is **llama.cpp with GGUF models**.

This is not the same as Ollama's MLX preview. Local inspection of `/Volumes/Data/Users/paul/development/src/github/ollama` shows:

- Ollama uses `llama-server` / llama.cpp for GGUF models.
- Ollama uses a separate `x/mlxrunner` subprocess for safetensors MLX models.
- Ollama routes by model format:
  - `model_format == "safetensors"` means MLX.
  - `model_format == "gguf"` or empty means llama.cpp.

So this project should first become a good llama.cpp wrapper, then later grow an Ollama-style backend abstraction that can host MLX.

## Current mlx-rs State To Discard

The current workspace contains these MLX-specific crates:

- `crates/mlx-sys`
- `crates/mlx-core`
- `crates/mlx-nn`
- `crates/mlx-models`
- `crates/mlx-vlm`
- `crates/mlx-lm`

Only `crates/mlx-lm` should survive as the public facade name. Its internals should be rewritten.

The current APIs that expose `mlx_core::Array` are not portable and should be removed or replaced:

- `CausalLM::forward_last_token_logits(&Array) -> Array`
- `EmbeddingModel::forward_hidden_states(&Array) -> Array`
- multimodal methods that take `pixel_values: Option<&Array>`

Public compatibility target is not the MLX tensor API. Public compatibility target is CLI, server, JSON routes, config, prompts, callbacks, metrics, and the `mlx_lm` facade.

## Reference Code To Revisit Before Work

### Current mlx-rs

- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/Cargo.toml`
  - Current workspace membership and binary declarations.
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-lm/src/lib.rs`
  - Current facade exports to preserve or deliberately replace.
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-lm/src/generate.rs`
  - Existing `GenerationPipeline`, `GenerationMetrics`, callback flow, stop handling.
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-lm/src/server.rs`
  - Existing HTTP server behavior, config parsing, auth, rate limiting, embeddings response shape.
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/src/bin/generate.rs`
  - Current CLI flags and stdout behavior.
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/src/bin/mlx-server.rs`
  - Current server CLI flags.

### Ollama llama.cpp Path

- `/Volumes/Data/Users/paul/development/src/github/ollama/llm/server.go`
  - `LlamaServer` interface. Key methods:
    - `Completion(ctx, req, fn)`
    - `Chat(ctx, req, fn)`
    - `ApplyChatTemplate(ctx, req)`
    - `Embedding(ctx, input)`
    - `Tokenize(ctx, content)`
- `/Volumes/Data/Users/paul/development/src/github/ollama/llm/llama_server.go`
  - llama.cpp subprocess wrapper.
  - Important areas:
    - launch config and command args
    - `appendBatchArgs`
    - `Completion`
    - `Chat`
    - `ApplyChatTemplate`
    - `Embedding`
    - `Tokenize`
- `/Volumes/Data/Users/paul/development/src/github/ollama/discover/llama_server.go`
  - How Ollama discovers llama-server GPU devices.
- `/Volumes/Data/Users/paul/development/src/github/ollama/server/sched.go`
  - Scheduler branch between llama.cpp and MLX.
  - Around the `if !req.model.IsMLX()` branch: non-MLX uses `llm.LoadModel` and `newServerFn`; MLX uses `mlxrunner.NewClient`.
- `/Volumes/Data/Users/paul/development/src/github/ollama/server/images.go`
  - `Model.IsMLX()` returns `m.Config.ModelFormat == "safetensors"`.
  - `isGGUF()` returns empty or `gguf`.
- `/Volumes/Data/Users/paul/development/src/github/ollama/server/routes.go`
  - `chatModeForModel`: MLX uses rendered chat, GGUF can use native llama-server chat template.

### Ollama MLX Path

- `/Volumes/Data/Users/paul/development/src/github/ollama/runner/runner.go`
  - Dispatches `ollama runner --mlx-engine` to `mlxrunner.Execute`.
- `/Volumes/Data/Users/paul/development/src/github/ollama/x/mlxrunner/client.go`
  - MLX subprocess client implementing `llm.LlamaServer`.
  - Starts `ollama runner --mlx-engine --model <name> --port <port>`.
  - Does not support native llama-server chat templates or embeddings yet.
- `/Volumes/Data/Users/paul/development/src/github/ollama/x/mlxrunner/server.go`
  - MLX runner HTTP server.
  - Endpoints:
    - `GET /v1/status`
    - `GET|POST|DELETE /v1/models`
    - `POST /v1/completions`
    - `POST /v1/tokenize`
    - redirects `/health`, `/load`, `/completion`
- `/Volumes/Data/Users/paul/development/src/github/ollama/x/mlxrunner/runner.go`
  - Loads manifest, model, tensors, pins arrays, enables MLX compile.
- `/Volumes/Data/Users/paul/development/src/github/ollama/x/mlxrunner/pipeline.go`
  - Text generation pipeline, prefill chunking, sampling, decode loop.
- `/Volumes/Data/Users/paul/development/src/github/ollama/x/mlxrunner/cache.go`
  - Prefix trie KV cache paging and snapshot strategy.
- `/Volumes/Data/Users/paul/development/src/github/ollama/x/mlxrunner/mlx/dynamic.go`
  - Dynamic loading of MLX-C runtime libraries.
  - Searches `mlx_*` dirs such as `mlx_metal_v3`, `mlx_metal_v4`.
  - Honors `OLLAMA_LLM_LIBRARY` only for `mlx_*` variants.
- `/Volumes/Data/Users/paul/development/src/github/ollama/x/mlxrunner/mlx/generated.h`
  - Generated MLX-C wrapper declarations.
- `/Volumes/Data/Users/paul/development/src/github/ollama/x/mlxrunner/model/base/base.go`
  - MLX model interface:
    - `Forward`
    - `Unembed`
    - `NumLayers`
    - `Tokenizer`
    - `MaxContextLength`
    - `LoadWeights`
- `/Volumes/Data/Users/paul/development/src/github/ollama/x/models/llama/llama.go`
  - Example Go MLX model implementation.
- `/Volumes/Data/Users/paul/development/src/github/ollama/x/mlxrunner/imports.go`
  - Registers supported MLX model families:
    - Gemma3
    - Gemma4
    - GLM4 MoE Lite
    - Laguna
    - Llama
    - Qwen3
    - Qwen3.5
    - Qwen3.5 MoE
- `/Volumes/Data/Users/paul/development/src/github/ollama/cmake/local.cmake`
  - Fetches pinned llama.cpp, MLX, and MLX-C sources.
- `/Volumes/Data/Users/paul/development/src/github/ollama/cmake/mlx/CMakeLists.txt`
  - Builds and installs MLX runtime libraries.
- `/Volumes/Data/Users/paul/development/src/github/ollama/CMakePresets.json`
  - `MLX Metal` preset sets `OLLAMA_MLX_BACKENDS=metal_v3;metal_v4`.

### llama.cpp C API

Verify against the actual pinned binding before implementation because llama.cpp APIs move. Current upstream `include/llama.h` references:

- Backend lifecycle:
  - `llama_backend_init`
  - `llama_backend_free`
- Model and context:
  - `llama_model_default_params`
  - `llama_model_load_from_file`
  - `llama_model_free`
  - `llama_context_default_params`
  - `llama_init_from_model`
  - `llama_free`
  - `llama_model_get_vocab`
  - `llama_model_n_embd`
  - `llama_model_n_ctx_train`
  - `llama_n_ctx`
- Batch and decode:
  - `llama_batch_init`
  - `llama_batch_free`
  - `llama_batch_get_one`
  - `llama_decode`
  - `llama_encode`
  - `llama_get_logits_ith`
- Sampling:
  - `llama_sampler_chain_default_params`
  - `llama_sampler_chain_init`
  - `llama_sampler_chain_add`
  - `llama_sampler_init_top_k`
  - `llama_sampler_init_top_p`
  - `llama_sampler_init_min_p`
  - `llama_sampler_init_temp`
  - `llama_sampler_init_greedy`
  - `llama_sampler_init_dist`
  - `llama_sampler_sample`
  - `llama_sampler_free`
- Tokenization and detokenization:
  - `llama_tokenize`
  - `llama_token_to_piece`
  - `llama_vocab_is_eog`
  - `llama_vocab_eos`
  - `llama_vocab_get_text`
- Chat templates:
  - `llama_chat_apply_template`
  - `llama_chat_builtin_templates`
- Embeddings:
  - `llama_context_params.embeddings`
  - `llama_context_params.pooling_type`
  - `llama_set_embeddings`
  - `llama_get_embeddings`
  - `llama_get_embeddings_ith`
  - `llama_get_embeddings_seq`
- State and cache:
  - `llama_get_memory`
  - `llama_memory_clear`
  - `llama_memory_seq_rm`
  - `llama_memory_seq_cp`
  - `llama_memory_seq_keep`
  - `llama_memory_seq_pos_min`
  - `llama_memory_seq_pos_max`
  - `llama_state_get_size`
  - `llama_state_get_data`
  - `llama_state_set_data`
  - `llama_state_seq_get_size`
  - `llama_state_seq_get_data`
  - `llama_state_seq_set_data`

Upstream reference:

- https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h

## Milestone Rules

Each milestone must be small enough to finish in one focused session.

For every milestone:

1. Read this file first.
2. Run `git status --short`.
3. Do not preserve old MLX code unless that milestone explicitly says to.
4. Keep public behavior tested before moving to the next milestone.
5. Update this file after completing a milestone:
   - mark status
   - add notes
   - add any newly discovered reference paths or API corrections

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` complete
- `[!]` blocked or needs decision

## Phase 1: Clean Break To llama.cpp

### Milestone 1.1 - Workspace Demolition And Skeleton

Status: `[x]`

Objective:

Delete the MLX implementation and leave a minimal compiling Rust workspace skeleton.

Actions:

- Replace root `Cargo.toml` workspace members with only:
  - `crates/mlx-lm`
- Keep root package only if needed for binaries:
  - `src/bin/generate.rs`
  - `src/bin/mlx-server.rs`
- Delete active use of:
  - `crates/mlx-sys`
  - `crates/mlx-core`
  - `crates/mlx-nn`
  - `crates/mlx-models`
  - `crates/mlx-vlm`
- Delete or quarantine MLX-specific binaries:
  - `compare_all`
  - `compare_llm_layers`
  - `compare_logits`
  - `diagnose_first_token`
  - `diffusion_gemma_trace`
  - `generate_diag`
  - `simple_vision_test`
  - `test_gemma4`
  - `test_gemma4_vision`
  - `embed_bench`
- Delete or quarantine MLX parity scripts and Python tests.
- Keep `README.md`, `config.toml`, `AGENTS.md`, and this `PLAN.md`.
- Make `crates/mlx-lm/src/lib.rs` compile with placeholder modules:
  - `config`
  - `runtime`
  - `sampler`
  - `server`
  - `types`

Acceptance:

- `cargo check --workspace` compiles.
- Running old removed binaries is no longer possible.
- `README.md` says the project is being rewritten around llama.cpp/GGUF.

Do not implement llama.cpp yet in this milestone.

Completion notes (2026-06-16):

- Deleted crates: mlx-sys, mlx-core, mlx-nn, mlx-models, mlx-vlm
- Deleted binaries: check_tokens, compare_all, compare_llm_layers, compare_logits, diagnose_first_token, diffusion_gemma_trace, embed_bench, generate_diag, simple_vision_test, test_gemma4, test_gemma4_vision
- Deleted directories: python_tests/, scripts/, test_harness/, tests/, config/, crates/mlx-lm/src/bin/
- Root Cargo.toml: workspace members reduced to `crates/mlx-lm` only; root package kept for generate + mlx-server binaries
- mlx-lm placeholder modules: config, runtime, sampler, server, types
- Binaries stub with "not yet implemented" exit
- README updated to reflect llama.cpp rewrite status

### Milestone 1.2 - Public Types And Config Contract

Status: `[ ]`

Objective:

Define the backend-neutral public API and config surface without real inference.

Actions:

- Define public types in `mlx-lm`:
  - `ServerConfig`
  - `Sampler`
  - `GenerationOptions`
  - `GenerationMetrics`
  - `GenerateOutput`
  - `ChatMessage`
  - `EmbeddingOutput`
  - `LoadedModelInfo`
- Preserve these facade exports where practical:
  - `run_server`
  - `run_server_from_toml_path`
  - `GenerationPipeline`
  - `Sampler`
  - `ServerConfig`
- Config keys to preserve:
  - `bind`
  - `port`
  - `model_path`
  - `model`
  - `api_key`
  - `rate_limit_rpm`
  - `thinking`
  - `embeddings_batch_size`
- Add llama.cpp config keys:
  - `n_ctx`
  - `n_batch`
  - `n_ubatch`
  - `n_gpu_layers`
  - `n_threads`
  - `n_threads_batch`
  - `embedding`
  - `pooling`
  - `use_mmap`
  - `use_mlock`
  - `flash_attn`
- Write config parsing tests.
- Return explicit `not implemented` errors for generation and embedding.

Acceptance:

- `cargo test -p mlx-lm config` passes.
- `cargo check --workspace` passes.
- Existing config files parse.

### Milestone 1.3 - GGUF Model Path Resolution

Status: `[ ]`

Objective:

Make model loading resolve GGUF paths and reject old MLX/safetensors directories clearly.

Actions:

- Implement `resolve_model_dir` replacement as `resolve_model_path`.
- Supported input:
  - direct `.gguf` file
  - directory containing exactly one `.gguf`
  - directory containing split GGUF files when naming is unambiguous
- Rejected input:
  - missing path
  - directory with only `config.json` / `.safetensors`
  - directory with multiple unrelated `.gguf` files
- Error message for safetensors:
  - "safetensors/MLX model directories are not supported by the llama.cpp runtime; provide a GGUF model file"
- Leave Hugging Face resolution out unless it is already trivial from current `hf-hub` cache logic.

Acceptance:

- Unit tests cover all path cases.
- Server `/llm/load` can validate a path but still does not run inference.

### Milestone 1.4 - Choose And Pin llama.cpp Binding

Status: `[ ]`

Objective:

Select the actual Rust integration path and pin it.

Recommended default:

- Use `llama-cpp-2` if it exposes the needed current llama.cpp API.
- If it does not expose enough API or build reliably, use a local `llama-sys` crate with bindgen over pinned llama.cpp.

Actions:

- Inspect `llama-cpp-2` APIs locally after adding the dependency.
- Confirm support for:
  - model load
  - context init
  - tokenization
  - decode
  - sampler chain or equivalent sampling
  - embeddings
  - Metal / GPU layer configuration
- Pin versions in `Cargo.toml`.
- Add a `build.rs` only if needed.
- Do not download or install external system dependencies manually.

Acceptance:

- `cargo check -p mlx-lm` links or at least compiles wrapper types.
- Document exact crate version and API mapping in this file.

### Milestone 1.5 - llama.cpp Model Load And Metadata

Status: `[ ]`

Objective:

Load a GGUF model into a llama.cpp model/context and expose metadata.

Actions:

- Add `LlamaBackend` / `LlamaRuntime`.
- On runtime creation:
  - call `llama_backend_init` once per process or via binding equivalent.
  - configure model params:
    - `n_gpu_layers`
    - `use_mmap`
    - `use_mlock`
  - call `llama_model_load_from_file`.
  - configure context params:
    - `n_ctx`
    - `n_batch`
    - `n_ubatch`
    - `n_threads`
    - `n_threads_batch`
    - `embeddings`
    - `pooling_type`
  - call `llama_init_from_model`.
- Expose:
  - model path
  - context length
  - embedding dimension
  - vocab size if available
  - whether embeddings are enabled
- Ensure `Drop` frees context and model.

Acceptance:

- Env-gated real model test loads a tiny GGUF if `MLX_RS_TEST_GGUF` is set.
- Non-env unit tests do not require a real model.
- `cargo check --workspace` passes.

### Milestone 1.6 - Tokenization And Detokenization

Status: `[ ]`

Objective:

Replace `tokenizers` dependency for generation with llama.cpp tokenizer, so GGUF metadata drives tokenization.

Actions:

- Implement:
  - `Runtime::tokenize(&str, add_bos: bool) -> Vec<i32>`
  - `Runtime::detokenize_piece(token) -> String`
  - incremental UTF-8 buffering if llama.cpp binding returns partial bytes.
- Reference llama.cpp:
  - `llama_model_get_vocab`
  - `llama_tokenize`
  - `llama_token_to_piece`
  - `llama_vocab_is_eog`
- Keep existing `ChatTemplate` fallback only if needed for rendered prompts.

Acceptance:

- Unit tests with mocked runtime or env-gated GGUF validate:
  - non-empty tokenization
  - piece detokenization
  - EOS/EOG detection

### Milestone 1.7 - Non-Streaming Text Generation

Status: `[ ]`

Objective:

Implement `GenerationPipeline::generate` and `/v1/chat/completions` non-streaming path.

Actions:

- Implement prefill:
  - tokenize prompt
  - build batch
  - call `llama_decode`
  - request logits for the last token
- Implement decode loop:
  - sample next token
  - stop on EOG/EOS
  - stop on `max_tokens`
  - detokenize incrementally
  - track `GenerationMetrics`
- Implement sampler mapping:
  - temperature `0` means greedy
  - temperature > 0 uses top-k/top-p/min-p/temp/dist chain
- Reference llama.cpp:
  - `llama_batch_init`
  - `llama_decode`
  - `llama_get_logits_ith`
  - `llama_sampler_chain_init`
  - `llama_sampler_chain_add`
  - `llama_sampler_sample`
  - `llama_sampler_free`

Acceptance:

- `generate` CLI works with `MLX_RS_TEST_GGUF`.
- `/v1/chat/completions` non-streaming returns OpenAI-style JSON.
- Unit tests cover stop reasons and response shape without requiring a real model.

### Milestone 1.8 - Streaming Text Generation

Status: `[ ]`

Objective:

Restore streaming callback and server streaming behavior.

Actions:

- Implement `generate_with_callback`.
- Implement server streaming chunks.
- Preserve cancellation:
  - stop signal for CLI
  - client disconnect for server
- Preserve metrics in final chunk where current API expects them.

Acceptance:

- CLI streaming output appears token-by-token or piece-by-piece.
- Server streaming test validates:
  - multiple chunks
  - final chunk
  - cancellation does not panic

### Milestone 1.9 - Server Load, Models, Health, Auth, Rate Limit

Status: `[ ]`

Objective:

Rebuild the server around the llama.cpp runtime.

Actions:

- Preserve routes:
  - `GET /health`
  - `GET /v1/models`
  - `POST /llm/load`
  - `POST /v1/chat/completions`
  - `POST /v1/embeddings`
- Preserve:
  - API key via `x-api-key` or `Authorization: Bearer`
  - fixed-window RPM limiting
  - startup preload from config
- Use one loaded runtime guarded by a mutex.
- Do not implement concurrency beyond current safety until the simple path is stable.

Acceptance:

- Server tests cover:
  - health without model
  - model list before/after load
  - auth failures
  - rate limiting
  - load invalid model path

### Milestone 1.10 - Embeddings

Status: `[ ]`

Objective:

Implement `/v1/embeddings` using llama.cpp embeddings.

Actions:

- Support request input:
  - string
  - array of strings
- Reject token arrays as current server does.
- Set context params:
  - `embeddings = true`
  - appropriate `pooling_type`
- Reference llama.cpp:
  - `llama_set_embeddings`
  - `llama_get_embeddings`
  - `llama_get_embeddings_ith`
  - `llama_get_embeddings_seq`
  - `llama_model_n_embd`
- Return OpenAI-style response:
  - `object: "list"`
  - `data[*].embedding`
  - `usage.prompt_tokens`
- Normalize vectors unless llama.cpp/GGUF metadata already guarantees normalized output. Match current server behavior by default.

Acceptance:

- Unit tests cover JSON shape and validation.
- Env-gated GGUF embedding test passes if `MLX_RS_TEST_EMBED_GGUF` is set.
- Non-embedding model returns a clear unsupported error.

### Milestone 1.11 - Chat Template Strategy

Status: `[ ]`

Objective:

Make chat behavior robust for GGUF models.

Actions:

- Prefer llama.cpp / GGUF native chat template if exposed by binding.
- Fallback to existing or simplified `ChatTemplate` rendering.
- Preserve current message roles:
  - system
  - user
  - assistant
  - tool only if already supported by server surface
- Preserve `thinking` strip behavior if it remains relevant.
- Reference llama.cpp:
  - `llama_chat_apply_template`
  - `llama_chat_builtin_templates`
- Reference Ollama:
  - `/Volumes/Data/Users/paul/development/src/github/ollama/server/routes.go` `chatModeForModel`.

Acceptance:

- Chat tests cover:
  - simple user message
  - system + user
  - assistant history
  - fallback when native template unavailable

### Milestone 1.12 - Documentation And Cleanup

Status: `[ ]`

Objective:

Make the llama.cpp rewrite understandable and remove stale MLX claims.

Actions:

- Rewrite `README.md`.
- Update `config.toml`.
- Remove stale benchmark configs that reference MLX safetensors unless preserved under a `legacy/` note.
- Document:
  - GGUF requirement
  - supported routes
  - example `generate`
  - example `mlx-server`
  - Metal/GPU layer config
  - embeddings support
  - unsupported VLM/diffusion status

Acceptance:

- A new user can run one GGUF model from README instructions.
- No docs claim safetensors/MLX support in Phase 1.

## Phase 2: Ollama-Style Multi-Backend Foundation

Phase 2 starts only after Phase 1 is stable.

### Milestone 2.1 - Backend Trait And Runtime Registry

Status: `[ ]`

Objective:

Introduce a backend abstraction that can host both llama.cpp and MLX.

Actions:

- Define trait similar in spirit to Ollama's `llm.LlamaServer`, but Rust-native:
  - `load`
  - `model_info`
  - `tokenize`
  - `generate`
  - `generate_stream`
  - `embedding`
  - `supports_chat_template`
  - `apply_chat_template`
  - `memory_info`
- Existing llama.cpp runtime becomes `LlamaCppBackend`.
- Add a registry:
  - `gguf` -> llama.cpp
  - `safetensors` -> reserved for MLX
- Keep server code talking only to the trait.

Acceptance:

- No behavior changes from Phase 1.
- Tests pass with llama.cpp backend through the trait.

### Milestone 2.2 - Runner Process Protocol

Status: `[ ]`

Objective:

Create an internal runner HTTP protocol inspired by Ollama, so MLX can run in a subprocess later.

Actions:

- Define internal endpoints:
  - `GET /v1/status`
  - `POST /v1/load`
  - `POST /v1/completions`
  - `POST /v1/tokenize`
  - `POST /v1/embeddings` reserved
- Define JSON structs for:
  - completion request
  - completion response
  - status response
  - tokenizer response
- Add a `RunnerClient` that implements the backend trait over HTTP.
- Initially test it against an in-process mock runner.

Reference:

- Ollama `x/mlxrunner/client.go`
- Ollama `x/mlxrunner/server.go`

Acceptance:

- Mock runner can serve one completion through the main server.
- Main server does not know whether backend is in-process or subprocess.

### Milestone 2.3 - MLX Dynamic Loader Skeleton

Status: `[ ]`

Objective:

Create a Rust equivalent of Ollama's dynamic MLX-C loader without model inference.

Actions:

- Add `crates/mlx-runner` or `crates/mlx-backend`.
- Add FFI boundary for MLX-C.
- Implement library search:
  - project `lib/ollama`-style dirs
  - `mlx_*` dirs
  - env override for `MLX_RS_MLX_LIBRARY` or `OLLAMA_LLM_LIBRARY` compatibility
- Implement:
  - `check_init`
  - `loaded_library_path`
  - version query if available
  - device availability check
- Do not implement tensors or models yet.

Reference:

- Ollama `x/mlxrunner/mlx/dynamic.go`
- Ollama `x/mlxrunner/mlx/dynamic.c`
- Ollama `x/mlxrunner/mlx/generated.h`

Acceptance:

- On a machine with MLX-C runtime, loader reports version/device.
- Without MLX-C runtime, loader returns a clear unavailable error.

### Milestone 2.4 - MLX Array And Ops Minimal Binding

Status: `[ ]`

Objective:

Implement only the MLX array operations needed for a tiny smoke path.

Actions:

- Add safe wrappers for:
  - array creation from ints/floats
  - shape/dtype
  - eval/async eval
  - memory sweep/cache clear
  - basic ops needed by embeddings/linear smoke tests
- Keep API private to MLX backend; do not recreate public `mlx-core`.

Reference:

- Ollama `x/mlxrunner/mlx/array.go`
- Ollama `x/mlxrunner/mlx/ops.go`
- Ollama `x/mlxrunner/mlx/memory.go`
- Current mlx-rs `crates/mlx-core/src/array.rs` only as cautionary prior art.

Acceptance:

- Unit tests create arrays and evaluate basic operations when MLX is available.
- Tests skip cleanly when MLX is unavailable.

### Milestone 2.5 - MLX Manifest And Safetensors Loader

Status: `[ ]`

Objective:

Load safetensors model assets in a backend-private format.

Actions:

- Decide model package format:
  - direct Hugging Face snapshot directory
  - Ollama-style manifest
  - project-local manifest
- Implement config and tensor discovery.
- Read quantization metadata where available.
- Load tensors into MLX arrays.
- Do not instantiate a transformer yet.

Reference:

- Ollama `x/mlxrunner/model/root.go`
- Ollama `x/mlxrunner/runner.go` `loadTensorsFromManifest`
- Current mlx-rs `crates/mlx-nn/src/var_builder.rs`

Acceptance:

- Can list tensors and config for one safetensors model.
- Can load selected tensor into MLX array.

### Milestone 2.6 - MLX Llama Minimal Inference

Status: `[ ]`

Objective:

Implement one small dense Llama-style model in MLX behind the backend trait.

Actions:

- Port only the minimum Llama path.
- Implement:
  - embedding
  - RMSNorm
  - RoPE
  - attention
  - MLP
  - final logits
  - KV cache
- Do not support MoE, VLM, Gemma, Qwen, diffusion, speculative decoding yet.

Reference:

- Ollama `x/models/llama/llama.go`
- Ollama `x/models/nn`
- Ollama `x/mlxrunner/model/base/base.go`
- Current mlx-rs `crates/mlx-models/src/llama.rs`

Acceptance:

- Env-gated tiny Llama safetensors model produces logits.
- One-token generation works through backend trait.

### Milestone 2.7 - MLX Runner Subprocess

Status: `[ ]`

Objective:

Run the MLX backend in a subprocess like Ollama.

Actions:

- Add a runner binary mode:
  - `mlx-rs-runner --mlx-engine --model <path> --port <port>`
  - or reuse root binary with a hidden runner subcommand.
- Serve internal runner endpoints from Milestone 2.2.
- Main server starts subprocess for safetensors models.
- Main server uses `RunnerClient` for subprocess communication.

Reference:

- Ollama `runner/runner.go`
- Ollama `x/mlxrunner/client.go`
- Ollama `x/mlxrunner/server.go`

Acceptance:

- Main server can load a safetensors model by spawning MLX runner.
- Process cleanup works on unload/server exit.
- llama.cpp GGUF path still works.

### Milestone 2.8 - MLX Chat Rendering And Tokenizer

Status: `[ ]`

Objective:

Make MLX backend usable through chat APIs.

Actions:

- Use tokenizer files from safetensors snapshots.
- Apply rendered chat in main server, not native llama.cpp chat templates.
- Mirror Ollama's strategy:
  - MLX uses rendered prompt path.
  - llama.cpp/GGUF can use native template where available.

Reference:

- Ollama `server/routes.go` `chatModeForModel`
- Ollama `x/mlxrunner/client.go` returns unsupported for native chat template.
- Current mlx-rs `crates/mlx-lm/src/chat_template.rs`

Acceptance:

- `/v1/chat/completions` works with MLX Llama backend.
- Existing llama.cpp chat path unaffected.

### Milestone 2.9 - MLX KV Cache And Prefix Reuse

Status: `[ ]`

Objective:

Add practical cache behavior after basic MLX generation works.

Actions:

- Start with simple per-request KV cache.
- Then add prefix cache only if baseline works.
- Use Ollama compressed trie design as reference, not first implementation.

Reference:

- Ollama `x/mlxrunner/cache.go`
- Ollama `x/mlxrunner/cache/`
- Current mlx-rs `crates/mlx-nn/src/kv_cache.rs`

Acceptance:

- Multi-turn generation does not leak memory.
- Repeated prompt smoke test has stable output and memory.

### Milestone 2.10 - Additional MLX Model Families

Status: `[ ]`

Objective:

Add model families one at a time only after MLX Llama path is stable.

Order:

1. Qwen dense
2. Gemma3 text
3. Qwen3
4. Qwen3.5
5. Gemma4 text
6. MoE families
7. VLM
8. diffusion

Reference:

- Ollama `x/mlxrunner/imports.go`
- Ollama `x/models/*`
- Current mlx-rs `crates/mlx-models/src/*`

Acceptance:

- Each family has:
  - one env-gated real model smoke test
  - one config detection test
  - one generation smoke

## Phase 3: Polish, Performance, And Release

### Milestone 3.1 - Performance Knobs

Status: `[ ]`

Objective:

Expose stable runtime tuning without leaking backend internals.

Actions:

- llama.cpp:
  - `n_ctx`
  - `n_batch`
  - `n_ubatch`
  - `n_gpu_layers`
  - `n_threads`
  - `flash_attn`
- MLX:
  - cache limit
  - compile enable/disable
  - prefill chunk size
- Server:
  - concurrency remains conservative until proven safe.

Acceptance:

- Config docs and tests cover every option.

### Milestone 3.2 - Compatibility Matrix

Status: `[ ]`

Objective:

Document what works and what does not.

Actions:

- Create `docs/compatibility.md`.
- Matrix columns:
  - backend
  - model format
  - text generation
  - chat
  - embeddings
  - tools
  - vision
  - tested model
  - notes

Acceptance:

- README links to compatibility matrix.

### Milestone 3.3 - Regression Benchmarks

Status: `[ ]`

Objective:

Add practical smoke benchmarks without recreating old parity machinery.

Actions:

- Add simple benchmark script:
  - load time
  - TTFT
  - tokens/sec
  - embeddings/sec
- Require explicit model paths via env vars.
- Do not assume network access.

Acceptance:

- Benchmarks run locally with provided GGUF paths.

## Non-Goals

These are explicitly out of scope for the first llama.cpp rewrite:

- Reproducing Python `mlx_lm` parity.
- Keeping `mlx_core::Array` public APIs.
- Supporting safetensors in Phase 1.
- Supporting Gemma4 VLM in Phase 1.
- Supporting diffusion generation in Phase 1.
- Building a full Ollama clone.
- Implementing scheduler-level concurrent model residency before one-model server behavior is stable.

## Implementation Defaults

When in doubt:

- Prefer deleting old MLX code over adapting it.
- Prefer small backend-neutral structs over generic trait towers.
- Prefer direct, tested behavior over broad architecture claims.
- Prefer one loaded model at a time until concurrency is justified.
- Prefer explicit unsupported errors over partial hidden behavior.
- Prefer GGUF for Phase 1.
- Prefer subprocess isolation for MLX in Phase 2.

## Quick Resume Checklist

Before implementing anything:

1. Read `PLAN.md`.
2. Check which milestone is next.
3. Run `git status --short`.
4. Run `cargo check --workspace` to see current baseline.
5. Complete only the next milestone.
6. Run the milestone acceptance checks.
7. Update milestone status and notes in this file.

