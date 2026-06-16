# llama-rs Clean Break Plan: llama.cpp First, Ollama-Style MLX Later

Date: 2026-06-16

This file is the durable working plan for replacing the current `llama-rs` implementation. It is intentionally detailed because future sessions may have little context. Re-read this file before doing any implementation work.

## Goal

Make a clean break from the current hand-written Rust MLX stack.

Phase 1 replaces the project with a llama.cpp-backed runtime that preserves the useful public API shape:

- `generate` CLI
- `llama-server` CLI
- `llama_lm` high-level Rust facade
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

## Current llama-rs State To Discard

The current workspace contains these MLX-specific crates:

- `crates/mlx-sys`
- `crates/mlx-core`
- `crates/mlx-nn`
- `crates/mlx-models`
- `crates/mlx-vlm`
- `crates/llama-lm`

Only `crates/llama-lm` should survive as the public facade name. Its internals should be rewritten.

The current APIs that expose `mlx_core::Array` are not portable and should be removed or replaced:

- `CausalLM::forward_last_token_logits(&Array) -> Array`
- `EmbeddingModel::forward_hidden_states(&Array) -> Array`
- multimodal methods that take `pixel_values: Option<&Array>`

Public compatibility target is not the MLX tensor API. Public compatibility target is CLI, server, JSON routes, config, prompts, callbacks, metrics, and the `llama_lm` facade.

## Reference Code To Revisit Before Work

### Current llama-rs

- `/Volumes/Data/Users/paul/development/src/github/llama-rs/Cargo.toml`
  - Current workspace membership and binary declarations.
- `/Volumes/Data/Users/paul/development/src/github/llama-rs/crates/llama-lm/src/lib.rs`
  - Current facade exports to preserve or deliberately replace.
- `/Volumes/Data/Users/paul/development/src/github/llama-rs/crates/llama-lm/src/generate.rs`
  - Existing `GenerationPipeline`, `GenerationMetrics`, callback flow, stop handling.
- `/Volumes/Data/Users/paul/development/src/github/llama-rs/crates/llama-lm/src/server.rs`
  - Existing HTTP server behavior, config parsing, auth, rate limiting, embeddings response shape.
- `/Volumes/Data/Users/paul/development/src/github/llama-rs/src/bin/generate.rs`
  - Current CLI flags and stdout behavior.
- `/Volumes/Data/Users/paul/development/src/github/llama-rs/src/bin/llama-server.rs`
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
  - `crates/llama-lm`
- Keep root package only if needed for binaries:
  - `src/bin/generate.rs`
  - `src/bin/llama-server.rs`
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
- Make `crates/llama-lm/src/lib.rs` compile with placeholder modules:
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
- Deleted directories: python_tests/, scripts/, test_harness/, tests/, config/, crates/llama-lm/src/bin/
- Root Cargo.toml: workspace members reduced to `crates/llama-lm` only; root package kept for generate + llama-server binaries
- llama-lm placeholder modules: config, runtime, sampler, server, types
- Binaries stub with "not yet implemented" exit
- README updated to reflect llama.cpp rewrite status

### Milestone 1.2 - Public Types And Config Contract

Status: `[x]`

Objective:

Define the backend-neutral public API and config surface without real inference.

Actions:

- Define public types in `llama-lm`:
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

- `cargo test -p llama-lm config` passes.
- `cargo check --workspace` passes.
- Existing config files parse.

Completion notes (2026-06-16):

- Public types defined in `types.rs`: ChatMessage (with system/user/assistant constructors), GenerationOptions, GenerationMetrics, GenerateOutput, StopReason, EmbeddingData, EmbeddingUsage, EmbeddingOutput, LoadedModelInfo
- ServerConfig extended with all llama.cpp keys: n_ctx, n_batch, n_ubatch, n_gpu_layers, n_threads, n_threads_batch, embedding, pooling, use_mmap, use_mlock, flash_attn
- ServerConfig::to_llamacpp_config() bridge method
- LlamaCppConfig kept in config.rs for runtime use
- GenerationPipeline facade stub in lib.rs
- Sampler has builder methods: with_top_k, with_min_p
- 7 config parsing tests: empty, basic, llamacpp keys, comments/blanks, partial, to_llamacpp_config, toml value types
- Added tempfile dev-dependency for tests

### Milestone 1.3 - GGUF Model Path Resolution

Status: `[x]`

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

Completion notes (2026-06-16):

- `resolve_model_path` implemented in `loader.rs`
- Handles: direct .gguf file, directory with single .gguf, directory with split GGUF shards (NNNNN-of-NNNNN pattern)
- Handles Hugging Face GGUF file refs like `unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf` by downloading into a local cache when missing
- Rejects: missing path, non-gguf file, safetensors/config.json directories, empty dirs, ambiguous multi-gguf dirs
- Shard detection: parses `-NNNNN-of-NNNNN` suffix, groups by base name, returns first shard sorted by index
- 19 loader tests: Hugging Face ref parsing/URL encoding plus direct file, nonexistent, non-gguf, single in dir, split shards, multiple unrelated, safetensors, config.json only, empty dir, shard name parsing, not-shard patterns, unsorted shards
- Re-exported as `llama_lm::resolve_model_path`

### Milestone 1.4 - Choose And Pin llama.cpp Binding

Status: `[x]`

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

- `cargo check -p llama-lm` links or at least compiles wrapper types.
- Document exact crate version and API mapping in this file.

Completion notes (2026-06-16):

- Selected exact `llama-cpp-2` v0.1.146 pin (with `llama-cpp-sys-2` v0.1.146 in `Cargo.lock`)
- No `build.rs` needed; crate builds llama.cpp from source via cmake
- API mapping to llama.cpp C API:
  - `llama_backend_init` -> `llama_cpp_2::llama_backend::LlamaBackend`
  - `llama_model_load_from_file` -> `llama_cpp_2::model::LlamaModel`
  - `llama_init_from_model` -> `llama_cpp_2::context::LlamaContext`
  - `llama_tokenize` -> `LlamaModel::tokenize()`
  - `llama_token_to_piece` -> `LlamaModel::token_to_str()`
  - `llama_decode` -> `LlamaContext::decode()`
  - `llama_get_logits_ith` -> `LlamaContext::get_logits_ith()`
  - `llama_sampler_chain_init/add/sample/free` -> `llama_cpp_2::sampling::LlamaSampler`
  - `llama_chat_apply_template` -> `LlamaModel::apply_chat_template()`
  - `llama_get_embeddings` -> `LlamaContext::embeddings()`
  - `llama_model_n_embd` -> `LlamaModel::n_embd()`
  - `llama_model_n_ctx_train` -> `LlamaModel::n_ctx_train()`
  - `llama_n_ctx` -> `LlamaContext::n_ctx()`
  - `llama_vocab_is_eog` -> `LlamaModel::is_eog_token()`
  - `llama_vocab_eos` -> `LlamaModel::token_eos()`
  - `list_llama_ggml_backend_devices` -> GPU/Metal device discovery
  - `llama_memory_clear/seq_rm/seq_cp` -> `LlamaContext::kv_cache_*`
- Modules confirmed: model, context, context::params, context::kv_cache, context::session, llama_batch, sampling, token, gguf, llama_backend, openai
- `openai` module provides `OpenAIChatTemplateParams` for OpenAI-compatible chat templates

### Milestone 1.5 - llama.cpp Model Load And Metadata

Status: `[x]`

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

Completion notes (2026-06-16):

- Runtime struct holds `LlamaContext<'static>` followed by a boxed `LlamaModel`; the boxed model allocation is stable across `Runtime` moves and the context drops before the model
- Backend initialization via `OnceLock<LlamaBackend>` (thread-safe, once-only)
- Model params: n_gpu_layers, use_mmap, use_mlock via `LlamaModelParams::with_*`
- Context params: n_ctx, n_batch, n_ubatch, n_threads, n_threads_batch, embeddings, pooling_type via `LlamaContextParams::with_*`
- Pooling type mapping: "mean"->Mean, "cls"->Cls, "last"->Last, "none"->None, else Unspecified
- Metadata exposed: model_path, context_length (n_ctx), embedding_dimension (n_embd), vocab_size (n_vocab), n_ctx_train, embeddings_enabled
- model() and context() accessors for downstream use
- 4 runtime tests: backend_init, nonexistent_model (catch_unwind for panic), real_model (env-gated), embeddings (env-gated)
- Note: LlamaModel::load_from_file panics on nonexistent paths (upstream behavior)

### Milestone 1.6 - Tokenization And Detokenization

Status: `[x]`

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

Completion notes (2026-06-16):

- `Runtime::tokenize(&str, add_bos)` uses `LlamaModel::str_to_token` with `AddBos::Always` / `AddBos::Never`
- `Runtime::detokenize(&[i32])` and `Runtime::detokenize_piece(i32)` use `LlamaModel::token_to_piece`
- Generation reuses one UTF-8 decoder across sampled tokens so partial UTF-8 pieces can be buffered across token boundaries
- EOS/EOG helpers exposed as `Runtime::token_eos()` and `Runtime::is_eog(i32)`
- Env-gated GGUF tests cover tokenization, detokenization, single-piece detokenization, BOS behavior, and EOG detection

### Milestone 1.7 - Non-Streaming Text Generation

Status: `[x]`

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

Completion notes:

- `sampler.rs`: Added `build_llama_sampler()` that builds a `LlamaSampler::chain_simple` with top-k, top-p, min-p, temp, dist. Greedy when temperature <= 0.
- `runtime.rs`: Added `generate(&mut self, prompt, options) -> Result<GenerateOutput>` with prefill batch, decode loop, EOG detection, stop sequence support, timing metrics.
- `runtime.rs`: `generate` now preserves actual stop reasons (`Eos`, `MaxTokens`, `Cancelled`) instead of always reporting EOS.
- `lib.rs`: `GenerationPipeline` now wraps `Runtime` with `new(model_path, config)` and `generate(prompt, options)`.
- `server.rs`: Added `from_toml_str` method alongside `from_toml_path`.
- `server.rs`: Added non-model tests for chat completion finish-reason mapping and OpenAI-style response shape.
- `generate.rs`: CLI wired up with model resolution, config loading, chat mode, and metrics output.
- 28 tests pass (2 new env-gated generate tests: `test_generate_non_streaming`, `test_generate_with_stop_sequence`).

### Milestone 1.8 - Streaming Text Generation

Status: `[x]`

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

Completion notes:

- `runtime.rs`: Added `generate_with_callback(prompt, options, on_token: FnMut(&str) -> bool) -> Result<GenerationMetrics>`. Callback returns `true` to continue, `false` to cancel. Refactored `generate` to delegate to `generate_with_callback`.
- `runtime.rs`: Added `generate_with_callback_output` so server streaming can preserve final stop reason and usage metrics.
- `lib.rs`: Added `GenerationPipeline::generate_stream` that delegates to `runtime.generate_with_callback`.
- `generate.rs`: Added `--stream` flag. When set, prints tokens to stdout as they arrive, flushes after each piece, metrics to stderr.
- `server.rs`: Streaming chunks now send raw SSE event data through Axum, include usage metrics on the final chunk, map final finish reason from runtime stop reason, and cancel generation when the receiver is closed.
- Tests cover streaming chunk shape, final usage chunk, raw SSE event data, and closed-channel cancellation.

### Milestone 1.9 - Server Load, Models, Health, Auth, Rate Limit

Status: `[x]`

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

Completion notes:

- `server.rs`: Full rewrite with axum HTTP framework.
  - `GET /health` returns `{status, model_loaded}`.
  - `GET /v1/models` returns OpenAI-style model list from loaded runtime.
  - `POST /llm/load` loads a model path into the runtime.
  - `POST /v1/chat/completions` supports non-streaming and SSE streaming.
  - Auth: `x-api-key` header or `Authorization: Bearer` token checked against config.
  - Rate limiting: fixed-window RPM via `RateLimiter`.
  - Startup preload: loads model from `model_path` or `model` config key on startup.
  - `unsafe impl Send for Runtime` for single-threaded tokio runtime compatibility.
- Dependencies: added `axum`, `tokio`, `tokio-stream`.
- `llama-server.rs`: Wired up with `#[tokio::main]`, CLI overrides for bind/port/model/api_key/rpm.
- Tests cover auth helpers and handler failure, rate limiting, health without model, model list before load, env-gated model list after load, invalid load path, and prompt building.

### Milestone 1.10 - Embeddings

Status: `[x]`

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

Completion notes:

- `runtime.rs`: Added `embed(text) -> Result<Vec<f32>>`. Tokenizes input, clears KV cache, runs decode, gets embeddings via `context.embeddings_seq_ith(0)`, normalizes L2. Returns error if embeddings not enabled.
- `server.rs`: Added `POST /v1/embeddings` endpoint. Accepts string or array of strings. Returns OpenAI-style response with `object: "list"`, `data[*].embedding`, `usage.prompt_tokens`. Auth and rate limiting applied.
- `types.rs`: `EmbeddingData`, `EmbeddingOutput`, `EmbeddingUsage` already defined in milestone 1.2.
- Tests cover embedding request single/array input deserialization, token-array rejection, response JSON shape, non-embedding model unsupported error, and env-gated real embedding generation via `MLX_RS_TEST_EMBED_GGUF`.

### Milestone 1.11 - Chat Template Strategy

Status: `[x]`

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

Completion notes:

- `runtime.rs`: Added `apply_chat_template(messages) -> Result<String>`. Uses `model.chat_template(None)` to get native GGUF template, falls back to tested Llama2-style `[INST]` rendering. Uses `add_ass=true` to include assistant prefix.
- `server.rs`: Updated both streaming and non-streaming chat completions to use `runtime.apply_chat_template` instead of hardcoded `build_prompt_from_messages`. Moved fallback helper to `#[cfg(test)]` only.
- Tests cover simple user, system + user, assistant history, unsupported tool role fallback behavior, and env-gated native GGUF template rendering.

### Milestone 1.12 - Documentation And Cleanup

Status: `[x]`

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
  - example `llama-server`
  - Metal/GPU layer config
  - embeddings support
  - unsupported VLM/diffusion status

Acceptance:

- A new user can run one GGUF model from README instructions.
- No docs claim safetensors/MLX support in Phase 1.

Completion notes:

- `README.md`: Full rewrite. Documents GGUF requirement, all server endpoints, generate CLI flags, config.toml format, Metal/GPU layer config, embeddings, non-goals.
- `README.md`: Quick start now uses a `MODEL=/path/to/model.gguf` variable, documents accepted GGUF path forms, includes `/llm/load`, API key header behavior, and all llama.cpp config knobs from `config.toml`.
- `config.toml`: Updated with llama.cpp engine options (n_ctx, n_batch, n_gpu_layers, etc.). All commented out as examples.
- `generate.rs`: Replaced hardcoded `build_chat_prompt` with `pipeline.apply_chat_template` for native GGUF template support.
- `lib.rs`: Exposed `apply_chat_template` on `GenerationPipeline`.
- `llama-server` CLI help no longer describes the server as an MLX chat server.
- Removed stale MLX/parity ignore entries from `.gitignore`.
- User-facing docs now describe Phase 1 as GGUF/llama.cpp only and list safetensors/MLX as unsupported.

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
- Current llama-rs `crates/mlx-core/src/array.rs` only as cautionary prior art.

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
- Current llama-rs `crates/mlx-nn/src/var_builder.rs`

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
- Current llama-rs `crates/mlx-models/src/llama.rs`

Acceptance:

- Env-gated tiny Llama safetensors model produces logits.
- One-token generation works through backend trait.

### Milestone 2.7 - MLX Runner Subprocess

Status: `[ ]`

Objective:

Run the MLX backend in a subprocess like Ollama.

Actions:

- Add a runner binary mode:
  - `llama-rs-runner --mlx-engine --model <path> --port <port>`
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
- Current llama-rs `crates/llama-lm/src/chat_template.rs`

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
- Current llama-rs `crates/mlx-nn/src/kv_cache.rs`

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
- Current llama-rs `crates/mlx-models/src/*`

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

- Reproducing Python `llama_lm` parity.
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
