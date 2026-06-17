# mlx-rs vs Ollama MLX: Comparison Findings

## Methodology

Added identical debug logging to both codebases at the same execution points:
- **Ollama**: `x/internal/debug/dbg.go` writing to `/tmp/ollama_debug.log` (18901 lines)
- **mlx-rs**: `eprintln!` writing to stderr, captured to `/tmp/mlx_rs_trace.log` (421 lines)

Debug points instrumented in both:
1. `Model.Forward` — B, L, positions_dims
2. `Attention.Forward` — headDim, ropeDims, ropeBase, is_sliding, scale, positions_dims, has_donor
3. `KvCache.Update` — k/v dims, offset_before, offset_after
4. `SDPA` — q/k/v dims, scale, mask_mode
5. Prefill/decode step — position, token

## Model Under Test

- **Model**: `mlx-community/gemma-4-e2b-it-4bit`
- **Path**: `/Volumes/Data/Users/paul/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/2c3e507453b4f218d05fe3cc97bea5c5a654257e`
- **Architecture**: Gemma4ForCausalLM, 35 layers, hidden_dim=1536
- **Ollama command**: `go run . run gemma4-mlx "Hello"`
- **mlx-rs command**: `cargo run --release --bin generate -- --model mlx-community/gemma-4-e2b-it-4bit --chat --prompt "Hello"`

## Architecture Details (from Ollama trace)

### Layer Types
- **Sliding layers** (is_sliding=true): layers 0-3, 5-8, 10-13, 15-18, 20-23, 25-28, 30-33
  - headDim=256, ropeDims=256, ropeBase=10000.0
  - Cache: RotatingKVCache(maxSize=512)
  - 1 KV head, 8 Q heads
- **Full layers** (is_sliding=false): layers 4, 9, 14, 19, 24, 29, 34
  - headDim=512, ropeDims=512, ropeBase=1000000.0
  - Cache: KVCache (unbounded)
  - 1 KV head, 8 Q heads

### KV Sharing
- Layers 15-34 share KV from donor layers (has_donor=true, cache=<nil>/false)
- Donor mapping: each shared layer reuses the last non-shared layer of the same type
- Example: layer 15 (sliding) → donor layer 13 (sliding)

### Scale
- All layers use scale=1.0 (Q/K norms handle magnitude)

## Side-by-Side Comparison

### Prefill Phase

| Point | Ollama | mlx-rs | Match? |
|-------|--------|--------|--------|
| tokens processed | 16 (chunked from 17 total) | 77 (all at once) | ❌ no chunking |
| B | 1 | 1 | ✅ |
| L | 16 | 77 | ✅ (different chunk size) |
| positions_dims | [1] | [1] | ✅ |
| sliding headDim | 256 | 256 | ✅ |
| sliding ropeDims | 256 | 256 | ✅ |
| sliding ropeBase | 10000.0 | 10000.0 | ✅ |
| full headDim | 512 | 512 | ✅ |
| full ropeDims | 512 | 512 | ✅ |
| full ropeBase | 1000000.0 | 1000000.0 | ✅ |
| scale | 1.0 | 1 | ✅ |
| SDPA mode | "causal" | "causal" | ✅ |
| KV cache offset | 0→16 | 0→77 | ✅ (proportional to chunk size) |
| RotatingKVCache offset | 0→16, idx 0→16 | N/A (no RotatingKvCache) | ❌ |

### Decode Phase (L=1)

| Point | Ollama | mlx-rs | Match? |
|-------|--------|--------|--------|
| B | 1 | 1 | ✅ |
| L | 1 | 1 | ✅ |
| positions_dims | [1] | [1] | ✅ |
| sliding headDim | 256 | 256 | ✅ |
| full headDim | 512 | 512 | ✅ |
| SDPA mode | "" (empty = no mask) | "causal" | **❌ BUG** |
| KV cache offset | 16→17 | 77→78 | ✅ (proportional) |
| has_donor (layer 15+) | true | true | ✅ |
| has_cache (donor layers) | false | false | ✅ |
| K dims (sliding) | [1, 1, 17, 256] | [1, 1, 78, 256] | ✅ (different prompt len) |
| K dims (full) | [1, 1, 17, 512] | [1, 1, 78, 512] | ✅ |

### SDPA Masking Behavior (from Ollama trace)

| Phase | Sliding Layers | Full Layers |
|-------|---------------|-------------|
| Prefill (L>1) | mode="causal" | mode="causal" |
| Decode (L=1) | mode="" (no mask) | mode="" (no mask) |

Ollama's `rotatingApplier.ApplyMask` strips causal masking at L=1:
- If `logical.IsCausal()` and L==1 → return empty mask → mode=""
- mlx-rs always passes "causal" regardless of L

## Divergences Found

### 1. SDPA Mode During Decode (Bug)
- **Ollama**: `mode=""` (no mask) for ALL layers when L=1
- **mlx-rs**: `mode="causal"` always
- **Impact**: Incorrect attention masking during decode. At L=1 causal is equivalent to no mask, so this may not affect output quality, but it diverges from Ollama's behavior.

### 2. GPU Stream Crash (Blocker)
- **Symptom**: `MLX error: There is no Stream(gpu, 60457072) in current thread` at `transforms.cpp:73`
- **When**: After all 35 layers complete (prefill + 1 decode step), during `argmax` → `mlx_eval`
- **Root cause**: The forward pass builds a lazy computation graph. `argmax` calls `mlx_eval` which needs to execute the graph on GPU. The GPU stream's command encoder is not registered in the thread-local encoder map for the current thread.
- **Ollama avoids this because**:
  1. Calls `mlx.Sweep()` + `mlx.AsyncEval()` after each forward pass (materializes KV cache arrays)
  2. Uses `mlx_load_safetensors` for tensor loading (registers CPU stream in thread-local)
  3. Runs on a dedicated MLX thread (`mlxthread.Thread`)
- **mlx-rs state**:
  - `init_streams()` calls `mlx_default_cpu_stream_new()` + `mlx_default_gpu_stream_new()` which should register streams
  - CPU stream crash was fixed by adding `mlx_default_cpu_stream_new()`
  - GPU stream crash persists despite `mlx_default_gpu_stream_new()`
  - No `eval()` call after forward pass to materialize arrays

### 3. No RotatingKvCache for Sliding Window Layers
- **Ollama**: `RotatingKVCache(maxSize=512)` for sliding layers, `KVCache` for full layers
- **mlx-rs**: Simple `KvCache` (concat-based, unbounded) for ALL layers
- **Impact**: Sliding window layers cache grows unbounded, wasting memory. No ring-buffer eviction.

### 4. No Prefill Chunking
- **Ollama**: Processes prompt in chunks of 2048 tokens, calling `mlx.Sweep()` + `materializeCaches()` between chunks
- **mlx-rs**: Sends all tokens at once in a single forward pass
- **Impact**: For long prompts (>2048 tokens), mlx-rs may OOM or be slower. For short prompts, no difference.

### 5. No eval() After Forward Passes
- **Ollama**: Calls `mlx.Sweep()` + `mlx.AsyncEval(sample.Arrays()...)` after each decode step
- **mlx-rs**: Never calls eval() explicitly — relies on argmax to trigger eval, which crashes
- **Impact**: KV cache arrays are never materialized, causing the GPU stream crash

### 6. EOS Detection Too Narrow
- **Ollama**: Checks multiple EOG tokens including `<|end_of_turn|>`, `<|eot_id|>`, etc.
- **mlx-rs**: Only checks `<|end_of_text|>` and `</s>`
- **Impact**: Model may generate runaway output until max_tokens for models that use `<|end_of_turn|>` as stop token

## Files Modified

### mlx-rs
| File | Changes |
|------|---------|
| `crates/mlx-backend/src/ffi.rs` | Added `MlxFastRopeDynamicFn`, `MlxLoadSafetensorsFn`, map types, all loaded in `MlxSymbols` |
| `crates/mlx-backend/src/ops.rs` | Added `fast_rope_dynamic()`, fixed `init_streams()` with CPU+GPU stream init, debug logging in rope functions |
| `crates/mlx-backend/src/tensors.rs` | Rewritten to use native `mlx_load_safetensors` |
| `crates/mlx-backend/src/manifest.rs` | Updated for `Result` returns from tensors.rs |
| `crates/mlx-backend/src/gemma4.rs` | Uses `fast_rope_dynamic` with positions array, debug logging in attention/model forward |
| `crates/mlx-backend/src/llama.rs` | Uses `fast_rope_dynamic` with positions array, debug logging in KvCache.update |
| `crates/mlx-backend/src/gemma3.rs` | Uses `fast_rope_dynamic` with positions array |
| `crates/mlx-backend/src/qwen3.rs` | Uses `fast_rope_dynamic` with positions array |
| `crates/mlx-backend/src/mlx_backend.rs` | `make_positions` returns shape [1], debug logging in generate loop |

### Ollama (debug logging only — to be reverted)
| File | Changes |
|------|---------|
| `x/internal/debug/dbg.go` | New file: file-based debug logger to `/tmp/ollama_debug.log` |
| `x/models/gemma4/gemma4.go` | debug.Dbg in Model.Forward, DecoderLayer.Forward, Attention.Forward |
| `x/mlxrunner/pipeline.go` | debug.Dbg in prefill loop, decode step |
| `x/mlxrunner/cache/kvcache.go` | debug.Dbg in KVCache.Update |
| `x/mlxrunner/cache/rotating.go` | debug.Dbg in RotatingKVCache.Update |
| `x/models/nn/sdpa.go` | debug.Dbg in ScaledDotProductAttention |

## Fix Priority

1. **GPU stream crash** — Blocker, nothing works without this. Need to either:
   - Call `eval()` on logits before argmax to materialize the graph
   - Or fix stream registration for the eval thread
2. **RotatingKvCache** — Required for correct sliding window behavior
3. **SDPA mode switching** — Use "" for L=1 decode, "causal" for prefill
4. **EOS detection** — Add `<|end_of_turn|>` and other EOG tokens
5. **eval() after forward** — Materialize KV cache arrays (may fix #1)
6. **Prefill chunking** — Only needed for long prompts

## Debug Trace Files

- Ollama: `/tmp/ollama_debug.log` (18901 lines)
- mlx-rs: `/tmp/mlx_rs_trace.log` (421 lines, crashes after 1 decode step)
