# Fix Plan: GPU Stream Crash & Gemma4 Correctness

Validated against actual code in `crates/mlx-backend/src/` and Ollama's implementation at `/Volumes/Data/Users/paul/development/src/github/ollama/`.

---

## Fix 1: GPU Stream Crash (BLOCKER) — `ops.rs`

**Root cause**: `init_streams()` (ops.rs:16-47) creates custom streams via `mlx_stream_new_device` and `default_stream()` (ops.rs:60-71) returns the custom GPU stream pointer. All ops create arrays tagged with this custom stream. But `eval()` (ops.rs:239-257) re-registers only the default streams via `mlx_default_*_stream_new()`, so arrays tagged with the custom stream crash with `There is no Stream(gpu, ...) in current thread`.

**Ollama's approach** (x/mlxrunner/mlx/stream.go, x/imagegen/mlx/mlx.go): Never uses `mlx_stream_new_device` for ops. Sets default device to GPU via `mlx_set_default_device(mlx_device_new_type(MLX_GPU, 0))`, then gets the default stream via `mlx_get_default_stream(device)`. All ops pass this default stream.

**Changes to `ops.rs`**:

1. `init_streams()`:
   - Remove `mlx_stream_new_device` calls entirely
   - Call `mlx_set_default_device(mlx_device_new_type(GPU, 0))` to set GPU as default
   - Call `mlx_get_default_stream(device)` to get the default GPU stream, store its ctx in `DEFAULT_STREAM_PTR`
   - Call `mlx_default_gpu_stream_new()` once to register its encoder
   - Remove `CPU_STREAM_PTR`, `GPU_STREAM_PTR` thread-locals
   - Free the device handle after setting default (Ollama: `C.mlx_device_free(dev)`)

2. `default_stream()`:
   - Return `DEFAULT_STREAM_PTR` (now populated from `mlx_get_default_stream`, not `mlx_stream_new_device`)

3. `cpu_stream()`:
   - Return `mlx_default_cpu_stream_new()` directly (matches Ollama's CPU stream for safetensors loading on macOS)
   - Cache in a thread-local since tensors.rs:37 references it for loaded arrays

4. `eval()`: Keep as-is (calling `mlx_default_cpu_stream_new()` + `mlx_default_gpu_stream_new()` before `mlx_eval` is correct — these re-register default stream encoders)

---

## Fix 2: `make_positions` broken for prefill — `mlx_backend.rs:119-121`

**Bug**: `make_positions()` always returns shape `[1]` with a single value `start`. For prefill with multiple tokens (L>1), positions should be an array `[start, start+1, ..., start+length-1]` with shape `[length]`. All tokens in a prefill chunk currently get the same position offset, producing incorrect RoPE encodings.

**Ollama's approach** (x/mlxrunner/pipeline.go): Passes `SeqOffsets` as a proper positions array with one offset per token.

**Changes to `mlx_backend.rs`**:

```rust
fn make_positions(start: usize, length: usize) -> Result<Array> {
    let positions: Vec<i32> = (start..start + length).map(|i| i as i32).collect();
    Array::from_data_i32(&positions, &[length])
}
```

Also update `prefill_chunked` call at line 144 to pass `chunk.len()` as length.

---

## Fix 3: Add `mlx_async_eval` binding — `ffi.rs` + `ops.rs`

**Ollama's approach** (x/mlxrunner/pipeline.go): Uses `mlx.AsyncEval()` after decode steps for non-blocking GPU dispatch. First decode token uses sync `Eval()` for timing.

**Changes to `ffi.rs`**:
- Add type: `pub type MlxAsyncEvalFn = unsafe extern "C" fn(MlxVectorArray) -> c_int;`
- Add field to `MlxSymbols`: `pub mlx_async_eval: MlxAsyncEvalFn,`
- Add symbol load: `mlx_async_eval: load_sym!(lib, b"mlx_async_eval\0", MlxAsyncEvalFn),`

**Changes to `ops.rs`**:
- Add `pub fn async_eval(arrays: &[&Array]) -> anyhow::Result<()>` (same as `eval` but calls `mlx_async_eval`)

**Changes to `mlx_backend.rs`**:
- In decode loops (lines 245, 310, 373): Use sync `eval()` for first decode step, `async_eval()` for subsequent steps (matching Ollama's pattern)

---

## Fix 4: Add `mlx_clear_cache` after prefill — `mlx_backend.rs`

**Current state**: `mlx_backend.rs:262` calls `crate::memory::clear_cache()` after generation. `ffi.rs` already has `mlx_clear_cache` loaded. Need to verify `memory::clear_cache` calls it.

**Ollama's approach** (x/mlxrunner/pipeline.go): Calls `mlx.ClearCache()` after each prefill chunk.

**Changes to `mlx_backend.rs`**:
- In `prefill_chunked` (line 149): Add `crate::memory::clear_cache()` after each chunk's eval
- Verify `memory::clear_cache()` actually calls `mlx_clear_cache` (check `memory.rs`)

---

## Fix 5: `argmax` optimization — `llama.rs:515-526`

**Current state**: `argmax()` calls `logits.data_f32()` which materializes the entire logits array into CPU memory. For large vocab (256K), this copies ~1MB per step.

**Better approach**: Use `ops::argmax_op()` (already defined at ops.rs:577-585) which runs on GPU and returns a scalar index. Then call `.data_i32()` on the single result.

**Changes to `llama.rs`**:
```rust
pub fn argmax(logits: &Array) -> anyhow::Result<i32> {
    let idx_arr = ops::argmax_op(logits)?;
    idx_arr.eval()?;
    idx_arr.item_i32()
}
```

Note: Requires adding `item_i32()` method to `Array` if not present (check `array.rs`).

---

## Fix 6: Debug `eprintln!` cleanup

Remove all debug logging after fixes confirmed working:

| File | Lines | Content |
|------|-------|---------|
| `ops.rs` | 389-390 | `[rope]` debug in `fast_rope_with_freqs` |
| `ops.rs` | 427-428 | `[rope_dynamic]` debug in `fast_rope_dynamic` |
| `gemma4.rs` | 343-344 | `[MLX_ATTN]` debug in `Gemma4Attention::forward` |
| `gemma4.rs` | 409-410 | `[MLX_SDPA]` debug in `Gemma4Attention::forward` |
| `gemma4.rs` | 412 | `[MLX_SDPA] out_dims` debug |
| `gemma4.rs` | 570 | `[MLX_FWD]` debug in `Gemma4Model::forward` |
| `gemma4.rs` | 574 | `[MLX_FWD] after_embed` debug |
| `gemma4.rs` | 610-611 | `[MLX_LAYER]` debug in layer loop |
| `llama.rs` | 306-307 | `[MLX_KVCACHE]` debug in `KvCache::update` |
| `mlx_backend.rs` | 145 | `[MLX_PREFILL]` debug |
| `mlx_backend.rs` | 149 | `[MLX_PREFILL] chunk done` debug |
| `mlx_backend.rs` | 242 | `[MLX_DECODE]` debug |

---

## Already Fixed (Verify)

| Issue | Status | Location |
|-------|--------|----------|
| RotatingKvCache for sliding window | Done | `gemma4.rs:658-668` — `KvCache::new_rotating(sw)` for sliding layers |
| SDPA mode for decode | Done | `gemma4.rs:408` — `let sdpa_mode = if l > 1 { "causal" } else { "" }` |
| eval() after forward | Done | `mlx_backend.rs:147,245,310,373` |
| Prefill chunking | Done | `mlx_backend.rs:128-152` with `PREFILL_CHUNK_SIZE=2048` |
| EOS detection | Done | `mlx_backend.rs:108-116` with multiple EOG tokens |
| KV sharing with donor map | Done | `gemma4.rs:588-626` |
| Q/K norms | Done | `gemma4.rs:350,388` |
| V normalization | Done | `gemma4.rs:400` — `rms_norm_weightless` |
| PLE (Per-Layer Embeddings) | Done | `gemma4.rs:443-496,534-537` |
| Logit softcapping | Done | `gemma4.rs:633-640` |
| Native safetensors loading | Done | `tensors.rs` — uses `mlx_load_safetensors` |
| `fast_rope_dynamic` with positions array | Done | `gemma4.rs:359-367` |
| Partial rotary factor / custom freqs | Done | `gemma4.rs:149-166` |

---

## Execution Order

1. **Fix 1** (stream crash) + **Fix 2** (positions) — correctness bugs, must fix together
2. **Build & test**: `cargo run --release --bin generate -- --model mlx-community/gemma-4-e2b-it-4bit --chat --prompt "Hello"`
3. **Fix 3** (async_eval) + **Fix 4** (clear_cache timing) — performance improvements
4. **Fix 5** (argmax optimization) — optional polish
5. **Fix 6** (debug cleanup) — after all fixes confirmed working
