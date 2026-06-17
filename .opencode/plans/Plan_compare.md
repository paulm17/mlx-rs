# Plan: Empirical Debug Comparison — Ollama MLX vs mlx-rs

## Goal

Stop guessing. Add concrete debug logging to Ollama's MLX path, run `go run . run gemma4-mlx "Hello"`, capture the trace, then run the same model in mlx-rs with equivalent logging, and compare the two traces side-by-side to find exactly where mlx-rs diverges.

## Model Under Test

```
/Volumes/Data/Users/paul/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/2c3e507453b4f218d05fe3cc97bea5c5a654257e
```

Ollama model name: `gemma4-mlx`

---

## Debug Loop Per Item

For each of the 7 items, the loop is:

1. **Instrument Ollama** — Add `slog.Debug` calls to the relevant Go functions
2. **Run Ollama** — `OLLAMA_DEBUG=1 go run . run gemma4-mlx "Hello" 2>&1 | tee /tmp/ollama_trace.log`
3. **Instrument mlx-rs** — Add `eprintln!` tracing to equivalent Rust functions
4. **Run mlx-rs** — `cargo run --release --bin generate -- --model <path> --prompt "Hello" 2>&1 | tee /tmp/mlxrs_trace.log`
5. **Compare** — Diff the two traces, identify divergence point
6. **Fix** — Change mlx-rs to match Ollama's behavior at the divergence point
7. **Verify** — Re-run mlx-rs, confirm output is sane

---

## Item 1: RoPE Position Handling

**Hypothesis**: mlx-rs calls `mlx_fast_rope` with `offset=0` (scalar). Ollama calls `mlx_fast_rope_dynamic` with a positions array `[0, 1, 2, ...]`. Every token in mlx-rs gets RoPE at position 0.

### Ollama instrumentation

File: `x/models/gemma4/gemma4.go` in `Attention.Forward` (~line 1288)

```go
slog.Debug("rope",
    "layer", "q",
    "is_sliding", isSliding,
    "rope_dims", ropeDims,
    "rope_base", ropeBase,
    "positions_shape", positions.Dims(),
    "positions_vals", positions.Ints(),
    "rope_freqs", ropeFreqs != nil,
    "q_shape_before", q.Dims(),
)
q = mlx.RoPEWithFreqs(q, ropeDims, false, ropeBase, 1.0, positions, ropeFreqs)
slog.Debug("rope", "layer", "q", "q_shape_after", q.Dims())
```

Same for K (~line 1316).

File: `x/mlxrunner/mlx/ops_extra.go` in `RoPEWithFreqs` (~line 421)

```go
slog.Debug("mlx_fast_rope_dynamic",
    "x_shape", x.Dims(),
    "dims", dims,
    "traditional", traditional,
    "base", base,
    "scale", scale,
    "offsets_shape", offsets.Dims(),
    "has_freqs", freqs != nil,
)
```

### mlx-rs instrumentation

File: `crates/mlx-backend/src/ops.rs` in `fast_rope_with_freqs` (~line 346)

```rust
eprintln!("[rope] x.shape={:?} dims={} traditional={} base={:?} scale={} offset={} has_freqs={}",
    a.shape(), dims, traditional, base, scale, offset, freqs.is_some());
```

File: `crates/mlx-backend/src/gemma4.rs` in `Gemma4Attention::forward` (~line 356)

```rust
eprintln!("[gemma4_attn] rope q: is_sliding={} dims={} base={:?} freqs={:?}",
    self.is_sliding, self.rope_dims, rope_base, self.full_rope_freqs.is_some());
```

### Expected divergence

- Ollama: `offsets_shape=[1, N]` where N = sequence length, values = `[0, 1, 2, ...]`
- mlx-rs: `offset=0` (scalar, always 0)

### Fix

Add `mlx_fast_rope_dynamic` to mlx-rs FFI bindings. Build a positions array `[prefix_len, prefix_len+1, ...]` and pass it instead of scalar `0`.

---

## Item 2: KV Cache — Rotating vs Simple Concat

**Hypothesis**: Sliding-window layers in Ollama use `RotatingKVCache` with ring-buffer eviction. mlx-rs uses simple `KvCache` that concatenates forever.

### Ollama instrumentation

File: `x/mlxrunner/cache/kvcache.go` in `Update` (~line 47)

```go
slog.Debug("kvcache_update", "type", "simple", "offset", c.offset, "k_shape", keys.Dims(), "v_shape", values.Dims())
```

File: `x/mlxrunner/cache/rotating.go` in `Update` (~line 36)

```go
slog.Debug("kvcache_update", "type", "rotating", "offset", c.offset, "max_size", c.maxSize, "k_shape", keys.Dims(), "v_shape", values.Dims())
```

File: `x/models/gemma4/gemma4.go` in `NewCaches` (~line 1154)

```go
slog.Debug("new_cache", "layer", i, "is_sliding", layer.IsSliding, "type", cacheType)
```

### mlx-rs instrumentation

File: `crates/mlx-backend/src/llama.rs` in `KvCache::update` (~line 275)

```rust
eprintln!("[kvcache_update] offset={} k_cache={:?} new_k={:?} new_v={:?}",
    self.len(), self.k_cache.as_ref().map(|k| k.shape()), k.shape(), v.shape());
```

### Expected divergence

- Ollama: sliding layers log `type=rotating` with bounded `max_size` (e.g. 512)
- mlx-rs: all layers log unbounded growth

### Fix

Implement `RotatingKvCache` in mlx-rs for sliding-window layers. Create caches based on `is_layer_sliding()`.

---

## Item 3: SDPA Masking

**Hypothesis**: Ollama builds explicit masks (causal + sliding window + padding). mlx-rs passes `"causal"` mode only with no sliding window mask.

### Ollama instrumentation

File: `x/mlxrunner/nn/sdpa.go` in `ScaledDotProductAttention` (~line 54)

```go
slog.Debug("sdpa",
    "q_shape", q.Dims(),
    "k_shape", k.Dims(),
    "v_shape", v.Dims(),
    "scale", scale,
    "mode", d.mode,
    "has_mask", d.arr != nil,
    "has_applier", d.applier != nil,
)
```

File: `x/models/gemma4/gemma4.go` in `Attention.Forward` (~line 1369)

```go
slog.Debug("sdpa_call",
    "head_dim", headDim,
    "scale", scale,
    "is_sliding", isSliding,
    "has_history", kv.history != nil,
    "sliding_window", cfg.SlidingWindow,
)
```

### mlx-rs instrumentation

File: `crates/mlx-backend/src/ops.rs` in `fast_sdpa` (~line 382)

```rust
eprintln!("[sdpa] q={:?} k={:?} v={:?} scale={} mode={} mask={}",
    queries.shape(), keys.shape(), values.shape(), scale, mask_mode, mask.is_some());
```

File: `crates/mlx-backend/src/gemma4.rs` in `Gemma4Attention::forward` (~line 404)

```rust
eprintln!("[gemma4_sdpa] is_sliding={} scale={} head_dim={}",
    self.is_sliding, self.scale, self.head_dim);
```

### Expected divergence

- Ollama: `mode=array` or `mode=causal` with `has_mask=true` for sliding layers, `has_applier=true` for rotating cache
- mlx-rs: always `mode=causal` with `mask=false`

### Fix

For sliding-window layers, build a sliding window mask array and pass it to `fast_sdpa` with `mode="array"`.

---

## Item 4: EOS Detection

**Hypothesis**: mlx-rs only checks `<|end_of_text|>` and `</s>`. Gemma 4 uses `<|end_of_turn|>` as a stop token.

### Ollama instrumentation

File: `x/tokenizer/tokenizer.go` — find `IsEOS` function

```go
slog.Debug("is_eos", "token_id", tokenID, "token_text", tok.Decode([]int32{tokenID}), "is_eos", result)
```

### mlx-rs instrumentation

File: `crates/mlx-backend/src/mlx_backend.rs` in `is_eog` (~line 128)

```rust
fn is_eog(&self, token_id: i32) -> bool {
    let result = token_id == self.token_eos();
    let piece = self.detokenize_piece(token_id).unwrap_or_default();
    eprintln!("[is_eog] token_id={} piece={:?} is_eog={}", token_id, piece, result);
    result
}
```

### Expected divergence

- Ollama: recognizes `<|end_of_turn|>` (token_id varies) as EOS
- mlx-rs: does not recognize it, keeps generating

### Fix

Check for `<|end_of_turn|>`, `<|eot_id|>`, and other model-specific stop tokens.

---

## Item 5: Thread-Local Stream Initialization

**Hypothesis**: `init_streams()` is called on the main thread during `MlxBackend::load()`. When `generate_stream` runs via `spawn_blocking`, the blocking thread has no GPU streams initialized.

### mlx-rs instrumentation

File: `crates/mlx-backend/src/ops.rs` in `init_streams` (~line 14) and `default_stream` (~line 33)

```rust
pub fn init_streams() {
    STREAM_INIT.with(|init| {
        eprintln!("[init_streams] thread={:?} already_init={}", std::thread::current().id(), init.get());
        // ... existing code
        eprintln!("[init_streams] thread={:?} DONE gpu_stream_ptr={:?}", std::thread::current().id(), gpu_stream.ctx);
    });
}

fn default_stream() -> *const MlxStream {
    let already = STREAM_INIT.with(|init| init.get());
    if !already {
        eprintln!("[default_stream] LAZY INIT on thread={:?}", std::thread::current().id());
    }
    // ... existing code
}
```

### Test

Run a streaming request and check if `[init_streams]` and `[default_stream]` show different thread IDs.

### Fix

Call `init_streams()` at the start of every `generate` / `generate_stream` / `generate_stream_output` call, or use a `OnceLock` per thread.

---

## Item 6: Prefill Chunking

**Hypothesis**: Ollama chunks prefill at 2048 tokens with eval/sweep between chunks. mlx-rs sends all tokens at once.

### Ollama instrumentation

File: `x/mlxrunner/pipeline.go` in `TextGenerationPipeline` (~line 113)

Already has: `slog.Info("Prompt processing progress", "processed", processed, "total", total)`

Add:

```go
slog.Debug("prefill_chunk", "chunk_size", n, "position", position, "memory", mlx.Memory{})
```

### mlx-rs instrumentation

File: `crates/mlx-backend/src/mlx_backend.rs` in `generate` (~line 168)

```rust
eprintln!("[prefill] tokens={} prefix_len={} total_positions={}", prefill_tokens.len(), prefix_len, prefill_tokens.len());
```

### Expected divergence

- Ollama: multiple `prefill_chunk` logs, each ≤ 2048 tokens
- mlx-rs: single `[prefill]` log with all tokens at once

### Fix

Chunk prefill into 2048-token batches with eval/sweep between chunks.

---

## Item 7: Eval/Sweep Discipline

**Hypothesis**: mlx-rs never calls `eval()` on KV cache arrays after forward passes. Ollama calls `materializeCaches()` (which does `mlx.Eval(state...)`) after each prefill chunk and during decode.

### Ollama instrumentation

File: `x/mlxrunner/pipeline.go` in `TextGenerationPipeline`

```go
// After materializeCaches():
slog.Debug("materialize_caches", "arrays", len(state), "memory", mlx.Memory{})
```

### mlx-rs instrumentation

File: `crates/mlx-backend/src/ops.rs` in `eval` (~line 212)

```rust
pub fn eval(arrays: &[&Array]) -> anyhow::Result<()> {
    eprintln!("[eval] arrays={}", arrays.len());
    // ... existing code
}
```

File: `crates/mlx-backend/src/llama.rs` in `argmax` (~line 484)

```rust
pub fn argmax(logits: &Array) -> anyhow::Result<i32> {
    eprintln!("[argmax] logits.shape={:?}", logits.shape());
    // ... existing code
}
```

### Expected divergence

- Ollama: `eval` called after every prefill chunk AND during decode
- mlx-rs: `eval` only called inside `argmax`

### Fix

Call `ops::eval()` on cache state arrays after each forward pass.

---

## Execution Order

1. **Item 1 (RoPE)** — Most likely root cause of garbage output. Fix first.
2. **Item 5 (Streams)** — Could cause silent failures. Verify early.
3. **Item 7 (Eval/Sweep)** — Could cause unmaterialized arrays. Verify early.
4. **Item 3 (SDPA Masking)** — Affects correctness of attention.
5. **Item 2 (Rotating Cache)** — Affects sliding window layers.
6. **Item 6 (Prefill Chunking)** — Memory/performance issue.
7. **Item 4 (EOS)** — Output quality issue (runaway generation).

## Files to Create/Modify

### Ollama (debug logging only — do NOT commit)
- `x/mlxrunner/mlx/ops_extra.go` — Add slog.Debug to RoPEWithFreqs
- `x/mlxrunner/nn/sdpa.go` — Add slog.Debug to ScaledDotProductAttention
- `x/models/gemma4/gemma4.go` — Add slog.Debug to Forward, Attention.Forward, NewCaches
- `x/mlxrunner/cache/kvcache.go` — Add slog.Debug to Update
- `x/mlxrunner/cache/rotating.go` — Add slog.Debug to Update (already has trace)
- `x/mlxrunner/pipeline.go` — Add slog.Debug to prefill loop

### mlx-rs (tracing + fixes)
- `crates/mlx-backend/src/ffi.rs` — Add `mlx_fast_rope_dynamic` binding
- `crates/mlx-backend/src/ops.rs` — Add `fast_rope_dynamic` wrapper, add tracing to all ops
- `crates/mlx-backend/src/gemma4.rs` — Fix RoPE to use positions array, add tracing
- `crates/mlx-backend/src/llama.rs` — Add `RotatingKvCache`, add tracing
- `crates/mlx-backend/src/mlx_backend.rs` — Fix EOS, fix streams, fix prefill chunking
