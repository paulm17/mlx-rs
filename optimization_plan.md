# Python MLX vs Rust MLX Optimization Plan

## Goal

Bring the Rust MLX inference stack close enough to the Python `mlx_lm` stack that the two are within touching distance on the current enclosed benchmark suite, while keeping correctness and maintainability intact.

"Within touching distance" in this document means:

- Dense models: Rust reaches roughly 85% to 95% of Python decode throughput on aligned benchmarks.
- MoE models: Rust reaches roughly 75% to 90% of Python decode throughput on aligned benchmarks.
- TTFT: Rust stays within about 10% to 25% of Python for dense models, and avoids pathological outliers like the current LFM2 result.

Current benchmark snapshot from `2026-03-05`:

- `Llama-3.2-1B-Instruct-4bit`: Rust is about 64% of Python throughput.
- `Qwen3-1.7B-MLX-4bit`: Rust is about 79% of Python throughput.
- `Qwen1.5-MoE-A2.7B-4bit`: Rust is about 59% of Python throughput.
- `LFM2-24B-A2B-MLX-4bit`: Rust is about 38% of Python throughput and has a much worse TTFT.

The largest conclusions from the code inspection are:

- The benchmark is not fully aligned today.
- The Rust decode loop is more synchronous than the Python path.
- Rust sampling leaves the device for non-greedy decoding.
- Rust MoE routing is heavily host-driven and likely dominates the MoE gap.

## Summary of the Main Gaps

### 1. Benchmark mismatch

The current harness benchmarks Python through streaming SSE but benchmarks Rust through the non-streaming path.

Relevant files:

- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/scripts/benchmark_python_vs_rust_servers.py`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-lm/src/server.rs`

Why it matters:

- TTFT is measured differently in practice.
- Transport and buffering differences are mixed into the runtime comparison.
- Python often stops before 256 tokens while Rust was measured through full non-stream completion.

### 2. Decode loop structure

Python `mlx_lm` uses a dedicated generation stream and overlaps work with `mx.async_eval`. Rust currently forces more synchronous evaluation.

Relevant files:

- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/.venv/lib/python3.14/site-packages/mlx_lm/generate.py`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-lm/src/generate.rs`

### 3. Sampling path

Rust sampling pulls logits to the CPU for non-greedy paths. Python keeps sampling on-device and compiles top-p and related transforms.

Relevant files:

- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-lm/src/sampler.rs`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/.venv/lib/python3.14/site-packages/mlx_lm/sample_utils.py`

### 4. KV cache implementation

Rust KV cache updates are functional but likely not as cheap as the Python stack's cache path.

Relevant files:

- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-nn/src/kv_cache.rs`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-models/src/llama.rs`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-models/src/qwen3.rs`

### 5. MoE routing and expert execution

Rust MoE implementations move routing probabilities to the host and perform substantial control flow and accumulation in Rust loops.

Relevant files:

- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-models/src/qwen3_moe.rs`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-models/src/lfm2_moe.rs`

## Phase 0: Align the Benchmark

### Objective

Make the benchmark compare equivalent serving paths and produce metrics that distinguish runtime gaps from harness artifacts.

### Files to Change

- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/scripts/benchmark_python_vs_rust_servers.py`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/scripts/run_enclosed_benchmarks.sh`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/tests/benchmark_cases.json`
- Optional: `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-lm/src/server.rs`

### Planned Changes

1. Add a benchmark mode switch so both Python and Rust can be measured via:
   - streaming path
   - non-streaming path
   - an explicit "aligned" default that uses the same mode for both

2. Change the default enclosed benchmark to use streaming for both backends.

3. Capture and report:
   - TTFT
   - total latency
   - completion tokens
   - decode-only throughput excluding TTFT
   - stop reason (`stop`, `length`, local repetition guard, error)

4. Ensure token counts are computed consistently:
   - Prefer server-reported completion token counts when available.
   - Fall back to tokenizer-based counting only when server debug metadata is missing.

5. Add a fixed prompt shape matrix:
   - short prompt, greedy decode
   - short prompt, non-greedy decode
   - long prompt, greedy decode
   - long prompt, non-greedy decode

6. Add a benchmark output section that reports:
   - raw numbers
   - Rust/Python ratio
   - decode-only ratio
   - TTFT ratio

7. Make sure the Rust benchmark path uses the existing Rust streaming API instead of the non-streaming completion path.

### Expected Gain

This phase does not improve real model performance. It improves measurement quality.

Expected effect on reported numbers:

- Dense-model gap may shrink modestly once transport mode is aligned.
- TTFT comparisons will become much more trustworthy.
- Some current MoE and LFM2 gap will remain even after alignment.

### Exit Criteria

- One command produces aligned Python vs Rust benchmark results.
- Streaming and non-streaming can both be benchmarked for both backends.
- Reports include decode-only throughput and stop reasons.

### Risks

- None of the core runtime issues are solved here.
- Some previously reported wins or losses may disappear after measurement cleanup.

## Phase 1: Instrument the Rust Decode Path

### Objective

Identify where time is going inside the Rust generation loop before changing execution structure.

### Files to Change

- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-lm/src/generate.rs`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-lm/src/server.rs`
- Possibly `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-core/src/stream.rs`
- Possibly `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-core/src/metal.rs`

### Planned Changes

1. Add optional generation profiling behind an environment variable.

2. Break per-request latency into:
   - tokenization
   - prompt prefill forward
   - first-token sample
   - per-token forward
   - per-token sample
   - per-token decode to text
   - stream write time

3. Add debug counters for:
   - number of `eval()` calls
   - number of CPU data extractions
   - KV cache reallocations
   - repetition-guard triggers

4. Make profiling metadata available in server debug responses when enabled.

### Expected Gain

- Direct throughput improvement: 0% to 3%
- Indirect gain: high

This phase mostly reduces uncertainty and prevents wasted optimization work.

### Exit Criteria

- A profile from one request clearly shows where time is spent.
- We can answer whether dense-model losses are dominated by forward, sampling, or synchronization.

## Phase 2: Remove Avoidable Synchronization in Rust Generation

### Objective

Restructure the Rust decode loop so it resembles Python's execution model more closely.

### Files to Change

- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-lm/src/generate.rs`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-core/src/array.rs`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-core/src/stream.rs`
- Possibly `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-core/src/ops.rs`

### Planned Changes

1. Audit every `eval()` in the generation loop and remove those that are only serving as eager barriers.

2. Introduce a dedicated generation stream on the Rust side, matching the Python pattern more closely.

3. Overlap token `n+1` compute with token `n` delivery where possible.

4. Replace synchronous result handling with a minimal synchronization boundary:
   - only block when the sampled token value is required on the host
   - avoid forcing evaluation of intermediate reshapes or normalized logits

5. Review `normalized_last_token_logits` to ensure it does not force avoidable materialization.

### Expected Gain

- Dense models: about 8% to 18%
- MoE models: about 5% to 12%
- TTFT: about 0% to 10% depending on how many sync points occur before first token

This is the most likely place to recover a large portion of the dense-model gap.

### Exit Criteria

- Dense-model decode-only throughput improves materially on aligned benchmarks.
- Profiling shows fewer synchronization points and fewer blocking evaluations.

### Risks

- Async behavior can make latent correctness bugs harder to diagnose.
- If MLX C bindings expose fewer async hooks than Python, achievable overlap may be limited without API work in `mlx-sys` or upstream MLX.

## Phase 3: Move Sampling and Logits Processing On-Device

### Objective

Eliminate CPU round-trips for non-greedy sampling and repetition processing.

### Files to Change

- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-lm/src/sampler.rs`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-core/src/ops.rs`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-core/src/array.rs`
- Possibly `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-sys/src/lib.rs`

### Planned Changes

1. Preserve current greedy path, which already uses on-device `argmax`.

2. Replace `to_vec_f32()` sampling for temperature/top-p with device-side MLX ops:
   - temperature scaling
   - log-softmax or logprob normalization
   - top-p masking
   - optional top-k support
   - categorical sampling

3. Move repetition penalty onto the device:
   - represent recent-token history as an MLX array
   - scatter or mask penalty adjustments on-device

4. Keep a host fallback path for unsupported configurations during bring-up.

5. Add parity tests against the current sampler for:
   - greedy
   - temperature-only
   - top-p
   - repetition penalty

### Expected Gain

- Greedy benchmark from `tests/benchmark_cases.json`: little to no improvement if temperature stays `0.0`
- Non-greedy serving workloads: about 10% to 30%
- TTFT: small improvement
- Tail latency: moderate improvement

This phase is mandatory for broad parity with Python even if it does not move the current greedy benchmark much.

### Exit Criteria

- Non-greedy decode no longer copies full vocab logits to the host.
- Sampler parity tests pass.

### Risks

- Device-side categorical sampling may require new MLX bindings or a slightly different API design.

## Phase 4: Improve Dense-Model KV Cache Efficiency

### Objective

Reduce per-token cache maintenance overhead in the dense-model path.

### Files to Change

- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-nn/src/kv_cache.rs`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-models/src/llama.rs`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-models/src/qwen3.rs`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-models/src/qwen3_5.rs`
- Potentially `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-models/src/lfm2_moe.rs`

### Planned Changes

1. Review whether `KvCache::update()` can return cache views without repeated prefix slicing each token.

2. Avoid unnecessary buffer replacement in the common decode path.

3. Consider larger initial allocation or model-configurable growth factors to reduce resize events.

4. Ensure cache update path does not trigger hidden copies due to shape/view layout issues.

5. Re-profile prompt prefill TTFT after cache changes for large models.

### Expected Gain

- Dense models: about 3% to 10%
- TTFT on large models: about 5% to 15%
- Long generations: moderate benefit

### Exit Criteria

- Cache reallocations during typical runs are rare or eliminated.
- Dense-model throughput improves without regressions in correctness.

### Risks

- Cache code is shared by several architectures. A buggy optimization here can create subtle correctness failures.

## Phase 5: Close the Remaining Dense-Model Gap

### Objective

Tune the dense-model path after the major structural fixes are in place.

### Files to Change

- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-models/src/llama.rs`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-models/src/qwen3.rs`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-models/src/qwen3_5.rs`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-nn/src/linear.rs`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-nn/src/embedding.rs`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-lm/src/loader.rs`

### Planned Changes

1. Audit weight-loading and quantization handling for any unnecessary conversions or transposes in hot paths.

2. Re-check grouped-query attention usage to ensure Rust is taking the cheapest available path for each architecture.

3. Review cache policy tuning in the loader for large-model serving stability.

4. Revisit prompt-template and tokenizer overhead for TTFT-sensitive cases.

5. Add model-specific benchmark slices for:
   - Llama
   - Qwen3
   - Qwen3.5 if used later

### Expected Gain

- Dense models: about 3% to 8%
- TTFT: about 0% to 5%

### Exit Criteria

- Qwen3 reaches near-parity territory.
- Llama narrows to a single-digit or low-double-digit percentage gap.

## Phase 6: Rewrite Qwen MoE Routing and Accumulation

### Objective

Remove host-driven routing and accumulation from `qwen3_moe`.

### Files to Change

- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-models/src/qwen3_moe.rs`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-core/src/ops.rs`
- Potentially `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-sys/src/lib.rs`

### Planned Changes

1. Replace `softmax(...).to_vec_f32()` router extraction with device-side top-k routing.

2. Batch token assignment by expert on device rather than constructing assignment lists on the host.

3. Replace per-token accumulation loops with batched scatter/add or equivalent MLX operations.

4. Keep the shared-expert path on-device as well.

5. Add focused correctness tests for:
   - routing equivalence
   - top-k weight normalization
   - shared-expert contribution

### Expected Gain

- Qwen1.5-MoE: about 15% to 30%
- LFM2: little direct benefit unless shared helper code is introduced

This is the phase most likely to move Qwen MoE from "far behind" to "competitive."

### Exit Criteria

- MoE router probabilities never need to become a host `Vec<f32>` in the hot path.
- Qwen MoE throughput materially improves on aligned benchmarks.

### Risks

- Requires more MLX-side primitive support than the dense path.
- Debugging correctness for sparse routing is harder than dense attention.

## Phase 7: Rewrite LFM2 Expert Execution

### Objective

Eliminate the obviously expensive per-row expert fallback and host-driven LFM2 routing path.

### Files to Change

- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-models/src/lfm2_moe.rs`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-core/src/ops.rs`
- Potentially `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-sys/src/lib.rs`

### Planned Changes

1. Remove the row-by-row `forward_selected()` fallback that slices one expert weight at a time and transposes it in the hot path.

2. Make expert dispatch consistently use the batched quantized or gathered matmul path.

3. Move router top-k selection and normalization on-device.

4. Replace token contribution accumulation loops with batched tensor ops.

5. Re-evaluate short-conv and attention interaction only after expert execution is fixed, because current routing overhead may be masking other costs.

### Expected Gain

- LFM2 decode throughput: about 20% to 45%
- LFM2 TTFT: about 10% to 25%

This is the single most important phase for LFM2.

### Exit Criteria

- No row-by-row expert matmul remains in the hot path.
- LFM2 no longer has a TTFT outlier relative to the other models.

### Risks

- High implementation complexity.
- High chance of introducing correctness or memory regressions if done too aggressively.

## Phase 8: Hardening, Validation, and Regression Protection

### Objective

Lock in the gains and prevent performance regressions from future model work.

### Files to Change

- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/scripts/benchmark_python_vs_rust_servers.py`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/scripts/run_enclosed_benchmarks.sh`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/tests/`
- Possibly CI config if benchmark automation exists later

### Planned Changes

1. Add benchmark baselines for:
   - dense greedy
   - dense non-greedy
   - MoE greedy
   - long-prompt TTFT

2. Add correctness tests for:
   - generation stop behavior
   - stream and non-stream parity
   - sampler parity
   - MoE routing parity

3. Add optional perf regression checks that run manually or in a non-blocking CI job.

4. Document benchmark methodology in the repo so future comparisons stay aligned.

### Expected Gain

- Direct speed gain: 0%
- Long-term value: very high

### Exit Criteria

- Performance improvements are measurable and repeatable.
- Future changes cannot silently break the benchmark methodology.

## Expected Cumulative Outcome

These gains are not strictly additive, but a reasonable planning estimate is:

- After Phase 0: clearer numbers, same real performance
- After Phases 2 through 5: dense models likely recover about 15% to 30% total
- After Phases 6 through 7: MoE models likely recover about 20% to 50% total depending on architecture

Most realistic target by model:

- `Qwen3-1.7B-MLX-4bit`
  - Likely to reach touching distance after benchmark alignment plus Phases 2 through 5

- `Llama-3.2-1B-Instruct-4bit`
  - Plausible to get close, but may require more decode-loop cleanup than Qwen3

- `Qwen1.5-MoE-A2.7B-4bit`
  - Unlikely to get close without Phase 6

- `LFM2-24B-A2B-MLX-4bit`
  - Unlikely to get close without Phase 7

## Recommended Execution Order

1. Phase 0: align the benchmark
2. Phase 1: add instrumentation
3. Phase 2: remove avoidable synchronization
4. Phase 3: move sampling on-device
5. Phase 4: improve KV cache efficiency
6. Phase 5: tune dense-model path
7. Phase 6: rewrite Qwen MoE routing
8. Phase 7: rewrite LFM2 expert execution
9. Phase 8: harden and protect against regressions

## Recommended First Deliverable

The first deliverable should be a benchmark-alignment PR that:

- updates the harness to benchmark both backends through the same transport mode
- reports decode-only throughput and stop reasons
- leaves model code unchanged

That gives a clean before-and-after baseline for every later optimization phase and avoids spending engineering time against misleading numbers.

## Qwen1.5-MoE Router Rewrite Addendum

### Status as of March 6, 2026

Dense-model work materially succeeded:

- `Llama-3.2-1B-Instruct-4bit` reached roughly 95% of Python decode throughput on aligned streaming runs.
- `Qwen3-1.7B-MLX-4bit` reached and in some runs exceeded Python decode throughput.

`Qwen1.5-MoE-A2.7B-4bit` did not follow the same pattern. The most useful stable profile from the current instrumentation shows:

- `decode_forward_s` is effectively the whole request.
- `moe_router_host_s` dominates `decode_forward_s`.
- `moe_expert_forward_s` is small relative to router cost.
- `moe_single_token_fast_path_hits` is very high, confirming that the decode hot path is overwhelmingly single-token sparse routing.

The strongest stable signal from the current work is:

- the remaining `Qwen1.5-MoE` gap is a router-host synchronization problem first
- expert matmul cost is not the first-order bottleneck yet

Two partial device-side top-k attempts regressed behavior or throughput. That means the next step must be a validated rewrite, not another ad hoc local optimization.

### What We Learned From the Failed Attempts

1. Small host-side cleanups are not enough.

- Removing full per-token expert sorting helped only marginally.
- Rewriting the sparse combine path helped only marginally.
- A single-token fast path helped slightly, but not enough to move the model into competitive range.

2. Partial device-side top-k is easy to get wrong.

- Reducing the transferred router payload without a fully validated top-k path caused either behavioral divergence or major throughput regressions.
- The problem is not just “get fewer floats to the host.” It is “preserve routing semantics exactly while also reducing synchronization.”

3. The current router path serializes every sparse layer.

Today `qwen3_moe` still does:

- device gate projection
- host extraction of router values
- host top-k routing
- host-driven control flow

for effectively every decode token across every sparse layer. That is why `moe_router_host_s` tracks almost the whole decode time.

### Rewrite Objective

Replace the current host-driven single-token MoE router path in `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-models/src/qwen3_moe.rs` with a validated MLX-side routing path that:

- keeps top-k selection on device
- preserves exact routing semantics and deterministic tie-breaking
- materializes at most the final selected expert ids and weights, or ideally nothing until expert dispatch
- leaves the multi-token fallback path in place until parity is proven

### Rewrite Scope

Primary file:

- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-models/src/qwen3_moe.rs`

Likely supporting files:

- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-core/src/ops.rs`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-core/src/array.rs`
- possibly `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-sys/src/lib.rs` if bindings are missing

Validation and harness files:

- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/scripts/benchmark_python_vs_rust_servers.py`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-lm/src/generate.rs`
- `/Volumes/Data/Users/paul/development/src/github/mlx-rs/crates/mlx-lm/src/server.rs`

### Proposed Rewrite Phases

#### Phase 6A: Isolated Top-k Validation Layer

Build a small, isolated routing helper that is not immediately wired into the hot path.

Planned work:

1. Add a dedicated helper for single-row top-k routing semantics:
   - input: router logits or probabilities for one token
   - output: selected expert ids and weights
   - deterministic tie-breaking by expert index

2. Validate this helper against the current host implementation on the same row.

3. Run the helper in a “shadow mode” first:
   - compute host result
   - compute device result
   - compare expert ids and weights
   - fall back to host result on mismatch

4. Gate this behind an environment variable for bring-up, for example:
   - `MLX_EXPERIMENTAL_MOE_ROUTER=1`
   - `MLX_VERIFY_MOE_ROUTER=1`

Exit criteria:

- the device-side top-k helper matches the host path on repeated benchmark runs
- no stop-behavior or repeat-guard regressions appear in verification mode

Expected gain:

- direct speed gain in shadow mode: none
- indirect value: very high

#### Phase 6B: Single-Token Device Router

After semantic parity is proven, switch the single-token decode path to the device-side router helper.

Planned work:

1. Keep gate projection and top-k selection on device.

2. Materialize only the selected expert ids and weights needed for dispatch.

3. Keep the current host multi-token path for prefill until the decode path is proven stable.

4. Keep the current profiling counters and add one more if needed:
   - `moe_router_device_path_hits`

Exit criteria:

- `Qwen1.5-MoE` no longer spends most of decode time in `moe_router_host_s`
- stop behavior remains stable
- no `repeat_guard` regressions are introduced

Expected gain:

- about 10% to 25% decode throughput if the router synchronization is truly collapsed

#### Phase 6C: Batched Selected-Expert Dispatch

Once routing is stable, reduce expert dispatch overhead without changing semantics.

Planned work:

1. Revisit the earlier idea of grouped selected-expert execution, but only after top-k parity is proven.

2. Do this on a separate experimental branch or behind a flag first.

3. Reuse patterns that already exist in `lfm2_moe` only after confirming the tensor layout assumptions for `qwen3_moe`.

4. Keep the single-token fast path, but make it dispatch selected experts in a grouped way if the gather path is validated.

Exit criteria:

- `moe_expert_forward_s` stays small or shrinks further
- no process crashes
- no generation-quality regression

Expected gain:

- about 3% to 10% additional decode throughput

#### Phase 6D: Multi-Token Router Migration

Only after single-token decode is stable and clearly faster:

1. port the prefill path from host routing to device routing
2. remove duplicate host-side helper code
3. keep a host fallback for correctness debugging

Expected gain:

- small to moderate TTFT improvement
- lower code divergence between prefill and decode routing

### Explicit Non-Goals for the Next Iteration

The next iteration should not:

- change both router semantics and expert dispatch at once
- remove the current host routing path before parity is proven
- optimize `LFM2` and `Qwen1.5-MoE` in the same patch set
- touch dense-model paths unless a regression needs to be reverted

### Recommended Implementation Order From Here

1. Keep the current stable `Qwen1.5-MoE` baseline in tree.
2. Implement a shadow-mode single-token device top-k helper.
3. Add explicit parity checks against the current host router path.
4. Only after parity is proven, switch decode to the device router path.
5. Re-benchmark before any grouped-expert dispatch work.

### Current Practical Expectation

Given the measured dominance of `moe_router_host_s`, a correct device-side single-token router is still the best remaining lever for `Qwen1.5-MoE`.

However, based on the failed attempts so far, the rewrite should be treated as an experimental subsystem bring-up, not as a normal local optimization patch.
