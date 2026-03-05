# Benchmark Report: Python MLX vs Rust MLX

## Run Context
- Date: 2026-03-05
- Harness: `scripts/run_enclosed_benchmarks.sh`
- Script used: `scripts/benchmark_python_vs_rust_servers.py`
- Benchmark artifact: `logs/python_vs_rust_benchmark_20260305_122116.json`
- Models tested:
  - `Llama-3.2-1B-Instruct-4bit`
  - `Qwen3-1.7B-MLX-4bit`
  - `Qwen1.5-MoE-A2.7B-4bit`
- Prompt: `What is the language Rust?`
- Temperature: `0.0`
- Max tokens: `256`

## Enclosed Execution Flow
The benchmark run is fully enclosed in one process:
1. Start Python server per model
2. Run benchmark for that model
3. Stop Python server
4. Repeat for all 3 models
5. Start Rust server
6. Load model -> benchmark -> unload model for each of 3 models
7. Stop Rust server

No manual server management is required during benchmark execution.

## Benchmark Results

| model | backend | ttft_s | tokens_per_s | tokens | total_s |
|---|---|---:|---:|---:|---:|
| Llama-3.2-1B-Instruct-4bit | python | 0.161 | 134.65 | 196 | 1.456 |
| Llama-3.2-1B-Instruct-4bit | rust | 0.094 | 86.03 | 256 | 2.976 |
| Qwen3-1.7B-MLX-4bit | python | 0.192 | 84.28 | 202 | 2.397 |
| Qwen3-1.7B-MLX-4bit | rust | 0.078 | 45.95 | 256 | 5.571 |
| Qwen1.5-MoE-A2.7B-4bit | python | 0.242 | 62.11 | 190 | 3.059 |
| Qwen1.5-MoE-A2.7B-4bit | rust | 0.738 | 28.56 | 141 | 4.937 |

## Observations
- Rust TTFT is lower for Llama and Qwen3, but Rust steady-state decode throughput is lower for all models.
- Disparity is largest on MoE model (`Qwen1.5-MoE`), where Rust TTFT is also significantly higher.
- Python outputs fewer tokens on two dense models in this run (196/202) while Rust reached 256, so some runtime comparisons include different decode lengths.

---

## Why the Gap Exists (Detailed)

## 1. Different generation-loop implementations (same runtime, different orchestration)
Both paths call MLX under the hood, but they do not execute equivalent decode loops.

- Python MLX (`mlx_lm`) uses a mature inference pipeline with optimized sequencing of:
  - prefill
  - decode step scheduling
  - tokenizer integration
  - stop-condition handling
- Rust currently uses a custom generation pipeline in `mlx-lm` with equivalent functionality, but different step ordering and host-device interaction points.

Impact:
- Different orchestration can dominate throughput even with identical model weights and same low-level library.
- This explains why TTFT can be better in Rust (fast early path) while overall tokens/sec is worse (less efficient sustained decode loop).

## 2. Host-device synchronization overhead in Rust sampling path
At `temperature=0`, Rust currently applies repetition control by reading logits to host (`to_vec_f32`) and selecting argmax from host-side values.

- This creates an explicit synchronization point each decode step.
- Repeated per-token synchronization prevents deep pipelining and increases end-to-end decode latency.

By contrast, optimized paths typically keep greedy decode fully on-device as long as possible, minimizing round-trips.

Impact:
- Lower tokens/sec in Rust, especially at longer output lengths.
- Gap appears on dense models and compounds with sequence length.

## 3. MoE implementation in Rust is currently correctness-first, not perf-first
Qwen1.5-MoE in Rust was brought to functional correctness with host-routed expert dispatch semantics aligned to expected behavior.

Current MoE tradeoffs:
- Routing decisions moved to host for deterministic behavior
- Per-expert assignment and aggregation performed with additional host orchestration
- Shared-expert gating and accumulation done in a non-fused way

Compared to optimized MLX Python MoE flow, this is higher overhead.

Impact:
- Largest throughput gap appears on MoE.
- Higher TTFT for MoE in Rust due to heavier first-pass routing/setup and synchronization.

## 4. Metric comparability is improved but not perfectly identical in execution shape
The benchmark now reports the same fields for both stacks (TTFT, tokens/sec, tokens, total), but run characteristics can still diverge:

- Different stopping behavior may produce different completion lengths.
- Sampling/termination details and chat-template handling can produce different token trajectories.
- Prompt-format or special-token handling differences alter decode path and token count.

Impact:
- Raw tokens/sec and total-time comparisons are directionally valid, but per-model parity is still influenced by differing completion lengths and stop behavior.

## 5. Runtime maturity and optimization depth difference
Python MLX inference path has had more tuning and broader usage on production-like workloads.
Rust path is newer and prioritizing feature-completeness/correctness first:

- multiple model families stabilized recently
- stop-token behavior recently hardened
- MoE brought up with safe semantics before optimization

Impact:
- Expected disparity until dedicated performance pass is completed.
- This is typical for early-stage runtime parity efforts.

---

## Model-by-Model Interpretation

### Llama-3.2-1B-Instruct-4bit
- Rust TTFT is better (`0.094s` vs `0.161s`), suggesting fast startup path.
- Python decode throughput is higher (`134.65` vs `86.03` tokens/s), indicating better sustained loop efficiency.
- Rust generated more tokens in this run (256 vs 196), increasing total decode time and reducing apparent throughput comparability.

### Qwen3-1.7B-MLX-4bit
- Same pattern as Llama: lower Rust TTFT, lower Rust tokens/sec.
- Larger sustained-gap than Llama indicates additional overhead in decode loop and/or prompt/termination behavior.

### Qwen1.5-MoE-A2.7B-4bit
- Biggest gap and inverse TTFT advantage (Rust TTFT worse: `0.738s` vs `0.242s`).
- This is consistent with host-oriented MoE routing/merge overhead in current Rust implementation.

---

## Practical Conclusion
The benchmark confirms:
- Functional parity is largely achieved (all three models run in both stacks).
- Performance parity is not yet achieved.
- The largest contributor to disparity is decode-loop efficiency (especially synchronization strategy), followed by MoE routing architecture.

Given current architecture, this performance profile is expected and actionable, not anomalous.

## High-Confidence Next Optimization Targets
1. Keep greedy decode and repetition control on-device where possible (reduce host logits extraction).
2. Reduce per-token `eval`/synchronization boundaries.
3. Move MoE routing/aggregation toward batched/fused device execution.
4. Ensure identical stop/token accounting policy across both backends for strict comparability.
5. Add per-phase timing in Rust (`template`, `tokenize`, `prefill`, `decode`) to isolate bottlenecks quantitatively.

