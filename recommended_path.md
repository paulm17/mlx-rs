# Recommended Path: Candle-Style Parity on MLX Runtime

## Executive Summary
You already have the correct high-level decomposition for an inference-focused stack:
- `mlx-core` (tensor/runtime primitives)
- `mlx-nn` (layers, builders, utility blocks)
- `mlx-lm` (loading/tokenization/generation UX)
- `mlx-models` (architecture-specific model code)

This is close enough to Candle's design philosophy that a full directory-structure rewrite is not the highest-ROI move right now.

The recommended path is:
1. Keep the current crate split.
2. Increase API/behavior parity with Candle where it improves portability and correctness.
3. Add missing features only when required by target models.
4. Use a checklist-driven parity process, validated by model-level golden tests.

---

## Current State vs Candle

### What already matches Candle patterns
- Layered crate boundaries (`core`, `nn`, model/inference layer split).
- `Module`-style composability.
- `VarBuilder` + `pp()` hierarchical weight access.
- Per-architecture model files and config structs.
- Shared generation pipeline separated from model internals.

### What is intentionally different (and acceptable)
- Runtime is MLX C API (`mlx.c`) instead of Candle's own backend stack.
- Inference-centric scope (you are not trying to replicate all Candle functionality).
- Model registry lives in `mlx-models` rather than `candle-transformers`/examples style.

### What is missing relative to full Candle breadth
- Broader tensor op coverage and edge-case shape semantics.
- Wider backend/feature matrix (CUDA/Metal CPU modes, kernel specializations).
- Richer utility and testing ecosystem.
- Some advanced generation features and strict parity harnesses.

---

## Recommended Path (Phased)

## Phase 1: Stabilize Architecture Parity (Now)
Goal: make supported models robust and predictable before expanding surface area.

1. Freeze model scope for parity work
- Llama 3
- Qwen3
- Qwen1.5-MoE (`qwen2_moe` family)

2. Define a strict compatibility contract per model
- Loadability: all tensors resolved with clear missing-key errors.
- Prompting: chat template behavior defined and deterministic.
- Stop behavior: EOS/chat terminators handled without user flags.
- Decode behavior: no runaway loops for `temperature=0` default usage.
- KV cache correctness: prefill vs decode token parity checks.

3. Add model-level regression tests
- One smoke prompt per architecture.
- One deterministic prompt with expected token prefix.
- One long-answer prompt to detect loops/repetition pathologies.
- One stop-token test for chat models.

4. Lock known-good defaults in `mlx-lm`
- Internal repetition controls and repeat-window defaults.
- Auto chat-template application policy.
- Stop-token augmentation policy (`eos_token_id` + known chat end markers).

Exit criteria:
- All 3 target model families produce stable outputs repeatedly across runs.
- No required CLI workaround for basic prompt quality.

---

## Phase 2: API Parity Hardening (Candle-Like Surface)
Goal: make porting Candle model logic easier with minimal restructuring.

1. Normalize core API naming/semantics
- Audit `mlx-core` tensor ops for shape/dtype semantics against Candle expectations.
- Add explicit error cases where currently permissive behavior can hide bugs.
- Document exact behavior for:
  - `slice`
  - `take` / `take_along_axis`
  - `argpartition`
  - `softmax` axis behavior
  - `contiguous` guarantees

2. Expand `VarBuilder` ergonomics toward parity
- Add optional convenience methods commonly expected by model ports:
  - typed getters
  - optional getters
  - clear missing-key diagnostics with nearest-prefix hints
- Keep existing `pp()` flow unchanged (already good).

3. Add architecture-neutral attention/feed-forward utilities
- Reusable helpers for:
  - QKV reshape/transpose
  - KV head repetition
  - last-token slicing
  - RoPE application patterns
- Reduce model-file divergence for similar families.

Exit criteria:
- Porting a new dense decoder architecture requires mostly model-file work, minimal infra edits.

---

## Phase 3: Inference UX and Reliability Parity
Goal: production-friendly defaults without CLI burden.

1. Generation safety rails
- Keep optional `max_tokens` but avoid hidden truncation when omitted.
- Add internal loop detection metrics/log hooks (off by default).
- Add optional `--debug-decode` for token/id tracing.

2. Better diagnostics
- On bad outputs, provide architecture + stop-token + template summary.
- Add structured error context for missing tensors and shape mismatches.

3. Deterministic mode policy
- Define exact deterministic mode semantics (`temperature=0`, top-p ignored, repetition still applied).

Exit criteria:
- Basic inference behaves sensibly with only `model-dir`, `prompt`, `temperature`.

---

## Phase 4: Expand Model Support Safely
Goal: add new families with controlled risk.

1. Introduce model onboarding template
For each new model family:
- Config mapping table (`config.json` -> Rust config struct).
- Tensor key mapping coverage report.
- Attention style (MHA/GQA/MQA/sliding window).
- FFN style (dense/MoE/shared experts).
- Chat template and stop policy.
- Golden prompt tests.

2. Add only required missing ops
- Avoid broad preemptive `candle-core` parity implementation.
- Track op additions to concrete model need.

Exit criteria:
- New model support additions are predictable and checklist-driven.

---

## Candle-Parity Checklist

Use this as the operational checklist for each release increment.

## A. `mlx-core` parity checklist
- [ ] DType coverage audited for model-required types.
- [ ] Tensor creation and scalar extraction behavior documented.
- [ ] `reshape`, `transpose`, `squeeze`, `expand_dims`, `flatten` semantics tested.
- [ ] `slice` semantics validated for multi-dim edge cases.
- [ ] `take`/`take_along_axis` semantics validated.
- [ ] `argmax`, `argpartition`, `softmax` axis semantics validated.
- [ ] Quantized matmul/dequantization edge cases tested.
- [ ] SDPA path validated for causal and decode modes.
- [ ] Stream/device default behavior documented and stable.
- [ ] Safetensors loading path robust on target environments.

## B. `mlx-nn` parity checklist
- [ ] `Linear` supports required quantized layouts for target checkpoints.
- [ ] `Embedding` behavior aligned for tied/untied head use.
- [ ] `RmsNorm` numerical stability validated.
- [ ] `RoPE` variants (scaling, theta, max-pos rules) validated per family.
- [ ] KV cache update and offset behavior unit-tested.
- [ ] `VarBuilder` prefix-scoping and key diagnostics validated.

## C. `mlx-models` parity checklist
- [ ] Each architecture has clear Config -> Model mapping doc.
- [ ] Tensor key fallbacks handled (e.g., `o_proj` vs `out_proj` where needed).
- [ ] Last-token logits extraction consistent and tested.
- [ ] MoE router semantics verified against reference behavior.
- [ ] Shared expert and shared gate behavior validated (where applicable).
- [ ] Layer-by-layer forward shape checks in debug tests.

## D. `mlx-lm` parity checklist
- [ ] Architecture detection robust (`model_type` + `architectures` fallback).
- [ ] Tokenizer load path handles eos and multiple stop IDs.
- [ ] Known chat terminators included automatically where appropriate.
- [ ] Chat template auto-apply policy documented and predictable.
- [ ] Greedy decoding (`temperature=0`) includes anti-loop controls.
- [ ] Sampling defaults tuned for production-safe behavior.
- [ ] Optional decode debug instrumentation available.

## E. CLI / UX checklist
- [ ] Minimal-command success (`--model-dir`, `--prompt`, `--temperature`).
- [ ] No hidden hard truncation when user omits max tokens.
- [ ] Streaming output responsive (no “appears hung” behavior).
- [ ] Generation summary reports meaningful metrics.

## F. Testing and validation checklist
- [ ] Unit tests for core op invariants.
- [ ] Unit tests for tokenizer stop logic.
- [ ] Unit tests for sampler deterministic behavior.
- [ ] Integration smoke tests per supported architecture.
- [ ] Golden prompt tests for quality regression detection.
- [ ] Failure-mode tests (missing tensor, bad config, incompatible dtype).

## G. Release readiness checklist
- [ ] Changelog entry for behavior/default changes.
- [ ] Model support matrix updated.
- [ ] Known limitations documented.
- [ ] Repro commands listed for all supported families.

---

## What Not To Do (High-Churn, Low ROI)
- Do not do a full Candle source-layout clone just for naming symmetry.
- Do not implement broad unused `candle-core` functionality ahead of model need.
- Do not mix large refactors with model bring-up in the same PR/patch.

---

## Suggested Work Plan for Next 2 Iterations

## Iteration 1 (Stability)
1. Add regression tests for Llama3/Qwen3/Qwen1.5-MoE prompts.
2. Add decode-loop/stop-token test coverage in `mlx-lm`.
3. Add structured missing-key diagnostics in `VarBuilder`.
4. Add an architecture support matrix document.

Definition of done:
- All 3 model families pass smoke + repeatability checks.

## Iteration 2 (Parity hardening)
1. Audit and document core op semantics used by current models.
2. Add small API improvements in `VarBuilder` and model utilities.
3. Reduce duplicated attention/reshape logic across model files.
4. Add one new model onboarding using the checklist (pilot).

Definition of done:
- New architecture bring-up requires minimal infra changes and follows a repeatable template.

---

## Final Recommendation
Keep the current crate structure. Treat this as a Candle-inspired MLX runtime implementation, then drive parity with a checklist and regression tests rather than a directory rewrite.

This gives you faster model onboarding, fewer regressions, and better long-term maintainability than attempting an exact structural clone of Candle.
