# Models Support Plan — mlx-rs MLX Backend

Ollama's MLX engine (Go, `x/models/`) supports architectures that mlx-rs does not yet have.
Each milestone adds one model family, with all file changes needed in both projects.

---

## Milestone 1: Qwen3.5 / Qwen3.5MoE / Qwen3Next

**Ollama reference files:**
- `x/models/qwen3_5/qwen3_5.go` — main Qwen3.5 dense model
- `x/models/qwen3_5/qwen3_5_test.go`
- `x/models/qwen3_5_moe/qwen3_5_moe.go` — MoE variant (registers `Qwen3_5MoeForConditionalGeneration`, `Qwen3_5MoeForCausalLM`, `Qwen3NextMoeForConditionalGeneration`, `Qwen3NextMoeForCausalLM`)
- `x/models/qwen3/qwen3.go` — base Qwen3 (already in mlx-rs)

**mlx-rs changes:**
- `crates/mlx-backend/src/qwen3_5.rs` — new file: Qwen3.5 dense model (config struct, model struct, forward pass, KvCache)
- `crates/mlx-backend/src/qwen3_5_moe.rs` — new file: MoE variant with expert routing
- `crates/mlx-backend/src/registry.rs` — add `"Qwen3_5ForCausalLM"`, `"Qwen3_5ForConditionalGeneration"`, `"Qwen3NextForCausalLM"`, `"Qwen3NextForConditionalGeneration"`, `"Qwen3_5MoeForConditionalGeneration"`, `"Qwen3_5MoeForCausalLM"`, `"Qwen3NextMoeForConditionalGeneration"`, `"Qwen3NextMoeForCausalLM"` to `create_model()` match and `supported_architectures()`
- `crates/mlx-backend/src/lib.rs` — add `pub mod qwen3_5;` and `pub mod qwen3_5_moe;`
- `crates/mlx-backend/src/chat_template.rs` — no change needed (falls back to llama3 template; Qwen models ship `chat_template` in tokenizer_config.json)

**Key differences from Qwen3:** Qwen3.5 adds thinking mode (chain-of-thought toggle), MoE uses expert routing with shared+specialist experts. Need `MoeLayer` abstraction in mlx-rs.

---

## Milestone 2: Glm4MoeLite (MoE)

**Ollama reference files:**
- `x/models/glm4_moe_lite/glm4_moe_lite.go` — model implementation
- `x/models/glm4_moe_lite/parser.go` — response parser
- `x/models/glm4_moe_lite/render.go` — chat template renderer
- `x/models/glm4_moe_lite/render_test.go`
- `x/models/glm4_moe_lite/parser_test.go`

**mlx-rs changes:**
- `crates/mlx-backend/src/glm4_moe_lite.rs` — new file: GLM4 MoE Lite model with expert routing, SwiGLU MLP, RMSNorm, RoPE
- `crates/mlx-backend/src/registry.rs` — add `"Glm4MoeLiteForCausalLM"` and `"GLM4MoeLite"` to match arm and `supported_architectures()`
- `crates/mlx-backend/src/lib.rs` — add `pub mod glm4_moe_lite;`
- `crates/mlx-backend/src/chat_template.rs` — add GLM4 chat template default fallback (or rely on tokenizer_config.json)

**Note:** This is the first MoE model. The `MoeLayer` / expert-routing abstraction from Milestone 1 should be reused here.

---

## Milestone 3: Laguna

**Ollama reference files:**
- `x/models/laguna/laguna.go` — model implementation
- `x/models/laguna/laguna_test.go`
- `model/renderers/laguna.go` — chat template renderer (Go TEMPLATE)
- `model/renderers/laguna_test.go`
- `model/parsers/laguna.go` — response parser
- `model/parsers/laguna_test.go`

**mlx-rs changes:**
- `crates/mlx-backend/src/laguna.rs` — new file: Laguna model (recurrent architecture, not standard transformer)
- `crates/mlx-backend/src/registry.rs` — add `"LagunaForCausalLM"` to match arm and `supported_architectures()`
- `crates/mlx-backend/src/lib.rs` — add `pub mod laguna;`
- `crates/mlx-backend/src/chat_template.rs` — add Laguna chat template as default fallback
- `crates/mlx-backend/src/cache.rs` — may need `RecurrentCache` type (Laguna is recurrent, uses different cache than standard KV)

**Note:** Laguna is a recurrent model, not a standard transformer. It uses a different cache type (`RecurrentCache` / `RotatingCache` in ollama). This requires a new cache variant in mlx-rs.

---

## Milestone 4: Gemma4Unified Variants

**Ollama reference files:**
- `x/models/gemma4/gemma4.go` — already ported in mlx-rs as `gemma4.rs`
- `x/models/gemma4/assistant.go` — MTP draft model (Gemma4AssistantForCausalLM)

**mlx-rs changes:**
- `crates/mlx-backend/src/gemma4.rs` — add `"Gemma4UnifiedForCausalLM"`, `"Gemma4UnifiedForConditionalGeneration"`, `"gemma4_unified"` to registry match (these likely use the same Gemma4Model with different config parsing)
- `crates/mlx-backend/src/registry.rs` — add the 3 new architecture strings
- No new model files needed — Unified variants reuse the existing Gemma4 implementation

---

## Milestone 5: MTP Draft Model Support (Speculative Decoding)

**Ollama reference files:**
- `x/models/gemma4/assistant.go` — `Gemma4AssistantForCausalLM` draft model (MTP: multi-token prediction)
- `x/mlxrunner/mtp.go` — MTP speculative decoding orchestration
- `x/mlxrunner/model/base/base.go` — `RegisterDraft()`, `NewDraft()`, `DraftModel` interface
- `x/mlxrunner/pipeline.go` — speculative decoding integration in generation loop

**mlx-rs changes:**
- `crates/mlx-backend/src/model.rs` — add `DraftModel` trait: `fn forward_from_embedding(&self, embeds: &Array, position: i32, caches: &mut [KvCache]) -> Result<Array>;` `fn token_embeddings(&self, ids: &Array) -> Result<Array>;`
- `crates/mlx-backend/src/gemma4.rs` — add `Gemma4AssistantModel` struct implementing `DraftModel` (loads assistant weights, runs MTP forward pass using target embeddings + target hidden states)
- `crates/mlx-backend/src/registry.rs` — add `"Gemma4AssistantForCausalLM"`, `"Gemma4UnifiedAssistantForCausalLM"`, `"gemma4_assistant"`, `"gemma4_unified_assistant"` draft registrations
- `crates/mlx-backend/src/mlx_backend.rs` — add speculative decoding to `generate()` and `generate_stream()`: after target model forward, run draft model to predict N tokens, verify against target model's logits
- `crates/mlx-backend/src/manifest.rs` — support loading draft model config alongside target model (draft config from `Draft` field in manifest or `draft/config.json`)
- `crates/backend-trait/src/backend.rs` — optionally expose draft model configuration in `Backend` trait

---

## Summary: Architecture Strings to Add

| Architecture String | Milestone |
|---|---|
| `Qwen3_5ForCausalLM` | 1 |
| `Qwen3_5ForConditionalGeneration` | 1 |
| `Qwen3NextForCausalLM` | 1 |
| `Qwen3NextForConditionalGeneration` | 1 |
| `Qwen3_5MoeForConditionalGeneration` | 1 |
| `Qwen3_5MoeForCausalLM` | 1 |
| `Qwen3NextMoeForConditionalGeneration` | 1 |
| `Qwen3NextMoeForCausalLM` | 1 |
| `Glm4MoeLiteForCausalLM` | 2 |
| `GLM4MoeLite` | 2 |
| `LagunaForCausalLM` | 3 |
| `Gemma4UnifiedForCausalLM` | 4 |
| `Gemma4UnifiedForConditionalGeneration` | 4 |
| `gemma4_unified` | 4 |
| `Gemma4AssistantForCausalLM` (draft) | 5 |
| `Gemma4UnifiedAssistantForCausalLM` (draft) | 5 |
| `gemma4_assistant` (draft) | 5 |
| `gemma4_unified_assistant` (draft) | 5 |