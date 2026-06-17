# New Model Porting Plan

Port models from deprecated `mlx-rs_repo` to current `mlx-rs`. Models already present (Llama, Qwen3, Qwen3.5, Gemma3, Gemma4 text) are skipped.

## Decisions

- **BERT batch embed:** Single-string `embed()` initially. The server already handles multiple inputs by calling `embed()` in a loop. True padded-batch optimization (like the deprecated repo's `supports_padded_embedding_batching()`) is a follow-up.
- **Qwen MoE:** Port native Rust only (`qwen3_moe.rs`), skip Python port.
- **LFM2 MoE:** Port native Rust only (`lfm2_moe.rs`), skip Python port.
- **Gemma4 multimodal:** New `MultimodalModel` trait (separate from `Model`).
- **Conv state:** Reuse existing `RecurrentCache` from `llama.rs`.
- **All models have been downloaded in /Volumes/Data/Users/paul/.cache/huggingface/hub/ directory**

---

## Phase 1: Qwen2 (Registry-Only)

Qwen2 is architecturally identical to Llama. Only registry changes needed.

**Files to modify:**
- `crates/mlx-backend/src/registry.rs`

**Changes:**
- Add `"Qwen2ForCausalLM"` to the Llama match arm in `create_model()`
- Add `"Qwen2ForCausalLM"` to `supported_architectures()`

**Verify:** Load `Qwen/Qwen2-1.5B-MLX-4bit`, run text generation.

cargo run --release \
--bin generate -- \
--model mlx-community/Qwen2-1.5B-Instruct-4bit \
--chat \
--prompt "Hello"

---

## Phase 2: BERT Encoder/Embedding Model

Add BERT model and embedding infrastructure.

**Files to create:**
- `crates/mlx-backend/src/bert.rs`

**Files to modify:**
- `crates/mlx-backend/src/model.rs` — Add `EncoderModel` trait
- `crates/mlx-backend/src/lib.rs` — Add `pub mod bert;`
- `crates/mlx-backend/src/registry.rs` — Add BERT architecture detection
- `crates/mlx-backend/src/mlx_backend.rs` — Add `encoder_model` field, implement `embed()`

### model.rs — New trait

```rust
pub trait EncoderModel: Send {
    fn encode(&self, input_ids: &Array) -> anyhow::Result<Array>;
    fn encode_masked(&self, input_ids: &Array, attention_mask: Option<&Array>) -> anyhow::Result<Array>;
    fn hidden_size(&self) -> i32;
}
```

### bert.rs — Structure (adapted from deprecated repo)

| Component | Source prefix | Description |
|-----------|--------------|-------------|
| `BertConfig` | config.json | vocab_size, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, hidden_act (GeLU), layer_norm_eps |
| `BertEmbeddings` | `embeddings.*` | word + position + token_type embeddings + LayerNorm |
| `BertSelfAttention` | `encoder.layer.N.attention.self.*` | Q/K/V projections + output.dense, scaled dot product attention |
| `BertAttention` | `encoder.layer.N.attention.*` | self-attention + residual + LayerNorm |
| `BertMlp` | `encoder.layer.N.*` | intermediate.dense (GeLU) + output.dense + LayerNorm |
| `BertLayer` | `encoder.layer.N.*` | attention + MLP |
| `BertModel` | top-level | embeddings + N layers, implements `EncoderModel` |

**Key adaptations from deprecated repo:**
- Use current repo's `Linear`, `Embedding` from `llama.rs` (not `mlx_nn`)
- Use current repo's `Array` and `ops` (not `mlx_core`)
- Add `LayerNorm` to `llama.rs` (currently missing — BERT uses LayerNorm, not RmsNorm)
- Implement `EncoderModel` trait (not deprecated `ModelRuntime`)
- Handle quantized weights via `QuantizedLinear` / `QuantizedEmbedding`
- Weight prefix: `resolve_weight_prefix()` checks for `bert.` prefix

### mlx_backend.rs — Changes

```rust
pub struct MlxBackend {
    model: Box<dyn Model>,
    encoder_model: Option<Box<dyn EncoderModel>>,  // NEW
    // ... existing fields
}
```

In `load()`:
- Detect BERT architecture → load as `encoder_model` instead of `model`
- For non-BERT models, `encoder_model` stays `None`

`embed()` implementation:
- Tokenize input (no BOS for BERT)
- Call `encoder_model.encode()`
- Mean pooling over sequence dimension
- L2 normalize
- Return `EmbeddingOutput`

`embeddings_enabled()`:
- Return `true` when `encoder_model.is_some()`

### Registry additions

```rust
"BertModel" | "BertForMaskedLM" | "BertForSequenceClassification" => {
    let cfg = BertConfig::from_json(config)?;
    let model = BertModel::load_from_tensors(tensors, cfg)?;
    // Return as encoder model — need separate handling
}
```

**Problem:** `create_model()` returns `Box<dyn Model>`, but BERT implements `EncoderModel`, not `Model`. Two options:
1. BERT also implements `Model` with stub `forward()` that returns error. `MlxBackend` detects BERT and also stores it as `encoder_model`.
2. Change `create_model()` to return an enum: `ModelOrEncoder(Box<dyn Model>, Option<Box<dyn EncoderModel>>)`.

**Recommendation:** Option 1 — simpler, no API change. BERT's `Model::forward()` returns error. `MlxBackend::load()` detects BERT from architecture string and stores the model in both fields.

**Verify:** Load `mlx-community/mxbai-embed-large-v1`, run embedding benchmark.

cargo run --release \
--bin generate -- \
--model mlx-community/mxbai-embed-large-v1 \
--chat \
--prompt "Hello"

---

## Phase 3: Qwen MoE

Add Qwen MoE (qwen1.5_moe / qwen2_moe) sparse mixture-of-experts model.

**Files to create:**
- `crates/mlx-backend/src/qwen_moe.rs`

**Files to modify:**
- `crates/mlx-backend/src/lib.rs` — Add `pub mod qwen_moe;`
- `crates/mlx-backend/src/registry.rs` — Add MoE architecture strings

### qwen_moe.rs — Structure (adapted from deprecated `qwen3_moe.rs`)

| Component | Description |
|-----------|-------------|
| `QwenMoeConfig` | hidden_size, intermediate_size, moe_intermediate_size, num_experts, num_experts_per_tok, decoder_sparse_step, shared_expert_intermediate_size, norm_topk_prob |
| `SwitchGlu` | Expert-parallel FFN using `gather_mm` — gate_proj, up_proj, down_proj per expert |
| `SparseMoeBlock` | Router gate → top-k expert selection → `SwitchGlu` forward → optional shared expert with gating |
| `QwenMoeAttention` | Same as Llama attention (Q/K/V/O, RoPE, KvCache, GQA) |
| `QwenMoeLayer` | attention + MoE FFN (or dense FFN if not MoE layer per `decoder_sparse_step`) |
| `QwenMoeModel` | embed_tokens + layers + norm + lm_head, implements `Model` |

**Key adaptations:**
- Use `ops::gather_mm()` for expert-parallel execution
- Use `LinearLayer` / `EmbeddingLayer` trait objects for quantization
- `LayerCache::Attention(KvCache)` (same as Llama)
- Router: linear projection → softmax → top-k selection
- `SwitchLinear`: uses `gather_qmm` (quantized) or `gather_mm` (dense) based on weight type
- Expert profiling infrastructure (optional, gated by `MLX_TRACE_GENERATION` env var)

**Registry additions:**
```rust
"Qwen2MoeForCausalLM" | "Qwen1.5MoeForCausalLM" => {
    let cfg = QwenMoeConfig::from_json(config)?;
    let model = QwenMoeModel::load_from_tensors(tensors, cfg)?;
    Ok(Box::new(model))
}
```

**Verify:** Load `mlx-community/Qwen1.5-MoE-A2.7B-4bit`, run text generation.

cargo run --release \
--bin generate -- \
--model mlx-community/Qwen1.5-MoE-A2.7B-4bit \
--chat \
--prompt "Hello"

---

## Phase 4: LFM2 MoE

Add LFM2 hybrid conv+attention MoE model. Most complex model.

**Files to create:**
- `crates/mlx-backend/src/lfm2_moe.rs`

**Files to modify:**
- `crates/mlx-backend/src/lib.rs` — Add `pub mod lfm2_moe;`
- `crates/mlx-backend/src/registry.rs` — Add LFM2 architecture strings
- `crates/mlx-backend/src/llama.rs` — Add `Recurrent` variant to `LayerCache` (if not already present from Qwen3.5)

### lfm2_moe.rs — Structure (adapted from deprecated `lfm2_moe.rs`)

| Component | Description |
|-----------|-------------|
| `Lfm2MoeConfig` | hidden_size, num_attention_heads, num_key_value_heads, layer_types, conv_l_cache, conv_bias, num_dense_layers, num_experts, num_experts_per_tok, norm_eps, rope_theta |
| `ShortConv` | in_proj → split (B, C, X) → B*X → conv1d → C*conv → out_proj. Uses conv state in `RecurrentCache` |
| `Lfm2Attention` | Q/K/V/O with optional Q/K layernorm, RoPE |
| `MoeFeedForward` | Router + `SwitchGlu` experts + optional expert_bias |
| `DenseFeedForward` | Standard gate/up/down MLP (SiLU) |
| `Lfm2MoeLayer` | operator (attention OR conv) + FFN (MoE OR dense). Layer type from `layer_types` config |
| `Lfm2MoeModel` | embed_tokens + layers + norm + lm_head, implements `Model` |

**Key adaptations:**
- Reuse `RecurrentCache` from `llama.rs` for conv state
- `LayerCache::Recurrent(RecurrentCache)` for conv layers
- `LayerCache::Attention(KvCache)` for attention layers
- Weight paths: `feed_forward.switch_mlp.*` and `feed_forward.gate.*`
- Norm names: `operator_norm` / `ffn_norm` (not `input_layernorm` / `post_attention_layernorm`)

**Registry additions:**
```rust
"Lfm2MoeForCausalLM" | "Lfm2ForCausalLM" => {
    let cfg = Lfm2MoeConfig::from_json(config)?;
    let model = Lfm2MoeModel::load_from_tensors(tensors, cfg)?;
    Ok(Box::new(model))
}
```

**Verify:** Load `LiquidAI/LFM2-24B-A2B-MLX-4bit`, run text generation.

cargo run --release \
--bin generate -- \
--model LiquidAI/LFM2-24B-A2B-MLX-4bit \
--chat \
--prompt "Hello"

---

## Phase 5: Gemma4 Multimodal

Extend existing Gemma4 with vision tower for multimodal (vision + text) support.

**Files to modify:**
- `crates/mlx-backend/src/model.rs` — Add `MultimodalModel` trait
- `crates/mlx-backend/src/gemma4.rs` — Add vision tower + multimodal embedder + image token expansion
- `crates/mlx-backend/src/registry.rs` — Update Gemma4 config parsing for multimodal
- `crates/mlx-backend/src/mlx_backend.rs` — Add multimodal dispatch

### model.rs — New trait

```rust
pub trait MultimodalModel: Send {
    fn forward_multimodal(
        &self,
        input_ids: &Array,
        caches: &mut [LayerCache],
        positions: &Array,
        pixel_values: &Array,
    ) -> anyhow::Result<Array>;

    fn vision_hidden_size(&self) -> i32;
    fn supports_vision(&self) -> bool;
}
```

### gemma4.rs — New components

| Component | Description |
|-----------|-------------|
| `VisionConfig` | image_size, patch_size, num_hidden_layers, num_attention_heads, hidden_size, intermediate_size |
| `VisionPatchEmbed` | Conv2d patch embedding (patch_size x patch_size → hidden_size) |
| `VisionModel` | patch embed + RoPE + N encoder layers + pooler. Implements forward pass: patches → encoder → pooled vision features |
| `MultimodalEmbedder` | pre-projection LayerNorm + linear projection (vision_hidden_size → text_hidden_size) |
| `Gemma4MultimodalConfig` | text_config + vision_config + vision_soft_tokens_per_image |
| Image token expansion | boi/eoi tokens, expand `<image>` placeholder to `boi + vision_tokens * N + eoi` |

**Forward flow for multimodal:**
1. Tokenize text, expand image tokens to `boi + vision_soft_tokens_per_image + eoi`
2. Extract vision features: `pixel_values → VisionModel → MultimodalEmbedder`
3. Replace image token positions in hidden states with vision features
4. Continue normal text decoder forward pass

**Forward flow for text-only (pixel_values=None):**
- Same as current Gemma4 (no changes needed)

### mlx_backend.rs — Changes

```rust
pub struct MlxBackend {
    model: Box<dyn Model>,
    encoder_model: Option<Box<dyn EncoderModel>>,
    multimodal_model: Option<Box<dyn MultimodalModel>>,  // NEW
    // ...
}
```

`load()`:
- Detect multimodal Gemma4 (check for `vision_config` in config.json)
- Store as both `model` (for text-only) and `multimodal_model` (for vision+text)

**Verify:** Load `google/gemma-4-2b-it-mlx-4bit`, run:
1. Text-only generation (pixel_values=None)
2. Image+text generation (with pixel_values)

---

## Execution Order

| Phase | Model | Estimated Effort | Dependencies |
|-------|-------|-----------------|--------------|
| 1 | Qwen2 | 5 min | None |
| 2 | BERT | 2-3 hrs | None |
| 3 | Qwen MoE | 3-4 hrs | None |
| 4 | LFM2 MoE | 4-5 hrs | RecurrentCache (exists) |
| 5 | Gemma4 multimodal | 5-6 hrs | None |

Phases 1-4 are independent and can be done in any order. Phase 5 extends an existing model.
