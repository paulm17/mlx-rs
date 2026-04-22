# Gemma4 VLM Implementation Plan

## Overview

Add Gemma4 vision-language model support to mlx-rs. Three model variants targeted, implemented in order:

1. **gemma-4-E2B-it-UD-MLX-4bit** — smallest, most complex feature set (per-layer gating, KV sharing, double-wide MLP, ClippableLinear vision)
2. **gemma-4-31b-it-UD-MLX-4bit** — dense model, k_eq_v, ProportionalRoPE, 60 layers
3. **gemma-4-26b-a4b-it-4bit** — MoE model, 128 experts top-8, all layers MoE+dense MLP

## Model Variant Comparison

| Feature | 2B | 31B | 26B MoE |
|---------|-----|------|---------|
| hidden_size | 1536 | 5376 | 2816 |
| num_hidden_layers | 35 | 60 | 30 |
| num_attention_heads | 8 | 32 | 16 |
| num_key_value_heads | **1** | 16 | 8 |
| head_dim (sliding) | 256 | 256 | 256 |
| global_head_dim | 512 | 512 | 512 |
| num_global_kv_heads | None | 4 | 2 |
| attention_k_eq_v | **false** | **true** | **true** |
| num_kv_shared_layers | **20** | 0 | 0 |
| hidden_size_per_layer_input | **256** | 0 | 0 |
| use_double_wide_mlp | **true** | false | false |
| enable_moe_block | false | false | **true** |
| num_experts | — | — | **128** |
| top_k_experts | — | — | **8** |
| moe_intermediate_size | — | — | **704** |
| intermediate_size | 6144 | 21504 | 2112 |
| sliding_window | 512 | 1024 | 1024 |
| vision hidden_size | 768 | 1152 | 1152 |
| vision layers | 16 | 27 | 27 |
| vision use_clipped_linears | **true** | false | false |
| vision standardize | false | true | true |

## Deliverables

| # | Deliverable | File |
|---|------------|------|
| D1 | Gemma4 model implementation | `crates/mlx-models/src/gemma4.rs` |
| D2 | Module export | `crates/mlx-models/src/lib.rs` |
| D3 | VLM crate | `crates/mlx-vlm/` (new) |
| D4 | Workspace update | `Cargo.toml` |

---

## Phase 0: Infrastructure (Low Priority, Do First)

### D2: Update `crates/mlx-models/src/lib.rs`

Add `pub mod gemma4;` and re-export `Gemma4`, `Gemma4Config`.

```rust
pub mod gemma4;
pub use gemma4::{Gemma4, Gemma4Config};
```

### D4: Update workspace `Cargo.toml`

Add `crates/mlx-vlm` to workspace members list.

### D3: Create `crates/mlx-vlm/` crate

**Cargo.toml** dependencies:
- `mlx-core`, `mlx-nn`, `mlx-models` (path deps)
- `anyhow`, `serde`, `serde_json`
- `hf-hub`, `tokenizers`, `minijinja`
- `image` (for image loading/resize)

**Module structure:**
```
crates/mlx-vlm/src/
├── lib.rs          — pub mod loader; pub mod generate; pub mod processing;
├── loader.rs       — detect VLM arch, load multimodal config, construct model
├── generate.rs     — VLM generation pipeline (text + image tokens)
└── processing.rs   — image preprocessing (resize, rescale, normalize, patchify)
```

**loader.rs** responsibilities:
- Detect `model_type: "gemma4"` from config.json
- Load `Gemma4Config` (text_config + vision_config + top-level fields)
- Construct `Gemma4` model via `VarBuilder::from_dir()`
- Return model + tokenizer + processor

**generate.rs** responsibilities:
- `VlmGenerationPipeline` struct holding model + tokenizer
- `generate()` method: encode image → get vision features → inject into token stream → autoregressive decode
- Handle image token placeholders (boi/eoi tokens, expansion to vision_soft_tokens_per_image)

**processing.rs** responsibilities:
- `Gemma4ImageProcessor`: resize preserving aspect ratio, rescale to [0,1], channels-first
- Patch embedding computation (delegated to model's VisionPatchEmbedder)

---

## Phase 1: Shared Building Blocks (`gemma4.rs`)

All Gemma4 model code lives in a single file: `crates/mlx-models/src/gemma4.rs`.

Components shared across ALL three model variants.

### Config Structs

```rust
#[derive(Deserialize)]
pub struct Gemma4Config {
    pub text_config: Gemma4TextConfig,
    pub vision_config: Gemma4VisionConfig,
    pub audio_config: Option<Gemma4AudioConfig>,
    pub model_type: String,
    pub image_token_id: u32,
    pub audio_token_id: Option<u32>,
    pub boi_token_id: u32,
    pub eoi_token_id: u32,
    pub boa_token_id: Option<u32>,
    pub eoa_token_id: Option<u32>,
    pub vision_soft_tokens_per_image: usize,
}
```

**Gemma4TextConfig:**
```rust
#[derive(Deserialize)]
pub struct Gemma4TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub num_global_key_value_heads: Option<usize>,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub sliding_window: usize,
    pub sliding_window_pattern: usize,
    pub layer_types: Vec<String>,
    pub rope_parameters: HashMap<String, RopeParams>,
    pub final_logit_softcapping: Option<f32>,
    pub tie_word_embeddings: bool,
    pub rms_norm_eps: f64,
    pub hidden_activation: String,
    // MoE fields
    pub enable_moe_block: bool,
    pub num_experts: Option<usize>,
    pub top_k_experts: Option<usize>,
    pub moe_intermediate_size: Option<usize>,
    // Attention variants
    pub attention_k_eq_v: bool,
    // KV sharing
    pub num_kv_shared_layers: usize,
    // Per-layer input gating
    pub hidden_size_per_layer_input: usize,
    // Double-wide MLP
    pub use_double_wide_mlp: bool,
}
```

**Gemma4VisionConfig:**
```rust
#[derive(Deserialize)]
pub struct Gemma4VisionConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub position_embedding_size: usize,
    pub pooling_kernel_size: usize,
    pub default_output_length: usize,
    pub max_patches: usize,
    pub standardize: bool,
    pub rope_parameters: HashMap<String, RopeParams>,
    pub use_clipped_linears: bool,
}
```

### RMS Norm Variants

Three norm types used across all Gemma4 variants:

1. **RmsNormNoScale** — RMS norm with no learnable weight (eps only)
   - Used in: v_norm in attention, embedding_pre_projection_norm, router norm
   - `forward(x)`: normalize by RMS, no affine transform

2. **RmsNormZeroShift** — RMS norm with weight but no +1 offset (scale_shift=0.0)
   - Used in: post_attention_layernorm, post_feedforward_layernorm (all variants)
   - `forward(x)`: RMS normalize then multiply by weight (no +1 bias)

3. **GemmaRmsNorm** (reuse from gemma3.rs pattern) — weight stored as weight+1.0
   - Used in: input_layernorm, pre_feedforward_layernorm, final norm
   - `forward(x)`: `fast_rms_norm(x, weight_plus_one, eps)`

### ProportionalRoPE

Custom RoPE for full_attention layers. Standard RoPE for sliding_attention layers.

**ProportionalRoPE behavior** (from rope_utils.py):
- Computes frequencies relative to `global_head_dim` (512 for all variants)
- Only applies rotation to `partial_rotary_factor * global_head_dim` dims
  - e.g., 0.25 * 512 = 128 dims rotated, remaining 384 dims pass through
- Used by: full_attention layers in ALL variants
- rope_theta from rope_parameters["full_attention"]["rope_theta"] = 1000000

**Standard RoPE** for sliding_attention layers:
- rope_theta from rope_parameters["sliding_attention"]["rope_theta"] = 10000

**Implementation approach:**
```
fn proportional_rope(x: &Array, offset: usize, dims: usize, traditional: bool, theta: f32, partial_factor: f32) -> Array {
    let full_dims = x.shape()[/* head_dim axis */];
    let rotated_dims = (partial_factor * full_dims as f32) as usize;
    let (x_rot, x_pass) = split_along_head_dim(x, rotated_dims);
    let x_rotated = fast_rope(x_rot, offset, rotated_dims, traditional, theta, ...);
    concatenate(&[x_rotated, x_pass], head_dim_axis)
}
```

### Attention (Core — All Variants)

**Gemma4Attention** struct:
```
Fields:
- q_proj: Linear
- k_proj: Linear
- v_proj: Option<Linear>           // None when k_eq_v (uses k_proj)
- o_proj: Linear
- q_norm: GemmaRmsNorm
- k_norm: GemmaRmsNorm
- v_norm: RmsNormNoScale
- cache: KvCache
- n_heads: usize, n_kv_heads: usize
- head_dim: usize                   // 256 sliding, 512 global
- is_sliding: bool
- use_proportional_rope: bool       // true for full_attention layers
```

**Variant behavior:**

| Variant | k_eq_v | v_proj | num_kv_heads (full) | head_dim (full) |
|---------|--------|--------|---------------------|-----------------|
| 2B sliding | false | separate | 1 | 256 |
| 2B full | false | separate | 1 | 512 |
| 31B sliding | false | separate | 16 | 256 |
| 31B full | true | = k_proj | 4 | 512 |
| 26B sliding | false | separate | 8 | 256 |
| 26B full | true | = k_proj | 2 | 512 |

**Forward logic:**
1. Project q, k, v (if k_eq_v: v = k_proj(x))
2. Reshape to (batch, seq, heads, head_dim)
3. Apply q_norm, k_norm, v_norm
4. Apply RoPE (ProportionalRoPE for full, standard for sliding)
5. repeat_kv for GQA
6. Update KV cache
7. scaled_dot_product_attention with appropriate mask
8. Reshape and output projection

### MLP (GeGLU) — Core (All Variants)

**Gemma4Mlp** struct:
```
Fields:
- gate_proj: Linear
- up_proj: Linear
- down_proj: Linear
```

**Forward:** `down_proj(gelu_approx(gate_proj(x)) * up_proj(x))`

Note: `gelu_approx` = `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`. The existing `Activation::Gelu` uses `x * sigmoid(1.702 * x)`. For accuracy, implement the tanh-based approximation matching Python's `nn.gelu_approx`.

**Double-wide MLP (2B only):** When `use_double_wide_mlp=true`, intermediate_size is already doubled in config (6144). No code change needed — the Linear layers just have wider dimensions.

### DecoderLayer — Core (All Variants)

**Gemma4DecoderLayer** struct:
```
Fields:
- input_layernorm: GemmaRmsNorm
- post_attn_layernorm: RmsNormZeroShift
- pre_ffw_layernorm: GemmaRmsNorm
- post_ffw_layernorm: RmsNormZeroShift
- self_attn: Gemma4Attention
- mlp: Gemma4Mlp
- moe: Option<SparseMoeBlock>        // 26B only
- layer_scalar: Option<Array>
- is_sliding: bool
// Per-layer input gating (2B only):
- per_layer_input_gate: Option<Linear>
- per_layer_projection: Option<Linear>
- post_per_layer_input_norm: Option<Array>  // weight only, RmsNormNoScale
```

**Base forward (31B, 26B no gating):**
```
h = input_layernorm(x)
attn_out = self_attn(h, mask, cache)
h = post_attn_layernorm(attn_out)
h = x + h                           // residual
h2 = pre_ffw_layernorm(h)
mlp_out = mlp(h2)
if moe: mlp_out = mlp_out + moe(h2) // 26B only
h2 = post_ffw_layernorm(mlp_out)
h = h + h2                          // residual
if layer_scalar: h = h * layer_scalar
return h
```

**2B forward with per-layer gating:**
```
// Per-layer embedding gating (only when hidden_size_per_layer_input > 0)
if per_layer_input_gate:
    gate_input = embed_per_layer(layer_idx)  // from model-level embed_tokens_per_layer
    gate_input = per_layer_model_projection(gate_input)
    gate_input = per_layer_projection_norm(gate_input)
    gate_val = sigmoid(per_layer_input_gate(gate_input))
    proj_val = per_layer_projection(gate_input)
    x = x + gate_val * proj_val

// Then standard forward as above
```

### Gemma4TextModel — Core

```
Fields:
- embed_tokens: Embedding
- layers: Vec<Gemma4DecoderLayer>
- norm: GemmaRmsNorm
- embed_scale: f32                    // sqrt(hidden_size)
- final_logit_softcapping: Option<f32>
// Per-layer input gating (2B only):
- embed_tokens_per_layer: Option<Embedding>
- per_layer_model_projection: Option<Linear>
- per_layer_projection_norm: Option<Array>
// KV sharing (2B only):
- num_kv_shared_layers: usize
```

**Forward:**
1. `h = embed_tokens(tokens) * embed_scale`
2. Build masks: global_mask (causal) and sliding_window_mask
3. For each layer i:
   a. If KV sharing and i >= num_non_shared: use cache from layer (i - num_non_shared)
   b. Select mask based on layer_types[i]
   c. `h = layers[i](h, mask, cache)`
4. `h = norm(h)`
5. Return h

**Mask construction:**
- For sliding_attention layers: sliding window mask (attend within window only)
- For full_attention layers: standard causal mask
- Both masks handle cache offset for generation

**KV sharing (2B only, num_kv_shared_layers=20):**
- Layers 0-14: have their own KV caches
- Layers 15-34: share KV caches with layers 0-14
- Layer i (i >= 15) reads/writes cache of layer (i - 15)
- In Rust: pass `&mut cache[i]` for layers 0-14, and `&mut cache[i - 15]` for layers 15-34
- Total caches needed: 15 (not 35)

### LanguageModel (wrapper)

```
Fields:
- model: Gemma4TextModel
```

**Forward:** delegates to Gemma4TextModel, applies `final_logit_softcapping` via `logit_softcap()`, outputs via `embed_tokens.as_linear()`.

**logit_softcap(logits, cap):** `cap * tanh(logits / cap)`

---

## Phase 2: Vision Model Components (All Variants)

### ClippableLinear (2B only, but struct exists for all)

```
Fields:
- linear: Linear
- input_min: Option<Array>
- input_max: Option<Array>
- output_min: Option<Array>
- output_max: Option<Array>
```

**Forward:**
1. If input_min/max: `x = clamp(x, input_min, input_max)`
2. `x = linear(x)`
3. If output_min/max: `x = clamp(x, output_min, output_max)`

When `use_clipped_linears=false` (31B, 26B): no min/max weights loaded, forward is just `linear(x)`.

### VisionPatchEmbedder

```
Fields:
- input_proj: ClippableLinear  (3 * patch_size^2 → hidden_size)
- position_embedding_table: Embedding (position_embedding_size, hidden_size)
- patch_size: usize
```

**Forward:**
1. Reshape image to patches: (batch, num_patches, 3 * patch_size^2)
2. `features = input_proj(patches)`
3. Build position indices from valid patch positions
4. `pos_emb = position_embedding_table(position_indices)`
5. Return features + pos_emb

**Position embedding:** Uses one-hot encoding of spatial coordinates (row * width + col), then lookup in embedding table.

### VisionPooler

Reduces sequence length from max_patches to default_output_length (280) tokens.

```
Fields:
- pooling_kernel_size: usize
- default_output_length: usize
```

**Forward (avg_pool_by_positions):**
- Kernel-based average pooling over spatial positions
- Groups patches into pools of kernel_size, averages
- Outputs exactly default_output_length tokens per image

### VisionRMSNorm / VisionRMSNormNoScale

Full float32 versions of RMS norm (cast to f32, normalize, cast back):
- **VisionRMSNorm**: with weight, compute in f32, cast result back
- **VisionRMSNormNoScale**: without weight, compute in f32, cast result back

### apply_multidimensional_rope (2D RoPE for Vision)

Splits head_dim into two halves for spatial (height, width) dimensions:
1. Compute 2D frequency grid based on spatial positions and rope_theta
2. For each spatial dim, apply rotate_half independently to half of head_dim
3. Concatenate rotated halves

### VisionAttention

```
Fields:
- q_proj, k_proj, v_proj: ClippableLinear
- q_norm, k_norm: VisionRMSNorm
- v_norm: VisionRMSNormNoScale
- out_proj: ClippableLinear
```

**Forward:**
1. Project q, k, v via ClippableLinear
2. Apply norms
3. Apply 2D RoPE (apply_multidimensional_rope)
4. scaled_dot_product_attention (bidirectional mask from valid positions)
5. Output projection via ClippableLinear

### VisionMLP

Same GeGLU pattern as text MLP but using ClippableLinear:
```
Fields:
- gate_proj, up_proj, down_proj: ClippableLinear
```

### VisionEncoderLayer

```
Fields:
- input_layernorm: VisionRMSNorm
- post_attention_layernorm: VisionRMSNorm
- pre_feedforward_layernorm: VisionRMSNorm
- post_feedforward_layernorm: VisionRMSNorm
- self_attn: VisionAttention
- mlp: VisionMLP
```

### VisionModel

```
Fields:
- patch_embedder: VisionPatchEmbedder
- pooler: VisionPooler
- layers: Vec<VisionEncoderLayer>
- norm: VisionRMSNorm
- config: Gemma4VisionConfig
```

**Forward:**
1. `features, pos_emb = patch_embedder(images)`
2. `features = features + pos_emb`
3. Build bidirectional attention mask from valid positions
4. For each encoder layer: `features = layer(features, mask)`
5. `features = norm(features)`
6. `features = pooler(features, valid_positions)`
7. Strip padding, return features (shape: [batch, default_output_length, hidden_size])

---

## Phase 3: Multimodal Model (Top-Level)

### MultimodalEmbedder

```
Fields:
- embedding_pre_projection_norm: RmsNormNoScale
- embedding_projection: Linear (vision_hidden_size → text_hidden_size)
```

**Forward:** `embedding_projection(embedding_pre_projection_norm(features))`

### masked_scatter

Injects vision features into token embedding positions:
```
fn masked_scatter(inputs: &Array, source: &Array, mask: &Array) -> Array
```
1. Find positions where mask is true (image_token_id positions)
2. Expand those positions by vision_soft_tokens_per_image (280)
3. Replace token embeddings at those positions with vision features
4. Return modified embeddings

### Gemma4 (top-level Model)

```
pub struct Gemma4 {
    language_model: LanguageModel,
    vision_tower: VisionModel,
    embed_vision: MultimodalEmbedder,
    config: Gemma4Config,
}
```

**Public API:**
```rust
impl Gemma4 {
    pub fn new(vb: &VarBuilder, config: &Gemma4Config) -> Result<Self>;
    pub fn encode_image(&self, pixel_values: &Array) -> Result<Array>;
    pub fn forward_logits(&self, input_ids: &Array, pixel_values: Option<&Array>, cache: &mut [KvCache]) -> Result<Array>;
    pub fn forward_last_token_logits(&self, input_ids: &Array, pixel_values: Option<&Array>, cache: &mut [KvCache]) -> Result<Array>;
    pub fn clear_cache(cache: &mut [KvCache]);
}
```

**new() loading:**
- Load vision_tower from `vb.pp("vision_tower")`
- Load language_model from `vb.pp("language_model")`
- Load embed_vision from `vb.pp("embed_vision")`

**sanitize() — weight preprocessing:**
- MoE: stack individual expert weights into 3D tensors for gather_mm
- Vision: strip clipping params when use_clipped_linears=false
- Per-layer quantization: already handled by VarBuilder

---

## Phase 1B: 2B-Specific Implementation

**Model path:** `~/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it-UD-MLX-4bit/`

### 2B Unique Features

#### Per-Layer Input Gating (`hidden_size_per_layer_input=256`)

The 2B model has a learned per-layer input modulation mechanism:

**Model-level components (Gemma4TextModel):**
- `embed_tokens_per_layer`: Embedding(num_hidden_layers=35, hidden_size_per_layer_input=256)
- `per_layer_model_projection`: Linear(256 → 1536)
- `per_layer_projection_norm`: weight array for RmsNormNoScale

**Per-layer components (Gemma4DecoderLayer):**
- `per_layer_input_gate`: Linear(256 → 1536) — sigmoid gate
- `per_layer_projection`: Linear(256 → 1536) — projected value
- `post_per_layer_input_norm`: weight array for RmsNormNoScale

**Forward per layer:**
```
layer_emb = embed_tokens_per_layer(layer_index)         // [256]
layer_emb = per_layer_model_projection(layer_emb)        // [1536]
layer_emb = per_layer_projection_norm(layer_emb)         // [1536]

gate = sigmoid(per_layer_input_gate(layer_emb))          // [1536]
proj = per_layer_projection(layer_emb)                   // [1536]

x = x + gate * proj                                      // modulate input
```

#### KV Sharing (`num_kv_shared_layers=20`)

- Layers 0-14: own their KV caches (15 caches)
- Layers 15-34: reuse caches from layers 0-14
- Layer 15→cache[0], Layer 16→cache[1], ..., Layer 34→cache[14]
- Total KV caches needed: 15 (not 35)

**Implementation in Rust:**
```rust
fn cache_index(layer_idx: usize, num_non_shared: usize, num_kv_shared: usize) -> usize {
    if num_kv_shared > 0 && layer_idx >= num_non_shared {
        layer_idx - num_non_shared
    } else {
        layer_idx
    }
}
```

#### Double-Wide MLP (`use_double_wide_mlp=true`)

- intermediate_size=6144 (4x hidden_size=1536, which is 2x the normal 3072)
- No code change — config already has doubled intermediate_size
- Linear layers automatically get wider dimensions

#### Extreme GQA (`num_key_value_heads=1`)

- 8 query heads, 1 KV head
- repeat_kv factor = 8
- All 35 layers use num_kv_heads=1 (no separate global_kv_heads)

#### ClippableLinear for Vision (`use_clipped_linears=true`)

Vision attention and MLP projections have clamping:
- Weight keys: `self_attn.q_proj.linear.weight`, `self_attn.q_proj.input_min`, etc.
- MLP keys: `mlp.gate_proj.linear.weight`, `mlp.gate_proj.input_min`, etc.
- Must load ClippableLinear with sub-scoped `vb.pp("q_proj").pp("linear")` for the actual Linear

### 2B Weight Structure

```
language_model.model.embed_tokens.weight
language_model.model.embed_tokens_per_layer.weight
language_model.model.per_layer_model_projection.weight
language_model.model.per_layer_projection_norm.weight
language_model.model.layers.N.input_layernorm.weight
language_model.model.layers.N.post_attention_layernorm.weight
language_model.model.layers.N.pre_feedforward_layernorm.weight
language_model.model.layers.N.post_feedforward_layernorm.weight
language_model.model.layers.N.post_per_layer_input_norm.weight
language_model.model.layers.N.layer_scalar.weight
language_model.model.layers.N.self_attn.q_proj.weight
language_model.model.layers.N.self_attn.k_proj.weight
language_model.model.layers.N.self_attn.v_proj.weight
language_model.model.layers.N.self_attn.o_proj.weight
language_model.model.layers.N.self_attn.q_norm.weight
language_model.model.layers.N.self_attn.k_norm.weight
language_model.model.layers.N.self_attn.v_norm.weight
language_model.model.layers.N.mlp.gate_proj.weight
language_model.model.layers.N.mlp.up_proj.weight
language_model.model.layers.N.mlp.down_proj.weight
language_model.model.layers.N.per_layer_input_gate.weight
language_model.model.layers.N.per_layer_projection.weight
language_model.model.norm.weight
vision_tower.patch_embedder.input_proj.weight
vision_tower.patch_embedder.position_embedding_table.weight
vision_tower.encoder.layers.N.input_layernorm.weight
vision_tower.encoder.layers.N.pre_feedforward_layernorm.weight
vision_tower.encoder.layers.N.post_attention_layernorm.weight
vision_tower.encoder.layers.N.post_feedforward_layernorm.weight
vision_tower.encoder.layers.N.self_attn.q_proj.linear.weight
vision_tower.encoder.layers.N.self_attn.q_proj.input_min
vision_tower.encoder.layers.N.self_attn.q_proj.input_max
vision_tower.encoder.layers.N.self_attn.q_proj.output_min
vision_tower.encoder.layers.N.self_attn.q_proj.output_max
... (same for k_proj, v_proj, o_proj)
vision_tower.encoder.layers.N.self_attn.q_norm.weight
vision_tower.encoder.layers.N.self_attn.k_norm.weight
vision_tower.encoder.layers.N.mlp.gate_proj.linear.weight
vision_tower.encoder.layers.N.mlp.gate_proj.input_min/max/output_min/max
... (same for up_proj, down_proj)
embed_vision.embedding_pre_projection_norm.weight
embed_vision.embedding_projection.weight
```

### 2B Implementation Checklist

- [ ] Config structs handle all 2B fields (per_layer, kv_shared, double_wide, clipped)
- [ ] RmsNormNoScale, RmsNormZeroShift, GemmaRmsNorm
- [ ] Standard RoPE for sliding_attention (theta=10000)
- [ ] ProportionalRoPE for full_attention (partial_rotary_factor=0.25, theta=1000000)
- [ ] Attention with num_kv_heads=1, no k_eq_v, dual head_dim
- [ ] MLP GeGLU with intermediate_size=6144
- [ ] DecoderLayer with per-layer input gating fields
- [ ] Gemma4TextModel with embed_tokens_per_layer, per_layer_model_projection, per_layer_projection_norm
- [ ] KV sharing: 15 caches, layers 15-34 → cache[i-15]
- [ ] ClippableLinear struct, VisionAttention/VisionMLP using it
- [ ] VisionPatchEmbedder, VisionPooler, VisionModel
- [ ] MultimodalEmbedder, masked_scatter, Gemma4 top-level
- [ ] 2D RoPE (apply_multidimensional_rope) for vision

---

## Phase 2B: 31B-Specific Implementation

**Model path:** `~/.cache/huggingface/hub/models--unsloth--gemma-4-31b-it-UD-MLX-4bit/`

### 31B Unique Features

#### k_eq_v Attention (`attention_k_eq_v=true`)

For full_attention layers ONLY:
- v_proj not loaded as separate weight
- `v = k_proj(x)` — same projection as k
- k and v have separate norms (k_norm: GemmaRmsNorm, v_norm: RmsNormNoScale)
- num_global_key_value_heads=4 (separate from num_key_value_heads=16)

For sliding_attention layers:
- Standard separate k_proj and v_proj
- num_key_value_heads=16

#### Large Model (60 layers)

- 48 sliding_attention layers (4 per pattern × 12 patterns)
- 12 full_attention layers (1 per pattern × 12 patterns)
- Memory management critical — 60 caches
- Per-layer quantization: attn at 8-bit, MLP at 4-bit (varies), embed at 6-bit

#### No Per-Layer Gating, No KV Sharing, No Double-Wide MLP

Simpler than 2B:
- `hidden_size_per_layer_input=0` → no per-layer embedding gating
- `num_kv_shared_layers=0` → each layer has its own cache (60 caches)
- `use_double_wide_mlp=false` → standard intermediate_size=21504

#### Vision (standardize=true, use_clipped_linears=false)

- VisionRMSNorm applies input standardization (subtract mean, divide by std before norm)
- ClippableLinear loads without min/max → behaves as regular Linear

### 31B Implementation Checklist

- [ ] k_eq_v: conditionally skip v_proj load for full_attention layers
- [ ] Dual head_dim: 256 for sliding, 512 for full (already in shared Attention)
- [ ] ProportionalRoPE for full attention (shared, already implemented)
- [ ] 60-layer cache management
- [ ] Large config.json with 3600+ lines of per-layer quantization overrides
- [ ] Vision standardize=true mode

---

## Phase 3B: 26B MoE-Specific Implementation

**Model path:** `~/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/`

### 26B Unique Features

#### MoE: Every Layer is MoE + Dense MLP

ALL 30 layers have both:
1. Dense MLP (intermediate_size=2112)
2. Sparse MoE (128 experts, top-8, moe_intermediate_size=704)

Combined: `output = dense_mlp(x) + moe(x)`

#### Router (per layer)

```rust
struct Router {
    norm: RmsNormNoScale,
    scale: Array,
    proj: Linear,
    top_k: usize,  // 8
}
```

**Forward:**
1. `h = norm(x)` — RmsNormNoScale
2. `h = h * scale` — router scalar
3. `logits = proj(h)` — Linear(hidden_size → num_experts)
4. `top_k_indices = argpartition(logits, k=top_k)` — get top-8 expert indices
5. `top_k_logits = take_along_axis(logits, top_k_indices)`
6. `weights = softmax(top_k_logits)`
7. Return (weights, top_k_indices)

#### SwitchLinear (per expert weight)

Following existing `qwen3_moe.rs` patterns:

```rust
struct SwitchLinear {
    weight: Array,  // [num_experts, out_features, in_features]
    bias: Option<Array>,
}
```

- Uses `gather_mm` for float, `gather_qmm` for quantized
- Forward: `gather_mm(x, weight, indices)`

#### SwitchGlu (GeGLU per expert)

```rust
struct SwitchGlu {
    gate_proj: SwitchLinear,   // [128, moe_intermediate_size, hidden_size]
    up_proj: SwitchLinear,     // [128, moe_intermediate_size, hidden_size]
    down_proj: SwitchLinear,   // [128, hidden_size, moe_intermediate_size]
}
```

**Forward:**
1. `gate = gate_proj(x, indices)` — gather to selected experts, project
2. `up = up_proj(x, indices)`
3. `gate = gelu_approx(gate)`
4. `output = down_proj(gate * up, indices)` — combine and project back

#### SparseMoeBlock

```rust
struct SparseMoeBlock {
    router: Router,
    switch_glu: SwitchGlu,
}
```

**Forward:**
1. `(weights, indices) = router(x)`
2. `expert_out = switch_glu(x, indices)`
3. `output = expert_out * weights` — weighted combination
4. Sum across selected experts
5. Return output

#### sanitize() for MoE

Python's sanitize() splits stacked expert weights:
- Individual expert weights like `block.sparse_mlp.gate_proj.expert_0.weight` through `expert_127.weight`
- Must stack into 3D tensor `[128, out_features, in_features]` for gather_mm
- Handle quantized weights: stack quantized params appropriately

### 26B Implementation Checklist

- [ ] Router struct with norm → scale → project → top-k
- [ ] SwitchLinear following qwen3_moe.rs gather_mm pattern
- [ ] SwitchGlu with GeGLU per expert
- [ ] SparseMoeBlock combining router + switch_glu
- [ ] DecoderLayer.mo = Some(SparseMoeBlock) when enable_moe_block
- [ ] Layer forward: `output = dense_mlp_out + moe_out`
- [ ] sanitize() to stack individual expert weights into 3D tensors
- [ ] k_eq_v for full_attention layers (shared with 31B)

---

## Implementation Order

| Step | What | Variant | Files |
|------|------|---------|-------|
| 1 | Update lib.rs, workspace Cargo.toml | — | lib.rs, Cargo.toml |
| 2 | Create mlx-vlm crate skeleton | — | crates/mlx-vlm/ |
| 3 | Config structs | All | gemma4.rs |
| 4 | RMS norm variants (3 types) | All | gemma4.rs |
| 5 | ClippableLinear | All (2B uses it) | gemma4.rs |
| 6 | Standard RoPE + ProportionalRoPE | All | gemma4.rs |
| 7 | Attention (k_eq_v, dual head_dim, GQA) | All | gemma4.rs |
| 8 | MLP (GeGLU) | All | gemma4.rs |
| 9 | DecoderLayer (post-norm, per-layer gating) | All | gemma4.rs |
| 10 | Gemma4TextModel (KV sharing, per-layer embed) | All | gemma4.rs |
| 11 | LanguageModel (logit softcapping) | All | gemma4.rs |
| 12 | 2D RoPE, VisionAttention, VisionMLP | All | gemma4.rs |
| 13 | VisionPatchEmbedder, VisionPooler, VisionModel | All | gemma4.rs |
| 14 | MultimodalEmbedder, masked_scatter, Gemma4 top-level | All | gemma4.rs |
| 15 | Router, SwitchLinear, SwitchGlu, SparseMoeBlock | 26B | gemma4.rs |
| 16 | sanitize() for MoE weight stacking | 26B | gemma4.rs |
| 17 | VLM loader + generation pipeline | All | crates/mlx-vlm/ |
| 18 | Image processor | All | crates/mlx-vlm/ |
| 19 | Build, test 2B | 2B | All |
| 20 | Test 31B | 31B | All |
| 21 | Test 26B MoE | 26B | All |

## Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| ProportionalRoPE correctness | Port from Python rope_utils.py, test with known inputs before full model |
| k_eq_v shared k/v projection | Verify weight loading matches Python; separate norms for k vs v |
| Per-layer input gating (2B) | Unique to 2B, test independently; gate sigmoid must be element-wise |
| KV sharing (2B) | Careful cache indexing; 15 caches for 35 layers |
| MoE gather_mm with quantized weights | Follow existing qwen3_moe.rs patterns exactly |
| No RotatingKVCache in Rust | Use standard KvCache + manual sliding_window_mask; add RotatingKVCache later |
| 2D RoPE for vision | Port apply_multidimensional_rope carefully, test with grid inputs |
| ClippableLinear weight scoping | Vision weights nested: `q_proj.linear.weight` not `q_proj.weight` |
| Large 31B config (3600+ lines) | Use HashMap<String,Value> for quantization overrides; don't try to enumerate |
| gelu_approx accuracy | Implement tanh-based approximation matching Python, not sigmoid-based |

## Files Changed Summary

| File | Action |
|------|--------|
| `crates/mlx-models/src/gemma4.rs` | CREATE — all Gemma4 model code |
| `crates/mlx-models/src/lib.rs` | EDIT — add `pub mod gemma4;` + re-exports |
| `crates/mlx-vlm/Cargo.toml` | CREATE — new crate manifest |
| `crates/mlx-vlm/src/lib.rs` | CREATE — module structure |
| `crates/mlx-vlm/src/loader.rs` | CREATE — VLM model loading |
| `crates/mlx-vlm/src/generate.rs` | CREATE — VLM generation pipeline |
| `crates/mlx-vlm/src/processing.rs` | CREATE — image preprocessing |
| `Cargo.toml` (workspace root) | EDIT — add mlx-vlm to members |
