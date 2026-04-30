#!/usr/bin/env python3
"""
test_gemma4.py

Loads unsloth/gemma-4-E2B-it-UD-MLX-4bit, processes ./red_square.png,
and captures 7 intermediate tensors during the forward pass, saving each
as a float32 safetensors file to /tmp/python_*.safetensors.

Tensor capture list:
  1. input_ids              - raw token ids from processor
  2. pixel_values           - image tensor from processor
  3. text_embeddings        - embed_tokens(input_ids) * embed_scale  (pre-vision scatter)
  4. vision_tower_output    - output of model.vision_tower(pixel_values)
  5. vision_embeddings      - projected vision features (after multi-modal projector)
  6. combined_embeddings    - text_embeddings with vision patches scattered in
  7. final_logits           - full-model output logits

Run with:
  .venv/bin/python test_gemma4.py
"""

import sys
import os
import numpy as np
from PIL import Image
import mlx.core as mx

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_tensor(name: str, arr: mx.array) -> None:
    """Cast to float32, eval, print stats, save to /tmp/python_<name>.safetensors."""
    arr = arr.astype(mx.float32)
    mx.eval(arr)
    np_arr = np.array(arr)
    mn, mx_val, mean = float(np_arr.min()), float(np_arr.max()), float(np_arr.mean())
    has_nan = bool(np.isnan(np_arr).any())
    has_inf = bool(np.isinf(np_arr).any())
    print(f"  [{name}]  shape={arr.shape}  min={mn:.6f}  max={mx_val:.6f}  "
          f"mean={mean:.6f}  nan={has_nan}  inf={has_inf}")
    path = f"/tmp/python_{name}.safetensors"
    mx.save_safetensors(path, {name: arr})
    print(f"    -> saved {path}")


def find_attr(obj, *candidates):
    """Return the first attribute name from candidates that exists on obj, or None."""
    for c in candidates:
        if hasattr(obj, c):
            return getattr(obj, c)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

MODEL_ID = "unsloth/gemma-4-E2B-it-UD-MLX-4bit"
IMAGE_PATH = "./red_square.png"

print(f"Loading model: {MODEL_ID}")
from mlx_vlm import load
model, processor = load(MODEL_ID)
print("Model loaded.\n")

# ── Inspect model structure so we can find the right attribute paths ──────────
print("=== Model type hierarchy ===")
print(f"  model type : {type(model)}")
lm = find_attr(model, "language_model", "model")
if lm is None:
    raise RuntimeError("Cannot find language_model / model on loaded model object")
print(f"  language_model type : {type(lm)}")
inner = find_attr(lm, "model", "transformer", "decoder")
if inner is None:
    raise RuntimeError("Cannot find inner language model on language_model")
print(f"  inner model type    : {type(inner)}")
print()

# ── 1. Processor → input_ids + pixel_values ──────────────────────────────────
print("=== Step 1: Running processor ===")
image = Image.open(IMAGE_PATH).convert("RGB")

tokenizer = getattr(processor, "tokenizer", processor)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# apply_chat_template returns a string prompt
prompt_str = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
)

# Process image + text together
inputs = processor(
    text=prompt_str,
    images=image,
    return_tensors="np",
)

# Pull out the arrays we need
input_ids_np = inputs["input_ids"]           # (1, seq_len)
pixel_values_np = inputs.get("pixel_values") # may be None for text-only fallback

input_ids = mx.array(input_ids_np)

print(f"  input_ids shape : {input_ids.shape}")
save_tensor("input_ids", input_ids)

if pixel_values_np is not None:
    pixel_values = mx.array(pixel_values_np)
    print(f"  pixel_values shape : {pixel_values.shape}")
    save_tensor("pixel_values", pixel_values)
else:
    print("  WARNING: processor returned no pixel_values – image may not have been tokenised.")
    print("  Attempting to re-process with explicit image list …")
    inputs2 = processor(
        images=[image],
        text=prompt_str,
        return_tensors="np",
    )
    pixel_values_np = inputs2.get("pixel_values")
    if pixel_values_np is None:
        raise RuntimeError("Still no pixel_values after retry – check model/processor version.")
    pixel_values = mx.array(pixel_values_np)
    print(f"  pixel_values shape : {pixel_values.shape}")
    save_tensor("pixel_values", pixel_values)

print()

# ── 2. text_embeddings = embed_tokens(input_ids) * embed_scale ───────────────
# Attribute names differ between mlx-vlm versions; try common ones.
print("=== Step 2: text_embeddings (pre-scatter) ===")

embed_tokens = find_attr(inner, "embed_tokens", "embedding", "tok_embeddings")
if embed_tokens is None:
    raise RuntimeError(
        "Cannot find embed_tokens on inner model. "
        f"Available attrs: {[a for a in dir(inner) if not a.startswith('_')]}"
    )

# embed_scale lives on the language model or inner model as a scalar
embed_scale_raw = find_attr(inner, "embed_scale", "embedding_scale")
if embed_scale_raw is None:
    embed_scale_raw = find_attr(lm, "embed_scale", "embedding_scale")
if embed_scale_raw is None:
    # Try config
    cfg = getattr(model, "config", None)
    if cfg is not None:
        embed_scale_raw = getattr(cfg, "embed_scale", None)
if embed_scale_raw is None:
    print("  WARNING: embed_scale not found – using 1.0")
    embed_scale = 1.0
else:
    embed_scale = embed_scale_raw
    print(f"  embed_scale = {embed_scale}")

raw_embeds = embed_tokens(input_ids)    # (1, seq_len, hidden_dim)
text_embeddings = raw_embeds * embed_scale
mx.eval(text_embeddings)
save_tensor("text_embeddings", text_embeddings)
print()

# ── 3. vision_tower_output ────────────────────────────────────────────────────
print("=== Step 3: vision_tower_output ===")

vision_tower = find_attr(model, "vision_tower", "vision_model", "visual")
if vision_tower is None:
    raise RuntimeError(
        "Cannot find vision_tower on model. "
        f"Available attrs: {[a for a in dir(model) if not a.startswith('_')]}"
    )

# pixel_values shape from processor is typically (1, num_patches, C, H, W) or (B, C, H, W)
# mlx-vlm gemma4 vision tower expects (batch, channels, height, width)
pv_in = pixel_values
if pv_in.ndim == 5:
    # (1, num_patches, C, H, W) → collapse batch*patches
    b, n, c, h, w = pv_in.shape
    pv_in = pv_in.reshape(b * n, c, h, w)

vision_tower_output = vision_tower(pv_in)
# The vision tower may return a named tuple / object – grab .last_hidden_state or index [0]
if hasattr(vision_tower_output, "last_hidden_state"):
    vision_tower_output = vision_tower_output.last_hidden_state
elif isinstance(vision_tower_output, (tuple, list)):
    vision_tower_output = vision_tower_output[0]

mx.eval(vision_tower_output)
save_tensor("vision_tower_output", vision_tower_output)
print()

# ── 4. vision_embeddings = projection of vision_tower_output ─────────────────
print("=== Step 4: vision_embeddings (after projector) ===")

# Common names for the multi-modal projector in mlx-vlm
mm_proj = find_attr(
    model,
    "multi_modal_projector",
    "mm_projector",
    "vision_projector",
    "embed_vision",
    "visual_projector",
)
if mm_proj is None:
    # Try on language_model
    mm_proj = find_attr(
        lm,
        "multi_modal_projector",
        "mm_projector",
        "vision_projector",
        "embed_vision",
    )

if mm_proj is not None:
    vision_embeddings = mm_proj(vision_tower_output)
    if hasattr(vision_embeddings, "last_hidden_state"):
        vision_embeddings = vision_embeddings.last_hidden_state
    elif isinstance(vision_embeddings, (tuple, list)):
        vision_embeddings = vision_embeddings[0]
    mx.eval(vision_embeddings)
    save_tensor("vision_embeddings", vision_embeddings)
else:
    print("  WARNING: cannot find mm_projector – attempting get_input_embeddings path instead")
    # Fall back: use model.get_input_embeddings which internally does vision+projection+scatter
    # We'll call it and save its output as combined_embeddings directly
    vision_embeddings = None

print()

# ── 5. combined_embeddings via get_input_embeddings ──────────────────────────
print("=== Step 5: combined_embeddings (text + vision scattered) ===")

# mlx-vlm VLMs expose get_input_embeddings(input_ids, pixel_values, ...)
# Signature varies by version – try both
combined_embeddings = None
get_input_embeds_fn = getattr(model, "get_input_embeddings", None)

if get_input_embeds_fn is not None:
    try:
        combined_embeddings = get_input_embeds_fn(
            input_ids=input_ids,
            pixel_values=pixel_values,
        )
    except TypeError:
        try:
            combined_embeddings = get_input_embeds_fn(input_ids, pixel_values)
        except Exception as e:
            print(f"  get_input_embeddings failed: {e}")

if combined_embeddings is None:
    # Manual scatter: find image token id from config
    cfg = getattr(model, "config", None)
    img_token_id = None
    if cfg is not None:
        img_token_id = getattr(cfg, "image_token_index", None)
        if img_token_id is None:
            img_token_id = getattr(cfg, "image_token_id", None)
    if img_token_id is None:
        # Try to get it from the tokenizer/processor
        try:
            img_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        except Exception:
            pass
    if img_token_id is None:
        print("  WARNING: image_token_index unknown, saving text_embeddings as combined")
        combined_embeddings = text_embeddings
    elif vision_embeddings is not None:
        # Find positions in input_ids where image tokens occur
        ids_np = np.array(input_ids[0].tolist())
        image_positions = np.where(ids_np == int(img_token_id))[0]
        print(f"  image_token_index={img_token_id}, positions count={len(image_positions)}")
        # Scatter vision embeddings into text embeddings at those positions
        combined_np = np.array(text_embeddings)  # (1, seq_len, hidden)
        vis_np = np.array(vision_embeddings)      # (num_patches, hidden) or (1, num_patches, hidden)
        if vis_np.ndim == 3:
            vis_np = vis_np[0]  # → (num_patches, hidden)
        n_img_pos = min(len(image_positions), vis_np.shape[0])
        combined_np[0, image_positions[:n_img_pos]] = vis_np[:n_img_pos]
        combined_embeddings = mx.array(combined_np)
    else:
        combined_embeddings = text_embeddings

if hasattr(combined_embeddings, "last_hidden_state"):
    combined_embeddings = combined_embeddings.last_hidden_state
elif hasattr(combined_embeddings, "inputs_embeds"):
    combined_embeddings = combined_embeddings.inputs_embeds
elif isinstance(combined_embeddings, (tuple, list)):
    combined_embeddings = combined_embeddings[0]

mx.eval(combined_embeddings)
save_tensor("combined_embeddings", combined_embeddings)
print()

# ── 6. final_logits via full model forward pass ───────────────────────────────
print("=== Step 6: final_logits (full model forward pass) ===")

# Use model.__call__ with the combined embeddings as inputs_embeds
# mlx-vlm models typically accept inputs_embeds kwarg on the language model
try:
    outputs = lm(
        input_ids=None,
        inputs_embeds=combined_embeddings,
    )
except TypeError:
    # Try passing positionally or with different kwarg name
    try:
        outputs = lm(inputs_embeds=combined_embeddings)
    except TypeError:
        # Last resort: use the full model with input_ids + pixel_values
        print("  inputs_embeds path failed, falling back to model(input_ids, pixel_values)")
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
        )

if hasattr(outputs, "logits"):
    final_logits = outputs.logits
elif isinstance(outputs, (tuple, list)):
    final_logits = outputs[0]
else:
    final_logits = outputs

mx.eval(final_logits)
save_tensor("final_logits", final_logits)

print()
print("=== All tensors saved successfully ===")
print("Files written:")
names = [
    "input_ids", "pixel_values", "text_embeddings",
    "vision_tower_output", "vision_embeddings",
    "combined_embeddings", "final_logits",
]
for n in names:
    p = f"/tmp/python_{n}.safetensors"
    exists = os.path.exists(p)
    size = os.path.getsize(p) if exists else 0
    print(f"  {p}  exists={exists}  size={size:,} bytes")