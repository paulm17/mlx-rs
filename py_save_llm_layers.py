#!/usr/bin/env python3
"""Save per-LLM-layer hidden states by patching Gemma4TextModel.__call__.
Uses the SAME input_ids/pixel_values as compare_all.rs."""
import mlx.core as mx
import numpy as np
from mlx_vlm import load
from mlx_vlm.models.gemma4.language import Gemma4TextModel
import safetensors.numpy as stn

MODEL_ID = "unsloth/gemma-4-E2B-it-UD-MLX-4bit"
OUT_DIR = "/tmp"

model, processor = load(MODEL_ID)

# Load the SAME inputs as compare_all.rs
input_ids = mx.array(stn.load_file(f"{OUT_DIR}/python_input_ids.safetensors")["input_ids"]).astype(mx.int32)
pixel_values = mx.array(stn.load_file(f"{OUT_DIR}/python_pixel_values.safetensors")["pixel_values"])

captured = {}
original_call = Gemma4TextModel.__call__

def patched_call(self, inputs=None, inputs_embeds=None, mask=None, cache=None,
                 per_layer_inputs=None, **kwargs):
    if inputs_embeds is None:
        h = self.embed_tokens(inputs)
        h = h * self.embed_scale
    else:
        h = inputs_embeds

    captured["layer_input"] = np.array(h.astype(mx.float32))

    if self.hidden_size_per_layer_input:
        if inputs is not None and per_layer_inputs is None:
            per_layer_inputs = self.get_per_layer_inputs(inputs)
        elif per_layer_inputs is not None:
            target_len = h.shape[1]
            if per_layer_inputs.shape[1] != target_len:
                cache_offset = 0
                for c in (cache or []):
                    if c is not None and hasattr(c, "offset"):
                        cache_offset = int(c.offset)
                        break
                max_start = max(per_layer_inputs.shape[1] - target_len, 0)
                start = min(cache_offset, max_start)
                per_layer_inputs = per_layer_inputs[:, start : start + target_len]
        if per_layer_inputs is not None or inputs is not None:
            per_layer_inputs = self.project_per_layer_inputs(h, per_layer_inputs)
        captured["per_layer_inputs"] = np.array(per_layer_inputs.astype(mx.float32))

    if cache is None:
        cache = [None] * self.first_kv_shared_layer_idx

    if mask is None:
        from mlx_vlm.models.gemma4.language import create_attention_mask
        global_mask = create_attention_mask(h,
            cache[self.first_full_cache_idx] if self.first_full_cache_idx < len(cache) else None)
        sliding_window_mask = create_attention_mask(h,
            cache[self.first_sliding_cache_idx] if self.first_sliding_cache_idx < len(cache) else None,
            window_size=self.window_size)

    for i, layer in enumerate(self.layers):
        c = cache[self.layer_idx_to_cache_idx[i]]
        is_global = layer.layer_type == "full_attention"
        local_mask = mask
        if mask is None and is_global:
            local_mask = global_mask
        elif mask is None:
            local_mask = sliding_window_mask

        pl_input = None
        if per_layer_inputs is not None:
            pl_input = per_layer_inputs[:, :, i, :]

        h = layer(h, local_mask, c, per_layer_input=pl_input)
        captured[f"layer_{i}_output"] = np.array(h.astype(mx.float32))

    captured["norm_weight"] = np.array(self.norm.weight.astype(mx.float32))
    h = self.norm(h)
    captured["final_norm_output"] = np.array(h.astype(mx.float32))
    return h

Gemma4TextModel.__call__ = patched_call
_ = model(input_ids, pixel_values=pixel_values)
Gemma4TextModel.__call__ = original_call

all_tensors = {}
for name, arr in captured.items():
    all_tensors[name] = arr
    print(f"Captured {name}: shape={arr.shape}")

stn.save_file(all_tensors, f"{OUT_DIR}/python_llm_layers.safetensors")
print(f"Saved {len(captured)} tensors")
