import numpy as np
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.utils import load_config, load_image_processor
from PIL import Image
import json
import os

model_path = "unsloth/gemma-4-E2B-it-UD-MLX-4bit"
model, processor = load(model_path, processor_config={"trust_remote_code": True})
config = load_config(model_path)

img = Image.new("RGB", (100, 100), color="red")
img.save("red_square.png")

messages = [{"role": "user", "content": "<img> Describe this image."}]
prompt = processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

from mlx_vlm.tokenizer_utils import load_image_processor
image_processor = load_image_processor(model_path, trust_remote_code=True)

pixel_values = image_processor([img], return_tensors="mlx")["pixel_values"]
print("pixel_values shape:", pixel_values.shape)
print("pixel_values dtype:", pixel_values.dtype)
print("pixel_values range:", mx.min(pixel_values).item(), mx.max(pixel_values).item())

# Save pixel values
np.save("debug_pixel_values.npy", np.array(pixel_values))

# Now run through vision tower manually and save intermediates
vision_tower = model.vision_tower

# Patch embedder
patch_embed = vision_tower.patch_embedder(pixel_values)
print("patch_embed shape:", patch_embed.shape)
np.save("debug_patch_embed.npy", np.array(patch_embed))

# Position embeddings
pos_embed = vision_tower.position_embedding(patch_embed)
print("pos_embed shape:", pos_embed.shape)
np.save("debug_pos_embed.npy", np.array(pos_embed))

# After adding pos embed
after_pos = patch_embed + pos_embed
np.save("debug_after_pos.npy", np.array(after_pos))

# Through encoder
encoder_out = vision_tower.encoder(after_pos)
print("encoder_out shape:", encoder_out.shape)
np.save("debug_encoder_out.npy", np.array(encoder_out))

# Pooler
pooled = vision_tower.pooler(encoder_out)
print("pooled shape:", pooled.shape)
np.save("debug_pooled.npy", np.array(pooled))

# Projection
projected = model.vision_embeddings(pooled)
print("projected shape:", projected.shape)
np.save("debug_projected.npy", np.array(projected))

# Full vision model output
full_out = model.encode_image(pixel_values)
print("full_out shape:", full_out.shape)
np.save("debug_full_vision.npy", np.array(full_out))

print("\nSaved all intermediates to debug_*.npy")
