#!/usr/bin/env python3
"""Python equivalent of simple_vision_test.rs."""
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.models.gemma4.processing_gemma4 import Gemma4ImageProcessor
from mlx_vlm.generate import generate
from PIL import Image

MODEL_ID = "unsloth/gemma-4-E2B-it-UD-MLX-4bit"

model, processor = load(MODEL_ID)
img = Image.open("./red_square.png").convert("RGB")
img_processor = Gemma4ImageProcessor()
tokenizer = processor.tokenizer

num_img = model.config.vision_soft_tokens_per_image

def run(label, image=None):
    if image is not None:
        img_ph = "<|image|>" * num_img
        px = mx.array(img_processor.preprocess([img])[0]["pixel_values"])
    else:
        img_ph = ""
        px = None

    user_msg = f"{img_ph}Describe this image in one sentence."
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        add_generation_prompt=True, tokenize=False,
    )
    input_ids = mx.array([tokenizer.encode(prompt)])

    kwargs = {"input_ids": input_ids, "max_tokens": 32, "temp": 0.0}
    if px is not None:
        kwargs["pixel_values"] = px
    result = generate(model, processor, prompt="ignored", **kwargs)
    print(f"{label}: {result.text}")

run("WITH IMAGE   ", img)
run("WITHOUT IMAGE")
