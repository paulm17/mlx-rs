#!/usr/bin/env python3
import sys
from pathlib import Path
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

MODEL = "unsloth/gemma-4-E2B-it-UD-MLX-4bit"
IMAGE = "./red_square.png"
PROMPT = "Describe this image."

def main():
    image_path = Path(IMAGE)
    if not image_path.exists():
        print(f"Error: {IMAGE} not found in current directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {MODEL}")
    model, processor = load(MODEL)
    config = load_config(MODEL)

    prompt = apply_chat_template(processor, config, PROMPT, num_images=1)

    print("Running inference...\n")
    output = generate(
        model,
        processor,
        prompt,
        image=str(image_path),
        max_tokens=512,
        verbose=False,
    )

    print("Response:")
    print(output)

if __name__ == "__main__":
    main()