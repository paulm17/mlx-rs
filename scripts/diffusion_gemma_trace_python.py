#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_vlm.utils import load
from mlx_vlm.models.diffusion_gemma.language import _cache_offset, _cache_state


def encode_prompt(processor, prompt: str, thinking: bool):
    tokenizer = getattr(processor, "tokenizer", processor)
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
    else:
        prompt_text = prompt
    encoded = tokenizer.encode(prompt_text)
    ids = encoded.ids if hasattr(encoded, "ids") else encoded
    return prompt_text, [int(x) for x in ids]


def softcap(logits, cap: float):
    return mx.tanh(logits.astype(mx.float32) / cap) * cap


def entropy_accept_mask(logits, entropy_bound: float):
    log_probs = logits.astype(mx.float32) - mx.logsumexp(
        logits.astype(mx.float32), axis=-1, keepdims=True
    )
    probs = mx.exp(log_probs)
    entropy = -mx.sum(probs * log_probs, axis=-1)
    entropy_np = np.array(entropy)
    mask = np.zeros(entropy_np.shape, dtype=np.int32)
    for b in range(entropy_np.shape[0]):
        order = np.argsort(entropy_np[b])
        cumulative = 0.0
        for rank, idx in enumerate(order):
            value = float(entropy_np[b, idx])
            cumulative += value
            if cumulative - value <= entropy_bound or rank == 0:
                mask[b, idx] = 1
            else:
                break
    return mx.array(mask)


def save_trace(out_dir: Path, tensors: dict, metadata: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {name: value.astype(mx.float32) for name, value in tensors.items()}
    mx.eval(*tensors.values())
    mx.save_safetensors(str(out_dir / "python_trace.safetensors"), tensors)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Dump DiffusionGemma Python reference tensors")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="What is Rust?")
    ap.add_argument("--out-dir", default="/tmp/diffusion_gemma_trace")
    ap.add_argument("--canvas-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cur-step", type=int, default=48)
    ap.add_argument("--thinking", action="store_true")
    args = ap.parse_args()

    model, processor = load(args.model, strict=False)
    model.eval()
    prompt_text, prompt_ids = encode_prompt(processor, args.prompt, args.thinking)
    input_ids = mx.array([prompt_ids], dtype=mx.int32)

    rng = np.random.default_rng(args.seed)
    vocab_size = int(model.config.text_config.vocab_size)
    canvas_np = rng.integers(0, vocab_size, size=(1, args.canvas_len), dtype=np.int32)
    canvas_ids = mx.array(canvas_np)

    traces = {
        "input_ids": input_ids,
        "canvas_ids": canvas_ids,
    }

    encoder = model.model.encoder
    decoder = model.model.decoder
    cache = encoder.make_cache()

    h = encoder._embed_inputs(input_ids)
    traces["encoder.embeddings"] = h
    masks = encoder._make_encoder_masks(h, cache)
    for i, (layer, c, mask) in enumerate(zip(decoder.layers, cache, masks)):
        if mask is not None and not isinstance(mask, str):
            traces[f"encoder.layer_{i}.mask"] = mask.astype(mx.float32)
        h = layer(
            h,
            mask,
            c,
            decoder=False,
            layer_scalar=encoder.language_model.layers[i].layer_scalar,
        )
        traces[f"encoder.layer_{i}.output"] = h
        state = _cache_state(c)
        if state is not None:
            traces[f"encoder.layer_{i}.cache_k"] = state[0]
            traces[f"encoder.layer_{i}.cache_v"] = state[1]
    traces["encoder.final_hidden"] = decoder.norm(h)

    canvas_embeddings = decoder.embed_tokens(canvas_ids) * decoder.embed_scale
    traces["decoder.canvas_embeddings_raw"] = canvas_embeddings
    zero_signal = mx.zeros_like(canvas_embeddings)
    traces["decoder.self_conditioning_signal"] = zero_signal
    h = decoder.self_conditioning(canvas_embeddings, zero_signal)
    traces["decoder.canvas_embeddings_conditioned"] = h
    traces["decoder.embeddings"] = h

    masks = decoder._make_decoder_masks(h, cache)
    offset = _cache_offset(cache[0]) if cache else 0
    for i, (layer, c) in enumerate(zip(decoder.layers, cache)):
        mask = masks.get(layer.layer_type)
        if mask is not None and not isinstance(mask, str):
            traces[f"decoder.layer_{i}.mask"] = mask.astype(mx.float32)
        h = layer(h, mask, c, decoder=True, offset=offset)
        traces[f"decoder.layer_{i}.output"] = h
    hidden = decoder.norm(h)
    traces["decoder.final_hidden"] = hidden

    logits = decoder.embed_tokens.as_linear(hidden)
    logits = softcap(logits, float(model.final_logit_softcapping))
    traces["decoder.raw_logits"] = logits

    generation_config = model.config.generation_config or {}
    t_min = float(generation_config.get("t_min", 0.4))
    t_max = float(generation_config.get("t_max", 0.8))
    max_steps = int(generation_config.get("max_denoising_steps", 48))
    temperature = t_min + ((t_max - t_min) * (args.cur_step / max(max_steps, 1)))
    processed = logits / temperature
    traces["decoder.processed_logits"] = processed
    traces["decoder.argmax_canvas"] = mx.argmax(processed, axis=-1).astype(mx.int32)
    sampler_config = generation_config.get("sampler_config") or {}
    entropy_bound = float(sampler_config.get("entropy_bound", 0.1))
    traces["decoder.entropy_accept_mask"] = entropy_accept_mask(processed, entropy_bound)

    out_dir = Path(args.out_dir)
    save_trace(
        out_dir,
        traces,
        {
            "model": args.model,
            "prompt": args.prompt,
            "prompt_text": prompt_text,
            "prompt_ids": prompt_ids,
            "canvas_len": args.canvas_len,
            "seed": args.seed,
            "cur_step": args.cur_step,
            "temperature": temperature,
            "entropy_bound": entropy_bound,
            "python_trace": str(out_dir / "python_trace.safetensors"),
        },
    )
    mx.save_safetensors(
        str(out_dir / "trace_inputs.safetensors"),
        {"input_ids": input_ids, "canvas_ids": canvas_ids},
    )
    print(f"Wrote Python trace to {out_dir}")


if __name__ == "__main__":
    main()
