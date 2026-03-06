#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import load as load_tokenizer
from mlx_lm.utils import load_model


def has_runaway_repeat(generated_tokens):
    for w in (16, 24, 32):
        if len(generated_tokens) < w * 3:
            continue
        n = len(generated_tokens)
        a = generated_tokens[n - w : n]
        b = generated_tokens[n - 2 * w : n - w]
        c = generated_tokens[n - 3 * w : n - 2 * w]
        if a == b and b == c:
            return True
    return False


def apply_template(tokenizer, user_prompt: str, thinking: bool) -> str:
    if not getattr(tokenizer, "has_chat_template", False):
        return user_prompt

    msgs = [{"role": "user", "content": user_prompt}]
    try:
        return tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
    except Exception:
        # Match Rust fallback behavior closely.
        if thinking:
            return (
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        return f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"


def get_eos_ids(config: dict):
    eos = config.get("eos_token_id")
    if eos is None:
        eos = (config.get("text_config") or {}).get("eos_token_id")
    if eos is None:
        return None
    if isinstance(eos, list):
        return [int(x) for x in eos]
    return [int(eos)]

def main():
    ap = argparse.ArgumentParser(description="Python generate script comparable to Rust generate")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--thinking", type=lambda x: str(x).lower() in ("1", "true", "yes", "on"), default=False)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--dump-json-out", default=None)
    ap.add_argument("--trace-step", type=int, default=None)
    ap.add_argument("--trace-window", type=int, default=4)
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    print(f'Loading model from "{model_dir}"...')
    model, config = load_model(model_dir, strict=False)
    tokenizer = load_tokenizer(model_dir, eos_token_ids=get_eos_ids(config))

    prompt = apply_template(tokenizer, args.prompt, args.thinking)

    enc = tokenizer.encode(prompt)
    ids = enc.ids if hasattr(enc, "ids") else enc
    if not ids:
        raise RuntimeError("Prompt tokenization returned no tokens")

    eos_ids = set(getattr(tokenizer, "eos_token_ids", []))
    sampler = make_sampler(temp=args.temperature, top_p=args.top_p)

    history_tokens = []
    generated_tokens = []
    output = ""
    total_generated = 0
    ttft_s = None
    t0 = time.perf_counter()
    stop_reason = "unknown"
    step_topk = []

    print("Generating...")
    prompt_arr = mx.array(ids, dtype=mx.int32)
    for token_arr, logprobs in generate_step(
        prompt_arr,
        model,
        max_tokens=(args.max_tokens if args.max_tokens is not None else 256),
        sampler=sampler,
    ):
        if args.trace_step is not None and abs(total_generated - args.trace_step) <= args.trace_window:
            arr = np.array(logprobs.astype(mx.float32))
            order = np.argsort(-arr)[: args.topk]
            step_topk.append(
                {
                    "step": total_generated,
                    "topk": [
                        {"token_id": int(i), "logprob": float(arr[i])}
                        for i in order.tolist()
                    ],
                }
            )
        token = int(np.array(token_arr)[()])
        if token in eos_ids:
            stop_reason = "stop"
            break

        generated_tokens.append(token)
        if has_runaway_repeat(generated_tokens):
            stop_reason = "repeat_guard"
            break

        piece = tokenizer.decode([token])
        print(piece, end="", flush=True)
        output += piece
        history_tokens.append(token)
        total_generated += 1
        if ttft_s is None:
            ttft_s = time.perf_counter() - t0
    else:
        if args.max_tokens is not None and total_generated >= args.max_tokens:
            stop_reason = "length"

    total_s = time.perf_counter() - t0
    last_token_id = generated_tokens[-1] if generated_tokens else None
    if stop_reason == "unknown" and args.max_tokens is not None and total_generated >= args.max_tokens:
        stop_reason = "length"

    if args.dump_json_out:
        Path(args.dump_json_out).write_text(
            json.dumps(
                {
                    "prompt": prompt,
                    "prompt_ids": ids,
                    "generated_token_ids": generated_tokens,
                    "last_token_id": last_token_id,
                    "eos_ids": sorted(eos_ids),
                    "stop_reason": stop_reason,
                    "ttft_s": ttft_s,
                    "total_s": total_s,
                    "step_topk": step_topk,
                    "output": output,
                },
                indent=2,
            )
        )
    print()
    print(
        f"\n--- Generated {len(output)} chars in {total_s:.2f}s "
        f"(tokens={total_generated}, ttft={0.0 if ttft_s is None else ttft_s:.3f}s, "
        f"peak_mem={mx.get_peak_memory()/1e9:.2f} GB) ---"
    )


if __name__ == "__main__":
    main()
