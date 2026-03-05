#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import types

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from mlx_lm.utils import load_model
from mlx_lm.tokenizer_utils import load as load_tokenizer
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.gated_delta import gated_delta_update


def to_summary(x, n=32):
    mx.eval(x)
    try:
        xf = x.astype(mx.float32)
    except Exception:
        xf = x
    a = np.array(xf)
    flat = a.reshape(-1)
    return {
        "shape": list(a.shape),
        "dtype": str(x.dtype),
        "first": flat[:n].tolist(),
        "mean": float(flat.mean()) if flat.size else 0.0,
        "std": float(flat.std()) if flat.size else 0.0,
    }


def last_token(x):
    if len(x.shape) >= 2 and x.shape[1] >= 1:
        return x[:, -1:]
    return x


def patch_linear_attn(layer_obj, layer_idx, traces):
    la = layer_obj.linear_attn
    target_id = id(la)
    cls = type(la)
    if not hasattr(cls, "_orig_call_for_trace"):
        cls._orig_call_for_trace = cls.__call__

    def wrapped(self, inputs, mask=None, cache=None):
        if id(self) != target_id:
            return cls._orig_call_for_trace(self, inputs, mask=mask, cache=cache)
        B, S, _ = inputs.shape
        # Support both qwen3_next and qwen3_5 variants.
        if hasattr(self, "fix_query_key_value_ordering"):
            q, k, v, z, b, a = self.fix_query_key_value_ordering(
                self.in_proj_qkvz(inputs), self.in_proj_ba(inputs)
            )
            mixed_qkv = mx.concatenate([q.reshape(B, S, -1), k.reshape(B, S, -1), v.reshape(B, S, -1)], axis=-1)
            conv_dim = self.conv_dim
            q_pre = to_summary(last_token(q))
            k_pre = to_summary(last_token(k))
            v_pre = to_summary(last_token(v))
            mixed_pre = None
        else:
            qkv = self.in_proj_qkv(inputs)
            z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
            b = self.in_proj_b(inputs)
            a = self.in_proj_a(inputs)
            mixed_qkv = qkv
            conv_dim = self.conv_dim
            q_pre = None
            k_pre = None
            v_pre = None
            mixed_pre = to_summary(last_token(qkv))

        phase = "prefill" if S > 1 else "decode"
        rec = {
            "kind": "linear_attn",
            "layer": layer_idx,
            "phase": phase,
            "S": int(S),
            "inputs": to_summary(last_token(inputs)),
            "q_pre": q_pre,
            "k_pre": k_pre,
            "v_pre": v_pre,
            "mixed_qkv_pre": mixed_pre,
            "z_pre": to_summary(last_token(z)),
            "a_pre": to_summary(last_token(a)),
            "b_pre": to_summary(last_token(b)),
        }

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros((B, self.conv_kernel_size - 1, conv_dim), dtype=inputs.dtype)

        if mask is not None:
            mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)

        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = conv_input[:, -n_keep:, :]

        conv_out = nn.silu(self.conv1d(conv_input))

        q2, k2, v2 = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        inv_scale = k2.shape[-1] ** -0.5
        q2 = (inv_scale**2) * mx.fast.rms_norm(q2, None, 1e-6)
        k2 = inv_scale * mx.fast.rms_norm(k2, None, 1e-6)

        out, state = gated_delta_update(
            q2, k2, v2, a, b, self.A_log, self.dt_bias, state, mask, use_kernel=False
        )

        if cache is not None:
            cache[1] = state
            if hasattr(cache, "advance"):
                cache.advance(S)

        out_norm = self.norm(out, z)
        y = self.out_proj(out_norm.reshape(B, S, -1))

        rec.update({
            "conv_out": to_summary(last_token(conv_out)),
            "q_post": to_summary(last_token(q2)),
            "k_post": to_summary(last_token(k2)),
            "v_post": to_summary(last_token(v2)),
            "out_delta": to_summary(last_token(out)),
            "state": to_summary(state[:, :1, :4, :4]) if state is not None else None,
            "out_norm": to_summary(last_token(out_norm)),
            "out_proj": to_summary(last_token(y)),
        })
        traces.append(rec)
        return y

    cls.__call__ = wrapped


def patch_full_attn(layer_obj, layer_idx, traces):
    attn = layer_obj.self_attn
    target_id = id(attn)
    cls = type(attn)
    if not hasattr(cls, "_orig_call_for_trace"):
        cls._orig_call_for_trace = cls.__call__

    def wrapped(self, x, mask=None, cache=None):
        if id(self) != target_id:
            return cls._orig_call_for_trace(self, x, mask=mask, cache=cache)
        B, L, _ = x.shape
        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, L, -1)

        keys, values = self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = self.o_proj(output * mx.sigmoid(gate))

        phase = "prefill" if L > 1 else "decode"
        traces.append({
            "kind": "full_attn",
            "layer": layer_idx,
            "phase": phase,
            "L": int(L),
            "q": to_summary(queries[:, :, -1:, :]),
            "k": to_summary(keys[:, :, -1:, :]),
            "v": to_summary(values[:, :, -1:, :]),
            "gate": to_summary(gate[:, -1:, :]),
            "attn_out": to_summary(output[:, -1:, :]),
            "proj_out": to_summary(out[:, -1:, :]),
        })
        return out

    cls.__call__ = wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", default="logs/qwen35_parity_dump.json")
    ap.add_argument("--chatml", action="store_true")
    args = ap.parse_args()

    model, config = load_model(Path(args.model_dir), strict=False)
    eos_ids = config.get("eos_token_id")
    tokenizer = load_tokenizer(Path(args.model_dir), eos_token_ids=eos_ids)

    traces = []
    layers = getattr(model, "layers", None)
    if layers is None:
        inner = getattr(model, "model", None)
        layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is None:
        raise RuntimeError("Could not locate decoder layers on loaded model")

    patch_linear_attn(layers[0], 0, traces)
    patch_full_attn(layers[3], 3, traces)

    prompt = args.prompt
    if args.chatml:
        prompt = f"<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"

    enc = tokenizer.encode(prompt)
    ids = enc.ids if hasattr(enc, "ids") else enc
    cache = model.make_cache()

    inp = mx.array([ids], dtype=mx.int32)
    logits = model(inp, cache=cache)
    logits_last = logits[:, -1, :]
    next_id = int(np.array(mx.argmax(logits_last, axis=-1))[0])

    inp2 = mx.array([[next_id]], dtype=mx.int32)
    logits2 = model(inp2, cache=cache)
    logits2_last = logits2[:, -1, :]

    out = {
        "model_dir": args.model_dir,
        "prompt_raw": args.prompt,
        "prompt_used": prompt,
        "input_len": len(ids),
        "prefill_next_id": next_id,
        "prefill_logits_last": to_summary(logits_last),
        "decode_logits_last": to_summary(logits2_last),
        "traces": traces,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
