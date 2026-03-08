#!/usr/bin/env python3
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from typing import Any


def http_json(method: str, url: str, payload: dict[str, Any] | None, timeout_s: float) -> tuple[int, dict[str, Any]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw) if raw.strip() else {}
        except Exception:
            return e.code, {"raw": raw}


def parse_batch_sizes(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def maybe_load_model(base_url: str, model_path: str | None, timeout_s: float) -> None:
    if not model_path:
        return
    status, body = http_json(
        "POST",
        f"{base_url.rstrip('/')}/llm/load",
        {"model_path": model_path},
        timeout_s,
    )
    if status >= 300:
        raise RuntimeError(f"/llm/load failed: status={status} body={body}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark mlx-server /v1/embeddings with repeated identical inputs.")
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    parser.add_argument("--model", default="mxbai-embed-large-v1")
    parser.add_argument("--model-path")
    parser.add_argument("--batch-sizes", default="1,8,32,64,128,256")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--text",
        default="king. a ruler over a nation or people.",
        help="Input text duplicated across the batch.",
    )
    args = parser.parse_args()

    maybe_load_model(args.base_url, args.model_path, args.timeout_s)

    print(f"Endpoint: {args.base_url.rstrip('/')}/v1/embeddings")
    print(f"Model: {args.model}")
    print("")
    print("batch_size\treq_s\titems_per_s\tprompt_toks_per_s\tms_per_item\tprompt_toks_per_item")

    for batch_size in parse_batch_sizes(args.batch_sizes):
        timings: list[float] = []
        prompt_tokens: list[int] = []

        payload = {
            "model": args.model,
            "input": [args.text] * batch_size,
            "encoding_format": "float",
        }
        for _ in range(args.repeat):
            started = time.perf_counter()
            status, body = http_json(
                "POST",
                f"{args.base_url.rstrip('/')}/v1/embeddings",
                payload,
                args.timeout_s,
            )
            elapsed = time.perf_counter() - started
            if status >= 300:
                raise RuntimeError(f"batch_size={batch_size} status={status} body={body}")
            data = body.get("data") or []
            if len(data) != batch_size:
                raise RuntimeError(f"batch_size={batch_size} expected {batch_size} vectors, got {len(data)}")
            usage = body.get("usage") or {}
            timings.append(elapsed)
            prompt_tokens.append(int(usage.get("prompt_tokens") or 0))

        avg_s = statistics.mean(timings)
        avg_prompt_tokens = statistics.mean(prompt_tokens)
        items_per_s = batch_size / avg_s
        prompt_tokens_per_s = avg_prompt_tokens / avg_s
        ms_per_item = (avg_s * 1000.0) / batch_size
        prompt_tokens_per_item = avg_prompt_tokens / batch_size
        print(
            f"{batch_size}\t{avg_s:.3f}\t{items_per_s:.2f}\t{prompt_tokens_per_s:.2f}\t"
            f"{ms_per_item:.2f}\t{prompt_tokens_per_item:.2f}"
        )


if __name__ == "__main__":
    main()
