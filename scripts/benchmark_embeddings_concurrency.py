#!/usr/bin/env python3
import argparse
import json
import math
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _http_json(method: str, url: str, payload: dict[str, Any] | None, timeout_s: float) -> tuple[int, dict[str, Any]]:
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
            body = json.loads(raw) if raw.strip() else {}
        except Exception:
            body = {"raw": raw}
        return e.code, body


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_values[lo]
    frac = rank - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def load_inputs(args: argparse.Namespace) -> list[str]:
    inputs = list(args.input or [])
    if args.input_file:
        raw = Path(args.input_file).read_text(encoding="utf-8")
        inputs.extend(line.strip() for line in raw.splitlines() if line.strip())
    if not inputs:
        raise SystemExit("provide at least one --input or --input-file")
    return inputs


def build_request_batch(corpus: list[str], start: int, batch_size: int) -> list[str]:
    out: list[str] = []
    for idx in range(batch_size):
        out.append(corpus[(start + idx) % len(corpus)])
    return out


@dataclass
class RequestMetric:
    status: int
    latency_s: float
    prompt_tokens: int
    embedding_count: int
    ok: bool
    error: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Concurrent benchmark for /v1/embeddings")
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--load-model-path")
    parser.add_argument("--input", action="append")
    parser.add_argument("--input-file")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests-per-worker", type=int, default=25)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    args = parser.parse_args()

    corpus = load_inputs(args)
    batch_size = max(args.batch_size, 1)
    concurrency = max(args.concurrency, 1)
    requests_per_worker = max(args.requests_per_worker, 1)

    if args.load_model_path:
        status, body = _http_json(
            "POST",
            f"{args.base_url.rstrip('/')}/llm/load",
            {"model_path": args.load_model_path},
            args.timeout_s,
        )
        if status >= 300:
            raise SystemExit(f"/llm/load failed: status={status} body={body}")

    for warmup_idx in range(max(args.warmup_requests, 0)):
        payload = {
            "model": args.model,
            "input": build_request_batch(corpus, warmup_idx * batch_size, batch_size),
            "encoding_format": "float",
        }
        status, body = _http_json(
            "POST",
            f"{args.base_url.rstrip('/')}/v1/embeddings",
            payload,
            args.timeout_s,
        )
        if status >= 300:
            raise SystemExit(f"warmup request failed: status={status} body={body}")

    barrier = threading.Barrier(concurrency)
    metrics: list[RequestMetric] = []
    metrics_lock = threading.Lock()

    def worker(worker_id: int) -> None:
        local: list[RequestMetric] = []
        barrier.wait()
        for req_idx in range(requests_per_worker):
            payload = {
                "model": args.model,
                "input": build_request_batch(
                    corpus,
                    (worker_id * requests_per_worker + req_idx) * batch_size,
                    batch_size,
                ),
                "encoding_format": "float",
            }
            t0 = time.perf_counter()
            status, body = _http_json(
                "POST",
                f"{args.base_url.rstrip('/')}/v1/embeddings",
                payload,
                args.timeout_s,
            )
            elapsed = time.perf_counter() - t0
            if status < 300:
                usage = body.get("usage", {})
                data = body.get("data", [])
                local.append(
                    RequestMetric(
                        status=status,
                        latency_s=elapsed,
                        prompt_tokens=int(usage.get("prompt_tokens") or 0),
                        embedding_count=len(data) if isinstance(data, list) else 0,
                        ok=True,
                        error="",
                    )
                )
            else:
                local.append(
                    RequestMetric(
                        status=status,
                        latency_s=elapsed,
                        prompt_tokens=0,
                        embedding_count=0,
                        ok=False,
                        error=json.dumps(body, ensure_ascii=True),
                    )
                )
        with metrics_lock:
            metrics.extend(local)

    threads = [threading.Thread(target=worker, args=(idx,), daemon=True) for idx in range(concurrency)]
    t0 = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall_s = time.perf_counter() - t0

    latencies = sorted(metric.latency_s for metric in metrics)
    ok_metrics = [metric for metric in metrics if metric.ok]
    status_counts = Counter(metric.status for metric in metrics)
    total_requests = len(metrics)
    total_embeddings = sum(metric.embedding_count for metric in ok_metrics)
    total_prompt_tokens = sum(metric.prompt_tokens for metric in ok_metrics)

    summary = {
        "base_url": args.base_url,
        "model": args.model,
        "load_model_path": args.load_model_path,
        "input_corpus_size": len(corpus),
        "batch_size": batch_size,
        "concurrency": concurrency,
        "requests_per_worker": requests_per_worker,
        "warmup_requests": max(args.warmup_requests, 0),
        "total_requests": total_requests,
        "ok_requests": len(ok_metrics),
        "error_requests": total_requests - len(ok_metrics),
        "status_counts": dict(sorted(status_counts.items())),
        "wall_s": wall_s,
        "requests_per_s": total_requests / max(wall_s, 1e-9),
        "embeddings_per_s": total_embeddings / max(wall_s, 1e-9),
        "prompt_tokens_per_s": total_prompt_tokens / max(wall_s, 1e-9),
        "latency_ms": {
            "min": min(latencies) * 1000.0 if latencies else 0.0,
            "p50": percentile(latencies, 0.50) * 1000.0,
            "p95": percentile(latencies, 0.95) * 1000.0,
            "p99": percentile(latencies, 0.99) * 1000.0,
            "max": max(latencies) * 1000.0 if latencies else 0.0,
        },
        "sample_error": next((metric.error for metric in metrics if metric.error), ""),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
