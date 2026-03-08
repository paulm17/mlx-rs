#!/usr/bin/env python3
import argparse
import io
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from rdflib import Graph, RDFS, URIRef
from tqdm import tqdm


DEFAULT_WORDNET_BASE = "/Volumes/Data/Users/paul/development/src/github/_archon_old/nlp/hebrew/bible-data"


class TqdmReader(io.BufferedReader):
    def __init__(self, raw, total_bytes: int, desc: str):
        super().__init__(raw)
        self._progress = tqdm(total=total_bytes, desc=desc, unit="B", unit_scale=True, unit_divisor=1024)

    def read(self, size: int = -1):
        chunk = super().read(size)
        if chunk:
            self._progress.update(len(chunk))
        return chunk

    def close(self):
        self._progress.close()
        return super().close()


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


def resolve_model_path(model: str, explicit_model_path: str | None) -> str | None:
    if explicit_model_path:
        return explicit_model_path

    env_model_path = os.environ.get("EMBEDDING_MODEL_PATH")
    if env_model_path:
        return env_model_path

    expanded = os.path.expanduser(model)
    if os.path.exists(expanded):
        return expanded

    hub_root = Path.home() / ".cache" / "huggingface" / "hub"
    candidate_dirs: list[Path] = []

    normalized = model.strip().lower()
    if "/" in normalized:
        candidate_dirs.append(hub_root / f"models--{normalized.replace('/', '--')}")
    else:
        candidate_dirs.append(hub_root / f"models--mlx-community--{normalized}")
        candidate_dirs.append(hub_root / f"models--mixedbread-ai--{normalized}")
        candidate_dirs.extend(sorted(hub_root.glob(f"models--*--{normalized}")))

    for candidate in candidate_dirs:
        snapshots = candidate / "snapshots"
        if snapshots.is_dir():
            dirs = sorted(path for path in snapshots.iterdir() if path.is_dir())
            if dirs:
                return str(dirs[-1])

    return None


def syn_id(node) -> str:
    return re.sub(r"^.*/", "", str(node))


def load_wordnet_texts(base_dir: str, limit: int | None) -> list[str]:
    ttl_path = os.path.join(base_dir, "SYNONYMS", "english-wordnet-2024.ttl")
    graph = Graph()
    ttl_size = os.path.getsize(ttl_path)
    with open(ttl_path, "rb") as handle:
        with TqdmReader(handle, ttl_size, "Parsing WordNet TTL") as wrapped:
            graph.parse(source=wrapped, format="turtle")

    syns: dict[str, dict[str, str]] = {}
    for s, _, o in graph.triples((None, URIRef("http://wordnet-rdf.princeton.edu/ontology#definition"), None)):
        sid = syn_id(s)
        syns.setdefault(sid, {})["gloss"] = str(o)
    for s, _, o in graph.triples((None, RDFS.label, None)):
        sid = syn_id(s)
        syns.setdefault(sid, {})["label"] = str(o)

    texts: list[str] = []
    for data in syns.values():
        text = (data.get("label", "") + ". " + data.get("gloss", "")).strip()
        if not text:
            continue
        texts.append(text)
        if limit is not None and len(texts) >= limit:
            break
    return texts


def build_batches(texts: list[str], batch_size: int, sort_mode: str) -> list[list[str]]:
    ordered = list(texts)
    if sort_mode == "char_len":
        ordered.sort(key=len)
    elif sort_mode == "word_len":
        ordered.sort(key=lambda text: len(text.split()))
    return [ordered[idx : idx + batch_size] for idx in range(0, len(ordered), batch_size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark mlx-server /v1/embeddings with WordNet synset texts.")
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    parser.add_argument("--model", default="mxbai-embed-large-v1")
    parser.add_argument("--model-path")
    parser.add_argument("--wordnet-base-dir", default=DEFAULT_WORDNET_BASE)
    parser.add_argument("--sample-size", type=int, default=4096, help="Number of WordNet texts to benchmark.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=1, help="Number of passes over the sample.")
    parser.add_argument("--sort-mode", choices=["none", "char_len", "word_len"], default="char_len")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    args = parser.parse_args()

    texts = load_wordnet_texts(os.path.abspath(args.wordnet_base_dir), args.sample_size)
    if not texts:
        raise RuntimeError("No WordNet texts loaded")

    model_path = resolve_model_path(args.model, args.model_path)
    maybe_load_model(args.base_url, model_path, args.timeout_s)

    print(f"Loaded {len(texts)} WordNet texts")
    print(f"Endpoint: {args.base_url.rstrip('/')}/v1/embeddings")
    print(f"Model: {args.model}")
    if model_path:
        print(f"Model path: {model_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sort mode: {args.sort_mode}")

    batch_latencies: list[float] = []
    batch_prompt_tokens: list[int] = []
    batch_items: list[int] = []
    total_embeddings = 0
    total_prompt_tokens = 0
    total_requests = 0

    started_all = time.perf_counter()
    for _ in range(args.repeat):
        batches = build_batches(texts, args.batch_size, args.sort_mode)
        for batch in tqdm(batches, desc="Embedding batches", unit="batch"):
            payload = {
                "model": args.model,
                "input": batch,
                "encoding_format": "float",
            }
            started = time.perf_counter()
            status, body = http_json(
                "POST",
                f"{args.base_url.rstrip('/')}/v1/embeddings",
                payload,
                args.timeout_s,
            )
            elapsed = time.perf_counter() - started
            if status >= 300:
                raise RuntimeError(f"status={status} body={body}")
            data = body.get("data") or []
            if len(data) != len(batch):
                raise RuntimeError(f"expected {len(batch)} vectors, got {len(data)}")
            usage = body.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            batch_latencies.append(elapsed)
            batch_prompt_tokens.append(prompt_tokens)
            batch_items.append(len(batch))
            total_embeddings += len(batch)
            total_prompt_tokens += prompt_tokens
            total_requests += 1
    wall_s = time.perf_counter() - started_all

    summary = {
        "base_url": args.base_url,
        "model": args.model,
        "wordnet_base_dir": os.path.abspath(args.wordnet_base_dir),
        "sample_size": len(texts),
        "batch_size": args.batch_size,
        "repeat": args.repeat,
        "sort_mode": args.sort_mode,
        "total_requests": total_requests,
        "total_embeddings": total_embeddings,
        "total_prompt_tokens": total_prompt_tokens,
        "wall_s": wall_s,
        "requests_per_s": total_requests / max(wall_s, 1e-9),
        "embeddings_per_s": total_embeddings / max(wall_s, 1e-9),
        "prompt_tokens_per_s": total_prompt_tokens / max(wall_s, 1e-9),
        "avg_batch_latency_s": statistics.mean(batch_latencies) if batch_latencies else 0.0,
        "avg_items_per_batch": statistics.mean(batch_items) if batch_items else 0.0,
        "avg_prompt_tokens_per_batch": statistics.mean(batch_prompt_tokens) if batch_prompt_tokens else 0.0,
        "avg_ms_per_item": ((statistics.mean(batch_latencies) * 1000.0) / statistics.mean(batch_items))
        if batch_latencies and batch_items
        else 0.0,
    }
    print("")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
