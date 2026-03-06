#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODELS_FILE = "tests/benchmark_models.json"
DEFAULT_CASES_FILE = "tests/benchmark_cases.json"


@dataclass
class ModelSpec:
    key: str
    name: str
    model_dir: str


@dataclass
class BenchmarkResult:
    backend: str
    mode: str
    model_key: str
    model_name: str
    model_dir: str
    ttft_s: float
    total_s: float
    tokens: int
    tokens_per_s: float
    decode_tokens_per_s: float
    stop_reason: str
    output_preview: str
    debug_profile: dict[str, Any] | None = None
    comparable: bool = True
    comparability_note: str = ""
    ok: bool = True
    error: str = ""


@dataclass
class ChatMetrics:
    ttft_s: float
    total_s: float
    tokens: int
    tokens_per_s: float
    decode_tokens_per_s: float
    stop_reason: str
    content: str
    debug_profile: dict[str, Any] | None = None


class ServerProcess:
    def __init__(self, cmd: list[str], host: str, port: int, ready_path: str | None = None, timeout_s: float = 120.0):
        self.cmd = cmd
        self.host = host
        self.port = port
        self.ready_path = ready_path
        self.timeout_s = timeout_s
        self.proc: subprocess.Popen[str] | None = None

    def __enter__(self) -> "ServerProcess":
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._wait_ready()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is not None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=8)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass

    def _socket_ready(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((self.host, self.port)) == 0

    def _http_ready(self) -> bool:
        if not self.ready_path:
            return True
        url = f"http://{self.host}:{self.port}{self.ready_path}"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                return 200 <= resp.status < 500
        except Exception:
            return False

    def _wait_ready(self) -> None:
        start = time.perf_counter()
        while time.perf_counter() - start < self.timeout_s:
            if self.proc is not None and self.proc.poll() is not None:
                out = ""
                if self.proc.stdout is not None:
                    try:
                        out = self.proc.stdout.read() or ""
                    except Exception:
                        pass
                raise RuntimeError(f"server exited early: {' '.join(self.cmd)}\n{out}")
            if self._socket_ready() and self._http_ready():
                return
            time.sleep(0.2)
        raise RuntimeError(f"server did not become ready within {self.timeout_s}s: {' '.join(self.cmd)}")


def _http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout_s: float = 240.0) -> tuple[int, dict[str, Any]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            if not raw.strip():
                return resp.status, {}
            return resp.status, json.loads(raw)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw}


def resolve_snapshot_path(model_dir: str) -> str:
    p = Path(os.path.expanduser(model_dir))
    snapshots = p / "snapshots"
    if snapshots.is_dir():
        dirs = sorted([d for d in snapshots.iterdir() if d.is_dir()])
        if dirs:
            return str(dirs[-1])
    return str(p)


def load_models(path: str) -> list[ModelSpec]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[ModelSpec] = []
    for item in raw:
        out.append(
            ModelSpec(
                key=item["key"],
                name=item["name"],
                model_dir=resolve_snapshot_path(item["model_dir"]),
            )
        )
    return out


def load_case(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("benchmark case file must be a JSON object")
    return data


def count_tokens(model_dir: str, text: str) -> int:
    try:
        from tokenizers import Tokenizer  # type: ignore

        tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        enc = tok.encode(text)
        return len(enc.ids)
    except Exception:
        # Fallback: rough token estimate
        return max(1, len(text.split())) if text.strip() else 0


def decode_tps(tokens: int, total_s: float, ttft_s: float) -> float:
    decode_s = max(total_s - ttft_s, 1e-9)
    return float(tokens) / decode_s if tokens > 0 else 0.0


def stream_chat(base_url: str, prompt: str, max_tokens: int, temperature: float) -> ChatMetrics:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": True,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    ttft = None
    parts: list[str] = []
    final_event: dict[str, Any] | None = None
    usage_event: dict[str, Any] | None = None
    finish_reason = ""

    try:
        with urllib.request.urlopen(req, timeout=600.0) as resp:
            if resp.status != 200:
                raise RuntimeError(f"chat status {resp.status}")
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    usage_event = obj
                choices = obj.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                maybe_finish_reason = choice.get("finish_reason")
                if maybe_finish_reason:
                    finish_reason = str(maybe_finish_reason)
                    final_event = obj
                delta = choice.get("delta", {}).get("content", "")
                if delta:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    parts.append(delta)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"stream chat status={e.code} body={body}") from e

    total = time.perf_counter() - t0
    if ttft is None:
        raise RuntimeError("no streamed tokens were produced")
    content = "".join(parts)
    usage = (usage_event or final_event or {}).get("usage", {})
    dbg = (final_event or {}).get("debug", {})
    tokens = int(usage.get("completion_tokens") or dbg.get("tokens") or 0)
    stop_reason = str(dbg.get("stop_reason") or finish_reason or "unknown")
    total_s = float(dbg.get("total_s") or total)
    ttft_s = float(dbg.get("ttft_s") or ttft)
    tps = float(dbg.get("tokens_per_s") or (tokens / max(total_s, 1e-9))) if tokens > 0 else 0.0
    return ChatMetrics(
        ttft_s=ttft_s,
        total_s=total_s,
        tokens=tokens,
        tokens_per_s=tps,
        decode_tokens_per_s=decode_tps(tokens, total_s, ttft_s),
        stop_reason=stop_reason,
        content=content,
        debug_profile=dbg.get("profile") if isinstance(dbg.get("profile"), dict) else None,
    )


def non_stream_chat(base_url: str, prompt: str, max_tokens: int, temperature: float) -> ChatMetrics:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
    }
    status, body = _http_json(
        "POST",
        f"{base_url.rstrip('/')}/v1/chat/completions",
        payload,
        timeout_s=900.0,
    )
    if status >= 300:
        raise RuntimeError(f"chat status={status} body={body}")
    content = (
        body.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    dbg = body.get("debug", {})
    ttft = float(dbg.get("ttft_s", 0.0))
    total = float(dbg.get("total_s", 0.0))
    usage = body.get("usage", {})
    tokens = int(dbg.get("tokens") or usage.get("completion_tokens") or 0)
    tps = float(dbg.get("tokens_per_s") or (tokens / max(total, 1e-9)))
    stop_reason = str(dbg.get("stop_reason") or body.get("choices", [{}])[0].get("finish_reason") or "unknown")
    if not content:
        raise RuntimeError("non-stream response returned empty content")
    return ChatMetrics(
        ttft_s=ttft,
        total_s=total,
        tokens=tokens,
        tokens_per_s=tps,
        decode_tokens_per_s=decode_tps(tokens, total, ttft),
        stop_reason=stop_reason,
        content=content,
        debug_profile=dbg.get("profile") if isinstance(dbg.get("profile"), dict) else None,
    )


def rust_load_model(base_url: str, model_dir: str) -> None:
    status, body = _http_json("POST", f"{base_url.rstrip('/')}/llm/load", {"model_path": model_dir}, timeout_s=900.0)
    if status >= 300:
        raise RuntimeError(f"/llm/load failed for {model_dir}: status={status} body={body}")


def rust_unload_model(base_url: str) -> None:
    _http_json("POST", f"{base_url.rstrip('/')}/llm/unload", {}, timeout_s=120.0)


def benchmark_python(
    models: list[ModelSpec],
    case: dict[str, Any],
    host: str,
    port: int,
    server_template: str,
    startup_timeout_s: float,
    mode: str,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    prompt = case["prompt"]
    max_tokens = int(case.get("max_tokens", 256))
    temperature = float(case.get("temperature", 0.0))

    for model in models:
        cmd = shlex.split(
            server_template.format(model_dir=model.model_dir, host=host, port=port)
        )
        print(f"[python] starting server for {model.name}")
        try:
            with ServerProcess(cmd, host, port, ready_path="/v1/models", timeout_s=startup_timeout_s):
                if mode == "stream":
                    metrics = stream_chat(f"http://{host}:{port}", prompt, max_tokens, temperature)
                else:
                    metrics = non_stream_chat(f"http://{host}:{port}", prompt, max_tokens, temperature)
                tokens = metrics.tokens
                tokens_per_s = metrics.tokens_per_s
                decode_tokens_per_s = metrics.decode_tokens_per_s
                if tokens <= 0:
                    tokens = count_tokens(model.model_dir, metrics.content)
                    tokens_per_s = tokens / max(metrics.total_s, 1e-9)
                    decode_tokens_per_s = decode_tps(tokens, metrics.total_s, metrics.ttft_s)
                results.append(
                    BenchmarkResult(
                        backend="python",
                        mode=mode,
                        model_key=model.key,
                        model_name=model.name,
                        model_dir=model.model_dir,
                        ttft_s=metrics.ttft_s,
                        total_s=metrics.total_s,
                        tokens=tokens,
                        tokens_per_s=tokens_per_s,
                        decode_tokens_per_s=decode_tokens_per_s,
                        stop_reason=metrics.stop_reason,
                        output_preview=metrics.content[:240],
                        debug_profile=metrics.debug_profile,
                    )
                )
        except Exception as e:
            results.append(
                BenchmarkResult(
                    backend="python",
                    mode=mode,
                    model_key=model.key,
                    model_name=model.name,
                    model_dir=model.model_dir,
                    ttft_s=0.0,
                    total_s=0.0,
                    tokens=0,
                    tokens_per_s=0.0,
                    decode_tokens_per_s=0.0,
                    stop_reason="error",
                    output_preview="",
                    ok=False,
                    error=str(e),
                )
            )
    return results


def benchmark_rust(
    models: list[ModelSpec],
    case: dict[str, Any],
    host: str,
    port: int,
    rust_server_cmd: str,
    startup_timeout_s: float,
    mode: str,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    prompt = case["prompt"]
    max_tokens = int(case.get("max_tokens", 256))
    temperature = float(case.get("temperature", 0.0))
    cmd = shlex.split(rust_server_cmd)

    print("[rust] starting server")
    with ServerProcess(cmd, host, port, ready_path="/health", timeout_s=startup_timeout_s):
        base_url = f"http://{host}:{port}"
        for model in models:
            print(f"[rust] loading {model.name}")
            try:
                rust_load_model(base_url, model.model_dir)
                if mode == "stream":
                    metrics = stream_chat(base_url, prompt, max_tokens, temperature)
                else:
                    metrics = non_stream_chat(base_url, prompt, max_tokens, temperature)
                tokens = metrics.tokens
                tokens_per_s = metrics.tokens_per_s
                decode_tokens_per_s = metrics.decode_tokens_per_s
                if tokens <= 0:
                    tokens = count_tokens(model.model_dir, metrics.content)
                    tokens_per_s = tokens / max(metrics.total_s, 1e-9)
                    decode_tokens_per_s = decode_tps(tokens, metrics.total_s, metrics.ttft_s)
                results.append(
                    BenchmarkResult(
                        backend="rust",
                        mode=mode,
                        model_key=model.key,
                        model_name=model.name,
                        model_dir=model.model_dir,
                        ttft_s=metrics.ttft_s,
                        total_s=metrics.total_s,
                        tokens=tokens,
                        tokens_per_s=tokens_per_s,
                        decode_tokens_per_s=decode_tokens_per_s,
                        stop_reason=metrics.stop_reason,
                        output_preview=metrics.content[:240],
                        debug_profile=metrics.debug_profile,
                    )
                )
            except Exception as e:
                results.append(
                    BenchmarkResult(
                        backend="rust",
                        mode=mode,
                        model_key=model.key,
                        model_name=model.name,
                        model_dir=model.model_dir,
                        ttft_s=0.0,
                        total_s=0.0,
                        tokens=0,
                        tokens_per_s=0.0,
                        decode_tokens_per_s=0.0,
                        stop_reason="error",
                        output_preview="",
                        ok=False,
                        error=str(e),
                    )
                )
            finally:
                try:
                    rust_unload_model(base_url)
                except Exception:
                    pass
    return results


def print_table(results: list[BenchmarkResult]) -> None:
    grouped: dict[str, dict[str, BenchmarkResult]] = {}
    for r in results:
        grouped.setdefault(r.model_key, {})[r.backend] = r

    headers = [
        "model",
        "backend",
        "mode",
        "ok",
        "ttft_s",
        "tokens_per_s",
        "decode_tps",
        "tokens",
        "total_s",
        "stop_reason",
        "comparable",
        "note",
        "error",
    ]
    print("\n" + " | ".join(headers))
    print(" | ".join(["---"] * len(headers)))

    for model_key in sorted(grouped.keys()):
        row = grouped[model_key]
        for backend in ("python", "rust"):
            r = row.get(backend)
            if not r:
                continue
            print(
                f"{r.model_name} | {backend} | {r.mode} | {str(r.ok).lower()} | {r.ttft_s:.3f} | {r.tokens_per_s:.2f} | {r.decode_tokens_per_s:.2f} | {r.tokens} | {r.total_s:.3f} | {r.stop_reason} | {str(r.comparable).lower()} | {r.comparability_note[:80]} | {r.error[:120]}"
            )

        py = row.get("python")
        rs = row.get("rust")
        if py and rs and py.ok and rs.ok and py.tokens_per_s > 0.0 and py.decode_tokens_per_s > 0.0:
            note = py.comparability_note or rs.comparability_note
            print(
                f"{py.model_name} | ratio | {py.mode} | true | {rs.ttft_s / py.ttft_s:.2f}x_ttft | {rs.tokens_per_s / py.tokens_per_s:.2%} | {rs.decode_tokens_per_s / py.decode_tokens_per_s:.2%} | - | - | - | {str(py.comparable and rs.comparable).lower()} | {note[:80]} | "
            )


def to_json(results: list[BenchmarkResult]) -> list[dict[str, Any]]:
    return [
        {
            "backend": r.backend,
            "mode": r.mode,
            "model_key": r.model_key,
            "model_name": r.model_name,
            "model_dir": r.model_dir,
            "ttft_s": r.ttft_s,
            "tokens_per_s": r.tokens_per_s,
            "decode_tokens_per_s": r.decode_tokens_per_s,
            "tokens": r.tokens,
            "total_s": r.total_s,
            "stop_reason": r.stop_reason,
            "comparable": r.comparable,
            "comparability_note": r.comparability_note,
            "output_preview": r.output_preview,
            "debug_profile": r.debug_profile,
            "ok": r.ok,
            "error": r.error,
        }
        for r in results
    ]


def annotate_comparability(results: list[BenchmarkResult]) -> None:
    grouped: dict[str, dict[str, BenchmarkResult]] = {}
    for r in results:
        grouped.setdefault(r.model_key, {})[r.backend] = r

    for row in grouped.values():
        py = row.get("python")
        rs = row.get("rust")
        if not py or not rs or not py.ok or not rs.ok:
            continue

        notes: list[str] = []
        if py.stop_reason != rs.stop_reason:
            notes.append(f"stop_reason differs: python={py.stop_reason}, rust={rs.stop_reason}")
        if py.tokens != rs.tokens:
            notes.append(f"token_count differs: python={py.tokens}, rust={rs.tokens}")
        if py.mode != rs.mode:
            notes.append(f"mode differs: python={py.mode}, rust={rs.mode}")

        if notes:
            note = "; ".join(notes)
            py.comparable = False
            rs.comparable = False
            py.comparability_note = note
            rs.comparability_note = note


def main() -> int:
    p = argparse.ArgumentParser(
        description="Enclosed benchmark: start python server, benchmark 3 models, stop; start rust server, benchmark 3 models, stop."
    )
    p.add_argument("--models-file", default=DEFAULT_MODELS_FILE)
    p.add_argument("--case-file", default=DEFAULT_CASES_FILE)

    p.add_argument("--python-host", default="127.0.0.1")
    p.add_argument("--python-port", type=int, default=5000)
    p.add_argument(
        "--python-server-template",
        default=".venv/bin/python -m mlx_lm.server --model {model_dir} --host {host} --port {port}",
        help="Command template for python server; supports {model_dir}, {host}, {port}",
    )

    p.add_argument("--rust-host", default="127.0.0.1")
    p.add_argument("--rust-port", type=int, default=3000)
    p.add_argument("--rust-server-cmd", default="target/debug/mlx-server")
    p.add_argument(
        "--mode",
        choices=("aligned_stream", "stream", "non_stream"),
        default="aligned_stream",
        help="Benchmark mode. 'aligned_stream' and 'stream' use streaming for both backends.",
    )

    p.add_argument("--startup-timeout-s", type=float, default=180.0)
    p.add_argument("--output-json", default=None)
    args = p.parse_args()

    models = load_models(args.models_file)
    case = load_case(args.case_file)

    if len(models) == 0:
        raise RuntimeError("no models provided")

    request_mode = "stream" if args.mode in {"aligned_stream", "stream"} else "non_stream"

    py_results = benchmark_python(
        models,
        case,
        host=args.python_host,
        port=args.python_port,
        server_template=args.python_server_template,
        startup_timeout_s=args.startup_timeout_s,
        mode=request_mode,
    )

    rust_results = benchmark_rust(
        models,
        case,
        host=args.rust_host,
        port=args.rust_port,
        rust_server_cmd=args.rust_server_cmd,
        startup_timeout_s=args.startup_timeout_s,
        mode=request_mode,
    )

    all_results = py_results + rust_results
    annotate_comparability(all_results)
    print_table(all_results)

    out_path = args.output_json
    if out_path is None:
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        Path("logs").mkdir(parents=True, exist_ok=True)
        out_path = f"logs/python_vs_rust_benchmark_{ts}.json"
    Path(out_path).write_text(json.dumps(to_json(all_results), indent=2), encoding="utf-8")
    print(f"\nSaved benchmark JSON: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
