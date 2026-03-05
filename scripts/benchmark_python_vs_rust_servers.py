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
    model_key: str
    model_name: str
    model_dir: str
    ttft_s: float
    total_s: float
    tokens: int
    tokens_per_s: float
    output_preview: str
    ok: bool = True
    error: str = ""


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


def stream_chat(base_url: str, prompt: str, max_tokens: int, temperature: float) -> tuple[float, float, str]:
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
            delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if delta:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                parts.append(delta)

    total = time.perf_counter() - t0
    if ttft is None:
        raise RuntimeError("no streamed tokens were produced")
    return ttft, total, "".join(parts)


def non_stream_chat(base_url: str, prompt: str, max_tokens: int, temperature: float) -> tuple[float, float, int, float, str]:
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
    tokens = int(dbg.get("tokens", 0))
    tps = float(dbg.get("tokens_per_s", 0.0))
    if not content:
        raise RuntimeError("non-stream response returned empty content")
    return ttft, total, tokens, tps, content


def rust_load_model(base_url: str, model_dir: str) -> None:
    status, body = _http_json("POST", f"{base_url.rstrip('/')}/llm/load", {"model_path": model_dir}, timeout_s=900.0)
    if status >= 300:
        raise RuntimeError(f"/llm/load failed for {model_dir}: status={status} body={body}")


def rust_unload_model(base_url: str) -> None:
    _http_json("POST", f"{base_url.rstrip('/')}/llm/unload", {}, timeout_s=120.0)


def benchmark_python(models: list[ModelSpec], case: dict[str, Any], host: str, port: int, server_template: str, startup_timeout_s: float) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    prompt = case["prompt"]
    max_tokens = int(case.get("max_tokens", 256))
    temperature = float(case.get("temperature", 0.0))

    for model in models:
        cmd = shlex.split(
            server_template.format(model_dir=model.model_dir, host=host, port=port)
        )
        print(f"[python] starting server for {model.name}")
        with ServerProcess(cmd, host, port, ready_path="/v1/models", timeout_s=startup_timeout_s):
            ttft, total, output = stream_chat(f"http://{host}:{port}", prompt, max_tokens, temperature)
            tokens = count_tokens(model.model_dir, output)
            tps = (tokens / max(total, 1e-9))
            results.append(
                BenchmarkResult(
                    backend="python",
                    model_key=model.key,
                    model_name=model.name,
                    model_dir=model.model_dir,
                    ttft_s=ttft,
                    total_s=total,
                    tokens=tokens,
                    tokens_per_s=tps,
                    output_preview=output[:240],
                )
            )
    return results


def benchmark_rust(models: list[ModelSpec], case: dict[str, Any], host: str, port: int, rust_server_cmd: str, startup_timeout_s: float) -> list[BenchmarkResult]:
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
                ttft, total, tokens, tps, output = non_stream_chat(base_url, prompt, max_tokens, temperature)
                if tokens <= 0:
                    tokens = count_tokens(model.model_dir, output)
                    tps = (tokens / max(total, 1e-9))
                results.append(
                    BenchmarkResult(
                        backend="rust",
                        model_key=model.key,
                        model_name=model.name,
                        model_dir=model.model_dir,
                        ttft_s=ttft,
                        total_s=total,
                        tokens=tokens,
                        tokens_per_s=tps,
                        output_preview=output[:240],
                    )
                )
            except Exception as e:
                results.append(
                    BenchmarkResult(
                        backend="rust",
                        model_key=model.key,
                        model_name=model.name,
                        model_dir=model.model_dir,
                        ttft_s=0.0,
                        total_s=0.0,
                        tokens=0,
                        tokens_per_s=0.0,
                        output_preview="",
                        ok=False,
                        error=str(e),
                    )
                )
            finally:
                rust_unload_model(base_url)
    return results


def print_table(results: list[BenchmarkResult]) -> None:
    grouped: dict[str, dict[str, BenchmarkResult]] = {}
    for r in results:
        grouped.setdefault(r.model_key, {})[r.backend] = r

    headers = [
        "model",
        "backend",
        "ok",
        "ttft_s",
        "tokens_per_s",
        "tokens",
        "total_s",
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
                f"{r.model_name} | {backend} | {str(r.ok).lower()} | {r.ttft_s:.3f} | {r.tokens_per_s:.2f} | {r.tokens} | {r.total_s:.3f} | {r.error[:120]}"
            )


def to_json(results: list[BenchmarkResult]) -> list[dict[str, Any]]:
    return [
        {
            "backend": r.backend,
            "model_key": r.model_key,
            "model_name": r.model_name,
            "model_dir": r.model_dir,
            "ttft_s": r.ttft_s,
            "tokens_per_s": r.tokens_per_s,
            "tokens": r.tokens,
            "total_s": r.total_s,
            "output_preview": r.output_preview,
            "ok": r.ok,
            "error": r.error,
        }
        for r in results
    ]


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

    p.add_argument("--startup-timeout-s", type=float, default=180.0)
    p.add_argument("--output-json", default=None)
    args = p.parse_args()

    models = load_models(args.models_file)
    case = load_case(args.case_file)

    if len(models) == 0:
        raise RuntimeError("no models provided")

    py_results = benchmark_python(
        models,
        case,
        host=args.python_host,
        port=args.python_port,
        server_template=args.python_server_template,
        startup_timeout_s=args.startup_timeout_s,
    )

    rust_results = benchmark_rust(
        models,
        case,
        host=args.rust_host,
        port=args.rust_port,
        rust_server_cmd=args.rust_server_cmd,
        startup_timeout_s=args.startup_timeout_s,
    )

    all_results = py_results + rust_results
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
