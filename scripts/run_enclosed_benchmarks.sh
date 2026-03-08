#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo build --bin mlx-server

python3 scripts/benchmark_python_vs_rust_servers.py \
  --models-file tests/benchmark_models.json \
  --case-file tests/benchmark_cases.json \
  --mode aligned_stream \
  --rust-server-cmd target/debug/mlx-server \
  "$@"

python3 scripts/benchmark_python_vs_rust_servers.py \
  --models-file tests/benchmark_models_embeddings.json \
  --case-file tests/benchmark_case_embeddings.json \
  --rust-server-cmd target/debug/mlx-server \
  "$@"
