#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 scripts/benchmark_python_vs_rust_servers.py \
  --models-file tests/benchmark_models.json \
  --case-file tests/benchmark_cases.json \
  "$@"
