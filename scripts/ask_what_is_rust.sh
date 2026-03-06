#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
MODEL_DIR="/Volumes/Data/Users/paul/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/d0b3cb793b1b12acf826571ae1bb2bc819a7a37f"
PROMPT="${2:-What is Rust?}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-0.9}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/generate_python.py" \
  --model-dir "${MODEL_DIR}" \
  --prompt "${PROMPT}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}"
