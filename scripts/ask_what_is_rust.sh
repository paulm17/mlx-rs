#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
MODEL="mlx-community/gemma-3-text-12b-it-4bit"
PROMPT="${2:-What is Rust?}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-0.9}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/generate_python.py" \
  --model "${MODEL}" \
  --prompt "${PROMPT}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}"
