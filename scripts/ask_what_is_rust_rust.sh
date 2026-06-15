#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-mlx-community/diffusiongemma-26B-A4B-it-4bit}"
PROMPT="${1:-What is Rust?}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-0.9}"

exec cargo run --manifest-path "${ROOT_DIR}/Cargo.toml" --bin generate -- \
  --model "${MODEL}" \
  --prompt "${PROMPT}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}"
