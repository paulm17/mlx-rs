#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-}"
PROMPT="${2:-What is Rust programming language?}"
MAX_TOKENS="${3:-512}"
TEMPERATURE="${4:-0}"
THINKING="${5:-false}"
INTERVAL_S="${6:-2}"

if [[ -z "${MODEL_DIR}" ]]; then
  echo "usage: $0 <model_dir> [prompt] [max_tokens] [temperature] [thinking] [interval_s]" >&2
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="logs/memory_trace_${TS}"
mkdir -p "${OUT_DIR}"

echo "trace output: ${OUT_DIR}"
echo "building generate binary..."
cargo build --bin generate >/dev/null

CMD=(
  target/debug/generate
  --model-dir "${MODEL_DIR}"
  --prompt "${PROMPT}"
  --max-tokens "${MAX_TOKENS}"
  --temperature "${TEMPERATURE}"
  --thinking "${THINKING}"
)

echo "starting: ${CMD[*]}"
MLX_TRACE_MEMORY=1 "${CMD[@]}" >"${OUT_DIR}/generate.stdout.log" 2>"${OUT_DIR}/generate.stderr.log" &
PID=$!
echo "${PID}" > "${OUT_DIR}/pid.txt"
echo "pid=${PID}"

{
  echo "ts_epoch_s,rss_kb,vsz_kb,pcpu,pmem"
  while kill -0 "${PID}" 2>/dev/null; do
    NOW="$(date +%s)"
    STATS="$(ps -o rss=,vsz=,pcpu=,pmem= -p "${PID}" | awk '{$1=$1;print}')"
    echo "${NOW},${STATS// /,}"
    sleep "${INTERVAL_S}"
  done
} > "${OUT_DIR}/process_stats.csv"

(
  SNAP=1
  while kill -0 "${PID}" 2>/dev/null; do
    vmmap -summary "${PID}" > "${OUT_DIR}/vmmap_summary_${SNAP}.txt" 2>&1 || true
    SNAP=$((SNAP + 1))
    sleep "${INTERVAL_S}"
  done
) &
VMMAP_PID=$!

wait "${PID}" || true
wait "${VMMAP_PID}" || true

echo "done. files:"
ls -1 "${OUT_DIR}"
