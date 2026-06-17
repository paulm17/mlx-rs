#!/usr/bin/env bash
# Smoke benchmarks for llama-rs
#
# Usage:
#   GGUF_MODEL=/path/to/model.gguf ./bench/run.sh
#   GGUF_MODEL=/path/to/model.gguf EMBED_GGUF=/path/to/embed.gguf ./bench/run.sh
#   SAFETENSORS_MODEL=/path/to/model/dir ./bench/run.sh
#
# All model paths must be provided via environment variables.
# No network access is required.

set -euo pipefail

GGUF_MODEL="${GGUF_MODEL:-}"
EMBED_GGUF="${EMBED_GGUF:-}"
SAFETENSORS_MODEL="${SAFETENSORS_MODEL:-}"
CONFIG="${CONFIG:-config.toml}"
MAX_TOKENS="${MAX_TOKENS:-128}"
STREAM_MAX_TOKENS="${STREAM_MAX_TOKENS:-64}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-5}"
SERVER_PORT="${SERVER_PORT:-18090}"
PROMPT="${PROMPT:-Explain the concept of recursion in programming.}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="$PROJECT_DIR/target/release"

# Build if needed
BUILT=false
for bin in generate llama-server; do
    if [ ! -x "$BIN_DIR/$bin" ]; then
        BUILT=true
    fi
done

if [ "$BUILT" = true ]; then
    echo "Building release binaries..."
    (cd "$PROJECT_DIR" && cargo build --release 2>&1)
fi

now_ms() { date +%s%3N; }

echo "========================================"
echo "  llama-rs Smoke Benchmarks"
echo "========================================"
date

CONFIG_ARG=""
if [ -f "$CONFIG" ]; then
    CONFIG_ARG="--config $CONFIG"
fi

# --- GGUF Text Generation ---
if [ -n "$GGUF_MODEL" ]; then
    echo ""
    echo "--- GGUF Text Generation ---"
    echo "Model: $GGUF_MODEL"

    echo ""
    echo "[1/4] Load + Non-streaming generation ($MAX_TOKENS tokens)..."
    T0=$(now_ms)
    "$BIN_DIR/generate" --model "$GGUF_MODEL" $CONFIG_ARG \
        --prompt "$PROMPT" --max-tokens "$MAX_TOKENS" --temperature 0.0 \
        2>&1 | tee /tmp/llama-rs-bench-1.log | grep -E "(Prompt tokens|Generated tokens|Time to first token|Total time|Tokens/sec|Stop reason)" || true
    T1=$(now_ms)
    echo "Wall time: $((T1 - T0))ms"

    echo ""
    echo "[2/4] Streaming generation ($STREAM_MAX_TOKENS tokens)..."
    T0=$(now_ms)
    "$BIN_DIR/generate" --model "$GGUF_MODEL" $CONFIG_ARG \
        --prompt "$PROMPT" --max-tokens "$STREAM_MAX_TOKENS" --temperature 0.0 --stream \
        2>&1 | tee /tmp/llama-rs-bench-2.log | grep -E "(Prompt tokens|Generated tokens|Time to first token|Total time|Tokens/sec)" || true
    T1=$(now_ms)
    echo "Wall time: $((T1 - T0))ms"

    echo ""
    echo "[3/4] Chat template generation..."
    T0=$(now_ms)
    "$BIN_DIR/generate" --model "$GGUF_MODEL" $CONFIG_ARG \
        --chat --prompt "What is 2+2?" --max-tokens 32 --temperature 0.0 \
        2>&1 | tee /tmp/llama-rs-bench-3.log | grep -E "(Prompt tokens|Generated tokens|Tokens/sec)" || true
    T1=$(now_ms)
    echo "Wall time: $((T1 - T0))ms"

    # --- Embeddings via server ---
    if [ -n "$EMBED_GGUF" ]; then
        echo ""
        echo "[4/4] Embeddings benchmark (server, $EMBED_BATCH_SIZE texts)..."

        EMBED_CONFIG=$(mktemp /tmp/llama-rs-bench-config.XXXXXX.toml)
        cat > "$EMBED_CONFIG" <<CONF
[server]
bind = "127.0.0.1"
port = $SERVER_PORT

embedding = true
pooling = "mean"
n_ctx = 512
n_gpu_layers = 99
CONF

        "$BIN_DIR/llama-server" --config "$EMBED_CONFIG" --model "$EMBED_GGUF" --port "$SERVER_PORT" &
        SERVER_PID=$!

        cleanup() {
            kill $SERVER_PID 2>/dev/null || true
            wait $SERVER_PID 2>/dev/null || true
            rm -f "$EMBED_CONFIG"
        }
        trap cleanup EXIT

        # Wait for server
        READY=false
        for i in $(seq 1 30); do
            if curl -sf "http://127.0.0.1:$SERVER_PORT/health" 2>/dev/null | grep -q "loaded"; then
                READY=true
                break
            fi
            sleep 1
        done

        if [ "$READY" = false ]; then
            echo "FAILED: Server did not start within 30s"
            cleanup
            exit 1
        fi

        T0=$(now_ms)
        for i in $(seq 1 "$EMBED_BATCH_SIZE"); do
            curl -sf "http://127.0.0.1:$SERVER_PORT/v1/embeddings" \
                -H "Content-Type: application/json" \
                -d "{\"input\": \"This is embedding test sentence number $i\", \"model\": \"local\"}" \
                -o /dev/null
        done
        T1=$(now_ms)
        EMBED_MS=$((T1 - T0))
        if [ "$EMBED_MS" -gt 0 ]; then
            EMBED_TPS=$(awk "BEGIN {printf \"%.2f\", $EMBED_BATCH_SIZE / ($EMBED_MS / 1000)}")
        else
            EMBED_TPS="inf"
        fi
        echo "Embedding calls: $EMBED_BATCH_SIZE in ${EMBED_MS}ms ($EMBED_TPS embeddings/sec)"

        cleanup
        trap - EXIT
    else
        echo ""
        echo "[4/4] Embeddings: SKIPPED (set EMBED_GGUF to benchmark)"
    fi
else
    echo ""
    echo "--- GGUF benchmarks SKIPPED (set GGUF_MODEL to run) ---"
fi

# --- Safetensors / MLX benchmark ---
if [ -n "$SAFETENSORS_MODEL" ]; then
    echo ""
    echo "--- MLX / Safetensors Text Generation ---"
    echo "Model: $SAFETENSORS_MODEL"

    echo ""
    echo "[1/2] Load + Non-streaming generation ($MAX_TOKENS tokens)..."
    T0=$(now_ms)
    "$BIN_DIR/generate" --model "$SAFETENSORS_MODEL" $CONFIG_ARG \
        --prompt "$PROMPT" --max-tokens "$MAX_TOKENS" --temperature 0.0 \
        2>&1 | tee /tmp/llama-rs-bench-mlx-1.log | grep -E "(Prompt tokens|Generated tokens|Time to first token|Total time|Tokens/sec|Stop reason)" || true
    T1=$(now_ms)
    echo "Wall time: $((T1 - T0))ms"

    echo ""
    echo "[2/2] Streaming generation ($STREAM_MAX_TOKENS tokens)..."
    T0=$(now_ms)
    "$BIN_DIR/generate" --model "$SAFETENSORS_MODEL" $CONFIG_ARG \
        --prompt "$PROMPT" --max-tokens "$STREAM_MAX_TOKENS" --temperature 0.0 --stream \
        2>&1 | tee /tmp/llama-rs-bench-mlx-2.log | grep -E "(Prompt tokens|Generated tokens|Time to first token|Total time|Tokens/sec)" || true
    T1=$(now_ms)
    echo "Wall time: $((T1 - T0))ms"
else
    echo ""
    echo "--- MLX benchmarks SKIPPED (set SAFETENSORS_MODEL to run) ---"
fi

echo ""
echo "========================================"
echo "  Benchmarks complete"
echo "========================================"