# Clef text decisions

The workspace pins llama.cpp to `99b95488cac0f00ce3f05af113a8c1e287753f87`
with a matched pair of vendored Rust wrappers. See [provenance](../vendor/README.md).
The native compatibility bridge uses the same headers and libraries.

Clef evaluates all questions jointly using its `systemone` template and model
calibration metadata. It returns probabilities without generating text. The full
prompt must fit `n_ctx`, `n_batch`, and `n_ubatch`; oversized requests return 400
rather than being truncated or split. Images/audio return 501 in this text-only
integration. Chat generation is unavailable for Clef.

Use the supplied `clef.toml`, which resolves the Q4_K_M weights from
`~/.cache/llama-rs/models/ggml-org--Clef-GGUF/`:

```toml
[server]
bind = "127.0.0.1"
port = 8080
model_path = "ggml-org/Clef-GGUF"
n_ctx = 4096
n_batch = 4096
n_ubatch = 4096
n_gpu_layers = 99
```

```bash
cargo run --release --offline --bin llama-server -- --config clef.toml
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"I was charged twice and nobody has replied.","questions":{
    "route":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":null,"shipping":null,"technical":null}},
    "angry":{"type":"noul","instructions":"Is the customer angry?"},
    "urgency":{"type":"score","instructions":"How urgent is this?","criteria":["can wait","this week","today","right now"]}
  }}'
```

`state` and `instructions` also accept structured JSON. Responses contain
`model`, `answers`, and `usage` (`input_tokens` and `output_tokens: 0`). Model
discovery reports `architecture.output_modalities: ["decisions"]`. Choice
options sort by key; score levels retain array order; noul returns the
probability of true. The temperatures come from GGUF metadata.

An explicit batch limit remains authoritative. If omitted, Clef defaults to a
physical batch equal to the context size and uses one sequence. Increasing the
physical batch size can substantially increase memory requirements.

## Rust

```rust
use llama_lm::{DecisionRequest, GenerationPipeline, LlamaCppConfig};

let mut pipeline = GenerationPipeline::new(
    "/absolute/path/to/Clef-Q4_K_M.gguf", LlamaCppConfig::default(),
)?;
let request: DecisionRequest = serde_json::from_value(serde_json::json!({
    "state": {"customer_message": "I was charged twice."},
    "questions": {"billing": {"type": "noul", "instructions": "Is this a billing issue?"}}
}))?;
let response = pipeline.decide(&request)?;
```

`LlamaCppBackend` implements `Backend::decide`; `Runtime::decide` is also
available. `RunnerClient::decide` sends the same typed request to `/v1/systemone`.
Other backends return an unsupported-capability error by default.

## Real-model validation

Tests use locally supplied weights and never download them:

```bash
MLX_RS_TEST_CLEF_GGUF=/absolute/path/to/Clef-Q4_K_M.gguf \
  cargo test -p llama-lm --offline --test clef clef_native -- --ignored
```

For numerical parity, run upstream llama-server from the pinned revision with
identical weights and Metal configuration (`-ngl 99 -c 4096 -b 4096 -ub 4096 -t 4 -tb 4`), then:

```bash
MLX_RS_TEST_CLEF_GGUF=/absolute/path/to/Clef-Q4_K_M.gguf \
MLX_RS_TEST_CLEF_REFERENCE_URL=http://127.0.0.1:8081 \
  cargo test -p llama-lm --offline --test clef clef_matches -- --ignored
```

On machines where loading two runtimes together would use too much memory, capture
the upstream response for the parity test request, stop the upstream server, and
set `MLX_RS_TEST_CLEF_REFERENCE_JSON` to that saved response instead of a URL.

Validated on Apple M1 Pro with `Clef-Q4_K_M.gguf` (SHA-256
`9997675e4c9e9016273372b5fdedbdda7ad14285bc722949e4b9daca62bf3822`)
and all 65 layers offloaded to Metal. Real-model tests cover all three question
types, repeat requests, model reload, context and physical batch rejection,
recovery after rejection, and structured-state HTTP requests. Native results
match the pinned upstream server within `1e-5`, with identical input-token usage.
The workspace chat, tool parsing, streaming, and embedding regressions pass.
