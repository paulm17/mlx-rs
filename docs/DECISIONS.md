# GGUF decision models

Start Kev-9B on Metal using the named profile:

```bash
cargo build --offline --bin llama-server
target/debug/llama-server --config kev.toml
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"kev-9b","state":{"message":"I was charged twice."},"questions":{"team":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":null,"shipping":null}},"refund":{"type":"noul","instructions":"Is a refund requested?"}}}'
```

Kev weights are cached at
`~/.cache/llama-rs/models/ggml-org--Kev-9B-GGUF/Kev-9B-Q4_K_M.gguf`.
The downloaded file is 6,358,923,744 bytes with SHA-256
`86a6084984a6ef6818eb12c07cdc67ed3c47b00208d41a8816c64301bc6bcbda`,
from Hugging Face revision `ec2bbfe6620218aee2e01cc93bb78dee2a96ed58`.

## Multiple models

Add profiles to the same configuration:

```toml
[server]
model = "kev-9b"
port = 8080

[models.kev-9b]
repository = "ggml-org/Kev-9B-GGUF"
file = "Kev-9B-Q4_K_M.gguf"
n_gpu_layers = 99

[models.openjev]
repository = "ggml-org/OpenJev-GGUF"
quantization = "Q4_K_M"
n_ctx = 4096
n_batch = 4096
n_ubatch = 4096
n_gpu_layers = 99
```

Profiles accept either `model_path` (local file/directory or an existing Hugging
Face reference), or `repository` with an optional `file` and `quantization`.
An explicit file selects that exact file; otherwise quantization selects a matching
GGUF. Cached files are reused and missing files are downloaded by the existing
loader. Profile defaults are Q4_K_M, context/batch/physical batch 4096, one sequence,
and 99 offloaded layers. All existing llama.cpp runtime configuration fields are
available within a profile. Native logging inherits the server destination unless
specified in the profile.

The request's `model` selects a configured profile. Omitted `model` selects
`server.model`, even after another profile was used. Unknown names return 400.
`POST /llm/load` with `{"model_path":"openjev"}` selects a profile explicitly.
`GET /v1/models` lists the configured profiles and reports the active model's
`architecture.decision_family`. Decision responses identify the selected profile.

One model is resident at a time. Selection and inference share the server mutex
and run in blocking workers. Switching frees the old native context and model
before loading the replacement. A failed load leaves no model resident; a later
request can retry or select another profile. Loading failures include their cause.

## Native adapters and Rust

GGUF `<architecture>.decision.type` metadata selects Kev, OpenJev, Lev, Nimble,
Laya, or the existing Clef adapter. Unknown families are rejected. No generated
text is used to approximate decisions. Ordinary generation models keep their
existing backend behavior.

All adapters expose `choice`, `score`, and `noul` through `DecisionRequest` and
`DecisionResponse`. Structured state/instructions and option ordering survive
serialization. Each model retains its upstream prompt formatting, calibration,
output interpretation and prompt variants; Lev's reversed choice variant and
rating-scale noul are handled explicitly. Kev uses the question/option embedding
halves and upstream scaled dot product. Clef retains its joint annotated batch.

Non-Clef prompts are evaluated independently, without shared-prefix caching.
Input-token usage sums all question/variant prompts, including repeated prefixes;
output-token usage is zero. Every prompt is checked against context, logical batch,
and physical batch limits before inference begins. Images/audio return 501, and
chat generation requests for decision models return 501.

Use `DecisionModels::new(profiles, default)` directly with `Backend::decide`, or
`GenerationPipeline::with_decision_models(profiles, default)`. Their mutable Rust
interface serializes selection and inference. `RunnerClient::decide` uses the same
HTTP request shape. Single-model `Runtime::decide` accepts an omitted model or its
loaded path; use profiles to select named models.

## Validation

Real-model tests use local weights and Metal; they never download models:

```bash
MLX_RS_TEST_KEV_GGUF=/absolute/path/to/Kev-9B-Q4_K_M.gguf \
  cargo test -p llama-lm --offline --test kev kev_native -- --ignored
MLX_RS_TEST_KEV_GGUF=/absolute/path/to/Kev-9B-Q4_K_M.gguf \
  cargo test -p llama-lm --offline kev_profiles_native_http_switch_and_errors -- --ignored
```

For parity, run upstream llama-server from pinned llama.cpp
`99b95488cac0f00ce3f05af113a8c1e287753f87` with
`-ngl 99 -np 1 -c 4096 -b 4096 -ub 4096 -t 4 -tb 4` and the same weights.
Set `MLX_RS_TEST_KEV_REFERENCE_URL=http://127.0.0.1:8081` for `kev_matches`.
Alternatively submit `crates/llama-lm/tests/fixtures/kev-request.json`, save the
response, stop upstream, and set `MLX_RS_TEST_KEV_REFERENCE_JSON` to that file.
Tests require matching input-token usage and numeric differences below `1e-5`.

Kev is validated with real Q4_K_M weights on Apple M1 Pro, all 33 layers on Metal.
Other adapter families have parser/calibration fixture coverage; validating new
weights against upstream is required before claiming equivalent numerical results.
