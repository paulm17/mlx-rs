# Native llama.cpp source

Rust crates: llama-cpp-2 and llama-cpp-sys-2 0.1.154, copied from the Cargo registry (llama-cpp-rs revision bed81ad4ab1a6c904b11d425608e50f976d8ea62).
llama.cpp: ggml-org/llama.cpp commit 99b95488cac0f00ce3f05af113a8c1e287753f87 (Clef merge).

EmbeddingGemma 2 text inference is backported from upstream PR #30054,
commit 1e91a0b054fcb839507671dc2ab9409821c12dc1 (merged as 4fbc76d).
Only native architecture changes are included; GGUF conversion tooling is omitted.
The graph uses explicit token-only embedding scaling because this pinned revision
has no scaling argument on build_inp_embd. Upstream's architecture test change
only excludes the model from a generic generation fixture; that test file is not
vendored here. Rust integration tests cover real text embeddings instead.
The model source glob tracks additions so cached CMake builds include new models.

Both Rust crates are patched together in the workspace. Local wrapper additions provide borrowed native pointers and joint decision output reservation. Compatibility adaptations use common_json, pass the new optional fit argument, convert speculative batches to common_batch, and update DRY/penalty sampler signatures. The vendored penalties constructor takes n_vocab as its first argument to match this native revision. The text adapter in crates/llama-lm is derived from this revision's tools/server/server-decision.cpp; its original source is retained for auditing. Upstream llama.cpp's MIT license is retained. Rust wrapper licensing is MIT OR Apache-2.0.

The reusable text adapter in crates/llama-lm/decision_compat.cpp also derives its model setup, prompt rendering, option handling, and calibrated answer formatting from the same server-decision.cpp. It omits media handling and shared-prefix scheduling and evaluates each question/variant independently. Kev pointer scoring follows server-context.cpp send_decision. The existing Clef adapter remains separate.
