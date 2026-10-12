# Native llama.cpp source

Rust crates: llama-cpp-2 and llama-cpp-sys-2 0.1.154, copied from the Cargo registry (llama-cpp-rs revision bed81ad4ab1a6c904b11d425608e50f976d8ea62).
llama.cpp: ggml-org/llama.cpp commit 99b95488cac0f00ce3f05af113a8c1e287753f87 (Clef merge).

Both Rust crates are patched together in the workspace. Local wrapper additions provide borrowed native pointers and joint decision output reservation. Compatibility adaptations use common_json, pass the new optional fit argument, convert speculative batches to common_batch, and update DRY/penalty sampler signatures. The vendored penalties constructor takes n_vocab as its first argument to match this native revision. The text adapter in crates/llama-lm is derived from this revision's tools/server/server-decision.cpp; its original source is retained for auditing. Upstream llama.cpp's MIT license is retained. Rust wrapper licensing is MIT OR Apache-2.0.

The reusable text adapter in crates/llama-lm/decision_compat.cpp also derives its model setup, prompt rendering, option handling, and calibrated answer formatting from the same server-decision.cpp. It omits media handling and shared-prefix scheduling and evaluates each question/variant independently. Kev pointer scoring follows server-context.cpp send_decision. The existing Clef adapter remains separate.
