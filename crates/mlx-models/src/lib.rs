//! mlx-models: Model implementations for the MLX Rust framework.
//!
//! Each model follows candle's pattern: Config → Layer → Block → Model.
//! Models are loaded via `VarBuilder` with `pp()` prefix scoping.

pub mod bert;
pub mod gemma3;
pub mod lfm2_moe;
pub mod lfm2_moe_python_port;
pub mod llama;
pub mod qwen3;
pub mod qwen3_5;
pub mod qwen3_moe;
pub mod qwen3_moe_python_port;

pub use bert::{Bert, BertConfig};
pub use gemma3::{Gemma3, Gemma3Config};
pub use lfm2_moe::{Lfm2Moe, Lfm2MoeConfig};
pub use lfm2_moe_python_port::{Lfm2MoePythonPort, Lfm2MoePythonPortConfig};
pub use llama::{Llama, LlamaConfig};
pub use qwen3::{Qwen3, Qwen3Config};
pub use qwen3_5::{Qwen35, Qwen35Config};
pub use qwen3_moe::{MoeProfileStats, Qwen3Moe, Qwen3MoeConfig};
pub use qwen3_moe_python_port::{Qwen3MoePythonPort, Qwen3MoePythonPortConfig};

pub fn reset_moe_profile_stats() {
    qwen3_moe::reset_moe_profile_stats();
    qwen3_moe_python_port::reset_qwen3_moe_python_port_profile_stats();
    lfm2_moe::reset_lfm2_moe_profile_stats();
    lfm2_moe_python_port::reset_lfm2_moe_python_port_profile_stats();
}

pub fn moe_profile_stats() -> MoeProfileStats {
    let mut stats = qwen3_moe::moe_profile_stats();
    let qwen_python_port = qwen3_moe_python_port::qwen3_moe_python_port_profile_stats();
    let lfm2 = lfm2_moe::lfm2_moe_profile_stats();
    let lfm2_python_port = lfm2_moe_python_port::lfm2_moe_python_port_profile_stats();
    stats.router_host_s += qwen_python_port.router_host_s;
    stats.routing_build_s += qwen_python_port.routing_build_s;
    stats.expert_forward_s += qwen_python_port.expert_forward_s;
    stats.shared_expert_s += qwen_python_port.shared_expert_s;
    stats.single_token_fast_path_hits += qwen_python_port.single_token_fast_path_hits;
    stats.device_router_shadow_checks += qwen_python_port.device_router_shadow_checks;
    stats.device_router_shadow_mismatches += qwen_python_port.device_router_shadow_mismatches;
    stats.router_host_s += lfm2.router_host_s;
    stats.routing_build_s += lfm2.routing_build_s;
    stats.expert_forward_s += lfm2.expert_forward_s;
    stats.shared_expert_s += lfm2.shared_expert_s;
    stats.single_token_fast_path_hits += lfm2.single_token_fast_path_hits;
    stats.device_router_shadow_checks += lfm2.device_router_shadow_checks;
    stats.device_router_shadow_mismatches += lfm2.device_router_shadow_mismatches;
    stats.router_host_s += lfm2_python_port.router_host_s;
    stats.routing_build_s += lfm2_python_port.routing_build_s;
    stats.expert_forward_s += lfm2_python_port.expert_forward_s;
    stats.shared_expert_s += lfm2_python_port.shared_expert_s;
    stats.single_token_fast_path_hits += lfm2_python_port.single_token_fast_path_hits;
    stats.device_router_shadow_checks += lfm2_python_port.device_router_shadow_checks;
    stats.device_router_shadow_mismatches += lfm2_python_port.device_router_shadow_mismatches;
    stats
}
