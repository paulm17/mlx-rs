use anyhow::Result;
use mlx_core::Array;
use mlx_lm::{GenerationPipeline, Sampler, Tokenizer};
use mlx_models::Gemma4;

/// Vision-language generation pipeline.
pub struct VlmGenerationPipeline {
    model: Gemma4,
    tokenizer: Tokenizer,
}

/// Generation options for VLM.
#[derive(Debug, Clone)]
pub struct VlmGenerateOptions {
    pub max_tokens: usize,
    pub temperature: f32,
}

impl Default for VlmGenerateOptions {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.6,
        }
    }
}

impl VlmGenerationPipeline {
    /// Create a new VLM generation pipeline.
    pub fn new(model: Gemma4, tokenizer: Tokenizer) -> Self {
        Self { model, tokenizer }
    }

    /// Generate token IDs given input token IDs and optional pixel values.
    ///
    /// Runs a full autoregressive loop:
    /// 1. Prefill: forward the full prompt (with pixel values if provided).
    /// 2. Decode: sample one token at a time, forwarding only the new token.
    ///
    /// Stops when `max_tokens` is reached or the EOS token is generated.
    pub fn generate_tokens(
        &mut self,
        input_ids: &Array,
        pixel_values: Option<&Array>,
        opts: &VlmGenerateOptions,
    ) -> Result<Vec<u32>> {
        let sampler = Sampler::new(opts.temperature, 1.0);
        let tokenizer = self.tokenizer.clone();
        let mut pipeline = GenerationPipeline::new(&mut self.model, tokenizer, sampler);
        let (_text, metrics) = pipeline.generate_from_ids_with_metrics(
            input_ids,
            pixel_values,
            Some(opts.max_tokens),
            |_, _| {},
        )?;
        Ok(metrics.generated_token_ids)
    }

    /// Generate text given input token IDs and optional pixel values.
    ///
    /// Returns the decoded text string (excluding the EOS token).
    pub fn generate(
        &mut self,
        input_ids: &Array,
        pixel_values: Option<&Array>,
        opts: &VlmGenerateOptions,
    ) -> Result<String> {
        let tokens = self.generate_tokens(input_ids, pixel_values, opts)?;
        self.tokenizer.decode(&tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_core::DType;
    use mlx_lm::Tokenizer;
    use mlx_models::{Gemma4, Gemma4Config};
    use mlx_models::gemma4::{Gemma4TextConfig, Gemma4VisionConfig};
    use mlx_nn::VarBuilder;
    use std::collections::HashMap;

    fn build_tiny_tokenizer() -> Tokenizer {
        use tokenizers::models::wordlevel::WordLevelBuilder;
        let mut vocab = HashMap::new();
        vocab.insert("<pad>".to_string(), 0);
        vocab.insert("a".to_string(), 1);
        vocab.insert("b".to_string(), 2);
        vocab.insert("</s>".to_string(), 3);
        let model = WordLevelBuilder::new()
            .vocab(vocab)
            .unk_token("<pad>".to_string())
            .build()
            .unwrap();
        Tokenizer::from_raw(tokenizers::Tokenizer::new(model))
    }

    fn make_tiny_config() -> Gemma4Config {
        Gemma4Config {
            model_type: "gemma4".to_string(),
            image_token_id: 0,
            audio_token_id: None,
            boi_token_id: 0,
            eoi_token_id: 0,
            boa_token_id: None,
            eoa_token_id: None,
            vision_soft_tokens_per_image: 0,
            text_config: Gemma4TextConfig {
                hidden_size: 4,
                num_hidden_layers: 1,
                num_attention_heads: 1,
                num_key_value_heads: 1,
                head_dim: 4,
                global_head_dim: 4,
                num_global_key_value_heads: None,
                intermediate_size: 8,
                vocab_size: 4,
                sliding_window: 512,
                sliding_window_pattern: 5,
                layer_types: vec!["sliding_attention".to_string()],
                rope_parameters: HashMap::new(),
                final_logit_softcapping: None,
                tie_word_embeddings: false,
                rms_norm_eps: 1e-6,
                hidden_activation: "gelu".to_string(),
                enable_moe_block: false,
                num_experts: None,
                top_k_experts: None,
                moe_intermediate_size: None,
                attention_k_eq_v: false,
                num_kv_shared_layers: 0,
                hidden_size_per_layer_input: 0,
                use_double_wide_mlp: false,
                vocab_size_per_layer_input: 262144,
            },
            vision_config: Gemma4VisionConfig {
                hidden_size: 4,
                num_hidden_layers: 1,
                num_attention_heads: 1,
                num_key_value_heads: 1,
                head_dim: 4,
                global_head_dim: 4,
                intermediate_size: 8,
                patch_size: 2,
                position_embedding_size: 4,
                pooling_kernel_size: 2,
                default_output_length: 1,
                max_patches: 4,
                standardize: false,
                rope_parameters: HashMap::new(),
                use_clipped_linears: false,
                rms_norm_eps: 1e-6,
            },
            audio_config: None,
        }
    }

    fn build_zero_weights(config: &Gemma4Config) -> HashMap<String, Array> {
        let mut weights = HashMap::new();
        let h = config.text_config.hidden_size as i32;
        let vocab = config.text_config.vocab_size as i32;
        let inter = config.text_config.intermediate_size as i32;

        // Text embeddings
        weights.insert(
            "language_model.model.embed_tokens.weight".to_string(),
            Array::zeros(&[vocab, h], DType::Float32).unwrap(),
        );

        // Layer 0
        let layer_prefix = "language_model.model.layers.0";
        for (name, shape) in [
            ("input_layernorm.weight", vec![h]),
            ("post_attention_layernorm.weight", vec![h]),
            ("pre_feedforward_layernorm.weight", vec![h]),
            ("post_feedforward_layernorm.weight", vec![h]),
            ("self_attn.q_proj.weight", vec![h, h]),
            ("self_attn.k_proj.weight", vec![h, h]),
            ("self_attn.v_proj.weight", vec![h, h]),
            ("self_attn.o_proj.weight", vec![h, h]),
            ("self_attn.q_norm.weight", vec![h]),
            ("self_attn.k_norm.weight", vec![h]),
            ("mlp.gate_proj.weight", vec![inter, h]),
            ("mlp.up_proj.weight", vec![inter, h]),
            ("mlp.down_proj.weight", vec![h, inter]),
        ] {
            weights.insert(
                format!("{layer_prefix}.{name}"),
                Array::zeros(&shape, DType::Float32).unwrap(),
            );
        }

        // Final norm
        weights.insert(
            "language_model.model.norm.weight".to_string(),
            Array::ones(&[h], DType::Float32).unwrap(),
        );

        // LM head: zero weights, strong bias for token 1
        let mut lm_head_bias = Array::zeros(&[vocab], DType::Float32).unwrap();
        lm_head_bias = lm_head_bias
            .slice_update(&Array::from_float(10.0).unwrap(), &[1], &[2], &[1])
            .unwrap();
        weights.insert(
            "language_model.lm_head.weight".to_string(),
            Array::zeros(&[vocab, h], DType::Float32).unwrap(),
        );
        weights.insert(
            "language_model.lm_head.bias".to_string(),
            lm_head_bias,
        );

        // Vision tower (needed because model construction loads it)
        let vp = config.vision_config.patch_size as i32;
        let vh = config.vision_config.hidden_size as i32;
        let vpos = config.vision_config.position_embedding_size as i32;
        weights.insert(
            "vision_tower.patch_embedder.input_proj.weight".to_string(),
            Array::zeros(&[vh, 3 * vp * vp], DType::Float32).unwrap(),
        );
        weights.insert(
            "vision_tower.patch_embedder.position_embedding_table".to_string(),
            Array::zeros(&[vpos, vh], DType::Float32).unwrap(),
        );

        let v_layer_prefix = "vision_tower.encoder.layers.0";
        for (name, shape) in [
            ("input_layernorm.weight", vec![vh]),
            ("post_attention_layernorm.weight", vec![vh]),
            ("pre_feedforward_layernorm.weight", vec![vh]),
            ("post_feedforward_layernorm.weight", vec![vh]),
            ("self_attn.q_proj.linear.weight", vec![vh, vh]),
            ("self_attn.k_proj.linear.weight", vec![vh, vh]),
            ("self_attn.v_proj.linear.weight", vec![vh, vh]),
            ("self_attn.o_proj.linear.weight", vec![vh, vh]),
            ("self_attn.q_norm.weight", vec![vh]),
            ("self_attn.k_norm.weight", vec![vh]),
            ("mlp.gate_proj.linear.weight", vec![inter, vh]),
            ("mlp.up_proj.linear.weight", vec![inter, vh]),
            ("mlp.down_proj.linear.weight", vec![vh, inter]),
        ] {
            weights.insert(
                format!("{v_layer_prefix}.{name}"),
                Array::zeros(&shape, DType::Float32).unwrap(),
            );
        }

        // Multimodal embedder
        weights.insert(
            "embed_vision.embedding_pre_projection_norm.weight".to_string(),
            Array::ones(&[vh], DType::Float32).unwrap(),
        );
        weights.insert(
            "embed_vision.embedding_projection.weight".to_string(),
            Array::zeros(&[h, vh], DType::Float32).unwrap(),
        );

        weights
    }

    #[test]
    fn greedy_generation_produces_expected_tokens() {
        let config = make_tiny_config();
        let weights = build_zero_weights(&config);
        let vb = VarBuilder::from_weights(weights, DType::Float32);
        let model = Gemma4::new(&vb, &config).unwrap();
        let tokenizer = build_tiny_tokenizer();
        let mut pipeline = VlmGenerationPipeline::new(model, tokenizer);

        // Prompt with a single token (id=1)
        let input_ids = Array::from_slice_i32(&[1]).unwrap().reshape(&[1, 1]).unwrap();
        let opts = VlmGenerateOptions {
            max_tokens: 5,
            temperature: 0.0,
        };

        let tokens = pipeline.generate_tokens(&input_ids, None, &opts).unwrap();
        // LM head bias strongly favors token 1, so greedy always picks 1
        // It should not stop immediately because EOS is token 3
        assert_eq!(tokens, vec![1, 1, 1, 1, 1]);
    }

    #[test]
    fn generation_stops_at_eos() {
        let mut config = make_tiny_config();
        // Set EOS token to be the one that gets highest logit
        config.text_config.vocab_size = 4;
        let mut weights = build_zero_weights(&config);
        // Make token 3 (EOS) have highest bias
        let mut lm_head_bias = Array::zeros(&[4], DType::Float32).unwrap();
        lm_head_bias = lm_head_bias
            .slice_update(&Array::from_float(10.0).unwrap(), &[3], &[4], &[1])
            .unwrap();
        weights.insert(
            "language_model.lm_head.bias".to_string(),
            lm_head_bias,
        );

        let vb = VarBuilder::from_weights(weights, DType::Float32);
        let model = Gemma4::new(&vb, &config).unwrap();
        let tokenizer = build_tiny_tokenizer();
        let mut pipeline = VlmGenerationPipeline::new(model, tokenizer);

        let input_ids = Array::from_slice_i32(&[1]).unwrap().reshape(&[1, 1]).unwrap();
        let opts = VlmGenerateOptions {
            max_tokens: 10,
            temperature: 0.0,
        };

        let tokens = pipeline.generate_tokens(&input_ids, None, &opts).unwrap();
        // First token should be EOS (3), so generation stops immediately
        assert!(tokens.is_empty());
    }

    #[test]
    fn generate_returns_decoded_text() {
        let config = make_tiny_config();
        let weights = build_zero_weights(&config);
        let vb = VarBuilder::from_weights(weights, DType::Float32);
        let model = Gemma4::new(&vb, &config).unwrap();
        let tokenizer = build_tiny_tokenizer();
        let mut pipeline = VlmGenerationPipeline::new(model, tokenizer);

        let input_ids = Array::from_slice_i32(&[1]).unwrap().reshape(&[1, 1]).unwrap();
        let opts = VlmGenerateOptions {
            max_tokens: 3,
            temperature: 0.0,
        };

        let text = pipeline.generate(&input_ids, None, &opts).unwrap();
        // Token 1 decodes to "a" in our tiny tokenizer
        assert_eq!(text, "a a a");
    }
}
