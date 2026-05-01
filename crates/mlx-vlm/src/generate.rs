use anyhow::Result;
use mlx_core::Array;
use mlx_models::Gemma4;
use tokenizers::Tokenizer;

/// Vision-language generation pipeline.
pub struct VlmGenerationPipeline {
    pub model: Gemma4,
    pub tokenizer: Tokenizer,
    eos_token_id: u32,
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
    pub fn new(model: Gemma4, tokenizer: Tokenizer, eos_token_id: u32) -> Self {
        Self {
            model,
            tokenizer,
            eos_token_id,
        }
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
        self.model.clear_cache();

        // Prefill
        let logits = self
            .model
            .forward_last_token_logits(input_ids, pixel_values)
            .map_err(|e| anyhow::anyhow!("prefill forward failed: {e}"))?;

        let mut generated = Vec::new();
        let mut next_token = self.sample_token(&logits, opts)?;

        for _ in 0..opts.max_tokens {
            if next_token == self.eos_token_id {
                break;
            }
            generated.push(next_token);

            // Decode step: forward only the new token
            let input = Array::from_int(next_token as i32)?.reshape(&[1, 1])?;
            let logits = self
                .model
                .forward_last_token_logits(&input, None)
                .map_err(|e| anyhow::anyhow!("decode forward failed: {e}"))?;
            next_token = self.sample_token(&logits, opts)?;
        }

        Ok(generated)
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
        let text = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;
        Ok(text)
    }

    fn sample_token(&self, logits: &Array, opts: &VlmGenerateOptions) -> Result<u32> {
        if opts.temperature <= 0.0 {
            // Greedy
            let logits = squeeze_last_token_logits(logits)?;
            let idx = logits.argmax(0)?;
            match idx.item_u32() {
                Ok(v) => Ok(v),
                Err(_) => Ok(idx.item_i32()? as u32),
            }
        } else {
            // Temperature sampling
            let logits = squeeze_last_token_logits(logits)?;
            let mut logits_vec = logits.to_vec_f32()?;
            let inv_temp = 1.0 / opts.temperature;
            for v in &mut logits_vec {
                *v *= inv_temp;
            }
            let probs = softmax(&logits_vec);
            categorical_sample(&probs)
        }
    }
}

fn squeeze_last_token_logits(logits: &Array) -> Result<Array> {
    let logits = match logits.ndim() {
        3 => logits.squeeze(0)?.squeeze(0)?.contiguous()?,
        2 => logits.squeeze(0)?.contiguous()?,
        _ => logits.contiguous()?,
    };
    Ok(squeeze_all_singletons(logits)?)
}

fn squeeze_all_singletons(mut arr: Array) -> Result<Array> {
    loop {
        let shape = arr.shape_raw();
        let mut squeezed = false;
        for axis in (0..shape.len()).rev() {
            if shape[axis] == 1 {
                arr = arr.squeeze(axis as i32)?;
                squeezed = true;
                break;
            }
        }
        if !squeezed {
            return Ok(arr);
        }
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_v = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps = Vec::with_capacity(logits.len());
    let mut sum = 0.0f32;
    for &v in logits {
        let e = (v - max_v).exp();
        exps.push(e);
        sum += e;
    }
    if sum <= 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

fn categorical_sample(probs: &[f32]) -> Result<u32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum > r {
            return Ok(i as u32);
        }
    }
    Ok((probs.len().saturating_sub(1)) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_core::DType;
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
        Tokenizer::new(model)
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
        // EOS is token 3 (</s>) in tiny tokenizer
        let mut pipeline = VlmGenerationPipeline::new(model, tokenizer, 3);

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
        // EOS is token 3 (</s>) in tiny tokenizer
        let mut pipeline = VlmGenerationPipeline::new(model, tokenizer, 3);

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
        // EOS is token 3 (</s>) in tiny tokenizer
        let mut pipeline = VlmGenerationPipeline::new(model, tokenizer, 3);

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
