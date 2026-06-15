//! Gemma4 block-diffusion model support.
//!
//! This module is intentionally separate from `gemma4.rs`: DiffusionGemma
//! checkpoints use Gemma4 components, but generation is block-diffusion rather
//! than autoregressive.

use crate::gemma4::{Gemma4TextConfig, LanguageModel};
use mlx_core::{Array, DType, Module, Result};
use mlx_nn::{Linear, QuantConfig, VarBuilder};
use rand::Rng;
use std::collections::HashMap;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Gemma4DiffusionConfig {
    pub canvas_length: usize,
    #[serde(default)]
    pub generation_config: Option<Gemma4DiffusionGenerationConfig>,
    pub text_config: Gemma4TextConfig,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Gemma4DiffusionGenerationConfig {
    pub max_denoising_steps: Option<usize>,
    pub max_new_tokens: Option<usize>,
    pub t_min: Option<f32>,
    pub t_max: Option<f32>,
    pub sampler_config: Option<Gemma4DiffusionSamplerConfig>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Gemma4DiffusionSamplerConfig {
    pub entropy_bound: Option<f32>,
}

pub struct Gemma4Diffusion {
    pub language_model: LanguageModel,
    self_conditioning: SelfConditioning,
    encoder_layer_scalars: Vec<Array>,
    config: Gemma4DiffusionConfig,
}

#[derive(Debug, Clone)]
pub struct Gemma4DiffusionDraft {
    pub finalized_token_ids: Vec<u32>,
    pub canvas_index: usize,
    pub denoising_step: usize,
    pub canvas_token_ids: Vec<u32>,
    pub accepted_mask: Vec<bool>,
}

struct SelfConditioning {
    pre_norm_weight: Array,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    eps: f32,
}

impl SelfConditioning {
    fn new(vb: &VarBuilder, cfg: &Gemma4TextConfig) -> anyhow::Result<Self> {
        let qc = QuantConfig::default();
        Ok(Self {
            pre_norm_weight: vb.pp("pre_norm").get("weight")?,
            gate_proj: Linear::new(&vb.pp("gate_proj"), &qc)?,
            up_proj: Linear::new(&vb.pp("up_proj"), &qc)?,
            down_proj: Linear::new(&vb.pp("down_proj"), &qc)?,
            eps: cfg.rms_norm_eps as f32,
        })
    }

    fn forward(&self, inputs_embeds: &Array, signal: &Array) -> Result<Array> {
        let normed = signal.fast_rms_norm(&self.pre_norm_weight, self.eps)?;
        let gate = self.gate_proj.forward(&normed)?;
        let up = self.up_proj.forward(&normed)?;
        let hidden = gelu_approx(&gate)?.multiply(&up)?;
        let signal = self.down_proj.forward(&hidden)?;
        let combined = inputs_embeds.add(&signal)?;
        let ones = Array::ones(
            &[combined.shape_raw().last().copied().unwrap_or(1)],
            combined.dtype(),
        )?;
        combined.fast_rms_norm(&ones, self.eps)
    }
}

fn gelu_approx(x: &Array) -> Result<Array> {
    let dt = x.dtype();
    let coeff = Array::from_float(0.044715f32)?.as_type(dt)?;
    let sqrt_2_over_pi = Array::from_float(0.7978845608f32)?.as_type(dt)?;
    let one = Array::from_float(1.0f32)?.as_type(dt)?;
    let half = Array::from_float(0.5f32)?.as_type(dt)?;

    let x_cubed = x.multiply(x)?.multiply(x)?;
    let inner = x.add(&x_cubed.multiply(&coeff)?)?;
    let tanh_arg = inner.multiply(&sqrt_2_over_pi)?;
    let tanh_val = tanh_arg.tanh()?;
    x.multiply(&half)?.multiply(&one.add(&tanh_val)?)
}

fn split_gate_up_tensor(value: Array) -> Option<(Array, Array)> {
    let shape = value.shape_raw();
    if shape.len() < 2 {
        return None;
    }

    let split_axis = shape.len() - 2;
    let split_dim = shape[split_axis] as usize;
    if split_dim == 0 || split_dim % 2 != 0 {
        return None;
    }

    let mid = split_dim / 2;

    let start_gate = vec![0i32; shape.len()];
    let mut stop_gate = shape.clone();
    stop_gate[split_axis] = mid as i32;
    let gate = value.slice(&start_gate, &stop_gate).ok()?;

    let mut start_up = vec![0i32; shape.len()];
    start_up[split_axis] = mid as i32;
    let up = value.slice(&start_up, &shape).ok()?;

    Some((gate, up))
}

pub fn sanitize_gemma4_diffusion_weights(
    weights: HashMap<String, Array>,
) -> HashMap<String, Array> {
    let mut sanitized = HashMap::with_capacity(weights.len());

    for (key, value) in weights {
        if let Some(stripped) = key.strip_prefix("model.decoder.self_conditioning.") {
            sanitized.insert(format!("self_conditioning.{stripped}"), value);
            continue;
        }

        if let Some(stripped) = key.strip_prefix("model.encoder.language_model.") {
            if stripped.ends_with(".layer_scalar") {
                sanitized.insert(format!("encoder.language_model.{stripped}"), value);
            }
            continue;
        }

        let Some(stripped) = key.strip_prefix("model.decoder.") else {
            continue;
        };

        let new_key = format!("language_model.model.{stripped}");

        if let Some(suffix) = new_key.strip_suffix(".experts.down_proj.weight") {
            sanitized.insert(
                format!("{suffix}.experts.switch_glu.down_proj.weight"),
                value,
            );
            continue;
        }
        if let Some(suffix) = new_key.strip_suffix(".experts.down_proj.scales") {
            sanitized.insert(
                format!("{suffix}.experts.switch_glu.down_proj.scales"),
                value,
            );
            continue;
        }
        if let Some(suffix) = new_key.strip_suffix(".experts.down_proj.biases") {
            sanitized.insert(
                format!("{suffix}.experts.switch_glu.down_proj.biases"),
                value,
            );
            continue;
        }

        if let Some(base) = new_key.strip_suffix(".experts.gate_up_proj.weight") {
            if let Some((gate, up)) = split_gate_up_tensor(value) {
                sanitized.insert(format!("{base}.experts.switch_glu.gate_proj.weight"), gate);
                sanitized.insert(format!("{base}.experts.switch_glu.up_proj.weight"), up);
            }
            continue;
        }
        if let Some(base) = new_key.strip_suffix(".experts.gate_up_proj.scales") {
            if let Some((gate, up)) = split_gate_up_tensor(value) {
                sanitized.insert(format!("{base}.experts.switch_glu.gate_proj.scales"), gate);
                sanitized.insert(format!("{base}.experts.switch_glu.up_proj.scales"), up);
            }
            continue;
        }
        if let Some(base) = new_key.strip_suffix(".experts.gate_up_proj.biases") {
            if let Some((gate, up)) = split_gate_up_tensor(value) {
                sanitized.insert(format!("{base}.experts.switch_glu.gate_proj.biases"), gate);
                sanitized.insert(format!("{base}.experts.switch_glu.up_proj.biases"), up);
            }
            continue;
        }

        sanitized.insert(new_key, value);
    }

    sanitized
}

impl Gemma4Diffusion {
    pub fn new(vb: &VarBuilder, config: &Gemma4DiffusionConfig) -> anyhow::Result<Self> {
        let language_model = LanguageModel::new(&vb.pp("language_model"), &config.text_config)?;
        let self_conditioning =
            SelfConditioning::new(&vb.pp("self_conditioning"), &config.text_config)?;
        let mut encoder_layer_scalars = Vec::with_capacity(config.text_config.num_hidden_layers);
        for i in 0..config.text_config.num_hidden_layers {
            encoder_layer_scalars.push(
                vb.pp(&format!("encoder.language_model.layers.{i}"))
                    .get("layer_scalar")?,
            );
        }
        Ok(Self {
            language_model,
            self_conditioning,
            encoder_layer_scalars,
            config: config.clone(),
        })
    }

    pub fn forward_hidden_states(&mut self, input_ids: &Array) -> Result<Array> {
        self.language_model
            .forward_hidden_states_with_layer_scalars(input_ids, &self.encoder_layer_scalars)
    }

    pub fn clear_cache(&mut self) {
        self.language_model.model.clear_cache();
    }

    pub fn default_max_tokens(&self) -> usize {
        self.config
            .generation_config
            .as_ref()
            .and_then(|cfg| cfg.max_new_tokens)
            .unwrap_or(256)
    }

    pub fn max_denoising_steps(&self) -> usize {
        self.config
            .generation_config
            .as_ref()
            .and_then(|cfg| cfg.max_denoising_steps)
            .unwrap_or(48)
    }

    fn temperature_for_step(&self, cur_step: usize) -> f32 {
        let generation_config = self.config.generation_config.as_ref();
        let t_min = generation_config.and_then(|cfg| cfg.t_min).unwrap_or(0.4);
        let t_max = generation_config.and_then(|cfg| cfg.t_max).unwrap_or(0.8);
        let max_steps = self.max_denoising_steps().max(1) as f32;
        t_min + ((t_max - t_min) * (cur_step as f32 / max_steps))
    }

    fn entropy_bound(&self) -> f32 {
        self.config
            .generation_config
            .as_ref()
            .and_then(|cfg| cfg.sampler_config.as_ref())
            .and_then(|cfg| cfg.entropy_bound)
            .unwrap_or(0.1)
    }

    fn initialize_canvas(
        &self,
        _canvas_index: usize,
        _step_index: usize,
        canvas_len: usize,
    ) -> Result<Array> {
        let vocab = self.config.text_config.vocab_size.max(1);
        let mut rng = rand::thread_rng();
        let mut ids = Vec::with_capacity(canvas_len);
        for _ in 0..canvas_len {
            ids.push(rng.gen_range(0..vocab) as i32);
        }
        Array::from_slice_i32(&ids)?.reshape(&[1, canvas_len as i32])
    }

    fn self_conditioning_embeddings_from_logits(&self, logits: &Array) -> Result<Array> {
        let probs = logits.softmax(-1)?;
        let mut embeddings = self
            .language_model
            .model
            .embed_tokens
            .embed_probabilities(&probs)?;
        embeddings =
            embeddings.multiply(&self.language_model.model.embed_scale)?;
        Ok(embeddings)
    }

    fn canvas_logits(
        &mut self,
        canvas: &Array,
        self_conditioning_signal: Option<&Array>,
    ) -> Result<Array> {
        let prefix_cache = self.language_model.model.caches.clone();
        let mut embeddings = self.language_model.model.embed_tokens.forward(canvas)?;
        embeddings =
            embeddings.multiply(&self.language_model.model.embed_scale)?;
        let zero_signal;
        let signal = if let Some(signal) = self_conditioning_signal {
            signal
        } else {
            zero_signal = Array::zeros(&embeddings.shape_raw(), embeddings.dtype())?;
            &zero_signal
        };
        embeddings = self.self_conditioning.forward(&embeddings, signal)?;
        let (logits, _) = self
            .language_model
            .forward_diffusion_decoder_logits(&embeddings, &prefix_cache)?;
        Ok(logits)
    }

    pub fn trace_one_diffusion_step(
        &mut self,
        input_ids: &Array,
        canvas_ids: &Array,
        cur_step: usize,
    ) -> anyhow::Result<HashMap<String, Array>> {
        let mut traces = HashMap::new();
        traces.insert("input_ids".to_string(), input_ids.clone());
        traces.insert("canvas_ids".to_string(), canvas_ids.clone());

        self.clear_cache();
        let _ = self.language_model.trace_encoder_with_layer_scalars(
            input_ids,
            &self.encoder_layer_scalars,
            &mut traces,
        )?;

        let prefix_cache = self.language_model.model.caches.clone();
        let mut canvas_embeddings = self.language_model.model.embed_tokens.forward(canvas_ids)?;
        canvas_embeddings =
            canvas_embeddings.multiply(&self.language_model.model.embed_scale)?;
        traces.insert(
            "decoder.canvas_embeddings_raw".to_string(),
            canvas_embeddings.clone(),
        );

        let zero_signal = Array::zeros(&canvas_embeddings.shape_raw(), canvas_embeddings.dtype())?;
        traces.insert("decoder.self_conditioning_signal".to_string(), zero_signal.clone());
        let conditioned_embeddings =
            self.self_conditioning.forward(&canvas_embeddings, &zero_signal)?;
        traces.insert(
            "decoder.canvas_embeddings_conditioned".to_string(),
            conditioned_embeddings.clone(),
        );

        let raw_logits = self.language_model.trace_diffusion_decoder_logits(
            &conditioned_embeddings,
            &prefix_cache,
            &mut traces,
        )?;
        let processed_logits =
            raw_logits.divide(&Array::from_float(self.temperature_for_step(cur_step))?)?;
        traces.insert(
            "decoder.processed_logits".to_string(),
            processed_logits.clone(),
        );
        traces.insert(
            "decoder.argmax_canvas".to_string(),
            processed_logits.argmax(-1)?.as_type(canvas_ids.dtype())?,
        );

        let (_sampled, accept_mask, _accepted) =
            Self::sample_canvas_and_accept_mask(&processed_logits, self.entropy_bound())?;
        traces.insert(
            "decoder.entropy_accept_mask".to_string(),
            accept_mask.as_type(DType::Int32)?,
        );

        Ok(traces)
    }

    fn sample_canvas_and_accept_mask(
        logits: &Array,
        entropy_bound: f32,
    ) -> anyhow::Result<(Array, Array, Vec<bool>)> {
        let shape = logits.shape_raw();
        anyhow::ensure!(
            shape.len() == 3,
            "diffusion logits must be [batch, canvas, vocab], got {shape:?}"
        );
        let batch = shape[0] as usize;
        let canvas_len = shape[1] as usize;
        let vocab = shape[2] as usize;
        anyhow::ensure!(vocab > 0, "cannot sample from empty vocabulary");

        let values = logits.as_type(DType::Float32)?.to_vec_f32()?;
        let mut rng = rand::thread_rng();
        let mut sampled_ids = Vec::with_capacity(batch * canvas_len);
        let mut entropies = Vec::with_capacity(batch * canvas_len);

        for row in values.chunks_exact(vocab) {
            let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            let mut weighted_logits = 0.0f32;
            for &logit in row {
                let exp = (logit - max_v).exp();
                sum_exp += exp;
                weighted_logits += exp * logit;
            }

            let total = sum_exp.max(f32::MIN_POSITIVE);
            let target = rng.gen::<f32>() * total;
            let mut cumulative = 0.0f32;
            let mut sampled = vocab - 1;
            for (idx, &logit) in row.iter().enumerate() {
                cumulative += (logit - max_v).exp();
                if cumulative > target {
                    sampled = idx;
                    break;
                }
            }
            sampled_ids.push(sampled as i32);

            let logsumexp = max_v + total.ln();
            let expected_logit = weighted_logits / total;
            entropies.push(logsumexp - expected_logit);
        }

        let mut mask = vec![0i32; batch * canvas_len];
        for b in 0..batch {
            let row_start = b * canvas_len;
            let mut indexed = entropies[row_start..row_start + canvas_len]
                .iter()
                .copied()
                .enumerate()
                .collect::<Vec<_>>();
            indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

            let mut cumulative = 0.0f32;
            for (rank, (idx, entropy)) in indexed.into_iter().enumerate() {
                cumulative += entropy;
                if cumulative - entropy <= entropy_bound || rank == 0 {
                    mask[row_start + idx] = 1;
                } else {
                    break;
                }
            }
        }

        let accepted_mask = mask.iter().map(|&value| value != 0).collect::<Vec<_>>();
        let sampled_canvas = Array::from_slice_i32(&sampled_ids)?.reshape(&[shape[0], shape[1]])?;
        let accept_mask = Array::from_slice_i32(&mask)?
            .reshape(&[shape[0], shape[1]])?
            .greater(&Array::from_int(0)?)?;
        Ok((sampled_canvas, accept_mask, accepted_mask))
    }

    pub fn generate_block_diffusion_token_ids(
        &mut self,
        input_ids: &Array,
        max_tokens: Option<usize>,
    ) -> anyhow::Result<Vec<u32>> {
        self.generate_block_diffusion_token_ids_with_drafts(input_ids, max_tokens, |_draft| Ok(()))
    }

    pub fn generate_block_diffusion_token_ids_with_drafts<F>(
        &mut self,
        input_ids: &Array,
        max_tokens: Option<usize>,
        mut on_draft: F,
    ) -> anyhow::Result<Vec<u32>>
    where
        F: FnMut(Gemma4DiffusionDraft) -> anyhow::Result<()>,
    {
        self.clear_cache();
        let _ = self.forward_hidden_states(input_ids)?;

        let max_tokens = max_tokens.unwrap_or_else(|| self.default_max_tokens());
        let max_denoising_steps = self.max_denoising_steps();
        let canvas_limit = self.config.canvas_length.max(1);
        let min_canvas = canvas_limit.min(64);
        let mut generated = Vec::with_capacity(max_tokens);
        let mut canvas_index = 0usize;

        while generated.len() < max_tokens {
            canvas_index += 1;
            let remaining = max_tokens - generated.len();
            let canvas_len = canvas_limit.min(remaining.max(min_canvas)).min(remaining);
            let mut canvas = self.initialize_canvas(canvas_index, 0, canvas_len)?;
            let mut self_conditioning_signal: Option<Array> = None;

            for cur_step in (1..=max_denoising_steps).rev() {
                let logits = self.canvas_logits(&canvas, self_conditioning_signal.as_ref())?;
                let processed_logits =
                    logits.divide(&Array::from_float(self.temperature_for_step(cur_step))?)?;
                let argmax_canvas = processed_logits.argmax(-1)?.as_type(input_ids.dtype())?;

                if cur_step == 1 {
                    canvas = argmax_canvas.clone();
                    let ids = canvas.as_type(DType::Int32)?.to_vec_i32()?;
                    on_draft(Gemma4DiffusionDraft {
                        finalized_token_ids: generated.clone(),
                        canvas_index,
                        denoising_step: cur_step,
                        canvas_token_ids: ids.into_iter().map(|id| id as u32).collect(),
                        accepted_mask: vec![true; canvas_len],
                    })?;
                } else {
                    let (denoiser_canvas, accept_mask, accepted_mask) =
                        Self::sample_canvas_and_accept_mask(
                            &processed_logits,
                            self.entropy_bound(),
                        )?;
                    let argmax_ids = argmax_canvas.as_type(DType::Int32)?.to_vec_i32()?;
                    on_draft(Gemma4DiffusionDraft {
                        finalized_token_ids: generated.clone(),
                        canvas_index,
                        denoising_step: cur_step,
                        canvas_token_ids: argmax_ids.into_iter().map(|id| id as u32).collect(),
                        accepted_mask,
                    })?;
                    let random_canvas =
                        self.initialize_canvas(canvas_index, cur_step, canvas_len)?;
                    canvas = accept_mask.where_cond(&denoiser_canvas, &random_canvas)?;
                }

                self_conditioning_signal =
                    Some(self.self_conditioning_embeddings_from_logits(&processed_logits)?);
            }

            let ids = canvas.as_type(DType::Int32)?.to_vec_i32()?;
            for id in ids {
                generated.push(id as u32);
                if generated.len() >= max_tokens {
                    break;
                }
            }

            let _ = self.forward_hidden_states(&canvas)?;
        }

        Ok(generated)
    }
}
