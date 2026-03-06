use anyhow::Result;
use mlx_core::Array;
use rand::Rng;
use std::collections::HashSet;

/// Token sampling strategies.
pub struct Sampler {
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
    repeat_last_n: usize,
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            temperature: 0.6,
            top_p: 0.9,
            repetition_penalty: 1.15,
            repeat_last_n: 256,
        }
    }
}

impl Sampler {
    pub fn new(temperature: f32, top_p: f32) -> Self {
        Self { temperature, top_p, ..Self::default() }
    }

    pub fn uses_host_sampling(&self) -> bool {
        self.temperature > 0.0
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    /// Sample a token from logits.
    pub fn sample(&self, logits: &Array) -> Result<u32> {
        self.sample_with_history(logits, &[])
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

    pub fn sample_raw_last_token_logits_array(&self, logits: &Array) -> Result<Array> {
        anyhow::ensure!(
            self.is_greedy(),
            "sample_raw_last_token_logits_array only supports greedy decoding"
        );
        let axis = logits.ndim().saturating_sub(1) as i32;
        let idx = logits.argmax(axis)?;
        Self::squeeze_all_singletons(idx)
    }

    pub fn sample_raw_last_token_logits(&self, logits: &Array, history: &[u32]) -> Result<u32> {
        if self.is_greedy() {
            let idx = self.sample_raw_last_token_logits_array(logits)?;
            return match idx.item_u32() {
                Ok(v) => Ok(v),
                Err(_) => Ok(idx.item_i32()? as u32),
            };
        }
        let logits = match logits.ndim() {
            3 => logits.squeeze(0)?.squeeze(0)?.contiguous()?,
            2 => logits.squeeze(0)?.contiguous()?,
            _ => logits.contiguous()?,
        };
        self.sample_with_history(&logits, history)
    }

    /// Sample a token from logits with optional repetition penalty history.
    pub fn sample_with_history(&self, logits: &Array, history: &[u32]) -> Result<u32> {
        if self.is_greedy() {
            // Greedy decoding should be pure argmax (no repetition penalty),
            // and should stay on-device to avoid per-token host allocations.
            let idx = logits.argmax(0)?;
            return match idx.item_u32() {
                Ok(v) => Ok(v),
                Err(_) => Ok(idx.item_i32()? as u32),
            };
        }

        let mut logits_vec = logits.to_vec_f32()?;
        self.apply_repetition_penalty(&mut logits_vec, history);

        // Temperature scaling
        let inv_temp = 1.0 / self.temperature;
        for v in &mut logits_vec {
            *v *= inv_temp;
        }
        let probs_vec = self.softmax(&logits_vec);

        if self.top_p >= 1.0 {
            return self.categorical_sample(&probs_vec);
        }

        // Top-p (nucleus) sampling
        self.top_p_sample(&probs_vec)
    }

    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        if logits.is_empty() {
            return Vec::new();
        }
        let max_v = logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
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

    fn apply_repetition_penalty(&self, logits: &mut [f32], history: &[u32]) {
        if self.repetition_penalty <= 1.0 || history.is_empty() {
            return;
        }
        let penalty = self.repetition_penalty;
        let window_start = history.len().saturating_sub(self.repeat_last_n);
        let seen: HashSet<usize> = history[window_start..].iter().map(|&t| t as usize).collect();
        for idx in seen {
            if let Some(v) = logits.get_mut(idx) {
                if *v > 0.0 {
                    *v /= penalty;
                } else {
                    *v *= penalty;
                }
            }
        }
    }

    fn categorical_sample(&self, probs: &[f32]) -> Result<u32> {
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen();
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum > r {
                return Ok(i as u32);
            }
        }
        Ok((probs.len() - 1) as u32)
    }

    fn top_p_sample(&self, probs: &[f32]) -> Result<u32> {
        // Sort indices by probability descending
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Accumulate until we exceed top_p
        let mut cumsum = 0.0;
        let mut filtered: Vec<(usize, f32)> = Vec::new();
        for (idx, p) in &indexed {
            cumsum += p;
            filtered.push((*idx, *p));
            if cumsum >= self.top_p {
                break;
            }
        }

        // Re-normalize and sample
        let total: f32 = filtered.iter().map(|(_, p)| p).sum();
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen::<f32>() * total;
        let mut cumsum = 0.0;
        for (idx, p) in &filtered {
            cumsum += p;
            if cumsum > r {
                return Ok(*idx as u32);
            }
        }
        Ok(filtered.last().map(|(i, _)| *i as u32).unwrap_or(0))
    }
}
