use rand::Rng;

pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
}

impl Sampler {
    pub fn new(temperature: f32, top_p: f32, top_k: usize, min_p: f32) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            min_p,
        }
    }

    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
        }
    }

    pub fn sample(&self, logits: &[f32]) -> usize {
        if self.temperature <= 0.0 {
            return argmax_idx(logits);
        }

        let vocab = logits.len();
        let temp = self.temperature.max(1e-7);

        // Build indexed array: (index, logit / temperature)
        let mut indexed: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v / temp))
            .collect();

        // Top-k filtering: keep only top k tokens
        let k = if self.top_k > 0 && self.top_k < vocab {
            self.top_k
        } else {
            vocab
        };

        if k < vocab {
            // Partial sort to find top-k by score (descending)
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.truncate(k);
        }

        // Compute softmax on the remaining logits
        let max_logit = indexed
            .iter()
            .map(|(_, v)| *v)
            .fold(f32::NEG_INFINITY, f32::max);

        let mut probs: Vec<f32> = indexed
            .iter()
            .map(|(_, v)| (*v - max_logit).exp())
            .collect();

        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }

        // Top-p (nucleus) filtering
        if self.top_p > 0.0 && self.top_p < 1.0 {
            let mut cumsum = 0.0f32;
            let cutoff = {
                let mut c = probs.len();
                for (i, &p) in probs.iter().enumerate() {
                    cumsum += p;
                    if cumsum >= self.top_p {
                        c = i + 1;
                        break;
                    }
                }
                c
            };
            for p in probs.iter_mut().skip(cutoff) {
                *p = 0.0;
            }
        }

        // Min-p filtering
        if self.min_p > 0.0 && self.min_p <= 1.0 {
            let max_prob = probs.iter().cloned().fold(0.0f32, f32::max);
            let threshold = max_prob * self.min_p;
            for p in probs.iter_mut() {
                if *p < threshold {
                    *p = 0.0;
                }
            }
        }

        // Renormalize
        let total: f32 = probs.iter().sum();
        if total > 0.0 {
            for p in probs.iter_mut() {
                *p /= total;
            }
        }

        // Sample from distribution
        let mut rng = rand::rng();
        let mut r = rng.random::<f32>();
        let mut chosen = indexed[0].0;
        for (i, &p) in probs.iter().enumerate() {
            r -= p;
            if r <= 0.0 {
                chosen = indexed[i].0;
                break;
            }
        }

        chosen
    }
}

fn argmax_idx(data: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in data.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_argmax() {
        let sampler = Sampler::greedy();
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        assert_eq!(sampler.sample(&logits), 1);
    }

    #[test]
    fn test_temperature_zero_greedy() {
        let sampler = Sampler::new(0.0, 0.9, 40, 0.0);
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        assert_eq!(sampler.sample(&logits), 1);
    }

    #[test]
    fn test_top_k_limits_candidates() {
        let sampler = Sampler::new(1.0, 1.0, 2, 0.0);
        let logits = vec![10.0, 1.0, 0.0, -1.0];
        let mut counts = vec![0usize; 4];
        for _ in 0..1000 {
            let idx = sampler.sample(&logits);
            counts[idx] += 1;
        }
        // Only top-2 tokens (indices 0 and 3, since top-k by logit value)
        // Wait, top-k sorts by logit descending: index 0 (10.0), index 3 (-1.0) is last
        // Actually: sorted desc: (0, 10.0), (2?, no) - let me re-think
        // logits: [10.0, 1.0, 0.0, -1.0] - indices 0,1,2,3
        // sorted desc: (0, 10.0), (1, 1.0), (2, 0.0), (3, -1.0)
        // top-2: (0, 10.0), (1, 1.0)
        // So only indices 0 and 1 should be sampled
        assert!(counts[2] == 0, "index 2 should not be sampled with top_k=2");
        assert!(counts[3] == 0, "index 3 should not be sampled with top_k=2");
    }

    #[test]
    fn test_top_p_nucleus_sampling() {
        let sampler = Sampler::new(1.0, 0.6, 0, 0.0);
        let logits = vec![5.0, 2.0, 1.0, 0.0];
        let mut counts = vec![0usize; 4];
        for _ in 0..1000 {
            let idx = sampler.sample(&logits);
            counts[idx] += 1;
        }
        // With top_p=0.6, the most probable token(s) should dominate
        // but we can't be 100% deterministic, just check the distribution makes sense
        assert!(counts[0] > counts[3], "token 0 should be sampled more than token 3");
    }

    #[test]
    fn test_min_p_zero_is_noop() {
        let sampler = Sampler::new(1.0, 1.0, 0, 0.0);
        let logits = vec![3.0, 1.0, 0.0];
        let mut counts = vec![0usize; 3];
        for _ in 0..1000 {
            let idx = sampler.sample(&logits);
            counts[idx] += 1;
        }
        assert!(counts[0] > counts[1], "token 0 should be sampled more");
        assert!(counts[1] > counts[2], "token 1 should be sampled more");
    }

    #[test]
    fn test_high_temperature_spreads_distribution() {
        let low_temp = Sampler::new(0.1, 1.0, 0, 0.0);
        let high_temp = Sampler::new(10.0, 1.0, 0, 0.0);
        let logits = vec![5.0, 0.0, 0.0, 0.0];

        let mut low_counts = vec![0usize; 4];
        let mut high_counts = vec![0usize; 4];
        for _ in 0..1000 {
            low_counts[low_temp.sample(&logits)] += 1;
            high_counts[high_temp.sample(&logits)] += 1;
        }
        // Low temp should heavily favor token 0
        assert!(low_counts[0] > 900, "low temp should strongly prefer highest logit");
        // High temp should spread more
        assert!(high_counts[0] < low_counts[0], "high temp should spread more than low temp");
    }
}