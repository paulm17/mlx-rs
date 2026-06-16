use llama_cpp_2::sampling::LlamaSampler;

pub struct Sampler {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
}

impl Sampler {
    pub fn new(temperature: f32, top_p: f32) -> Self {
        Self {
            temperature,
            top_p,
            top_k: 40,
            min_p: 0.0,
        }
    }

    pub fn greedy() -> Self {
        Self::new(0.0, 1.0)
    }

    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    pub fn with_min_p(mut self, min_p: f32) -> Self {
        self.min_p = min_p;
        self
    }

    pub fn build_llama_sampler(&self) -> LlamaSampler {
        if self.temperature <= 0.0 {
            return LlamaSampler::greedy();
        }

        let mut samplers = Vec::new();

        if self.top_k > 0 {
            samplers.push(LlamaSampler::top_k(self.top_k as i32));
        }

        if self.top_p > 0.0 && self.top_p < 1.0 {
            samplers.push(LlamaSampler::top_p(self.top_p, 1));
        }

        if self.min_p > 0.0 {
            samplers.push(LlamaSampler::min_p(self.min_p, 1));
        }

        samplers.push(LlamaSampler::temp(self.temperature));
        samplers.push(LlamaSampler::dist(42));

        LlamaSampler::chain_simple(samplers)
    }
}

impl Default for Sampler {
    fn default() -> Self {
        Self::new(0.6, 0.9)
    }
}
