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
}

impl Default for Sampler {
    fn default() -> Self {
        Self::new(0.6, 0.9)
    }
}
