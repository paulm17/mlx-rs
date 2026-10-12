//! Named GGUF decision profiles with one resident model and metadata dispatch.
use crate::types::*;
use crate::{Backend, LlamaCppConfig};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DecisionModelProfile {
    pub model_path: Option<String>,
    pub repository: Option<String>,
    pub file: Option<String>,
    pub quantization: Option<String>,
    #[serde(flatten)]
    pub runtime: LlamaCppConfig,
}
impl DecisionModelProfile {
    pub fn resolve(&self) -> Result<std::path::PathBuf> {
        match (&self.model_path, &self.repository) {
            (Some(path), None) if self.file.is_none() && self.quantization.is_none() => {
                crate::loader::resolve_model_path(path)
            }
            (None, Some(repo)) => crate::loader::resolve_gguf_profile(
                repo,
                self.file.as_deref(),
                self.quantization.as_deref().unwrap_or("Q4_K_M"),
            ),
            _ => anyhow::bail!(
                "profile requires either model_path or repository with optional file/quantization"
            ),
        }
    }
    pub fn config(&self) -> LlamaCppConfig {
        let mut config = self.runtime.clone();
        let ctx = config.n_ctx.unwrap_or(4096);
        config.n_ctx = Some(ctx);
        config.n_batch = Some(config.n_batch.unwrap_or(ctx));
        config.n_ubatch = Some(config.n_ubatch.unwrap_or(ctx));
        config.n_gpu_layers = Some(config.n_gpu_layers.unwrap_or(99));
        config.n_seq_max = Some(1);
        config
    }
}

type Factory = Box<dyn Fn(&DecisionModelProfile) -> Result<Box<dyn Backend>> + Send>;
/// Call through `&mut self` (or an external mutex) to serialize selection and inference.
pub struct DecisionModels {
    profiles: BTreeMap<String, DecisionModelProfile>,
    default: String,
    active: Option<String>,
    loaded: Option<Box<dyn Backend>>,
    factory: Factory,
}
impl DecisionModels {
    pub fn new(profiles: BTreeMap<String, DecisionModelProfile>, default: &str) -> Result<Self> {
        Self::with_factory(
            profiles,
            default,
            Box::new(|profile| {
                let path = profile.resolve()?;
                Ok(Box::new(crate::LlamaCppBackend::new(
                    &path.to_string_lossy(),
                    profile.config(),
                )?))
            }),
        )
    }
    fn with_factory(
        profiles: BTreeMap<String, DecisionModelProfile>,
        default: &str,
        factory: Factory,
    ) -> Result<Self> {
        if !profiles.contains_key(default) {
            anyhow::bail!("unknown default decision profile: {default}");
        }
        Ok(Self {
            profiles,
            default: default.into(),
            active: None,
            loaded: None,
            factory,
        })
    }
    pub fn profiles(&self) -> &BTreeMap<String, DecisionModelProfile> {
        &self.profiles
    }
    pub fn active_profile(&self) -> Option<&str> {
        self.active.as_deref()
    }
    pub fn select(&mut self, id: &str) -> Result<()> {
        if !self.profiles.contains_key(id) {
            return Err(DecisionError::InvalidRequest(format!(
                "unknown decision model profile: {id}"
            ))
            .into());
        }
        if self.active.as_deref() == Some(id) && self.loaded.is_some() {
            return Ok(());
        }
        // Release native model/context before creating the replacement. Failed loads leave no resident model.
        drop(self.loaded.take());
        self.active = None;
        let loaded = (self.factory)(&self.profiles[id])
            .with_context(|| format!("loading decision profile {id}"))?;
        if !loaded.supports_decisions() {
            return Err(DecisionError::Unsupported(format!(
                "profile {id} is not a supported decision model"
            ))
            .into());
        }
        self.loaded = Some(loaded);
        self.active = Some(id.into());
        Ok(())
    }
    fn current(&self) -> Result<&(dyn Backend + '_)> {
        match self.loaded.as_deref() {
            Some(b) => Ok(b),
            None => Err(DecisionError::Inference("no decision model loaded".into()).into()),
        }
    }
    fn current_mut(&mut self) -> Result<&mut (dyn Backend + '_)> {
        match self.loaded.as_deref_mut() {
            Some(b) => Ok(b),
            None => Err(DecisionError::Inference("no decision model loaded".into()).into()),
        }
    }
}
impl Backend for DecisionModels {
    fn supports_decisions(&self) -> bool {
        true
    }
    fn decide(&mut self, request: &DecisionRequest) -> Result<DecisionResponse> {
        request.validate_text()?;
        let id = request.model.as_deref().unwrap_or(&self.default).to_owned();
        self.select(&id)?;
        let mut native = request.clone();
        native.model = None;
        let mut response = self.current_mut()?.decide(&native)?;
        response.model = id;
        Ok(response)
    }
    fn select_model(&mut self, id: &str) -> Result<()> {
        self.select(id)
    }
    fn available_models(&self) -> Vec<LoadedModelInfo> {
        self.profiles
            .iter()
            .map(|(id, profile)| {
                if self.active.as_ref() == Some(id) {
                    self.model_info()
                } else {
                    LoadedModelInfo {
                        supports_decisions: true,
                        decision_family: None,
                        model_path: id.clone(),
                        context_length: Some(profile.config().n_ctx.unwrap() as usize),
                        embedding_dimension: None,
                        vocab_size: None,
                    }
                }
            })
            .collect()
    }
    fn model_info(&self) -> LoadedModelInfo {
        let mut info = self
            .loaded
            .as_ref()
            .map(|b| b.model_info())
            .unwrap_or(LoadedModelInfo {
                supports_decisions: true,
                decision_family: None,
                model_path: self.default.clone(),
                context_length: None,
                embedding_dimension: None,
                vocab_size: None,
            });
        if let Some(id) = &self.active {
            info.model_path = id.clone();
        }
        info
    }
    fn tokenize(&self, text: &str, add_bos: bool) -> Result<Vec<i32>> {
        self.current()?.tokenize(text, add_bos)
    }
    fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        self.current()?.detokenize(tokens)
    }
    fn detokenize_piece(&self, id: i32) -> Result<String> {
        self.current()?.detokenize_piece(id)
    }
    fn is_eog(&self, id: i32) -> bool {
        self.current().is_ok_and(|b| b.is_eog(id))
    }
    fn token_eos(&self) -> i32 {
        self.current().map_or(-1, |b| b.token_eos())
    }
    fn embeddings_enabled(&self) -> bool {
        false
    }
    fn supports_chat_template(&self) -> bool {
        false
    }
    fn apply_chat_template(&self, _: &[ChatMessage]) -> Result<String> {
        anyhow::bail!("decision models do not generate text; use /v1/systemone")
    }
    fn generate(&mut self, _: &str, _: &GenerationOptions) -> Result<GenerateOutput> {
        anyhow::bail!("decision models do not generate text; use /v1/systemone")
    }
    fn generate_stream(
        &mut self,
        _: &str,
        _: &GenerationOptions,
        _: Box<dyn FnMut(&str) -> bool + Send>,
    ) -> Result<GenerationMetrics> {
        anyhow::bail!("decision models do not generate text; use /v1/systemone")
    }
    fn generate_stream_output(
        &mut self,
        _: &str,
        _: &GenerationOptions,
        _: Box<dyn FnMut(&str) -> bool + Send>,
    ) -> Result<GenerateOutput> {
        anyhow::bail!("decision models do not generate text; use /v1/systemone")
    }
    fn embed(&mut self, _: &str) -> Result<EmbeddingOutput> {
        anyhow::bail!("decision models do not expose text embeddings")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    struct Stub {
        name: String,
        resident: Arc<AtomicUsize>,
        supported: bool,
    }
    impl Drop for Stub {
        fn drop(&mut self) {
            self.resident.fetch_sub(1, Ordering::SeqCst);
        }
    }
    impl Backend for Stub {
        fn supports_decisions(&self) -> bool {
            self.supported
        }
        fn decide(&mut self, _: &DecisionRequest) -> Result<DecisionResponse> {
            Ok(serde_json::from_value(
                serde_json::json!({"model":self.name,"answers":{"x":{"type":"noul","noul":0.75}},"usage":{"input_tokens":10,"output_tokens":0}}),
            )?)
        }
        fn model_info(&self) -> LoadedModelInfo {
            LoadedModelInfo {
                supports_decisions: self.supported,
                decision_family: Some("kev".into()),
                model_path: self.name.clone(),
                context_length: Some(4096),
                embedding_dimension: None,
                vocab_size: None,
            }
        }
        fn tokenize(&self, _: &str, _: bool) -> Result<Vec<i32>> {
            Ok(vec![])
        }
        fn detokenize(&self, _: &[i32]) -> Result<String> {
            Ok(String::new())
        }
        fn detokenize_piece(&self, _: i32) -> Result<String> {
            Ok(String::new())
        }
        fn is_eog(&self, _: i32) -> bool {
            false
        }
        fn token_eos(&self) -> i32 {
            0
        }
        fn embeddings_enabled(&self) -> bool {
            false
        }
        fn apply_chat_template(&self, _: &[ChatMessage]) -> Result<String> {
            unreachable!()
        }
        fn generate(&mut self, _: &str, _: &GenerationOptions) -> Result<GenerateOutput> {
            unreachable!()
        }
        fn generate_stream(
            &mut self,
            _: &str,
            _: &GenerationOptions,
            _: Box<dyn FnMut(&str) -> bool + Send>,
        ) -> Result<GenerationMetrics> {
            unreachable!()
        }
        fn generate_stream_output(
            &mut self,
            _: &str,
            _: &GenerationOptions,
            _: Box<dyn FnMut(&str) -> bool + Send>,
        ) -> Result<GenerateOutput> {
            unreachable!()
        }
        fn embed(&mut self, _: &str) -> Result<EmbeddingOutput> {
            unreachable!()
        }
    }
    fn request(model: Option<&str>) -> DecisionRequest {
        serde_json::from_value(serde_json::json!({"model":model,"state":{"text":"hi"},"questions":{"x":{"type":"noul","instructions":"greeting?"}}})).unwrap()
    }
    #[test]
    fn profiles_switch_drop_before_load_and_recover() {
        let resident = Arc::new(AtomicUsize::new(0));
        let loads = Arc::new(AtomicUsize::new(0));
        let profiles = ["a", "b", "broken", "ordinary"]
            .into_iter()
            .map(|id| {
                (
                    id.to_owned(),
                    DecisionModelProfile {
                        model_path: Some(id.into()),
                        ..Default::default()
                    },
                )
            })
            .collect();
        let count = resident.clone();
        let calls = loads.clone();
        let mut models = DecisionModels::with_factory(
            profiles,
            "a",
            Box::new(move |p| {
                assert_eq!(
                    count.load(Ordering::SeqCst),
                    0,
                    "old model must be freed before loading"
                );
                calls.fetch_add(1, Ordering::SeqCst);
                let name = p.model_path.clone().unwrap();
                if name == "broken" {
                    anyhow::bail!("fixture load failure");
                }
                count.fetch_add(1, Ordering::SeqCst);
                Ok(Box::new(Stub {
                    supported: name != "ordinary",
                    name,
                    resident: count.clone(),
                }))
            }),
        )
        .unwrap();
        assert_eq!(models.decide(&request(None)).unwrap().model, "a");
        models.decide(&request(None)).unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert!(models
            .decide(&request(Some("unknown")))
            .unwrap_err()
            .downcast_ref::<DecisionError>()
            .is_some());
        assert_eq!(resident.load(Ordering::SeqCst), 1);
        assert_eq!(models.decide(&request(Some("b"))).unwrap().model, "b");
        assert_eq!(models.model_info().decision_family.as_deref(), Some("kev"));
        assert_eq!(models.available_models().len(), 4);
        assert!(models
            .decide(&request(Some("broken")))
            .unwrap_err()
            .to_string()
            .contains("broken"));
        assert_eq!(models.active_profile(), None);
        assert_eq!(resident.load(Ordering::SeqCst), 0);
        assert!(models.decide(&request(Some("ordinary"))).is_err());
        assert_eq!(resident.load(Ordering::SeqCst), 0);
        assert_eq!(models.decide(&request(None)).unwrap().model, "a");
        drop(models);
        assert_eq!(resident.load(Ordering::SeqCst), 0);
    }
    #[test]
    fn defaults_and_profile_source_validation() {
        let p = DecisionModelProfile::default();
        assert_eq!(p.config().n_gpu_layers, Some(99));
        assert_eq!(p.config().n_batch, Some(4096));
        assert!(p.resolve().is_err());
        assert!(DecisionModels::new(BTreeMap::new(), "missing").is_err());
        let p = DecisionModelProfile {
            repository: Some("owner/repo".into()),
            file: Some("../escape.gguf".into()),
            ..Default::default()
        };
        assert!(p.resolve().is_err());
    }
}
