//! Real-model tests are opt-in and never download weights.
use llama_lm::runtime::Runtime;
use llama_lm::{DecisionAnswer, DecisionRequest, LlamaCppConfig};
use serde_json::json;

fn request(state: &str) -> DecisionRequest {
    serde_json::from_value(json!({"state":state,"questions":{
        "route":{"type":"choice","instructions":"Which team should handle this?","criteria":{"technical":null,"billing":null,"shipping":null}},
        "angry":{"type":"noul","instructions":"Is the customer angry?"},
        "urgency":{"type":"score","instructions":"How urgent is this?","criteria":["can wait","this week","today","right now"]}
    }})).unwrap()
}

#[test]
#[ignore = "requires MLX_RS_TEST_CLEF_GGUF local weights"]
fn clef_native_repeated_requests_reload_and_boundaries() {
    let path =
        std::env::var("MLX_RS_TEST_CLEF_GGUF").expect("set MLX_RS_TEST_CLEF_GGUF to local weights");
    let config = LlamaCppConfig {
        n_ctx: Some(4096),
        n_batch: Some(4096),
        n_ubatch: Some(4096),
        n_gpu_layers: Some(99),
        n_threads: Some(4),
        n_threads_batch: Some(4),
        logging: std::env::var("MLX_RS_TEST_CLEF_LOG").ok(),
        ..Default::default()
    };
    let first = request("I was charged twice for my order last week and nobody has replied.");
    let mut runtime = Runtime::new(&path, config.clone()).unwrap();
    assert!(runtime.supports_decisions());
    assert!(!runtime.embeddings_enabled());
    let output = runtime.decide(&first).unwrap();
    assert_eq!(output.answers.len(), 3);
    assert_eq!(output.usage.output_tokens, 0);
    for answer in output.answers.values() {
        match answer {
            DecisionAnswer::Choice {
                probabilities,
                confidence,
                ..
            }
            | DecisionAnswer::Score {
                probabilities,
                confidence,
                ..
            } => {
                assert!((probabilities.values().sum::<f64>() - 1.0).abs() < 1e-6);
                assert!((0.0..=1.0).contains(confidence));
            }
            DecisionAnswer::Noul { noul } => assert!((0.0..=1.0).contains(noul)),
        }
    }
    runtime
        .decide(&request("The parcel arrived safely. Thank you!"))
        .unwrap();
    let again = runtime.decide(&first).unwrap();
    assert_eq!(
        serde_json::to_value(&output).unwrap(),
        serde_json::to_value(&again).unwrap()
    );
    let oversized = request(&"long customer message ".repeat(5000));
    let error = runtime.decide(&oversized).unwrap_err();
    assert!(matches!(
        error.downcast_ref::<llama_lm::DecisionError>(),
        Some(llama_lm::DecisionError::InvalidRequest(_))
    ));
    assert_eq!(
        serde_json::to_value(runtime.decide(&first).unwrap()).unwrap(),
        serde_json::to_value(&output).unwrap()
    );
    assert!(runtime.generate("hello", &Default::default()).is_err());
    drop(runtime);
    let mut reloaded = Runtime::new(&path, config.clone()).unwrap();
    assert_eq!(
        serde_json::to_value(reloaded.decide(&first).unwrap()).unwrap(),
        serde_json::to_value(output).unwrap()
    );
    drop(reloaded);
    let mut bounded = Runtime::new(
        &path,
        LlamaCppConfig {
            n_batch: Some(256),
            n_ubatch: Some(256),
            ..config
        },
    )
    .unwrap();
    let error = bounded.decide(&first).unwrap_err();
    assert!(matches!(
        error.downcast_ref::<llama_lm::DecisionError>(),
        Some(llama_lm::DecisionError::InvalidRequest(_))
    ));
    let small: DecisionRequest = serde_json::from_value(json!({
        "state":"hello",
        "questions":{"greeting":{"type":"noul","instructions":"Is this a greeting?"}}
    }))
    .unwrap();
    assert!(bounded.decide(&small).unwrap().usage.input_tokens <= 256);
}

#[test]
#[ignore = "requires local Clef weights and a matching upstream server"]
fn clef_matches_upstream_systemone() {
    let path = std::env::var("MLX_RS_TEST_CLEF_GGUF").expect("set MLX_RS_TEST_CLEF_GGUF");

    let mut runtime = Runtime::new(
        &path,
        LlamaCppConfig {
            n_ctx: Some(4096),
            n_batch: Some(4096),
            n_ubatch: Some(4096),
            n_gpu_layers: Some(99),
            ..Default::default()
        },
    )
    .unwrap();
    let input = request("I was charged twice for my order last week and nobody has replied.");
    let native = runtime.decide(&input).unwrap();
    let upstream: llama_lm::DecisionResponse =
        if let Ok(file) = std::env::var("MLX_RS_TEST_CLEF_REFERENCE_JSON") {
            serde_json::from_slice(&std::fs::read(file).unwrap()).unwrap()
        } else {
            let url = std::env::var("MLX_RS_TEST_CLEF_REFERENCE_URL")
                .expect("set reference URL or captured JSON file");
            reqwest::blocking::Client::new()
                .post(format!("{}/v1/systemone", url.trim_end_matches('/')))
                .json(&input)
                .send()
                .unwrap()
                .error_for_status()
                .unwrap()
                .json()
                .unwrap()
        };
    assert_eq!(native.usage.input_tokens, upstream.usage.input_tokens);
    fn compare(a: &serde_json::Value, b: &serde_json::Value) {
        match (a, b) {
            (serde_json::Value::Number(a), serde_json::Value::Number(b)) => assert!(
                (a.as_f64().unwrap() - b.as_f64().unwrap()).abs() < 1e-5,
                "{a} != {b}"
            ),
            (serde_json::Value::Object(a), serde_json::Value::Object(b)) => {
                assert_eq!(a.len(), b.len());
                for (key, value) in a {
                    compare(value, &b[key]);
                }
            }
            _ => assert_eq!(a, b),
        }
    }
    compare(
        &serde_json::to_value(native.answers).unwrap(),
        &serde_json::to_value(upstream.answers).unwrap(),
    );
}
