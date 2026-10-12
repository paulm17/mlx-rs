//! Opt-in tests using local EmbeddingGemma 2 weights; never download models.
use llama_lm::runtime::Runtime;
use llama_lm::LlamaCppConfig;
use serde_json::{json, Value};

fn runtime() -> Runtime {
    let path = std::env::var("MLX_RS_TEST_EMBED_GGUF").expect("set MLX_RS_TEST_EMBED_GGUF");
    Runtime::new(
        &path,
        LlamaCppConfig {
            n_ctx: Some(512),
            n_batch: Some(512),
            n_ubatch: Some(512),
            n_seq_max: Some(4),
            embedding: Some(true),
            // Use the GGUF's pooling metadata.
            ..Default::default()
        },
    )
    .expect("load EmbeddingGemma 2")
}

fn valid(vector: &[f32]) {
    assert_eq!(vector.len(), 768);
    assert!(vector.iter().all(|x| x.is_finite()));
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "norm {norm}");
}

fn close(a: &[f32], b: &[f32]) {
    valid(a);
    valid(b);
    let delta = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    // Metal may select different kernels for single and batched inputs.
    assert!(delta < 1e-3, "maximum element difference {delta}");
    let cosine: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    assert!(cosine > 0.99999, "cosine similarity {cosine}");
}

const TEXTS: [&str; 2] = [
    "A gardener plants tomatoes and waters the vegetable garden.",
    "The database server rejected a transaction due to a deadlock.",
];

#[test]
#[ignore = "requires MLX_RS_TEST_EMBED_GGUF pointing to EmbeddingGemma 2"]
fn embeddinggemma2_single_batch_repeat_and_dimensions() {
    let mut rt = runtime();
    assert_eq!(rt.model().n_embd(), 512);
    assert_eq!(rt.embedding_dimension(), 768);
    assert_eq!(rt.model_info().embedding_dimension, Some(768));
    let singles: Vec<_> = TEXTS.iter().map(|text| rt.embed(text).unwrap()).collect();
    for vector in &singles {
        valid(vector);
    }
    let distance: f32 = singles[0]
        .iter()
        .zip(&singles[1])
        .map(|(x, y)| (x - y).powi(2))
        .sum();
    assert!(
        distance > 1e-3,
        "different text must yield different vectors"
    );
    let batch = rt.embed_batch(&TEXTS.map(String::from)).unwrap();
    assert_eq!(batch.len(), 2);
    for (single, batched) in singles.iter().zip(&batch) {
        close(single, batched);
    }
    close(&singles[0], &rt.embed(TEXTS[0]).unwrap());
    let reversed = rt.embed_batch(&[TEXTS[1].into(), TEXTS[0].into()]).unwrap();
    close(&singles[1], &reversed[0]);
    close(&singles[0], &reversed[1]);
}

#[test]
#[ignore = "requires MLX_RS_TEST_EMBED_GGUF and MLX_RS_TEST_EMBED_URL for a running server"]
fn embeddinggemma2_http_single_and_batch() {
    let url = std::env::var("MLX_RS_TEST_EMBED_URL").expect("set MLX_RS_TEST_EMBED_URL");
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .unwrap();
    let mut rt = runtime();
    for input in [json!(TEXTS[0]), json!(TEXTS)] {
        let texts: Vec<&str> = if input.is_string() {
            vec![TEXTS[0]]
        } else {
            TEXTS.to_vec()
        };
        let response: Value = client
            .post(format!("{url}/v1/embeddings"))
            .json(&json!({"model":"embeddinggemma-2","input":input}))
            .send()
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .unwrap();
        assert_eq!(response["object"], "list");
        assert_eq!(response["model"], "embeddinggemma-2");
        let data = response["data"].as_array().unwrap();
        assert_eq!(data.len(), texts.len());
        let mut tokens = 0;
        for (index, text) in texts.iter().enumerate() {
            assert_eq!(data[index]["index"], index);
            assert_eq!(data[index]["object"], "embedding");
            let vector: Vec<f32> =
                serde_json::from_value(data[index]["embedding"].clone()).unwrap();
            close(&vector, &rt.embed(text).unwrap());
            tokens += rt.tokenize(text, true).unwrap().len();
        }
        assert_eq!(response["usage"]["prompt_tokens"], tokens);
        assert_eq!(response["usage"]["total_tokens"], tokens);
    }
}
