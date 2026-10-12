use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A joint, non-streaming System One decision request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub state: Value,
    pub questions: indexmap::IndexMap<String, DecisionQuestion>,
    #[serde(
        default,
        deserialize_with = "media_or_empty",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub files: Vec<Value>,
    #[serde(
        default,
        deserialize_with = "media_or_empty",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub images: Vec<Value>,
}

fn media_or_empty<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<Value>, D::Error> {
    Ok(Option::<Vec<Value>>::deserialize(deserializer)?.unwrap_or_default())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum DecisionQuestion {
    Choice {
        instructions: Value,
        criteria: indexmap::IndexMap<String, Value>,
    },
    Score {
        instructions: Value,
        criteria: Vec<Value>,
    },
    Noul {
        instructions: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        criteria: Option<BTreeMap<String, Value>>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum DecisionAnswer {
    Choice {
        choice: String,
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
    Score {
        score: f64,
        legend: BTreeMap<String, Value>,
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
    Noul {
        noul: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionResponse {
    pub model: String,
    pub answers: BTreeMap<String, DecisionAnswer>,
    pub usage: DecisionUsage,
}

/// Errors distinguish bad input and missing capability from inference failures.
#[derive(Debug)]
pub enum DecisionError {
    InvalidRequest(String),
    Unsupported(String),
    Inference(String),
}
impl fmt::Display for DecisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(s) | Self::Unsupported(s) | Self::Inference(s) => f.write_str(s),
        }
    }
}
impl std::error::Error for DecisionError {}

impl DecisionRequest {
    pub fn validate_text(&self) -> Result<(), DecisionError> {
        // Only interpret media parts in the upstream-supported chat state forms.
        let messages = self
            .state
            .as_array()
            .or_else(|| self.state.get("messages").and_then(Value::as_array));
        let has_media = messages.into_iter().flatten().any(|m| {
            m.get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .any(|p| {
                    matches!(
                        p.get("type").and_then(Value::as_str),
                        Some("image_url" | "input_audio" | "video_url")
                    )
                })
        });
        if !self.files.is_empty() || !self.images.is_empty() || has_media {
            return Err(DecisionError::Unsupported(
                "Decision image/audio input is not supported by this text-only backend".into(),
            ));
        }
        if self.state.is_null() || self.questions.is_empty() {
            return Err(DecisionError::InvalidRequest(
                "state and a non-empty questions object are required".into(),
            ));
        }
        for (id, q) in &self.questions {
            let (instructions, valid) = match q {
                DecisionQuestion::Choice {
                    instructions,
                    criteria,
                } => (instructions, (1..=255).contains(&criteria.len())),
                DecisionQuestion::Score {
                    instructions,
                    criteria,
                } => (instructions, (2..=10).contains(&criteria.len())),
                DecisionQuestion::Noul { instructions, .. } => (instructions, true),
            };
            if instructions.is_null() || !valid {
                return Err(DecisionError::InvalidRequest(format!(
                    "questions.{id}: invalid instructions or criteria"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn mixed_questions_and_structured_state_roundtrip() {
        let wire = json!({"state":{"ticket":"charged twice"},"questions":{
            "route":{"type":"choice","instructions":["choose team"],"criteria":{"z":null,"a":{"description":"billing"}}},
            "urgent":{"type":"score","instructions":"urgency","criteria":["low","high"]},
            "angry":{"type":"noul","instructions":"angry?"}
        }});
        let req: DecisionRequest = serde_json::from_value(wire).unwrap();
        req.validate_text().unwrap();
        let DecisionQuestion::Choice { criteria, .. } = &req.questions["route"] else {
            panic!()
        };
        assert_eq!(criteria.keys().collect::<Vec<_>>(), vec!["z", "a"]);
        let again: DecisionRequest =
            serde_json::from_str(&serde_json::to_string(&req).unwrap()).unwrap();
        again.validate_text().unwrap();
        assert_eq!(again.questions.len(), 3);
    }

    #[test]
    fn question_order_is_preserved_from_wire() {
        let req: DecisionRequest = serde_json::from_str(r#"{"state":"s","files":null,"images":null,"questions":{"z":{"type":"noul","instructions":"last?"},"a":{"type":"noul","instructions":"first?"}}}"#).unwrap();
        assert_eq!(
            req.questions.keys().map(String::as_str).collect::<Vec<_>>(),
            vec!["z", "a"]
        );
        let serialized = serde_json::to_string(&req).unwrap();
        let again: DecisionRequest = serde_json::from_str(&serialized).unwrap();
        assert_eq!(
            again
                .questions
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            vec!["z", "a"]
        );
    }

    #[test]
    fn invalid_criteria_and_media_are_rejected() {
        for criteria in [
            json!([]),
            json!(["one"]),
            json!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ] {
            let req: DecisionRequest = serde_json::from_value(json!({"state":"s","questions":{"q":{"type":"score","instructions":"q","criteria":criteria}}})).unwrap();
            assert!(matches!(
                req.validate_text(),
                Err(DecisionError::InvalidRequest(_))
            ));
        }
        let mut req: DecisionRequest = serde_json::from_value(
            json!({"state":"s","questions":{"q":{"type":"noul","instructions":"q"}}}),
        )
        .unwrap();
        req.images.push(json!("data:image/png;base64,abc"));
        assert!(matches!(
            req.validate_text(),
            Err(DecisionError::Unsupported(_))
        ));
        req.images.clear();
        req.state =
            json!({"messages":[{"content":[{"type":"image_url","image_url":{"url":"data:abc"}}]}]});
        assert!(matches!(
            req.validate_text(),
            Err(DecisionError::Unsupported(_))
        ));
    }
}

#[cfg(test)]
mod option_order_tests {
    use super::*;
    #[test]
    fn choice_wire_order_survives_roundtrip() {
        let raw = r#"{"state":"x","questions":{"route":{"type":"choice","instructions":"pick","criteria":{"z":"last alphabetically","a":null,"m":null}}}}"#;
        let parsed: DecisionRequest = serde_json::from_str(raw).unwrap();
        let DecisionQuestion::Choice { criteria, .. } = &parsed.questions["route"] else {
            panic!()
        };
        assert_eq!(
            criteria.keys().map(String::as_str).collect::<Vec<_>>(),
            ["z", "a", "m"]
        );
        let serialized = serde_json::to_string(&parsed).unwrap();
        assert!(serialized.find("\"z\"").unwrap() < serialized.find("\"a\"").unwrap());
    }
}

#[cfg(test)]
mod state_order_tests {
    use super::*;
    #[test]
    fn structured_state_and_instructions_preserve_wire_order() {
        let raw = r#"{"state":{"z":{"last":1,"first":2},"a":true},"questions":{"x":{"type":"noul","instructions":{"z":"last","a":"first"}}}}"#;
        let request: DecisionRequest = serde_json::from_str(raw).unwrap();
        let state = serde_json::to_string(&request.state).unwrap();
        assert_eq!(state, r#"{"z":{"last":1,"first":2},"a":true}"#);
        let DecisionQuestion::Noul { instructions, .. } = &request.questions["x"] else {
            panic!()
        };
        assert_eq!(
            serde_json::to_string(instructions).unwrap(),
            r#"{"z":"last","a":"first"}"#
        );
    }
}
