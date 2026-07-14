//! Codex-compatible `request_user_input` argument validation.

use serde::{Deserialize, Serialize};

use crate::error::ToolError;

pub const MIN_AUTO_RESOLUTION_MS: u64 = 60_000;
pub const MAX_AUTO_RESOLUTION_MS: u64 = 240_000;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RequestUserInputQuestionOption {
    pub label: String,
    pub description: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RequestUserInputQuestion {
    pub id: String,
    pub header: String,
    pub question: String,
    #[serde(rename = "isOther", default)]
    pub is_other: bool,
    #[serde(rename = "isSecret", default)]
    pub is_secret: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<Vec<RequestUserInputQuestionOption>>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RequestUserInputArgs {
    pub questions: Vec<RequestUserInputQuestion>,
    #[serde(
        rename = "autoResolutionMs",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub auto_resolution_ms: Option<u64>,
}

pub fn validate_request_user_input(
    value: &serde_json::Value,
) -> Result<RequestUserInputArgs, ToolError> {
    match value.get("questions") {
        None => {
            return Err(ToolError::respond(
                "request_user_input requires at least one question",
            ));
        }
        Some(serde_json::Value::Array(items)) if items.is_empty() => {
            return Err(ToolError::respond(
                "request_user_input requires at least one question",
            ));
        }
        Some(serde_json::Value::Array(items)) => {
            for item in items {
                if !item.is_object() {
                    return Err(ToolError::respond("each question must be an object"));
                }
            }
        }
        Some(_) => {
            return Err(ToolError::respond("field 'questions' must be a list"));
        }
    }

    let auto_resolution_ms = parse_auto_resolution_ms(value)?;

    let mut cleaned = value.clone();
    if let Some(obj) = cleaned.as_object_mut() {
        obj.remove("autoResolutionMs");
        obj.remove("auto_resolution_ms");
    }

    let mut args: RequestUserInputArgs = serde_json::from_value(cleaned)
        .map_err(|e| ToolError::respond(format!("invalid request_user_input arguments: {e}")))?;
    args.auto_resolution_ms = auto_resolution_ms;
    normalize_request_user_input_strings(&mut args);
    let mut seen = std::collections::HashSet::new();
    for q in &args.questions {
        if q.id.is_empty() {
            return Err(ToolError::respond("each question requires a non-empty id"));
        }
        if q.question.is_empty() {
            return Err(ToolError::respond(format!(
                "question {:?} requires non-empty question text",
                q.id
            )));
        }
        if !seen.insert(q.id.clone()) {
            return Err(ToolError::respond(format!(
                "duplicate question id {:?}",
                q.id
            )));
        }
        if let Some(opts) = &q.options {
            if opts.is_empty() {
                return Err(ToolError::respond(format!(
                    "question {:?} has empty options array",
                    q.id
                )));
            }
            for opt in opts {
                if opt.label.is_empty() {
                    return Err(ToolError::respond(format!(
                        "question {:?} has option with empty label",
                        q.id
                    )));
                }
            }
        }
    }
    Ok(args)
}

fn normalize_request_user_input_strings(args: &mut RequestUserInputArgs) {
    for q in &mut args.questions {
        q.id = q.id.trim().to_string();
        q.question = q.question.trim().to_string();
        if let Some(opts) = &mut q.options {
            for opt in opts {
                opt.label = opt.label.trim().to_string();
            }
        }
    }
}

fn parse_auto_resolution_ms(value: &serde_json::Value) -> Result<Option<u64>, ToolError> {
    let raw = value
        .get("autoResolutionMs")
        .or_else(|| value.get("auto_resolution_ms"));
    let Some(raw) = raw else {
        return Ok(None);
    };
    let ms = match raw {
        serde_json::Value::Number(n) => n
            .as_u64()
            .or_else(|| n.as_i64().and_then(|i| u64::try_from(i).ok())),
        _ => None,
    };
    ms.map(normalize_auto_resolution_ms)
        .ok_or_else(|| {
            ToolError::respond(format!("autoResolutionMs must be an integer, got {raw:?}"))
        })
        .map(Some)
}

pub fn normalize_auto_resolution_ms(ms: u64) -> u64 {
    ms.clamp(MIN_AUTO_RESOLUTION_MS, MAX_AUTO_RESOLUTION_MS)
}

pub fn normalize_request_user_input(mut args: RequestUserInputArgs) -> RequestUserInputArgs {
    if let Some(ms) = args.auto_resolution_ms {
        args.auto_resolution_ms = Some(normalize_auto_resolution_ms(ms));
    }
    args
}

pub fn args_to_value(args: &RequestUserInputArgs) -> serde_json::Value {
    serde_json::to_value(args).expect("RequestUserInputArgs serializes")
}

pub fn default_auto_answers(args: &RequestUserInputArgs) -> serde_json::Value {
    let mut answers = serde_json::Map::new();
    for q in &args.questions {
        let default_answer = q
            .options
            .as_ref()
            .and_then(|o| o.first())
            .map(|o| o.label.clone())
            .unwrap_or_default();
        answers.insert(
            q.id.clone(),
            serde_json::json!({ "answers": [default_answer] }),
        );
    }
    serde_json::json!({ "answers": answers })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_questions() {
        let raw = serde_json::json!({ "questions": [] });
        assert!(validate_request_user_input(&raw).is_err());
    }

    #[test]
    fn accepts_valid_payload() {
        let raw = serde_json::json!({
            "questions": [{
                "id": "q1",
                "header": "Pick",
                "question": "Which?",
                "options": [{"label": "A", "description": "first"}]
            }],
            "autoResolutionMs": 5000
        });
        let args = validate_request_user_input(&raw).unwrap();
        assert_eq!(args.questions.len(), 1);
        assert_eq!(args.auto_resolution_ms, Some(MIN_AUTO_RESOLUTION_MS));
    }

    #[test]
    fn clamps_auto_resolution_to_max() {
        let raw = serde_json::json!({
            "questions": [{"id": "q1", "header": "H", "question": "Q?"}],
            "autoResolutionMs": 999_999_999
        });
        let args = validate_request_user_input(&raw).unwrap();
        assert_eq!(args.auto_resolution_ms, Some(MAX_AUTO_RESOLUTION_MS));
    }

    #[test]
    fn rejects_non_list_questions() {
        let raw = serde_json::json!({ "questions": "not-a-list" });
        let err = validate_request_user_input(&raw).unwrap_err();
        assert_eq!(err.to_string(), "field 'questions' must be a list");
    }

    #[test]
    fn rejects_malformed_auto_resolution_ms() {
        let raw = serde_json::json!({
            "questions": [{"id": "q1", "header": "H", "question": "Q?"}],
            "autoResolutionMs": "soon"
        });
        let err = validate_request_user_input(&raw).unwrap_err();
        assert!(
            err.to_string()
                .contains("autoResolutionMs must be an integer")
        );
    }

    #[test]
    fn accepts_auto_resolution_ms_snake_case_alias() {
        let raw = serde_json::json!({
            "questions": [{"id": "q1", "header": "H", "question": "Q?"}],
            "auto_resolution_ms": 90_000
        });
        let args = validate_request_user_input(&raw).unwrap();
        assert_eq!(args.auto_resolution_ms, Some(90_000));
    }

    #[test]
    fn default_auto_answers_matches_codex_response_shape() {
        let args = validate_request_user_input(&serde_json::json!({
            "questions": [{
                "id": " q1 ",
                "header": "Pick",
                "question": " Which? ",
                "options": [{"label": " A ", "description": "first"}]
            }]
        }))
        .unwrap();
        assert_eq!(args.questions[0].id, "q1");
        let out = default_auto_answers(&args);
        assert_eq!(out["answers"]["q1"]["answers"], serde_json::json!(["A"]));
    }

    #[test]
    fn args_to_value_emits_clamped_timeout() {
        let args = validate_request_user_input(&serde_json::json!({
            "questions": [{"id": "q1", "header": "H", "question": "Q?"}],
            "autoResolutionMs": 1
        }))
        .unwrap();
        let payload = args_to_value(&args);
        assert_eq!(payload["autoResolutionMs"], MIN_AUTO_RESOLUTION_MS);
    }
}
