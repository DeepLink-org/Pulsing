use regex::Regex;
use serde_json::{Value, json};
use uuid::Uuid;

use super::types::{LlmMessage, LlmUsage};

const DEMO_PEERS: &[&str] = &["bard", "smith", "sage", "guide"];

fn is_tool_result_user_message(msg: &Value) -> bool {
    let Some(content) = msg.get("content").and_then(|v| v.as_array()) else {
        return false;
    };
    !content.is_empty()
        && content
            .iter()
            .all(|b| b.get("type").and_then(|v| v.as_str()) == Some("tool_result"))
}

fn last_user_text(messages: &[Value]) -> String {
    for msg in messages.iter().rev() {
        if msg.get("role").and_then(|v| v.as_str()) != Some("user") {
            continue;
        }
        let content = msg.get("content").cloned().unwrap_or(Value::Null);
        if let Some(s) = content.as_str() {
            return s.trim().to_string();
        }
        if let Some(arr) = content.as_array() {
            if is_tool_result_user_message(msg) {
                continue;
            }
            let parts: Vec<String> = arr
                .iter()
                .filter_map(|b| {
                    if b.get("type").and_then(|v| v.as_str()) == Some("text") {
                        b.get("text").and_then(|v| v.as_str()).map(str::to_string)
                    } else {
                        None
                    }
                })
                .collect();
            return parts.join(" ").trim().to_string();
        }
    }
    String::new()
}

fn tool_names(tools: &[Value]) -> Vec<String> {
    tools
        .iter()
        .filter_map(|t| t.get("name").and_then(|v| v.as_str()).map(str::to_string))
        .collect()
}

enum DemoPlan {
    Text(String),
    Tool { name: String, input: Value },
}

fn plan_demo_turn(messages: &[Value], tools: &[Value]) -> DemoPlan {
    if messages.last().is_some_and(is_tool_result_user_message) {
        let original = last_user_text(messages);
        let snippet = Regex::new(r"\s+")
            .unwrap()
            .replace_all(original.trim(), " ")
            .chars()
            .take(100)
            .collect::<String>();
        return DemoPlan::Text(format!(
            "(demo) Finished tool run for: {}.",
            if snippet.is_empty() {
                "your request".to_string()
            } else {
                snippet
            }
        ));
    }

    let text = last_user_text(messages);
    let lower = text.to_lowercase();
    let allowed = tool_names(tools);

    let has = |keys: &[&str]| keys.iter().any(|k| lower.contains(k));

    if has(&["glob", "files", "list project", "directory"]) && allowed.iter().any(|n| n == "Glob") {
        return DemoPlan::Tool {
            name: "Glob".into(),
            input: json!({ "pattern": "*", "path": "." }),
        };
    }
    if (has(&["create", "write", "scaffold", "bootstrap", "init"]) || lower.contains("readme.md"))
        && allowed.iter().any(|n| n == "Write")
    {
        return DemoPlan::Tool {
            name: "Write".into(),
            input: json!({
                "file_path": "README.md",
                "content": "# Project\n\n(demo) Workspace initialized by Pulsing init guide.\n",
            }),
        };
    }
    if has(&["read", "readme", "summary"]) && allowed.iter().any(|n| n == "Read") {
        return DemoPlan::Tool {
            name: "Read".into(),
            input: json!({ "file_path": "README.md" }),
        };
    }
    if has(&["quest", "puzzle", "unit-test", "questreport"])
        && allowed.iter().any(|n| n == "QuestReport")
    {
        return DemoPlan::Tool {
            name: "QuestReport".into(),
            input: json!({
                "quest_id": "unit-tests",
                "status": "in_progress",
                "note": "demo chatter",
            }),
        };
    }
    if (lower.contains("messageclusteragent")
        || lower.contains("coordinate")
        || lower.contains("peer"))
        && allowed.iter().any(|n| n == "MessageClusterAgent")
    {
        let target = DEMO_PEERS
            .iter()
            .find(|p| lower.contains(**p))
            .unwrap_or(&"smith");
        return DemoPlan::Tool {
            name: "MessageClusterAgent".into(),
            input: json!({
                "agent": target,
                "message": "Demo ping — reply in one short sentence.",
                "wait": false,
            }),
        };
    }

    let snippet = Regex::new(r"\s+")
        .unwrap()
        .replace_all(text.trim(), " ")
        .chars()
        .take(100)
        .collect::<String>();
    DemoPlan::Text(format!(
        "(demo) Noted: {}",
        if snippet.is_empty() {
            "(empty)".to_string()
        } else {
            snippet
        }
    ))
}

pub struct DemoStream {
    plan: DemoPlan,
    text: String,
}

impl DemoStream {
    pub fn new(messages: &[Value], tools: &[Value]) -> Self {
        let plan = plan_demo_turn(messages, tools);
        let text = match &plan {
            DemoPlan::Text(t) => t.clone(),
            DemoPlan::Tool { .. } => String::new(),
        };
        Self { plan, text }
    }

    pub fn text_chunks(&self) -> Vec<String> {
        if self.text.is_empty() {
            vec![]
        } else {
            vec![self.text.clone()]
        }
    }

    pub fn final_message(&self) -> LlmMessage {
        match &self.plan {
            DemoPlan::Tool { name, input } => LlmMessage {
                content: vec![json!({
                    "type": "tool_use",
                    "id": format!("demo-{}", &Uuid::new_v4().simple().to_string()[..12]),
                    "name": name,
                    "input": input,
                })],
                usage: Some(LlmUsage {
                    input_tokens: 1,
                    output_tokens: 1,
                }),
                stop_reason: Some("tool_use".into()),
            },
            DemoPlan::Text(text) => LlmMessage {
                content: vec![json!({ "type": "text", "text": text })],
                usage: Some(LlmUsage {
                    input_tokens: 1,
                    output_tokens: (text.len().max(1) / 4) as u64,
                }),
                stop_reason: Some("end_turn".into()),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_intent_triggers_tool() {
        let messages = vec![json!({ "role": "user", "content": "list project files" })];
        let tools = vec![json!({ "name": "Glob", "input_schema": {} })];
        let stream = DemoStream::new(&messages, &tools);
        assert!(matches!(stream.plan, DemoPlan::Tool { .. }));
    }
}
