use serde_json::{Value, json};

pub fn tool_schema_to_openai(tool: &Value) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": tool.get("name").and_then(|v| v.as_str()).unwrap_or(""),
            "description": tool.get("description").and_then(|v| v.as_str()).unwrap_or(""),
            "parameters": tool.get("input_schema").cloned().unwrap_or_else(|| json!({})),
        }
    })
}

fn tool_result_to_text(content: &Value) -> String {
    if let Some(s) = content.as_str() {
        return s.to_string();
    }
    if content.is_null() {
        return String::new();
    }
    serde_json::to_string(content).unwrap_or_default()
}

fn is_tool_result_user_message(content: &[Value]) -> bool {
    !content.is_empty()
        && content
            .iter()
            .all(|b| b.get("type").and_then(|v| v.as_str()) == Some("tool_result"))
}

fn user_content_blocks_to_openai(content: &[Value]) -> Vec<Value> {
    let mut parts = Vec::new();
    for block in content {
        let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match block_type {
            "text" => {
                parts.push(json!({
                    "type": "text",
                    "text": block.get("text").and_then(|v| v.as_str()).unwrap_or(""),
                }));
            }
            "image" => {
                let source = block.get("source").cloned().unwrap_or_else(|| json!({}));
                let media_type = source
                    .get("media_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("image/png");
                let data = source.get("data").and_then(|v| v.as_str()).unwrap_or("");
                parts.push(json!({
                    "type": "image_url",
                    "image_url": { "url": format!("data:{media_type};base64,{data}") },
                }));
            }
            _ => {}
        }
    }
    if parts.is_empty() {
        parts.push(json!({ "type": "text", "text": "" }));
    }
    parts
}

pub fn to_openai_messages(system: Option<&str>, messages: &[Value]) -> Vec<Value> {
    let mut out = Vec::new();
    if let Some(sys) = system.filter(|s| !s.is_empty()) {
        out.push(json!({ "role": "system", "content": sys }));
    }

    for message in messages {
        let role = message
            .get("role")
            .and_then(|v| v.as_str())
            .unwrap_or("user");
        let content = message.get("content").cloned().unwrap_or(Value::Null);

        if role == "user"
            && let Some(arr) = content.as_array()
        {
            if is_tool_result_user_message(arr) {
                for block in arr {
                    out.push(json!({
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id").and_then(|v| v.as_str()).unwrap_or(""),
                            "content": tool_result_to_text(block.get("content").unwrap_or(&Value::Null)),
                        }));
                }
                continue;
            }
            out.push(json!({
                "role": "user",
                "content": user_content_blocks_to_openai(arr),
            }));
            continue;
        }

        if role == "assistant"
            && let Some(arr) = content.as_array()
        {
            let mut text_parts = Vec::new();
            let mut tool_calls = Vec::new();
            for block in arr {
                let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match block_type {
                    "text" => {
                        text_parts.push(
                            block
                                .get("text")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                        );
                    }
                    "tool_use" => {
                        let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                        tool_calls.push(json!({
                                "id": block.get("id").and_then(|v| v.as_str()).unwrap_or(""),
                                "type": "function",
                                "function": {
                                    "name": block.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                                    "arguments": serde_json::to_string(&input).unwrap_or_else(|_| "{}".into()),
                                }
                            }));
                    }
                    _ => {}
                }
            }
            let text = text_parts.concat();
            let mut assistant = json!({
                "role": "assistant",
                "content": if text.is_empty() { Value::Null } else { json!(text) },
            });
            if !tool_calls.is_empty() {
                assistant
                    .as_object_mut()
                    .expect("assistant object")
                    .insert("tool_calls".into(), Value::Array(tool_calls));
            }
            out.push(assistant);
            continue;
        }

        out.push(json!({ "role": role, "content": content }));
    }

    out
}

pub fn build_openai_request(req: &super::types::StreamRequest, stream: bool) -> Value {
    let mut body = json!({
        "model": req.model,
        "messages": to_openai_messages(req.system.as_deref(), &req.messages),
        "max_tokens": req.max_tokens,
        "stream": stream,
    });
    if !req.tools.is_empty() {
        let tools: Vec<Value> = req.tools.iter().map(tool_schema_to_openai).collect();
        body.as_object_mut()
            .expect("body object")
            .insert("tools".into(), Value::Array(tools));
    }
    body
}

pub fn normalize_openai_stop_reason(reason: Option<&str>) -> Option<String> {
    reason.map(|r| match r {
        "stop" => "end_turn".to_string(),
        "length" => "max_tokens".to_string(),
        "tool_calls" => "tool_use".to_string(),
        other => other.to_string(),
    })
}

pub fn build_anthropic_request(req: &super::types::StreamRequest, stream: bool) -> Value {
    let mut body = json!({
        "model": req.model,
        "max_tokens": req.max_tokens,
        "messages": req.messages,
        "stream": stream,
    });
    if let Some(sys) = req.system.as_deref().filter(|s| !s.is_empty()) {
        body.as_object_mut()
            .expect("body object")
            .insert("system".into(), json!(sys));
    }
    if !req.tools.is_empty() {
        body.as_object_mut()
            .expect("body object")
            .insert("tools".into(), Value::Array(req.tools.clone()));
    }
    body
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_tool_result_roundtrip_shape() {
        let messages = vec![json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": "ok",
            }],
        })];
        let out = to_openai_messages(None, &messages);
        assert_eq!(out[0]["role"], "tool");
        assert_eq!(out[0]["tool_call_id"], "t1");
    }
}
