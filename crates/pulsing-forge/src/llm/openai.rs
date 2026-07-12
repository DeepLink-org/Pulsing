use std::collections::BTreeMap;

use futures_util::StreamExt;
use reqwest::Client;
use reqwest_eventsource::{Event as SseEvent, EventSource};
use serde_json::{Value, json};

use super::error::LlmError;
use super::message::{build_openai_request, normalize_openai_stop_reason};
use super::types::{LlmMessage, LlmUsage, StreamRequest};

pub struct OpenAiStream {
    text_chunks: Vec<String>,
    final_message: LlmMessage,
}

impl OpenAiStream {
    pub async fn start(
        client: &Client,
        api_key: &str,
        base_url: &str,
        req: &StreamRequest,
    ) -> Result<Self, LlmError> {
        let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
        let body = build_openai_request(req, true);

        let request = client
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(&body);

        let mut es = EventSource::new(request).map_err(|e| LlmError::Stream(e.to_string()))?;
        let mut text_parts = String::new();
        let mut text_chunks = Vec::new();
        let mut tool_calls: BTreeMap<usize, (String, String, String)> = BTreeMap::new();
        let mut usage: Option<LlmUsage> = None;
        let mut finish_reason: Option<String> = None;

        while let Some(event) = es.next().await {
            match event {
                Ok(SseEvent::Open) => {}
                Ok(SseEvent::Message(message)) => {
                    let data = message.data.trim();
                    if data == "[DONE]" {
                        break;
                    }
                    let json: Value = serde_json::from_str(data)?;
                    if let Some(u) = json.get("usage") {
                        usage = Some(LlmUsage {
                            input_tokens: u
                                .get("prompt_tokens")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0),
                            output_tokens: u
                                .get("completion_tokens")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0),
                        });
                    }
                    let choices = json
                        .get("choices")
                        .and_then(|v| v.as_array())
                        .cloned()
                        .unwrap_or_default();
                    for choice in choices {
                        if let Some(reason) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                            finish_reason = Some(reason.to_string());
                        }
                        let delta = choice.get("delta").cloned().unwrap_or_else(|| json!({}));
                        if let Some(content) = delta.get("content").and_then(|v| v.as_str()) {
                            text_parts.push_str(content);
                            text_chunks.push(content.to_string());
                        }
                        if let Some(calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                            for call in calls {
                                let index = call.get("index").and_then(|v| v.as_u64()).unwrap_or(0)
                                    as usize;
                                let entry = tool_calls.entry(index).or_insert_with(|| {
                                    (String::new(), String::new(), String::new())
                                });
                                if let Some(id) = call.get("id").and_then(|v| v.as_str()) {
                                    entry.0 = id.to_string();
                                }
                                if let Some(function) = call.get("function") {
                                    if let Some(name) =
                                        function.get("name").and_then(|v| v.as_str())
                                    {
                                        entry.1 = name.to_string();
                                    }
                                    if let Some(args) =
                                        function.get("arguments").and_then(|v| v.as_str())
                                    {
                                        entry.2.push_str(args);
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => return Err(LlmError::Stream(e.to_string())),
            }
        }

        let mut content = Vec::new();
        if !text_parts.is_empty() {
            content.push(json!({ "type": "text", "text": text_parts }));
        }
        for (_idx, (id, name, args)) in tool_calls {
            let parsed: Value = serde_json::from_str(args.trim()).unwrap_or_else(|_| json!({}));
            let input = if parsed.is_object() {
                parsed
            } else {
                json!({})
            };
            content.push(json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": input,
            }));
        }

        Ok(Self {
            text_chunks,
            final_message: LlmMessage {
                content,
                usage,
                stop_reason: normalize_openai_stop_reason(finish_reason.as_deref()),
            },
        })
    }

    pub fn text_chunks(&self) -> &[String] {
        &self.text_chunks
    }

    pub fn final_message(&self) -> LlmMessage {
        self.final_message.clone()
    }
}
