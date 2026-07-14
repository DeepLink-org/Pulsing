use std::collections::BTreeMap;

use futures_util::StreamExt;
use reqwest::Client;
use reqwest_eventsource::{Event as SseEvent, EventSource};
use serde_json::{Value, json};

use super::error::LlmError;
use super::message::build_anthropic_request;
use super::types::{LlmMessage, LlmUsage, StreamRequest};

pub struct AnthropicStream {
    text_chunks: Vec<String>,
    final_message: LlmMessage,
}

impl AnthropicStream {
    pub async fn start(
        client: &Client,
        api_key: &str,
        base_url: &str,
        req: &StreamRequest,
    ) -> Result<Self, LlmError> {
        let url = format!("{}/messages", base_url.trim_end_matches('/'));
        let body = build_anthropic_request(req, true);

        let request = client
            .post(url)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(&body);

        let mut es = EventSource::new(request).map_err(|e| LlmError::Stream(e.to_string()))?;
        let mut text_parts = String::new();
        let mut text_chunks = Vec::new();
        let mut tool_blocks: BTreeMap<usize, (String, String, String)> = BTreeMap::new();
        let mut usage: Option<LlmUsage> = None;
        let mut stop_reason: Option<String> = None;

        while let Some(event) = es.next().await {
            match event {
                Ok(SseEvent::Open) => {}
                Ok(SseEvent::Message(message)) => {
                    let data: Value = serde_json::from_str(message.data.trim())?;
                    let event_type = data.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    match event_type {
                        "content_block_start" => {
                            let index =
                                data.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                            let block = data
                                .get("content_block")
                                .cloned()
                                .unwrap_or_else(|| json!({}));
                            if block.get("type").and_then(|v| v.as_str()) == Some("tool_use") {
                                let id = block
                                    .get("id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let name = block
                                    .get("name")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                tool_blocks.insert(index, (id, name, String::new()));
                            }
                        }
                        "content_block_delta" => {
                            let index =
                                data.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                            let delta = data.get("delta").cloned().unwrap_or_else(|| json!({}));
                            let delta_type =
                                delta.get("type").and_then(|v| v.as_str()).unwrap_or("");
                            match delta_type {
                                "text_delta" => {
                                    if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                                        text_parts.push_str(text);
                                        text_chunks.push(text.to_string());
                                    }
                                }
                                "input_json_delta" => {
                                    if let Some(entry) = tool_blocks.get_mut(&index)
                                        && let Some(partial) =
                                            delta.get("partial_json").and_then(|v| v.as_str())
                                    {
                                        entry.2.push_str(partial);
                                    }
                                }
                                _ => {}
                            }
                        }
                        "message_delta" => {
                            if let Some(delta) = data.get("delta") {
                                if let Some(reason) =
                                    delta.get("stop_reason").and_then(|v| v.as_str())
                                {
                                    stop_reason = Some(reason.to_string());
                                }
                                if let Some(u) = delta.get("usage") {
                                    usage = Some(LlmUsage {
                                        input_tokens: usage
                                            .as_ref()
                                            .map(|x| x.input_tokens)
                                            .unwrap_or(0)
                                            .max(
                                                u.get("input_tokens")
                                                    .and_then(|v| v.as_u64())
                                                    .unwrap_or(0),
                                            ),
                                        output_tokens: u
                                            .get("output_tokens")
                                            .and_then(|v| v.as_u64())
                                            .unwrap_or(0),
                                    });
                                }
                            }
                        }
                        "message_start" => {
                            if let Some(u) = data.get("message").and_then(|m| m.get("usage")) {
                                usage = Some(LlmUsage {
                                    input_tokens: u
                                        .get("input_tokens")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0),
                                    output_tokens: u
                                        .get("output_tokens")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0),
                                });
                            }
                        }
                        _ => {}
                    }
                }
                Err(e) => return Err(LlmError::Stream(e.to_string())),
            }
        }

        let mut content = Vec::new();
        if !text_parts.is_empty() {
            content.push(json!({ "type": "text", "text": text_parts }));
        }
        for (_idx, (id, name, args)) in tool_blocks {
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
                stop_reason,
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
