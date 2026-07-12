//! Multi-turn Forge agent loop (LLM + tools).

use std::path::PathBuf;
use std::sync::mpsc::Sender;

use serde_json::{Value, json};

use crate::agent::events::{AgentEvent, emit};
use crate::agent::tools::{DEFAULT_TOOL_NAMES, forge_tool_definitions};
use crate::context::LocalToolSession;
use crate::llm::{LlmClient, LlmMessage, LlmStream, StreamRequest};
use crate::result::ToolResult;
use crate::runtime::ToolRuntime;

const DEFAULT_SYSTEM: &str = "You are a capable coding agent with filesystem and shell tools.\n\
Use tools to inspect the workspace before answering.\n\
When multi-step work is needed, call update_plan first.\n\
Be concise in final replies.";

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub cwd: PathBuf,
    pub provider: String,
    pub model: String,
    pub max_tokens: u32,
    pub max_turns: usize,
    pub sandbox: String,
    pub tool_names: Vec<String>,
    pub system_prompt: Option<String>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            cwd: std::env::current_dir().unwrap_or_else(|_| ".".into()),
            provider: default_provider(),
            model: default_model_for_provider(&default_provider()),
            max_tokens: 8192,
            max_turns: 20,
            sandbox: "off".into(),
            tool_names: DEFAULT_TOOL_NAMES.iter().map(|s| s.to_string()).collect(),
            system_prompt: None,
        }
    }
}

pub fn default_provider() -> String {
    if std::env::var("ANTHROPIC_API_KEY")
        .map(|k| !k.is_empty())
        .unwrap_or(false)
    {
        return "anthropic".into();
    }
    if std::env::var("OPENAI_API_KEY")
        .map(|k| !k.is_empty())
        .unwrap_or(false)
    {
        return "openai".into();
    }
    "demo".into()
}

pub fn default_model_for_provider(provider: &str) -> String {
    match provider.trim().to_lowercase().as_str() {
        "demo" => "demo".into(),
        "openai" => std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".into()),
        _ => std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".into()),
    }
}

pub async fn run_agent_turn(config: &AgentConfig, prompt: &str) -> anyhow::Result<String> {
    run_agent_turn_observed(config, prompt, None).await
}

pub async fn run_agent_turn_observed(
    config: &AgentConfig,
    prompt: &str,
    event_tx: Option<Sender<AgentEvent>>,
) -> anyhow::Result<String> {
    let mut agent = ForgeAgent::new(config.clone());
    agent.event_tx = event_tx;
    match agent.run(prompt).await {
        Ok(text) => Ok(text),
        Err(err) => {
            emit(&agent.event_tx, AgentEvent::Error(err.to_string()));
            Err(err)
        }
    }
}

pub struct ForgeAgent {
    config: AgentConfig,
    client: LlmClient,
    runtime: ToolRuntime,
    messages: Vec<Value>,
    event_tx: Option<Sender<AgentEvent>>,
}

impl ForgeAgent {
    pub fn new(config: AgentConfig) -> Self {
        let client = LlmClient::new(&config.provider, None, None).expect("LLM client");
        let session = std::sync::Arc::new(LocalToolSession::default());
        let runtime = ToolRuntime::new(crate::runtime::ToolRuntimeConfig {
            cwd: config.cwd.clone(),
            sandbox_policy: config.sandbox.clone(),
            session,
            ..Default::default()
        });
        Self {
            config,
            client,
            runtime,
            messages: Vec::new(),
            event_tx: None,
        }
    }

    pub async fn run(&mut self, prompt: &str) -> anyhow::Result<String> {
        self.messages.clear();
        self.messages
            .push(json!({ "role": "user", "content": prompt }));

        let tool_names: Vec<&str> = self.config.tool_names.iter().map(String::as_str).collect();
        let tools = forge_tool_definitions(&tool_names);

        let mut final_msg: Option<LlmMessage> = None;
        for _ in 0..self.config.max_turns {
            final_msg = Some(self.stream_one_turn(&tools).await?);
            let msg = final_msg.as_ref().expect("final message");
            self.messages
                .push(json!({ "role": "assistant", "content": msg.content }));

            let tool_uses = extract_tool_uses(&msg.content);
            if tool_uses.is_empty() {
                let text = text_from_content(&msg.content);
                emit(&self.event_tx, AgentEvent::Done { text: text.clone() });
                return Ok(text);
            }

            let mut blocks = Vec::new();
            for (id, name, input) in tool_uses {
                emit(&self.event_tx, AgentEvent::ToolStart { name: name.clone() });
                let result = self.runtime.call_tool(&name, input).await;
                let summary = result.content.chars().take(200).collect::<String>();
                emit(
                    &self.event_tx,
                    AgentEvent::ToolEnd {
                        name: name.clone(),
                        ok: !result.is_error,
                        summary,
                    },
                );
                blocks.push(tool_result_block(&id, &result));
                if !result.is_error {
                    eprintln!("# tool {name} ok");
                } else {
                    eprintln!("# tool {name} error: {}", result.content);
                }
            }
            self.messages
                .push(json!({ "role": "user", "content": blocks }));
        }

        let text = final_msg
            .map(|m| text_from_content(&m.content))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "(max turns reached)".into());
        emit(&self.event_tx, AgentEvent::Done { text: text.clone() });
        Ok(text)
    }

    async fn stream_one_turn(&self, tools: &[Value]) -> anyhow::Result<LlmMessage> {
        let system = self
            .config
            .system_prompt
            .clone()
            .unwrap_or_else(|| DEFAULT_SYSTEM.to_string());
        let req = StreamRequest {
            model: self.config.model.clone(),
            max_tokens: self.config.max_tokens,
            messages: self.messages.clone(),
            system: Some(system),
            tools: tools.to_vec(),
        };
        let stream = self.client.stream_messages(req).await?;
        emit_stream_text(&stream, &self.event_tx);
        Ok(stream.final_message())
    }
}

fn emit_stream_text(stream: &LlmStream, event_tx: &Option<Sender<AgentEvent>>) {
    for chunk in stream.text_chunks() {
        if let Some(tx) = event_tx {
            let _ = tx.send(AgentEvent::TextDelta(chunk.to_string()));
        } else {
            print!("{chunk}");
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
    }
    if event_tx.is_none() && !stream.text_chunks().is_empty() {
        println!();
    }
}

fn text_from_content(content: &[Value]) -> String {
    let mut parts = Vec::new();
    for block in content {
        if block.get("type").and_then(|v| v.as_str()) == Some("text")
            && let Some(t) = block.get("text").and_then(|v| v.as_str())
        {
            parts.push(t);
        }
    }
    parts.concat()
}

fn extract_tool_uses(content: &[Value]) -> Vec<(String, String, Value)> {
    let mut out = Vec::new();
    for block in content {
        if block.get("type").and_then(|v| v.as_str()) != Some("tool_use") {
            continue;
        }
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
        let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
        if !name.is_empty() {
            out.push((id, name, input));
        }
    }
    out
}

fn tool_result_block(id: &str, result: &ToolResult) -> Value {
    json!({
        "type": "tool_result",
        "tool_use_id": id,
        "content": result.content,
        "is_error": result.is_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn demo_agent_glob_turn() {
        let cfg = AgentConfig {
            provider: "demo".into(),
            model: "demo".into(),
            cwd: std::env::current_dir().unwrap_or_else(|_| ".".into()),
            ..Default::default()
        };
        let out = run_agent_turn(&cfg, "list project files with Glob")
            .await
            .unwrap();
        assert!(!out.is_empty());
    }
}
