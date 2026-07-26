//! Multi-turn Forge agent loop (LLM + tools).

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::mpsc::Sender;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use tokio_util::sync::CancellationToken;

use crate::agent::events::{AgentEvent, emit};
use crate::agent::tools::DEFAULT_TOOL_NAMES;
use crate::approval::ApprovalPolicy;
use crate::context::LocalToolSession;
use crate::llm::{LlmClient, LlmMessage, LlmStream, StreamRequest};
use crate::result::ToolResult;
use crate::runtime::ToolRuntime;
use crate::turn::TurnExecutionContext;
use crate::{SessionId, TurnId};

const DEFAULT_SYSTEM: &str = "You are a capable coding agent with filesystem and shell tools.\n\
Use tools to inspect the workspace before answering.\n\
When multi-step work is needed, call update_plan first.\n\
Be concise in final replies.";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentConfig {
    pub cwd: PathBuf,
    pub provider: String,
    pub model: String,
    pub max_tokens: u32,
    pub max_turns: usize,
    pub sandbox: String,
    pub approval_policy: ApprovalPolicy,
    pub tool_names: Vec<String>,
    pub system_prompt: Option<String>,
}

#[derive(Debug, Error)]
#[error("agent turn cancelled")]
pub struct AgentCancelled;

pub type AgentEventHandler =
    Arc<dyn Fn(AgentEvent) -> Pin<Box<dyn Future<Output = ()> + Send + 'static>> + Send + Sync>;

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            cwd: std::env::current_dir().unwrap_or_else(|_| ".".into()),
            provider: default_provider(),
            model: default_model_for_provider(&default_provider()),
            max_tokens: 8192,
            max_turns: 20,
            sandbox: "off".into(),
            approval_policy: ApprovalPolicy::OnRequest,
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
    event_handler: Option<AgentEventHandler>,
}

impl ForgeAgent {
    pub fn new(config: AgentConfig) -> Self {
        Self::try_new(config).expect("LLM client")
    }

    pub fn try_new(config: AgentConfig) -> anyhow::Result<Self> {
        let client = LlmClient::new(
            &config.provider,
            None,
            provider_base_url_from_env(&config.provider),
        )?;
        let session = std::sync::Arc::new(
            LocalToolSession::default().with_approval_policy(config.approval_policy),
        );
        let runtime = ToolRuntime::new(crate::runtime::ToolRuntimeConfig {
            cwd: config.cwd.clone(),
            sandbox_policy: config.sandbox.clone(),
            session,
            ..Default::default()
        });
        Ok(Self {
            config,
            client,
            runtime,
            messages: Vec::new(),
            event_tx: None,
            event_handler: None,
        })
    }

    pub fn set_event_handler(&mut self, handler: Option<AgentEventHandler>) {
        self.event_handler = handler;
    }

    pub async fn run(&mut self, prompt: &str) -> anyhow::Result<String> {
        self.run_cancellable(prompt, CancellationToken::new()).await
    }

    pub async fn run_cancellable(
        &mut self,
        prompt: &str,
        cancel: CancellationToken,
    ) -> anyhow::Result<String> {
        let turn = Arc::new(TurnExecutionContext::with_cancellation(
            SessionId::new(),
            TurnId::new(),
            cancel,
        ));
        self.run_in_turn(prompt, turn).await
    }

    pub async fn run_in_turn(
        &mut self,
        prompt: &str,
        turn: Arc<TurnExecutionContext>,
    ) -> anyhow::Result<String> {
        let cancel = turn.cancellation();
        if cancel.is_cancelled() {
            self.emit_event(AgentEvent::Cancelled).await;
            return Err(AgentCancelled.into());
        }
        self.messages
            .push(json!({ "role": "user", "content": prompt }));

        let tools = self.runtime.tool_definitions(&self.config.tool_names)?;

        let mut final_msg: Option<LlmMessage> = None;
        for _ in 0..self.config.max_turns {
            let message = {
                let _model_resource = turn.resources().register_passive("model_request");
                tokio::select! {
                    _ = cancel.cancelled() => {
                        self.emit_event(AgentEvent::Cancelled).await;
                        return Err(AgentCancelled.into());
                    }
                    result = self.stream_one_turn(&tools) => result?,
                }
            };
            final_msg = Some(message);
            let msg = final_msg.as_ref().expect("final message");
            self.messages
                .push(json!({ "role": "assistant", "content": msg.content }));

            let tool_uses = extract_tool_uses(&msg.content);
            if tool_uses.is_empty() {
                let text = text_from_content(&msg.content);
                self.emit_event(AgentEvent::Done { text: text.clone() })
                    .await;
                return Ok(text);
            }

            let mut blocks = Vec::new();
            for (id, name, input) in tool_uses {
                self.emit_event(AgentEvent::ToolStart { name: name.clone() })
                    .await;
                let result = {
                    let _tool_resource = turn.resources().register_passive(format!("tool:{name}"));
                    tokio::select! {
                        _ = cancel.cancelled() => {
                            self.emit_event(AgentEvent::ToolCancelled { name }).await;
                            self.emit_event(AgentEvent::Cancelled).await;
                            return Err(AgentCancelled.into());
                        }
                        result = self.runtime.call_tool_in_turn(turn.clone(), &name, input) => result,
                    }
                };
                let summary = result.content.chars().take(200).collect::<String>();
                self.emit_event(AgentEvent::ToolEnd {
                    name: name.clone(),
                    ok: !result.is_error,
                    summary,
                })
                .await;
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
        self.emit_event(AgentEvent::Done { text: text.clone() })
            .await;
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
        self.emit_stream_text(&stream).await;
        Ok(stream.final_message())
    }

    async fn emit_stream_text(&self, stream: &LlmStream) {
        for chunk in stream.text_chunks() {
            if self.event_tx.is_some() || self.event_handler.is_some() {
                self.emit_event(AgentEvent::TextDelta(chunk.to_string()))
                    .await;
            } else {
                print!("{chunk}");
                let _ = std::io::Write::flush(&mut std::io::stdout());
            }
        }
        if self.event_tx.is_none()
            && self.event_handler.is_none()
            && !stream.text_chunks().is_empty()
        {
            println!();
        }
    }

    async fn emit_event(&self, event: AgentEvent) {
        emit(&self.event_tx, event.clone());
        if let Some(handler) = &self.event_handler {
            handler(event).await;
        }
    }
}

fn provider_base_url_from_env(provider: &str) -> Option<String> {
    let variable = match provider.trim().to_lowercase().as_str() {
        "openai" => "OPENAI_BASE_URL",
        "anthropic" => "ANTHROPIC_BASE_URL",
        _ => return None,
    };
    std::env::var(variable)
        .ok()
        .filter(|value| !value.is_empty())
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

    #[tokio::test]
    async fn forge_agent_preserves_messages_across_user_turns() {
        let cfg = AgentConfig {
            provider: "demo".into(),
            model: "demo".into(),
            ..Default::default()
        };
        let mut agent = ForgeAgent::new(cfg);
        agent.run("remember alpha").await.unwrap();
        let after_first = agent.messages.len();
        agent.run("remember beta").await.unwrap();
        assert!(agent.messages.len() > after_first);
        assert_eq!(
            agent
                .messages
                .last()
                .and_then(|message| message.get("role")),
            Some(&Value::String("assistant".into()))
        );
    }

    #[tokio::test]
    async fn pre_cancelled_turn_never_starts() {
        let cfg = AgentConfig {
            provider: "demo".into(),
            model: "demo".into(),
            ..Default::default()
        };
        let mut agent = ForgeAgent::new(cfg);
        let cancel = CancellationToken::new();
        cancel.cancel();
        let err = agent
            .run_cancellable("must not run", cancel)
            .await
            .unwrap_err();
        assert!(err.downcast_ref::<AgentCancelled>().is_some());
        assert!(agent.messages.is_empty());
    }
}
