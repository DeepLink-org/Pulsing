use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::{CommandId, ProtocolVersion, SessionId, TurnId};
use crate::agent::AgentConfig;
use crate::approval::ApprovalPolicy;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSpec {
    pub cwd: PathBuf,
    pub provider: String,
    pub model: String,
    pub max_tokens: u32,
    pub max_turns: usize,
    pub sandbox: String,
    #[serde(default)]
    pub approval_policy: ApprovalPolicy,
    pub tool_names: Vec<String>,
    pub system_prompt: Option<String>,
}

impl From<AgentConfig> for SessionSpec {
    fn from(value: AgentConfig) -> Self {
        Self {
            cwd: value.cwd,
            provider: value.provider,
            model: value.model,
            max_tokens: value.max_tokens,
            max_turns: value.max_turns,
            sandbox: value.sandbox,
            approval_policy: value.approval_policy,
            tool_names: value.tool_names,
            system_prompt: value.system_prompt,
        }
    }
}

impl From<SessionSpec> for AgentConfig {
    fn from(value: SessionSpec) -> Self {
        Self {
            cwd: value.cwd,
            provider: value.provider,
            model: value.model,
            max_tokens: value.max_tokens,
            max_turns: value.max_turns,
            sandbox: value.sandbox,
            approval_policy: value.approval_policy,
            tool_names: value.tool_names,
            system_prompt: value.system_prompt,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandEnvelope<T> {
    pub protocol: String,
    pub version: ProtocolVersion,
    pub command_id: CommandId,
    pub session_id: SessionId,
    pub expected_seq: Option<u64>,
    pub payload: T,
}

impl<T> CommandEnvelope<T> {
    pub fn new(command_id: CommandId, session_id: SessionId, payload: T) -> Self {
        Self {
            protocol: "forge.session".into(),
            version: ProtocolVersion::V1,
            command_id,
            session_id,
            expected_seq: None,
            payload,
        }
    }

    pub fn expecting(mut self, seq: u64) -> Self {
        self.expected_seq = Some(seq);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateSession {
    pub spec: SessionSpec,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StartTurn {
    pub turn_id: TurnId,
    pub input: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CancelTurn {
    pub turn_id: TurnId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandReceipt {
    pub command_id: CommandId,
    pub session_id: SessionId,
    pub turn_id: Option<TurnId>,
    pub accepted_seq: u64,
}
