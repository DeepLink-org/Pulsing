//! Error returned while executing a model-visible tool invocation.
//!
//! Adapted from `codex-tools` `FunctionCallError` (Apache-2.0).

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq, Clone)]
pub enum ToolError {
    #[error("{0}")]
    RespondToModel(String),
    #[error("Fatal error: {0}")]
    Fatal(String),
}

impl ToolError {
    pub fn respond(msg: impl Into<String>) -> Self {
        Self::RespondToModel(msg.into())
    }
}
