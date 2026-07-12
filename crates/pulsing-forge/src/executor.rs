//! Shared runtime contract for model-visible tools.
//!
//! Adapted from `codex-tools` `tool_executor.rs` (Apache-2.0).

use std::future::Future;
use std::pin::Pin;

use crate::error::ToolError;
use crate::result::ToolResult;

pub type ToolExecutorFuture<'a> =
    Pin<Box<dyn Future<Output = Result<ToolResult, ToolError>> + Send + 'a>>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolExposure {
    Direct,
    Deferred,
    DirectModelOnly,
    Hidden,
}

impl ToolExposure {
    pub fn is_direct(self) -> bool {
        matches!(self, Self::Direct | Self::DirectModelOnly)
    }
}

pub trait ToolExecutor: Send + Sync {
    fn tool_name(&self) -> &str;

    fn exposure(&self) -> ToolExposure {
        ToolExposure::Direct
    }

    fn supports_parallel(&self) -> bool {
        false
    }

    fn handle<'a>(
        &'a self,
        ctx: &'a crate::context::ToolCallContext,
        arguments: serde_json::Value,
    ) -> ToolExecutorFuture<'a>;
}
