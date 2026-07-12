//! **Pulsing Forge** — agent tool & environment runtime for the Pulsing ecosystem.
//!
//! Sandboxed shell, filesystem, and session tools. Handlers under [`handlers`];
//! host callbacks under [`context`].

pub mod approval;
pub mod cli;
pub mod context;
pub mod discovery;
pub mod error;
pub mod exec_output;
pub mod execpolicy;
pub mod executor;
pub mod handlers;
pub mod mcp;
pub mod patch;
pub mod pty_session;
pub mod result;
pub mod runtime;
pub mod sandbox;
pub mod session_input;
pub mod unified_exec;

pub use context::{
    LocalToolSession, NullToolSession, PlanItem, StepStatus, ToolCallContext, ToolSession,
    UpdatePlanArgs,
};

pub use error::ToolError;
pub use executor::{ToolExecutor, ToolExposure};
pub use result::ToolResult;
pub use runtime::ToolRuntime;
