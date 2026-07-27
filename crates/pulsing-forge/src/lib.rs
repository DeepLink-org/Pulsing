//! **Pulsing Forge** — agent tool & environment runtime for the Pulsing ecosystem.
//!
//! Sandboxed shell, filesystem, and session tools. Handlers under [`handlers`];
//! host callbacks under [`context`].

pub mod agent;
pub mod approval;
pub mod cli;
pub mod context;
pub mod discovery;
pub mod error;
pub mod exec_output;
pub mod execpolicy;
pub mod executor;
pub mod handlers;
pub mod llm;
pub mod mcp;
pub mod patch;
mod process_group;
pub mod protocol;
pub mod pty_session;
pub mod registry;
pub mod result;
pub mod runtime;
pub mod sandbox;
pub mod session;
pub mod session_input;
pub mod turn;
pub mod unified_exec;

pub use context::{
    LocalToolSession, NullToolSession, PlanItem, StepStatus, ToolCallContext, ToolSession,
    UpdatePlanArgs,
};

pub use agent::{
    AgentCancelled, AgentConfig, AgentEvent, AgentEventHandler, AgentEventTx, DEFAULT_TOOL_NAMES,
    INIT_TOOL_NAMES, InteractiveConfig, default_model_for_provider, default_provider,
    run_agent_turn, run_agent_turn_observed, run_init_guide, run_interactive, run_oneshot,
};
pub use error::ToolError;
pub use executor::{ToolExecutor, ToolExposure};
pub use llm::{LlmClient, LlmError, LlmMessage, LlmStream, LlmUsage, Provider, StreamRequest};
pub use protocol::{
    CandidateId, CommandId, CommandReceipt, EventId, ForgeEvent, ForgeEventKind,
    ForgeProtocolError, ProtocolVersion, SessionId, SessionSpec, TurnId,
};
pub use registry::{ToolName, ToolRegistry, ToolRegistryError, ToolSpec};
pub use result::ToolResult;
pub use runtime::ToolRuntime;
pub use session::{
    EventSubscription, ForgeService, InMemoryEventStore, LocalForgeClient, SessionSnapshot,
};
pub use turn::{TurnExecutionContext, TurnResourceGuard, TurnResourceRegistry};
