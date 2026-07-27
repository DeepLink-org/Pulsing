//! Forge coding agent — Codex-style LLM + tool loop (pure Rust).

mod events;
mod init_guide;
mod interactive;
mod r#loop;
mod tools;

pub use events::{AgentEvent, AgentEventTx};
pub use init_guide::run_init_guide;
pub use interactive::{InteractiveConfig, run_interactive, run_oneshot};
pub use r#loop::{
    AgentCancelled, AgentConfig, AgentEventHandler, ForgeAgent, default_model_for_provider,
    default_provider, run_agent_turn, run_agent_turn_observed,
};
pub use tools::{DEFAULT_TOOL_NAMES, INIT_TOOL_NAMES};
