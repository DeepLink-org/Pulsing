//! Default tool selections for Forge agent profiles.
//!
//! Schemas intentionally do not live here. The canonical definitions are
//! derived from the executable [`crate::registry::ToolRegistry`].

pub const DEFAULT_TOOL_NAMES: &[&str] = &["update_plan", "Glob", "Read", "Grep", "shell_command"];

/// Tools for LLM-guided ``pulsing init`` (includes write/edit).
pub const INIT_TOOL_NAMES: &[&str] = &[
    "update_plan",
    "Glob",
    "Read",
    "Grep",
    "Write",
    "Edit",
    "shell_command",
];
