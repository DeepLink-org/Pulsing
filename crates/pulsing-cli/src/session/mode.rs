//! Session mode — safe agent vs workflow (all UX lives in pulsing-cli).

use std::path::PathBuf;

#[derive(Debug, Clone)]
pub enum SessionMode {
    /// Safe mode: Forge agent only (no user Python).
    Safe,
    /// Workflow finished; script can be re-run from the session.
    WorkflowIdle { script: PathBuf, args: Vec<String> },
}

impl SessionMode {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Safe => "safe",
            Self::WorkflowIdle { .. } => "workflow",
        }
    }
}
