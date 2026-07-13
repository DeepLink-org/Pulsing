//! Workflow script resolution — session UX lives in [`crate::session`].

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;

use crate::codex::CodexOptions;
use crate::session::workspace;

pub struct WorkflowSession {
    pub script: PathBuf,
    pub script_args: Vec<String>,
    pub codex: CodexOptions,
    pub batch: bool,
}

pub fn run_session(session: WorkflowSession) -> Result<ExitCode> {
    crate::session::run_workflow(
        session.codex,
        session.script,
        session.script_args,
        session.batch,
    )
}

/// Resolve workflow script: explicit path, or default under `.pulsing/workflows/`.
pub fn resolve_script(explicit: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path);
    }
    workspace::resolve_workflow_script(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pulsing_workspace::{init_workspace, InitOptions, Template};
    use tempfile::tempdir;

    #[test]
    fn resolve_default_example() {
        let dir = tempdir().unwrap();
        init_workspace(
            dir.path(),
            InitOptions {
                template: Template::Minimal,
                name: None,
                force: false,
                guide: None,
            },
        )
        .unwrap();
        std::env::set_current_dir(dir.path()).unwrap();
        let path = resolve_script(None).unwrap();
        assert!(path.ends_with("example.py"));
    }
}
