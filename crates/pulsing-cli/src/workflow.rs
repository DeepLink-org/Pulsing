//! Workspace workflow path resolution.

use std::path::PathBuf;

use anyhow::Result;
use pulsing_workspace::{require_workspace_root, WorkspaceLayout};

pub fn resolve_script(explicit: Option<PathBuf>) -> Result<PathBuf> {
    match explicit {
        Some(path) => Ok(path),
        None => {
            let root = require_workspace_root(None)?;
            Ok(WorkspaceLayout::new(root)
                .workflows_dir()
                .join("example.py"))
        }
    }
}
