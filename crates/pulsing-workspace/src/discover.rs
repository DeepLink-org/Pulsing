use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::layout::{WorkspaceLayout, CLUSTER_FILE, PULSING_DIR};

/// Find directory containing ``.pulsing/cluster.json``.
pub fn find_workspace_root(start: Option<&Path>) -> Option<PathBuf> {
    let mut cur = start
        .unwrap_or_else(|| Path::new("."))
        .canonicalize()
        .ok()?;
    loop {
        if cur.join(PULSING_DIR).join(CLUSTER_FILE).is_file() {
            return Some(cur);
        }
        if !cur.pop() {
            return None;
        }
    }
}

pub fn require_workspace_root(start: Option<&Path>) -> Result<PathBuf> {
    find_workspace_root(start).with_context(|| {
        "not a Pulsing workspace — run `pulsing init` in this project directory first"
    })
}

#[allow(dead_code)]
pub fn layout_from_cwd() -> Result<WorkspaceLayout> {
    let root = require_workspace_root(None)?;
    Ok(WorkspaceLayout::new(root))
}
