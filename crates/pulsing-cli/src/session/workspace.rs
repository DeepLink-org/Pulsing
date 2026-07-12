//! Workspace actions invoked from the session (journal, workflow discovery).

use std::path::PathBuf;

use anyhow::{bail, Result};
use pulsing_workspace::{
    checkpoint, list_revisions, require_workspace_root, rollback, CheckpointOptions,
    RollbackOptions, WorkspaceLayout,
};

pub fn print_history() -> Result<()> {
    let root = require_workspace_root(None)?;
    let layout = WorkspaceLayout::new(root);
    let head = pulsing_workspace::current_head(&layout)?;
    let revs = list_revisions(&layout)?;
    if revs.is_empty() {
        eprintln!("no checkpoints yet — use /checkpoint");
        return Ok(());
    }
    for r in revs {
        let mark = if head.as_deref() == Some(r.id.as_str()) {
            "*"
        } else {
            " "
        };
        eprintln!(
            "{mark} {}  {}  {} files  {}",
            r.id, r.created_at, r.file_count, r.message
        );
    }
    Ok(())
}

pub fn save_checkpoint(message: Option<String>) -> Result<String> {
    let root = require_workspace_root(None)?;
    let layout = WorkspaceLayout::new(root);
    let manifest = checkpoint(
        &layout,
        CheckpointOptions {
            message,
            author: Some("pulsing".into()),
        },
    )?;
    Ok(format!(
        "checkpoint {} ({} files) — {}",
        manifest.id,
        manifest.files.len(),
        manifest.message
    ))
}

pub fn do_rollback(revision: Option<String>) -> Result<String> {
    let root = require_workspace_root(None)?;
    let layout = WorkspaceLayout::new(root);
    let manifest = rollback(
        &layout,
        RollbackOptions {
            revision_id: revision,
        },
    )?;
    Ok(format!(
        "rolled back to {} — {}",
        manifest.id, manifest.message
    ))
}

pub fn list_workflow_scripts() -> Result<Vec<PathBuf>> {
    let root = require_workspace_root(None)?;
    let dir = root.join(".pulsing").join("workflows");
    if !dir.is_dir() {
        bail!("no `.pulsing/workflows/` — run `pulsing init`");
    }
    let mut scripts: Vec<PathBuf> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "py"))
        .collect();
    scripts.sort();
    Ok(scripts)
}

pub fn resolve_workflow_script(explicit: Option<&str>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(PathBuf::from(path));
    }
    let scripts = list_workflow_scripts()?;
    let example = scripts
        .iter()
        .find(|p| p.file_name().is_some_and(|n| n == "example.py"));
    if let Some(p) = example {
        return Ok(p.clone());
    }
    scripts
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no workflow scripts in `.pulsing/workflows/`"))
}
