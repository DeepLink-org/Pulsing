use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use walkdir::WalkDir;

use crate::ignore::should_skip;
use crate::layout::WorkspaceLayout;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRecord {
    pub path: String,
    pub sha256: String,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevisionManifest {
    pub id: String,
    pub parent: Option<String>,
    pub created_at: String,
    pub message: String,
    pub author: String,
    pub files: Vec<FileRecord>,
}

#[derive(Debug, Clone)]
pub struct RevisionInfo {
    pub id: String,
    pub created_at: String,
    pub message: String,
    pub author: String,
    pub file_count: usize,
}

#[derive(Debug, Clone)]
pub struct CheckpointOptions {
    pub message: Option<String>,
    pub author: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RollbackOptions {
    pub revision_id: Option<String>,
}

pub fn list_revisions(layout: &WorkspaceLayout) -> Result<Vec<RevisionInfo>> {
    let mut out = Vec::new();
    let rev_dir = layout.revisions_dir();
    if !rev_dir.is_dir() {
        return Ok(out);
    }
    for entry in fs::read_dir(&rev_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let manifest_path = entry.path().join("manifest.json");
        if !manifest_path.is_file() {
            continue;
        }
        let manifest: RevisionManifest =
            serde_json::from_str(&fs::read_to_string(&manifest_path)?)?;
        out.push(RevisionInfo {
            id: manifest.id,
            created_at: manifest.created_at,
            message: manifest.message,
            author: manifest.author,
            file_count: manifest.files.len(),
        });
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

pub fn current_head(layout: &WorkspaceLayout) -> Result<Option<String>> {
    let head = layout.head_file();
    if !head.is_file() {
        return Ok(None);
    }
    let id = fs::read_to_string(&head)?.trim().to_string();
    if id.is_empty() {
        Ok(None)
    } else {
        Ok(Some(id))
    }
}

pub fn checkpoint(layout: &WorkspaceLayout, opts: CheckpointOptions) -> Result<RevisionManifest> {
    let parent = current_head(layout)?;
    let next_id = next_revision_id(layout)?;
    let rev_path = layout.revision_dir(&next_id);
    let files_dir = rev_path.join("files");
    fs::create_dir_all(&files_dir)?;

    let scanned = scan_workspace_files(layout)?;
    let mut records = Vec::new();
    for rel in scanned {
        let src = layout.root.join(&rel);
        let dest = files_dir.join(&rel);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&src, &dest)?;
        let data = fs::read(&src)?;
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let hash = format!("{:x}", hasher.finalize());
        records.push(FileRecord {
            path: rel.to_string_lossy().into_owned(),
            sha256: hash,
            size: data.len() as u64,
        });
    }

    let manifest = RevisionManifest {
        id: next_id.clone(),
        parent,
        created_at: Utc::now().to_rfc3339(),
        message: opts
            .message
            .unwrap_or_else(|| "checkpoint".to_string()),
        author: opts.author.unwrap_or_else(|| "user".to_string()),
        files: records,
    };

    fs::write(
        rev_path.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;
    fs::write(layout.head_file(), format!("{next_id}\n"))?;
    Ok(manifest)
}

pub fn rollback(layout: &WorkspaceLayout, opts: RollbackOptions) -> Result<RevisionManifest> {
    let id = match opts.revision_id {
        Some(id) => id,
        None => current_head(layout)?.context("no checkpoint to roll back to")?,
    };
    let rev_path = layout.revision_dir(&id);
    let manifest_path = rev_path.join("manifest.json");
    if !manifest_path.is_file() {
        bail!("revision {id} not found");
    }
    let manifest: RevisionManifest =
        serde_json::from_str(&fs::read_to_string(&manifest_path)?)?;
    let files_dir = rev_path.join("files");

    for file in &manifest.files {
        let rel = Path::new(&file.path);
        if should_skip(rel) {
            continue;
        }
        let src = files_dir.join(rel);
        let dest = layout.root.join(rel);
        if !src.is_file() {
            continue;
        }
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&src, &dest)?;
    }

    fs::write(layout.head_file(), format!("{id}\n"))?;
    Ok(manifest)
}

fn next_revision_id(layout: &WorkspaceLayout) -> Result<String> {
    let existing = list_revisions(layout)?;
    let next = existing
        .last()
        .map(|r| r.id.parse::<u32>().unwrap_or(0) + 1)
        .unwrap_or(1);
    Ok(format!("{next:04}"))
}

fn scan_workspace_files(layout: &WorkspaceLayout) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in WalkDir::new(&layout.root)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let rel = layout
            .rel_to_root(path)
            .with_context(|| format!("path outside workspace: {}", path.display()))?;
        if should_skip(&rel) {
            continue;
        }
        paths.push(rel);
    }
    paths.sort();
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init::{init_workspace, InitOptions, Template};
    use tempfile::tempdir;

    #[test]
    fn init_checkpoint_rollback_roundtrip() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("hello.txt"), b"v1").unwrap();

        init_workspace(
            root,
            InitOptions {
                template: Template::Minimal,
                name: None,
                force: false,
            },
        )
        .unwrap();

        let layout = WorkspaceLayout::new(root);
        fs::write(root.join("hello.txt"), b"v2").unwrap();
        checkpoint(
            &layout,
            CheckpointOptions {
                message: Some("v2".into()),
                author: None,
            },
        )
        .unwrap();

        fs::write(root.join("hello.txt"), b"v3").unwrap();
        rollback(&layout, RollbackOptions { revision_id: None }).unwrap();
        assert_eq!(fs::read(root.join("hello.txt")).unwrap(), b"v2");
    }
}
