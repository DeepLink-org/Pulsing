//! Workspace bootstrap and journal commands.

use std::path::PathBuf;

use anyhow::Result;
use clap::Subcommand;
use pulsing_workspace::{
    checkpoint, init_workspace, list_revisions, require_workspace_root, rollback,
    CheckpointOptions, InitOptions, RollbackOptions, Template, WorkspaceLayout,
};

#[derive(Subcommand)]
pub enum WorkspaceCommand {
    /// Bootstrap a workspace with hooks and journal history.
    Init {
        dir: Option<PathBuf>,
        #[arg(long, default_value = "minimal")]
        template: String,
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        force: bool,
    },
    /// List workspace checkpoints.
    History,
    /// Save a workspace snapshot.
    Checkpoint {
        #[arg(short, long)]
        message: Option<String>,
    },
    /// Restore a workspace checkpoint.
    Rollback { revision: Option<String> },
}

pub fn run(cmd: WorkspaceCommand) -> Result<()> {
    match cmd {
        WorkspaceCommand::Init {
            dir,
            template,
            name,
            force,
        } => {
            let root = match dir {
                Some(dir) => dir,
                None => std::env::current_dir()?,
            };
            let result = init_workspace(
                &root,
                InitOptions {
                    template: Template::parse(&template)?,
                    name,
                    force,
                    guide: None,
                },
            )?;
            let status = if result.created {
                "initialized"
            } else {
                "already initialized"
            };
            println!(
                "{status}: {} (cluster_id={})",
                result.root.display(),
                result.cluster_id
            );
        }
        WorkspaceCommand::History => {
            let layout = WorkspaceLayout::new(require_workspace_root(None)?);
            let head = pulsing_workspace::current_head(&layout)?;
            for rev in list_revisions(&layout)? {
                let mark = if head.as_deref() == Some(rev.id.as_str()) {
                    "*"
                } else {
                    " "
                };
                println!(
                    "{mark} {}  {}  {} files  {}",
                    rev.id, rev.created_at, rev.file_count, rev.message
                );
            }
        }
        WorkspaceCommand::Checkpoint { message } => {
            let layout = WorkspaceLayout::new(require_workspace_root(None)?);
            let manifest = checkpoint(
                &layout,
                CheckpointOptions {
                    message,
                    author: None,
                },
            )?;
            println!(
                "checkpoint {} ({} files) — {}",
                manifest.id,
                manifest.files.len(),
                manifest.message
            );
        }
        WorkspaceCommand::Rollback { revision } => {
            let layout = WorkspaceLayout::new(require_workspace_root(None)?);
            let manifest = rollback(
                &layout,
                RollbackOptions {
                    revision_id: revision,
                },
            )?;
            println!("rolled back to {} — {}", manifest.id, manifest.message);
        }
    }
    Ok(())
}
