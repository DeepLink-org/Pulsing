//! Workspace subcommands for the pulsing binary.

use std::path::PathBuf;

use anyhow::Result;
use clap::Subcommand;
use pulsing_workspace::{
    checkpoint, init_workspace, list_revisions, require_workspace_root, rollback,
    CheckpointOptions, InitOptions, RollbackOptions, Template, WorkspaceLayout,
};

#[derive(Subcommand)]
pub enum WorkspaceCommand {
    /// Bootstrap a Pulsing workspace (``.pulsing/`` + hooks + journal)
    Init {
        /// Target directory (default: current directory)
        #[arg(value_name = "DIR")]
        dir: Option<PathBuf>,
        /// Workspace template: minimal or agent
        #[arg(long, default_value = "agent")]
        template: String,
        /// Display name stored in cluster.json
        #[arg(long)]
        name: Option<String>,
        /// Re-initialize even if workspace exists
        #[arg(long)]
        force: bool,
    },
    /// List workspace checkpoints
    History,
    /// Save a new checkpoint of tracked files
    Checkpoint {
        /// Checkpoint message
        #[arg(short, long)]
        message: Option<String>,
    },
    /// Restore files from a checkpoint (default: HEAD)
    Rollback {
        /// Revision id (e.g. 0003); default is latest checkpoint
        revision: Option<String>,
    },
}

pub fn run(cmd: WorkspaceCommand) -> Result<()> {
    match cmd {
        WorkspaceCommand::Init {
            dir,
            template,
            name,
            force,
        } => {
            let root = dir.unwrap_or_else(|| std::env::current_dir().expect("cwd"));
            let template = Template::parse(&template)?;
            let result = init_workspace(
                &root,
                InitOptions {
                    template,
                    name,
                    force,
                },
            )?;
            if result.created {
                println!(
                    "initialized {}  (cluster_id={})",
                    result.root.display(),
                    result.cluster_id
                );
                if template == Template::Agent {
                    println!("  pulsing agent wake   # start agents");
                }
                println!("  pulsing history      # list checkpoints");
                println!("  pulsing checkpoint   # save workspace snapshot");
            } else {
                println!("already initialized: {}", result.root.display());
            }
        }
        WorkspaceCommand::History => {
            let root = require_workspace_root(None)?;
            let layout = WorkspaceLayout::new(root);
            let head = pulsing_workspace::current_head(&layout)?;
            let revs = list_revisions(&layout)?;
            if revs.is_empty() {
                println!("no checkpoints yet — run `pulsing checkpoint`");
                return Ok(());
            }
            for r in revs {
                let mark = if head.as_deref() == Some(r.id.as_str()) {
                    "*"
                } else {
                    " "
                };
                println!(
                    "{mark} {}  {}  {} files  {}",
                    r.id, r.created_at, r.file_count, r.message
                );
            }
        }
        WorkspaceCommand::Checkpoint { message } => {
            let root = require_workspace_root(None)?;
            let layout = WorkspaceLayout::new(root);
            let manifest = checkpoint(
                &layout,
                CheckpointOptions {
                    message,
                    author: None,
                },
            )?;
            println!(
                "checkpoint {}  ({} files) — {}",
                manifest.id,
                manifest.files.len(),
                manifest.message
            );
        }
        WorkspaceCommand::Rollback { revision } => {
            let root = require_workspace_root(None)?;
            let layout = WorkspaceLayout::new(root);
            let manifest = rollback(
                &layout,
                RollbackOptions {
                    revision_id: revision,
                },
            )?;
            println!(
                "rolled back to {} — {}",
                manifest.id, manifest.message
            );
        }
    }
    Ok(())
}

pub fn run_or_exit(cmd: WorkspaceCommand) {
    if let Err(err) = run(cmd) {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}
