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
        /// Natural-language goal — LLM customizes workspace after scaffold
        #[arg(short = 'g', long = "guide")]
        guide_flag: Option<String>,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        guide_words: Vec<String>,
        #[arg(long)]
        provider: Option<String>,
        #[arg(long)]
        model: Option<String>,
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
            guide_flag,
            guide_words,
            provider,
            model,
        } => {
            let root = dir.unwrap_or_else(|| std::env::current_dir().expect("cwd"));
            let template = Template::parse(&template)?;
            let guide = merge_guide(guide_flag, guide_words);
            let result = init_workspace(
                &root,
                InitOptions {
                    template,
                    name,
                    force,
                    guide: guide.clone(),
                },
            )?;
            if result.created {
                println!(
                    "initialized {}  (cluster_id={})",
                    result.root.display(),
                    result.cluster_id
                );
                if let Some(ref text) = guide {
                    run_init_guide_step(&result.root, text, provider.as_deref(), model.as_deref())?;
                }
                println!("  pulsing              # safe-mode agent (interactive)");
                println!("  pulsing \"your task\"  # safe-mode one-shot");
                println!("  pulsing run          # immersive workflow session");
                println!("  pulsing history      # list checkpoints");
                println!("  pulsing checkpoint   # save workspace snapshot");
            } else {
                println!("already initialized: {}", result.root.display());
                if let Some(text) = guide.filter(|_| force) {
                    run_init_guide_step(
                        &result.root,
                        &text,
                        provider.as_deref(),
                        model.as_deref(),
                    )?;
                }
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
            println!("rolled back to {} — {}", manifest.id, manifest.message);
        }
    }
    Ok(())
}

fn merge_guide(flag: Option<String>, words: Vec<String>) -> Option<String> {
    if let Some(g) = flag.filter(|s| !s.trim().is_empty()) {
        return Some(g);
    }
    if words.is_empty() {
        None
    } else {
        Some(words.join(" "))
    }
}

fn run_init_guide_step(
    root: &std::path::Path,
    guide: &str,
    provider: Option<&str>,
    model: Option<&str>,
) -> Result<()> {
    eprintln!("\n# LLM-guided bootstrap…");
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let summary = rt.block_on(pulsing_forge::run_init_guide(
        root.to_path_buf(),
        guide,
        provider,
        model,
    ))?;
    println!("\n{summary}\n");
    let layout = WorkspaceLayout::new(root);
    let _ = checkpoint(
        &layout,
        CheckpointOptions {
            message: Some("init guide".into()),
            author: Some("pulsing".into()),
        },
    );
    Ok(())
}

pub fn run_or_exit(cmd: WorkspaceCommand) {
    if let Err(err) = run(cmd) {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}
