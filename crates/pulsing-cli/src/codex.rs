//! Codex-style entry — delegates interactive UX to [`crate::session`].

use std::process::ExitCode;

use anyhow::{Context, Result};
use pulsing_workspace::{find_workspace_root, init_workspace, InitOptions, Template};

#[derive(Debug, Clone)]
pub struct CodexOptions {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub auto_init: bool,
}

pub fn run_default(prompt: Option<&str>, opts: CodexOptions) -> Result<ExitCode> {
    crate::session::run_safe(opts, prompt)
}

pub fn ensure_workspace(auto_init: bool) -> Result<()> {
    if find_workspace_root(None).is_some() {
        return Ok(());
    }
    if !auto_init {
        eprintln!("tip: `pulsing init` to bootstrap a workspace (or pass --init)");
        return Ok(());
    }
    let root = std::env::current_dir().context("cwd")?;
    init_workspace(
        &root,
        InitOptions {
            template: Template::Agent,
            name: None,
            force: false,
            guide: None,
        },
    )?;
    eprintln!("initialized workspace at {}", root.display());
    Ok(())
}
