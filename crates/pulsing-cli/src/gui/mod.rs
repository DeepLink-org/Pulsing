//! Desktop chat GUI — `pulsing gui` (GPUI).

use std::process::ExitCode;

use anyhow::Result;

use crate::codex::{ensure_workspace, CodexOptions};
use crate::session::config;

pub fn run(opts: CodexOptions) -> Result<ExitCode> {
    ensure_workspace(opts.auto_init)?;
    let agent = config::interactive_config(&opts)?;
    pulsing_gui::run(agent)?;
    Ok(ExitCode::SUCCESS)
}
