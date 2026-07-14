//! Desktop chat GUI — `pulsing gui` (egui).

use std::process::ExitCode;

use anyhow::Result;

use crate::codex::CodexOptions;

#[cfg(feature = "gui")]
pub fn run(opts: CodexOptions) -> Result<ExitCode> {
    use crate::codex::ensure_workspace;
    use crate::session::config;

    ensure_workspace(opts.auto_init)?;
    let agent = config::interactive_config(&opts)?;
    pulsing_gui::run(agent)?;
    Ok(ExitCode::SUCCESS)
}

#[cfg(not(feature = "gui"))]
pub fn run(_opts: CodexOptions) -> Result<ExitCode> {
    anyhow::bail!(
        "desktop GUI is not available in this build; rebuild with \
         `cargo build -p pulsing-cli --features gui`"
    )
}
