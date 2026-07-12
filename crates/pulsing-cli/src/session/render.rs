//! Plain-text rendering (future: TUI/GUI plug in here).

use std::path::Path;

use pulsing_forge::InteractiveConfig;

use super::mode::SessionMode;

pub fn print_session_header(mode: &SessionMode, agent: &InteractiveConfig) {
    let script_hint = match mode {
        SessionMode::Safe => String::new(),
        SessionMode::WorkflowIdle { script, .. } => {
            let name = script
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("workflow");
            format!(" · {name}")
        }
    };
    eprintln!(
        "Pulsing session · {}/{}{script_hint}",
        agent.provider, agent.model
    );
    eprintln!("type /help · /exit to leave · empty line does nothing");
    if agent.provider == "demo" {
        eprintln!("(demo LLM — set ANTHROPIC_API_KEY or OPENAI_API_KEY for live models)");
    }
    eprintln!();
}

pub fn print_help(mode: &SessionMode) {
    eprintln!("Commands:");
    eprintln!("  /help              this message");
    eprintln!("  /mode              show safe | workflow");
    eprintln!("  /exit              return to shell");
    eprintln!("  /history           list checkpoints");
    eprintln!("  /checkpoint [msg]  save checkpoint");
    eprintln!("  /rollback [id]     restore checkpoint");
    if matches!(mode, SessionMode::WorkflowIdle { .. }) {
        eprintln!("  /workflow run      re-run current workflow");
        eprintln!("  rerun              alias for /workflow run");
    } else {
        eprintln!("  /workflow list     list `.pulsing/workflows/*.py`");
        eprintln!("  /workflow run [f]  start workflow session");
    }
    eprintln!("  <text>             safe-mode agent task");
    eprintln!();
}

pub fn print_mode(mode: &SessionMode) {
    eprintln!("mode: {}", mode.label());
    if let SessionMode::WorkflowIdle { script, .. } = mode {
        eprintln!("workflow: {}", script.display());
    }
    eprintln!();
}

pub fn print_workflow_start(script: &Path) {
    let name = script
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("workflow");
    eprintln!("— running workflow {name} —");
}

pub fn print_workflow_ok() {
    eprintln!("\n✓ workflow complete\n");
}

pub fn print_workflow_err(err: &anyhow::Error) {
    eprintln!("\n✗ workflow failed: {err:#}\n");
}

pub fn print_failure_recovery(script: &Path) {
    let name = script
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("workflow.py");
    eprintln!("recovery: /rollback · ask agent to fix · /workflow run to retry");
    eprintln!("  pulsing \"fix {name}\"");
    eprintln!();
}
