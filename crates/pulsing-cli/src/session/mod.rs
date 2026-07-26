//! Unified immersive session — all interactive UX stays in pulsing-cli.
//!
//! pulsing-forge provides agent/tools; pulsing-workspace provides journal;
//! this module owns prompts, slash commands, workflow orchestration, and rendering.

mod commands;
pub mod config;
mod input;
mod mode;
mod render;
pub mod workspace;

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use pulsing_forge::{run_oneshot, AgentConfig, InteractiveConfig, LocalForgeClient, SessionId};

use commands::{parse_line, InputAction};
use mode::SessionMode;

use crate::codex::{ensure_workspace, CodexOptions};
use crate::embed;

const PROMPT: &str = "› ";

/// Entry: `pulsing` / `pulsing agent` (safe mode).
pub fn run_safe(opts: CodexOptions, initial_prompt: Option<&str>) -> Result<ExitCode> {
    ensure_workspace(opts.auto_init)?;

    let rt = config::tokio_runtime()?;
    let agent_cfg = config::interactive_config(&opts)?;

    if let Some(text) = initial_prompt.filter(|s| !s.trim().is_empty()) {
        let reply = rt.block_on(run_oneshot(agent_cfg, text.trim()))?;
        println!("{reply}");
        return Ok(ExitCode::SUCCESS);
    }

    let mut state = SessionState {
        mode: SessionMode::Safe,
        agent: agent_cfg,
    };
    run_loop(&mut state)?;
    Ok(ExitCode::SUCCESS)
}

/// Entry: `pulsing run` (workflow then immersive session, unless ``batch``).
pub fn run_workflow(
    opts: CodexOptions,
    script: PathBuf,
    script_args: Vec<String>,
    batch: bool,
) -> Result<ExitCode> {
    ensure_workspace(opts.auto_init)?;
    require_extension_mode()?;

    let script = script
        .canonicalize()
        .with_context(|| format!("workflow not found: {}", script.display()))?;

    if batch {
        return match embed::run_workflow_script(&script, &script_args) {
            Ok(()) => Ok(ExitCode::SUCCESS),
            Err(err) => {
                render::print_workflow_err(&err);
                Ok(ExitCode::FAILURE)
            }
        };
    }

    let agent_cfg = config::interactive_config(&opts)?;

    render::print_workflow_start(&script);
    match embed::run_workflow_script(&script, &script_args) {
        Ok(()) => {
            render::print_workflow_ok();
            let mut state = SessionState {
                mode: SessionMode::WorkflowIdle {
                    script: script.clone(),
                    args: script_args.clone(),
                },
                agent: agent_cfg,
            };
            run_loop(&mut state)?;
            Ok(ExitCode::SUCCESS)
        }
        Err(err) => {
            render::print_workflow_err(&err);
            render::print_failure_recovery(&script);
            if input::confirm("› retry workflow? [y/N] ", true)? {
                return run_workflow(opts, script, script_args, false);
            }
            Ok(ExitCode::FAILURE)
        }
    }
}

struct SessionState {
    mode: SessionMode,
    agent: InteractiveConfig,
}

fn run_loop(state: &mut SessionState) -> Result<()> {
    let rt = config::tokio_runtime()?;
    let forge = LocalForgeClient::default();
    let forge_session = rt.block_on(forge.create_session(agent_config(&state.agent)))?;
    render::print_session_header(&state.mode, &state.agent);

    while let Some(line) = input::read_line(PROMPT)? {
        let action = parse_line(&line);
        match dispatch(&rt, &forge, &forge_session, state, action)? {
            LoopControl::Continue => {}
            LoopControl::Break => break,
        }
    }
    Ok(())
}

enum LoopControl {
    Continue,
    Break,
}

fn dispatch(
    rt: &tokio::runtime::Runtime,
    forge: &LocalForgeClient,
    forge_session: &SessionId,
    state: &mut SessionState,
    action: InputAction,
) -> Result<LoopControl> {
    match action {
        InputAction::AgentTask { prompt } if prompt.is_empty() => Ok(LoopControl::Continue),
        InputAction::Help => {
            render::print_help(&state.mode);
            Ok(LoopControl::Continue)
        }
        InputAction::Mode => {
            render::print_mode(&state.mode);
            Ok(LoopControl::Continue)
        }
        InputAction::Exit => Ok(LoopControl::Break),
        InputAction::History => {
            workspace::print_history()?;
            Ok(LoopControl::Continue)
        }
        InputAction::Checkpoint { message } => {
            match workspace::save_checkpoint(message) {
                Ok(msg) => eprintln!("{msg}\n"),
                Err(err) => eprintln!("checkpoint failed: {err:#}\n"),
            }
            Ok(LoopControl::Continue)
        }
        InputAction::Rollback { revision } => {
            match workspace::do_rollback(revision) {
                Ok(msg) => eprintln!("{msg}\n"),
                Err(err) => eprintln!("rollback failed: {err:#}\n"),
            }
            Ok(LoopControl::Continue)
        }
        InputAction::WorkflowList => {
            match workspace::list_workflow_scripts() {
                Ok(scripts) => {
                    if scripts.is_empty() {
                        eprintln!("no workflows in `.pulsing/workflows/`\n");
                    } else {
                        for p in scripts {
                            eprintln!("  {}", p.display());
                        }
                        eprintln!();
                    }
                }
                Err(err) => eprintln!("{err:#}\n"),
            }
            Ok(LoopControl::Continue)
        }
        InputAction::WorkflowRun { script } => {
            require_extension_mode()?;
            let path = workspace::resolve_workflow_script(script.as_deref())?;
            let args = match &state.mode {
                SessionMode::WorkflowIdle { args, .. } if script.is_none() => args.clone(),
                _ => Vec::new(),
            };
            render::print_workflow_start(&path);
            match embed::run_workflow_script(&path, &args) {
                Ok(()) => {
                    render::print_workflow_ok();
                    state.mode = SessionMode::WorkflowIdle { script: path, args };
                }
                Err(err) => {
                    render::print_workflow_err(&err);
                    render::print_failure_recovery(&path);
                }
            }
            Ok(LoopControl::Continue)
        }
        InputAction::RerunWorkflow => match &state.mode {
            SessionMode::WorkflowIdle { script, args } => {
                render::print_workflow_start(script);
                match embed::run_workflow_script(script, args) {
                    Ok(()) => render::print_workflow_ok(),
                    Err(err) => {
                        render::print_workflow_err(&err);
                        render::print_failure_recovery(script);
                    }
                }
                Ok(LoopControl::Continue)
            }
            SessionMode::Safe => {
                eprintln!("not in workflow mode — use `/workflow run`\n");
                Ok(LoopControl::Continue)
            }
        },
        InputAction::AgentTask { prompt } => {
            match rt.block_on(forge.run_turn(forge_session.clone(), &prompt)) {
                Ok(reply) => println!("\n{reply}\n"),
                Err(err) => eprintln!("error: {err:#}\n"),
            }
            Ok(LoopControl::Continue)
        }
    }
}

fn agent_config(agent: &InteractiveConfig) -> AgentConfig {
    AgentConfig {
        cwd: agent.cwd.clone(),
        provider: agent.provider.clone(),
        model: agent.model.clone(),
        ..AgentConfig::default()
    }
}

fn require_extension_mode() -> Result<()> {
    if embed::extension_mode_available() {
        Ok(())
    } else {
        anyhow::bail!("{}", crate::help::EXTENSION_UNAVAILABLE.trim());
    }
}
