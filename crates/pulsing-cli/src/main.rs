//! Pulsing CLI — single binary: safe-mode agent + optional Python workflows.

mod codex;
mod embed;
mod gui;
mod help;
mod session;
mod workflow;
mod workspace;

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::{Parser, Subcommand};

use codex::CodexOptions;
use workspace::WorkspaceCommand;

#[derive(Parser)]
#[command(
    name = "pulsing",
    version,
    about = help::ABOUT,
    long_about = help::LONG_ABOUT,
    after_help = help::AFTER_HELP,
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// LLM provider: demo, anthropic, openai
    #[arg(long, global = true)]
    provider: Option<String>,

    /// Model id override
    #[arg(long, global = true)]
    model: Option<String>,

    /// Create `.pulsing/` automatically if missing (safe mode)
    #[arg(long, global = true)]
    init: bool,

    /// One-shot task when no subcommand is given (safe mode)
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    prompt: Vec<String>,
}

#[derive(Subcommand)]
enum Command {
    /// Safe mode: LLM agent with Forge tools (no user Python)
    Agent(AgentArgs),
    /// Bootstrap a Pulsing workspace (`.pulsing/` + journal)
    Init {
        #[arg(value_name = "DIR")]
        dir: Option<PathBuf>,
        #[arg(long, default_value = "agent")]
        template: String,
        #[arg(long)]
        name: Option<String>,
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
    /// Save a workspace checkpoint
    Checkpoint {
        #[arg(short, long)]
        message: Option<String>,
    },
    /// Restore workspace files from a checkpoint
    Rollback { revision: Option<String> },
    /// Immersive workflow session (extension mode; stays in CLI on success)
    #[command(visible_alias = "workflow")]
    Run(RunArgs),
    /// Desktop chat UI (GPUI)
    Gui,
    /// Low-level Forge tool REPL (safe mode)
    Forge {
        #[command(subcommand)]
        cmd: ForgeCommand,
    },
    /// Legacy Python CLI (actor, inspect, …) — extension mode
    #[command(external_subcommand)]
    Passthrough(Vec<OsString>),
}

#[derive(Parser)]
struct AgentArgs {
    /// Task prompt; omit for interactive session
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    prompt: Vec<String>,
}

#[derive(Subcommand)]
enum ForgeCommand {
    /// Interactive Forge tool REPL
    Repl {
        #[arg(long, default_value = ".")]
        cwd: PathBuf,
        #[arg(long, default_value = "off")]
        sandbox: String,
        #[arg(long)]
        dangerously_disable_sandbox: bool,
    },
}

#[derive(Parser)]
struct RunArgs {
    /// Workflow script (default: `.pulsing/workflows/example.py`)
    script: Option<PathBuf>,
    /// Arguments passed to the script
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    script_args: Vec<String>,
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<ExitCode> {
    let cli = Cli::parse();
    let codex_opts = CodexOptions {
        provider: cli.provider,
        model: cli.model,
        auto_init: cli.init,
    };

    match cli.command {
        Some(Command::Agent(args)) => {
            let prompt = join_prompt(args.prompt);
            codex::run_default(prompt.as_deref(), codex_opts)
        }
        Some(Command::Init {
            dir,
            template,
            name,
            force,
            guide_flag,
            guide_words,
            provider,
            model,
        }) => {
            workspace::run_or_exit(WorkspaceCommand::Init {
                dir,
                template,
                name,
                force,
                guide_flag,
                guide_words,
                provider,
                model,
            });
            Ok(ExitCode::SUCCESS)
        }
        Some(Command::History) => {
            workspace::run_or_exit(WorkspaceCommand::History);
            Ok(ExitCode::SUCCESS)
        }
        Some(Command::Checkpoint { message }) => {
            workspace::run_or_exit(WorkspaceCommand::Checkpoint { message });
            Ok(ExitCode::SUCCESS)
        }
        Some(Command::Rollback { revision }) => {
            workspace::run_or_exit(WorkspaceCommand::Rollback { revision });
            Ok(ExitCode::SUCCESS)
        }
        Some(Command::Run(args)) => {
            let script = workflow::resolve_script(args.script)?;
            workflow::run_session(workflow::WorkflowSession {
                script,
                script_args: args.script_args,
                codex: codex_opts,
            })
        }
        Some(Command::Gui) => gui::run(codex_opts),
        Some(Command::Forge {
            cmd:
                ForgeCommand::Repl {
                    cwd,
                    sandbox,
                    dangerously_disable_sandbox,
                },
        }) => {
            pulsing_forge::cli::run_repl(pulsing_forge::cli::ReplCliArgs {
                cwd,
                sandbox,
                dangerously_disable_sandbox,
                approve: "auto".into(),
                trace: None,
                record: None,
                replay_all: false,
                dry_run: false,
                verify: false,
            })?;
            Ok(ExitCode::SUCCESS)
        }
        Some(Command::Passthrough(os_args)) => {
            embed::warn_legacy_mode();
            if !embed::extension_mode_available() {
                anyhow::bail!("{}", help::EXTENSION_UNAVAILABLE.trim());
            }
            let args: Vec<String> = os_args
                .into_iter()
                .map(|s| s.to_string_lossy().into_owned())
                .collect();
            embed::delegate_to_python_cli(&args)
        }
        None => {
            let prompt = join_prompt(cli.prompt);
            codex::run_default(prompt.as_deref(), codex_opts)
        }
    }
}

fn join_prompt(words: Vec<String>) -> Option<String> {
    if words.is_empty() {
        None
    } else {
        Some(words.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_oneshot_prompt() {
        let cli = Cli::parse_from(["pulsing", "list", "README", "files"]);
        assert!(cli.command.is_none());
        assert_eq!(cli.prompt, vec!["list", "README", "files"]);
    }

    #[test]
    fn parse_agent_subcommand() {
        let cli = Cli::parse_from(["pulsing", "agent", "fix", "tests"]);
        let Command::Agent(AgentArgs { prompt }) = cli.command.expect("agent") else {
            panic!("expected Agent");
        };
        assert_eq!(prompt, vec!["fix", "tests"]);
    }

    #[test]
    fn parse_init_with_guide() {
        let cli = Cli::parse_from(["pulsing", "init", "-g", "Python ML project"]);
        let Command::Init { guide_flag, .. } = cli.command.expect("init") else {
            panic!("expected Init");
        };
        assert_eq!(guide_flag.as_deref(), Some("Python ML project"));
    }

    #[test]
    fn parse_workflow_alias() {
        let cli = Cli::parse_from(["pulsing", "workflow", "app.py", "arg"]);
        let Command::Run(RunArgs {
            script,
            script_args,
        }) = cli.command.expect("run")
        else {
            panic!("expected Run");
        };
        assert_eq!(script, Some(PathBuf::from("app.py")));
        assert_eq!(script_args, vec!["arg"]);
    }

    #[test]
    fn parse_run_without_script() {
        let cli = Cli::parse_from(["pulsing", "run"]);
        let Command::Run(RunArgs { script, .. }) = cli.command.expect("run") else {
            panic!("expected Run");
        };
        assert!(script.is_none());
    }

    #[test]
    fn parse_forge_repl() {
        let cli = Cli::parse_from(["pulsing", "forge", "repl"]);
        let Command::Forge {
            cmd: ForgeCommand::Repl { .. },
        } = cli.command.expect("forge")
        else {
            panic!("expected forge repl");
        };
    }
}
