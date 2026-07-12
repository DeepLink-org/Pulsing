//! Pulsing CLI — Rust entry with RustPython (`rustpython_vm`).

mod embed;
mod workspace;

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::{Parser, Subcommand};

use workspace::WorkspaceCommand;

#[derive(Parser)]
#[command(
    name = "pulsing",
    version,
    about = "Pulsing — AI agent runtime (local and distributed)",
    long_about = "Rust CLI with embedded RustPython VM.\n\
                  Use `pulsing init` to bootstrap a workspace, then `pulsing run app.py`."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Bootstrap a Pulsing workspace
    Init {
        #[arg(value_name = "DIR")]
        dir: Option<PathBuf>,
        #[arg(long, default_value = "agent")]
        template: String,
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        force: bool,
    },
    /// List workspace checkpoints
    History,
    /// Save a workspace checkpoint
    Checkpoint {
        #[arg(short, long)]
        message: Option<String>,
    },
    /// Restore workspace files from a checkpoint
    Rollback {
        revision: Option<String>,
    },
    /// Run a Python agent / application script
    Run(RunArgs),
    /// Pass-through to the Python CLI (actor, agent, inspect, forge, …)
    #[command(external_subcommand)]
    Passthrough(Vec<OsString>),
}

#[derive(Parser)]
struct RunArgs {
    /// Python script to execute (e.g. app.py)
    script: PathBuf,

    /// Arguments passed to the script (after ``--``)
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
    match cli.command {
        Some(Command::Init {
            dir,
            template,
            name,
            force,
        }) => {
            workspace::run_or_exit(WorkspaceCommand::Init {
                dir,
                template,
                name,
                force,
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
        Some(Command::Run(args)) => embed::run_python_script(&args.script, &args.script_args),
        Some(Command::Passthrough(os_args)) => {
            let args: Vec<String> = os_args
                .into_iter()
                .map(|s| s.to_string_lossy().into_owned())
                .collect();
            embed::delegate_to_python_cli(&args)
        }
        None => {
            // Inside a workspace, default to Python CLI; otherwise show init hint.
            if pulsing_workspace::find_workspace_root(None).is_some() {
                embed::delegate_to_python_cli(&[])
            } else {
                eprintln!("No workspace here. Run: pulsing init");
                eprintln!("Or: pulsing run <script.py>  ·  pulsing --help");
                Ok(ExitCode::from(2))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_run_args_no_extra() {
        let cli = Cli::parse_from([
            "pulsing",
            "run",
            "examples/python/forge_agent_quickstart.py",
        ]);
        let Command::Run(args) = cli.command.expect("run subcommand") else {
            panic!("expected Run");
        };
        assert_eq!(
            args.script,
            PathBuf::from("examples/python/forge_agent_quickstart.py")
        );
        assert!(args.script_args.is_empty(), "{:?}", args.script_args);
    }

    #[test]
    fn parse_init_agent_template() {
        let cli = Cli::parse_from(["pulsing", "init", "--template", "minimal"]);
        let Command::Init { template, .. } = cli.command.expect("init") else {
            panic!("expected Init");
        };
        assert_eq!(template, "minimal");
    }
}
