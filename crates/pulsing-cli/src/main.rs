//! Actor CLI and workspace workflows, without an embedded agent product.

mod embed;
mod help;
mod workflow;
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
    about = "Distributed actor runtime and workspace tools"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    #[command(flatten)]
    Workspace(WorkspaceCommand),
    /// Execute a Python workflow once.
    #[command(visible_alias = "workflow")]
    Run {
        script: Option<PathBuf>,
        /// Explicit one-shot mode (all workflows now run once).
        #[arg(long)]
        batch: bool,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        script_args: Vec<String>,
    },
    /// Python actor CLI (actor, inspect, examples).
    #[command(external_subcommand)]
    Passthrough(Vec<OsString>),
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
        Some(Command::Workspace(cmd)) => workspace::run(cmd)?,
        Some(Command::Run {
            script,
            script_args,
            batch: _,
        }) => {
            let script = workflow::resolve_script(script)?;
            embed::run_workflow_script(&script, &script_args)?;
        }
        Some(Command::Passthrough(args)) => {
            if !embed::extension_mode_available() {
                anyhow::bail!("{}", help::EXTENSION_UNAVAILABLE.trim());
            }
            let args = args
                .into_iter()
                .map(|s| s.to_string_lossy().into_owned())
                .collect::<Vec<_>>();
            return embed::delegate_to_python_cli(&args);
        }
        None => {
            use clap::CommandFactory;
            Cli::command().print_help()?;
            println!();
        }
    }
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_workspace_init() {
        let cli = Cli::parse_from(["pulsing", "init", "--template", "minimal"]);
        assert!(matches!(
            cli.command,
            Some(Command::Workspace(WorkspaceCommand::Init { .. }))
        ));
    }

    #[test]
    fn parse_workflow_alias() {
        let cli = Cli::parse_from(["pulsing", "workflow", "app.py", "arg"]);
        let Some(Command::Run {
            script,
            script_args,
            ..
        }) = cli.command
        else {
            panic!("expected Run");
        };
        assert_eq!(script, Some(PathBuf::from("app.py")));
        assert_eq!(script_args, vec!["arg"]);
    }

    #[test]
    fn actor_commands_pass_through() {
        let cli = Cli::parse_from(["pulsing", "actor", "my_app.Counter"]);
        assert!(matches!(cli.command, Some(Command::Passthrough(_))));
    }

    #[test]
    fn init_rejects_llm_options() {
        assert!(Cli::try_parse_from(["pulsing", "init", "--provider", "demo"]).is_err());
    }
}
