//! Forge session REPL (Rust-native). Invoked via ``pulsing forge repl``.

pub mod commands;
pub mod completer;
pub mod parse;
pub mod repl;
pub mod session;
pub mod trace;

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

pub use repl::{ForgeRepl, ReplConfig};

#[derive(Parser, Debug)]
#[command(
    name = "pulsing forge repl",
    about = "Forge session REPL (Rust-native shell)"
)]
pub struct ReplCliArgs {
    #[arg(long, default_value = ".")]
    pub cwd: PathBuf,
    #[arg(long, default_value = "off")]
    pub sandbox: String,
    #[arg(long)]
    pub dangerously_disable_sandbox: bool,
    #[arg(long, default_value = "auto")]
    pub approve: String,
    #[arg(long)]
    pub trace: Option<PathBuf>,
    #[arg(long)]
    pub record: Option<PathBuf>,
    #[arg(long)]
    pub replay_all: bool,
    #[arg(long)]
    pub dry_run: bool,
    #[arg(long)]
    pub verify: bool,
}

pub fn run_repl(args: ReplCliArgs) -> Result<()> {
    let approve_auto = !args.approve.eq_ignore_ascii_case("ask");
    let cfg = ReplConfig {
        cwd: args.cwd.canonicalize().unwrap_or(args.cwd),
        sandbox: args.sandbox,
        dangerously_disable_sandbox: args.dangerously_disable_sandbox,
        approve_auto,
        trace_path: args.trace.clone(),
        record_path: args.record,
    };

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    rt.block_on(async {
        let mut repl = ForgeRepl::new(cfg)?;
        if args.replay_all {
            if args.trace.is_none() {
                anyhow::bail!("--replay-all requires --trace");
            }
            for line in repl.replay_all(args.dry_run, args.verify).await? {
                println!("{line}");
            }
            Ok(())
        } else {
            repl.run_interactive()
        }
    })
}

pub fn run_repl_from_iter<I, T>(iter: I) -> Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    run_repl(ReplCliArgs::parse_from(iter))
}
