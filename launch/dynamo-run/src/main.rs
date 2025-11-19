// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;

use clap::{CommandFactory as _, Parser, Subcommand};
use dynamo_runtime::config::environment_names::logging as env_logging;

use dynamo_llm::entrypoint::input::Input;
use pulsing_cli::{benchmark::Benchmark, Output, Run};
use dynamo_runtime::logging;

const HELP: &str = r#"
pulsing is a single binary that wires together the various inputs (http, text, network) and workers (network, engine), that runs the services. It is the simplest way to use dynamo locally.

Verbosity:
- -v enables debug logs
- -vv enables full trace logs
- Default is info level logging

Example:
- cargo build --features cuda -p pulsing
- cd target/debug
- ./pulsing run Qwen/Qwen3-0.6B (OR ./pulsing run /data/hf-checkouts/Qwen3-0.6B)

See `docs/guides/dynamo_run.md` in the repo for full details.
"#;

const USAGE: &str = "USAGE: pulsing run in=[http|grpc|text|dyn://<path>|batch:<folder>] out=ENGINE_LIST|auto|dyn://<path> [--http-port 8080] [--model-path <path>] [--model-name <served-model-name>] [--context-length=N] [--kv-cache-block-size=16] [--extra-engine-args=args.json] [--router-mode random|round-robin|kv] [--kv-overlap-score-weight=2.0] [--router-temperature=0.0] [--use-kv-events] [--max-num-batched-tokens=1.0] [--migration-limit=0] [--verbosity (-v|-vv)]";

#[derive(Parser)]
#[command(name = "pulsing")]
#[command(about = "Pulsing CLI - Run inference services and benchmarks")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run inference services
    #[command(name = "run")]
    Run(Run),
    /// Run inference benchmarks
    #[command(name = "benchmark")]
    Benchmark(Benchmark),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Benchmark(benchmark)) => {
            // Initialize logging for benchmark
            env_logger::Builder::from_default_env()
                .filter_level(log::LevelFilter::Info)
                .init();
            
            // Run benchmark in async context
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(benchmark.run())
        }
        Some(Command::Run(run)) => {
            // New format with subcommand
            handle_run_command(run)
        }
        None => {
            // Handle legacy command line format (no subcommand)
            handle_legacy_format()
        }
    }
}

fn handle_legacy_format() -> anyhow::Result<()> {
    // Set log level based on verbosity flag
    let log_level = match Run::try_parse() {
        Ok(run) => match run.verbosity {
            0 => "info",
            1 => "debug",
            2 => "trace",
            _ => {
                return Err(anyhow::anyhow!(
                    "Invalid verbosity level. Valid values are v (debug) or vv (trace)"
                ));
            }
        },
        Err(_) => "info",
    };

    if log_level != "info" {
        unsafe { std::env::set_var(env_logging::DYN_LOG, log_level) };
    }

    logging::init();

    // max_worker_threads and max_blocking_threads from env vars or config file.
    let rt_config = dynamo_runtime::RuntimeConfig::from_settings()?;
    tracing::debug!("Runtime config: {rt_config}");

    // One per process. Wraps a Runtime with holds one or two tokio runtimes.
    let worker = dynamo_runtime::Worker::from_config(rt_config)?;

    worker.execute(wrapper)
}

fn handle_run_command(run: Run) -> anyhow::Result<()> {
    // Set log level based on verbosity flag
    let log_level = match run.verbosity {
        0 => "info",
        1 => "debug",
        2 => "trace",
        _ => {
            return Err(anyhow::anyhow!(
                "Invalid verbosity level. Valid values are v (debug) or vv (trace)"
            ));
        }
    };

    if log_level != "info" {
        unsafe { std::env::set_var(env_logging::DYN_LOG, log_level) };
    }

    logging::init();

    // max_worker_threads and max_blocking_threads from env vars or config file.
    let rt_config = dynamo_runtime::RuntimeConfig::from_settings()?;
    tracing::debug!("Runtime config: {rt_config}");

    // One per process. Wraps a Runtime with holds one or two tokio runtimes.
    let worker = dynamo_runtime::Worker::from_config(rt_config)?;

    worker.execute(move |runtime| async move {
        let mut in_opt = None;
        let mut out_opt = None;
        let args: Vec<String> = env::args().skip(1).collect();
        if args.is_empty()
            || args[0] == "-h"
            || args[0] == "--help"
            || (args.iter().all(|arg| arg == "-v" || arg == "-vv"))
        {
            let engine_list = Output::available_engines().join("|");
            let usage = USAGE.replace("ENGINE_LIST", &engine_list);
            println!("{usage}");
            println!("{HELP}");
            Run::command().print_long_help().unwrap();
            return Ok(());
        } else if args[0] == "--version" {
            if let Some(describe) = option_env!("VERGEN_GIT_DESCRIBE") {
                println!("pulsing {}", describe);
            } else {
                println!("Version not available (git describe not available)");
            }
            return Ok(());
        }
        for arg in env::args().skip(1).take(2) {
            let Some((in_out, val)) = arg.split_once('=') else {
                // Probably we're defaulting in and/or out, and this is a flag
                continue;
            };
            match in_out {
                "in" => {
                    in_opt = Some(val.try_into()?);
                }
                "out" => {
                    if val == "sglang" || val == "trtllm" || val == "vllm" {
                        tracing::error!(
                            "To run the {val} engine please use the Python interface, see root README or look in directory `examples/backends/`."
                        );
                        std::process::exit(1);
                    }

                    out_opt = Some(val.try_into()?);
                }
                _ => {
                    anyhow::bail!("Invalid argument, must start with 'in' or 'out. {USAGE}");
                }
            }
        }
        let mut non_flag_params = 1; // binary name
        let in_opt = match in_opt {
            Some(x) => {
                non_flag_params += 1;
                x
            }
            None => Input::default(),
        };
        if out_opt.is_some() {
            non_flag_params += 1;
        }

        // Clap skips the first argument expecting it to be the binary name, so add it back
        // Note `--model-path` has index=1 (in lib.rs) so that doesn't need a flag.
        let flags = Run::try_parse_from(
            ["pulsing".to_string(), "run".to_string()]
                .into_iter()
                .chain(env::args().skip(non_flag_params)),
        )?;

        if is_in_dynamic(&in_opt) && is_out_dynamic(&out_opt) {
            anyhow::bail!("Cannot use endpoint for both in and out");
        }

        pulsing_cli::run(runtime, in_opt, out_opt, flags).await
    })
}

async fn wrapper(runtime: dynamo_runtime::Runtime) -> anyhow::Result<()> {
    let mut in_opt = None;
    let mut out_opt = None;
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty()
        || args[0] == "-h"
        || args[0] == "--help"
        || (args.iter().all(|arg| arg == "-v" || arg == "-vv"))
    {
        let engine_list = Output::available_engines().join("|");
        let usage = USAGE.replace("ENGINE_LIST", &engine_list);
        println!("{usage}");
        println!("{HELP}");
        Run::command().print_long_help().unwrap();
        return Ok(());
    } else if args[0] == "--version" {
        if let Some(describe) = option_env!("VERGEN_GIT_DESCRIBE") {
            println!("pulsing {}", describe);
        } else {
            println!("Version not available (git describe not available)");
        }
        return Ok(());
    }
    for arg in env::args().skip(1).take(2) {
        let Some((in_out, val)) = arg.split_once('=') else {
            // Probably we're defaulting in and/or out, and this is a flag
            continue;
        };
        match in_out {
            "in" => {
                in_opt = Some(val.try_into()?);
            }
            "out" => {
                if val == "sglang" || val == "trtllm" || val == "vllm" {
                    tracing::error!(
                        "To run the {val} engine please use the Python interface, see root README or look in directory `examples/backends/`."
                    );
                    std::process::exit(1);
                }

                out_opt = Some(val.try_into()?);
            }
            _ => {
                anyhow::bail!("Invalid argument, must start with 'in' or 'out. {USAGE}");
            }
        }
    }
    let mut non_flag_params = 1; // binary name
    let in_opt = match in_opt {
        Some(x) => {
            non_flag_params += 1;
            x
        }
        None => Input::default(),
    };
    if out_opt.is_some() {
        non_flag_params += 1;
    }

    // Clap skips the first argument expecting it to be the binary name, so add it back
    // Note `--model-path` has index=1 (in lib.rs) so that doesn't need a flag.
    let flags = Run::try_parse_from(
        ["pulsing".to_string()]
            .into_iter()
            .chain(env::args().skip(non_flag_params)),
    )?;

    if is_in_dynamic(&in_opt) && is_out_dynamic(&out_opt) {
        anyhow::bail!("Cannot use endpoint for both in and out");
    }

    pulsing_cli::run(runtime, in_opt, out_opt, flags).await
}

fn is_in_dynamic(in_opt: &Input) -> bool {
    matches!(in_opt, Input::Endpoint(_))
}

fn is_out_dynamic(out_opt: &Option<Output>) -> bool {
    matches!(out_opt, Some(Output::Auto))
}
