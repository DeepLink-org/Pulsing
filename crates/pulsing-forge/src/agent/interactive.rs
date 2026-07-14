//! Codex-style interactive session (REPL prompt loop).

use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::Result;

use super::r#loop::{AgentConfig, default_model_for_provider, default_provider, run_agent_turn};

#[derive(Clone, PartialEq)]
pub struct InteractiveConfig {
    pub cwd: PathBuf,
    pub provider: String,
    pub model: String,
}

impl Default for InteractiveConfig {
    fn default() -> Self {
        let provider = default_provider();
        Self {
            cwd: std::env::current_dir().unwrap_or_else(|_| ".".into()),
            provider: provider.clone(),
            model: default_model_for_provider(&provider),
        }
    }
}

pub async fn run_interactive(cfg: InteractiveConfig) -> Result<()> {
    let agent_cfg = AgentConfig {
        cwd: cfg.cwd,
        provider: cfg.provider.clone(),
        model: cfg.model,
        ..AgentConfig::default()
    };

    eprintln!(
        "Pulsing safe mode ({}/{}) — Forge tools, no user Python",
        agent_cfg.provider, agent_cfg.model
    );
    eprintln!("type a task · empty line or Ctrl-D to exit · `pulsing run` for workflows");
    if agent_cfg.provider == "demo" {
        eprintln!("(demo LLM — set ANTHROPIC_API_KEY or OPENAI_API_KEY for live models)");
    }

    loop {
        print!("› ");
        let _ = io::stdout().flush();
        let mut line = String::new();
        let n = io::stdin().read_line(&mut line)?;
        if n == 0 {
            break;
        }
        let prompt = line.trim();
        if prompt.is_empty() {
            break;
        }
        if prompt == "exit" || prompt == "quit" {
            break;
        }
        match run_agent_turn(&agent_cfg, prompt).await {
            Ok(reply) => {
                println!("\n{reply}\n");
            }
            Err(e) => {
                eprintln!("error: {e:#}");
            }
        }
    }
    Ok(())
}

pub async fn run_oneshot(cfg: InteractiveConfig, prompt: &str) -> Result<String> {
    let agent_cfg = AgentConfig {
        cwd: cfg.cwd,
        provider: cfg.provider,
        model: cfg.model,
        ..AgentConfig::default()
    };
    run_agent_turn(&agent_cfg, prompt).await
}
