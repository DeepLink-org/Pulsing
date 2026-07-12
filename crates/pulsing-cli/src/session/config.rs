//! Session configuration (LLM provider, workspace paths).

use anyhow::{Context, Result};
use pulsing_forge::InteractiveConfig;
use pulsing_workspace::find_workspace_root;

use crate::codex::CodexOptions;

pub fn interactive_config(opts: &CodexOptions) -> Result<InteractiveConfig> {
    let mut cfg = workspace_interactive_config()?;
    if let Some(ref p) = opts.provider {
        cfg.provider = p.clone();
        cfg.model = pulsing_forge::default_model_for_provider(&cfg.provider);
    }
    if let Some(ref m) = opts.model {
        cfg.model = m.clone();
    }
    Ok(cfg)
}

fn workspace_interactive_config() -> Result<InteractiveConfig> {
    let cwd = std::env::current_dir().context("cwd")?;
    let mut cfg = InteractiveConfig {
        cwd: cwd.clone(),
        ..InteractiveConfig::default()
    };

    if let Some(root) = find_workspace_root(Some(&cwd)) {
        let cluster_path = root.join(".pulsing").join("cluster.json");
        if cluster_path.is_file() {
            let data: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&cluster_path)?)?;
            if let Some(p) = data.get("provider").and_then(|v| v.as_str()) {
                if !p.is_empty() {
                    cfg.provider = p.to_string();
                }
            }
            if let Some(m) = data.get("model").and_then(|v| v.as_str()) {
                if !m.is_empty() {
                    cfg.model = m.to_string();
                }
            } else {
                cfg.model = pulsing_forge::default_model_for_provider(&cfg.provider);
            }
        }
    }

    if !provider_has_credentials(&cfg.provider) {
        cfg.provider = pulsing_forge::default_provider();
        cfg.model = pulsing_forge::default_model_for_provider(&cfg.provider);
    }
    Ok(cfg)
}

fn provider_has_credentials(provider: &str) -> bool {
    match provider.trim().to_lowercase().as_str() {
        "demo" => true,
        "openai" => std::env::var("OPENAI_API_KEY")
            .map(|k| !k.is_empty())
            .unwrap_or(false),
        _ => std::env::var("ANTHROPIC_API_KEY")
            .map(|k| !k.is_empty())
            .unwrap_or(false),
    }
}

pub fn tokio_runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("tokio runtime")
}
