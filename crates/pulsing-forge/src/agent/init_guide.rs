//! LLM-guided workspace bootstrap after ``pulsing init``.

use std::path::PathBuf;

use super::r#loop::{AgentConfig, default_model_for_provider, default_provider, run_agent_turn};
use super::tools::INIT_TOOL_NAMES;

const INIT_SYSTEM: &str = "You are bootstrapping a new Pulsing AI workspace.\n\
The `.pulsing/` scaffold (cluster.json, hooks, journal) already exists.\n\
\n\
Customize the project to match the user's goal:\n\
1. Read `.pulsing/cluster.json` — adjust default_agents and puzzles if needed\n\
2. Create or update project files (README.md, tests/, configs) with Write/Edit\n\
3. Use Glob/Read to inspect before changing; keep changes minimal and practical\n\
4. Do not delete `.pulsing/history/`\n\
5. End with a short summary of what you configured\n";

pub async fn run_init_guide(
    root: PathBuf,
    guide: &str,
    provider: Option<&str>,
    model: Option<&str>,
) -> anyhow::Result<String> {
    let provider = provider
        .map(str::to_string)
        .unwrap_or_else(default_provider);
    let model = model
        .map(str::to_string)
        .unwrap_or_else(|| default_model_for_provider(&provider));

    let cfg = AgentConfig {
        cwd: root,
        provider,
        model,
        max_turns: 15,
        system_prompt: Some(INIT_SYSTEM.to_string()),
        tool_names: INIT_TOOL_NAMES.iter().map(|s| s.to_string()).collect(),
        ..AgentConfig::default()
    };

    let prompt = format!(
        "Workspace bootstrap goal:\n\n{guide}\n\n\
         Start by reading `.pulsing/cluster.json` and the project root, then apply changes."
    );
    run_agent_turn(&cfg, &prompt).await
}
