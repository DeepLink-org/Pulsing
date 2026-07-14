use pulsing_forge::{AgentConfig, DEFAULT_TOOL_NAMES};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatMode {
    Agent,
    AgentSafe,
    Ask,
    Plan,
}

impl ChatMode {
    pub const ALL: [ChatMode; 4] = [
        ChatMode::Agent,
        ChatMode::AgentSafe,
        ChatMode::Ask,
        ChatMode::Plan,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Agent => "Agent",
            Self::AgentSafe => "Agent · Safe",
            Self::Ask => "Ask",
            Self::Plan => "Plan",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Self::Agent => "Full tools — read, edit, and run commands in the workspace.",
            Self::AgentSafe => "Full tools with restricted sandbox for shell commands.",
            Self::Ask => "Chat only — no tools, best for quick questions.",
            Self::Plan => "Read-only exploration — browse files before acting.",
        }
    }

    pub fn apply(self, cfg: &mut AgentConfig) {
        match self {
            Self::Agent => {
                cfg.sandbox = "off".into();
                cfg.tool_names = DEFAULT_TOOL_NAMES
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect();
                cfg.system_prompt = None;
            }
            Self::AgentSafe => {
                cfg.sandbox = "restricted".into();
                cfg.tool_names = DEFAULT_TOOL_NAMES
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect();
                cfg.system_prompt = None;
            }
            Self::Ask => {
                cfg.sandbox = "off".into();
                cfg.tool_names = vec![];
                cfg.system_prompt = Some(
                    "You are a helpful coding assistant. Answer clearly and concisely without using tools."
                        .into(),
                );
            }
            Self::Plan => {
                cfg.sandbox = "off".into();
                cfg.tool_names = ["update_plan", "Glob", "Read", "Grep"]
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect();
                cfg.system_prompt = Some(
                    "You help plan work by reading the codebase. Use read-only tools; do not edit files or run shell commands."
                        .into(),
                );
            }
        }
    }
}

pub struct ModelPreset {
    pub provider: &'static str,
    pub section: &'static str,
    pub label: &'static str,
    pub model: &'static str,
}

pub const MODEL_PRESETS: &[ModelPreset] = &[
    ModelPreset {
        provider: "demo",
        section: "Demo",
        label: "Demo (offline)",
        model: "demo",
    },
    ModelPreset {
        provider: "anthropic",
        section: "Anthropic",
        label: "Claude Sonnet 4",
        model: "claude-sonnet-4-20250514",
    },
    ModelPreset {
        provider: "anthropic",
        section: "Anthropic",
        label: "Claude 3.5 Sonnet",
        model: "claude-3-5-sonnet-20241022",
    },
    ModelPreset {
        provider: "openai",
        section: "OpenAI",
        label: "GPT-4o",
        model: "gpt-4o",
    },
    ModelPreset {
        provider: "openai",
        section: "OpenAI",
        label: "GPT-4o mini",
        model: "gpt-4o-mini",
    },
];

pub fn provider_available(provider: &str) -> bool {
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

pub fn build_agent_config(
    cwd: &std::path::Path,
    provider: &str,
    model: &str,
    mode: ChatMode,
) -> AgentConfig {
    let mut cfg = AgentConfig {
        cwd: cwd.to_path_buf(),
        provider: provider.to_string(),
        model: model.to_string(),
        ..AgentConfig::default()
    };
    mode.apply(&mut cfg);
    cfg
}
