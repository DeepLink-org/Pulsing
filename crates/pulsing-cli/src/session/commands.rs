//! Slash commands and legacy aliases (parsed in pulsing-cli only).

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputAction {
    Help,
    Mode,
    Exit,
    History,
    Checkpoint { message: Option<String> },
    Rollback { revision: Option<String> },
    WorkflowList,
    WorkflowRun { script: Option<String> },
    RerunWorkflow,
    AgentTask { prompt: String },
}

pub fn parse_line(line: &str) -> InputAction {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return InputAction::AgentTask {
            prompt: String::new(),
        };
    }

    if let Some(rest) = trimmed.strip_prefix('/') {
        return parse_slash(rest);
    }

    // Legacy aliases (discoverable via /help).
    match trimmed.to_lowercase().as_str() {
        "exit" | "quit" | "q" => InputAction::Exit,
        "rerun" | "run" => InputAction::RerunWorkflow,
        "rollback" => InputAction::Rollback { revision: None },
        _ => InputAction::AgentTask {
            prompt: trimmed.to_string(),
        },
    }
}

fn parse_slash(rest: &str) -> InputAction {
    let mut parts = rest.split_whitespace();
    let cmd = parts.next().unwrap_or("").to_lowercase();
    match cmd.as_str() {
        "help" | "h" | "?" => InputAction::Help,
        "mode" => InputAction::Mode,
        "exit" | "quit" | "q" => InputAction::Exit,
        "history" => InputAction::History,
        "checkpoint" | "cp" => {
            let msg = parts.collect::<Vec<_>>().join(" ");
            InputAction::Checkpoint {
                message: if msg.is_empty() { None } else { Some(msg) },
            }
        }
        "rollback" | "rb" => {
            let rev = parts.next().map(str::to_string);
            InputAction::Rollback { revision: rev }
        }
        "workflow" | "wf" => {
            let sub = parts.next().unwrap_or("list").to_lowercase();
            match sub.as_str() {
                "list" => InputAction::WorkflowList,
                "run" => {
                    let script = parts.next().map(str::to_string);
                    InputAction::WorkflowRun { script }
                }
                _ => InputAction::Help,
            }
        }
        _ => InputAction::Help,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slash_help() {
        assert_eq!(parse_line("/help"), InputAction::Help);
    }

    #[test]
    fn slash_checkpoint_with_message() {
        assert_eq!(
            parse_line("/checkpoint after deploy"),
            InputAction::Checkpoint {
                message: Some("after deploy".into()),
            }
        );
    }

    #[test]
    fn legacy_rerun() {
        assert_eq!(parse_line("rerun"), InputAction::RerunWorkflow);
    }

    #[test]
    fn agent_task() {
        match parse_line("fix tests") {
            InputAction::AgentTask { prompt } => assert_eq!(prompt, "fix tests"),
            _ => panic!("expected agent task"),
        }
    }
}
