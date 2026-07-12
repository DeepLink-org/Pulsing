//! Slash + meta commands — Codex-compatible names with Forge REPL semantics.

/// What a slash command does in this REPL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlashAction {
    Meta(&'static str),
    Clear,
    Diff,
    Ps,
    Stop,
    Fork,
    Compact,
    Mcp,
    Plugins,
    Mention,
    Review,
    Rollout,
    Copy,
}

pub struct SlashDef {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub description: &'static str,
    pub action: SlashAction,
}

/// Presentation order: frequent first (Codex convention — do not alpha-sort).
pub const SLASH_DEFS: &[SlashDef] = &[
    SlashDef {
        name: "help",
        aliases: &[],
        description: "show commands and examples",
        action: SlashAction::Meta("help"),
    },
    SlashDef {
        name: "tools",
        aliases: &[],
        description: "list registered Forge tools",
        action: SlashAction::Meta("tools"),
    },
    SlashDef {
        name: "session",
        aliases: &["status"],
        description: "cwd, approval, replay progress (Codex /status)",
        action: SlashAction::Meta("session"),
    },
    SlashDef {
        name: "plan",
        aliases: &[],
        description: "show collaborative task plan",
        action: SlashAction::Meta("plan"),
    },
    SlashDef {
        name: "permissions",
        aliases: &["approve"],
        description: "approval mode: auto | ask (Codex /permissions)",
        action: SlashAction::Meta("approve"),
    },
    SlashDef {
        name: "replay",
        aliases: &[],
        description: "replay trace step or /replay all [dry] [verify]",
        action: SlashAction::Meta("replay"),
    },
    SlashDef {
        name: "trace",
        aliases: &[],
        description: "trace save PATH | trace show",
        action: SlashAction::Meta("trace"),
    },
    SlashDef {
        name: "fork",
        aliases: &[],
        description: "fork trace at step N then continue interactively",
        action: SlashAction::Fork,
    },
    SlashDef {
        name: "ps",
        aliases: &[],
        description: "list background unified-exec sessions (Codex /ps)",
        action: SlashAction::Ps,
    },
    SlashDef {
        name: "stop",
        aliases: &["clean"],
        description: "stop all background exec sessions (Codex /clean)",
        action: SlashAction::Stop,
    },
    SlashDef {
        name: "diff",
        aliases: &[],
        description: "git diff including untracked files (Codex /diff)",
        action: SlashAction::Diff,
    },
    SlashDef {
        name: "review",
        aliases: &[],
        description: "show workspace diff for manual review",
        action: SlashAction::Review,
    },
    SlashDef {
        name: "compact",
        aliases: &[],
        description: "request new context (maps to new_context tool)",
        action: SlashAction::Compact,
    },
    SlashDef {
        name: "mcp",
        aliases: &[],
        description: "list MCP-related tools",
        action: SlashAction::Mcp,
    },
    SlashDef {
        name: "plugins",
        aliases: &[],
        description: "list installable plugins",
        action: SlashAction::Plugins,
    },
    SlashDef {
        name: "mention",
        aliases: &[],
        description: "read file preview: /mention path/to/file",
        action: SlashAction::Mention,
    },
    SlashDef {
        name: "rollout",
        aliases: &[],
        description: "show active trace/record file paths",
        action: SlashAction::Rollout,
    },
    SlashDef {
        name: "copy",
        aliases: &[],
        description: "print last tool result (markdown-friendly)",
        action: SlashAction::Copy,
    },
    SlashDef {
        name: "clear",
        aliases: &[],
        description: "clear terminal screen",
        action: SlashAction::Clear,
    },
    SlashDef {
        name: "events",
        aliases: &[],
        description: "recent exec deltas / forge events",
        action: SlashAction::Meta("events"),
    },
    SlashDef {
        name: "call",
        aliases: &[],
        description: "call TOOL {json}",
        action: SlashAction::Meta("call"),
    },
    SlashDef {
        name: "quit",
        aliases: &["exit"],
        description: "exit REPL",
        action: SlashAction::Meta("quit"),
    },
];

pub const META_COMMANDS: &[(&str, &str)] = &[
    ("help", "show commands"),
    ("tools", "list tools"),
    ("session", "session snapshot"),
    ("plan", "task plan"),
    ("events", "exec deltas"),
    ("approve", "auto | ask"),
    ("replay", "step | all [dry] [verify]"),
    ("trace", "save PATH | show"),
    ("fork", "N — fork trace at step"),
    ("call", "TOOL {json}"),
    ("quit", "exit"),
];

pub const APPROVE_MODES: &[&str] = &["auto", "ask"];
pub const REPLAY_FLAGS: &[&str] = &["all", "dry", "verify"];
pub const TRACE_SUBCMDS: &[&str] = &["save", "show"];

pub const TOOL_FLAGS: &[(&str, &[&str])] = &[
    ("Read", &["--file_path", "--offset", "--limit"]),
    ("Glob", &["--pattern", "--path"]),
    ("Grep", &["--pattern", "--path", "--glob"]),
    ("Edit", &["--file_path", "--old_string", "--new_string"]),
    ("Write", &["--file_path", "--content"]),
    ("Bash", &["--command"]),
    (
        "shell_command",
        &[
            "--command",
            "--workdir",
            "--timeout_ms",
            "--login",
            "--sandbox_permissions",
        ],
    ),
    (
        "exec_command",
        &[
            "--cmd",
            "--workdir",
            "--tty",
            "--yield_time_ms",
            "--max_output_tokens",
        ],
    ),
    (
        "write_stdin",
        &["--session_id", "--chars", "--yield_time_ms"],
    ),
    ("apply_patch", &["--patch"]),
    ("view_image", &["--path", "--detail"]),
    ("tool_search", &["--query", "--limit"]),
    (
        "request_plugin_install",
        &[
            "--tool_type",
            "--action_type",
            "--tool_id",
            "--suggest_reason",
        ],
    ),
    ("request_permissions", &["--reason"]),
    ("list_mcp_resources", &["--server"]),
    ("list_mcp_resource_templates", &["--server", "--cursor"]),
    ("read_mcp_resource", &["--server", "--uri"]),
];

pub const HELP: &str = r#"Forge session REPL — Codex-aligned slash commands + ToolRuntime

Tools (Nushell-style):
  Read --file_path README.md
  @README.md                    mention → Read preview

Slash (Tab after /):
  /help /tools /session /plan /permissions auto
  /replay all verify  /fork 3  /ps  /stop  /diff  /compact
  /mcp  /plugins  /mention src/main.rs  /rollout  /copy  /clear

Meta:  tools | session | approve ask | replay | trace show | quit
Tips:  Tab · gray hints · A=auto K=ask · :cmd = bare meta alias
"#;

/// Parsed user input before dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedInput {
    Empty,
    SlashMenu,
    Slash { action: SlashAction, args: String },
    Mention(String),
    Meta(String),
    Bare(String),
}

pub fn parse_line(line: &str) -> ParsedInput {
    let line = line.trim();
    if line.is_empty() {
        return ParsedInput::Empty;
    }
    if line.starts_with('@') {
        let path = line[1..].trim();
        if path.is_empty() {
            return ParsedInput::SlashMenu;
        }
        return ParsedInput::Mention(path.to_string());
    }
    if line.starts_with('/') {
        let rest = line[1..].trim();
        if rest.is_empty() {
            return ParsedInput::SlashMenu;
        }
        let mut parts = rest.split_whitespace();
        let head = parts.next().unwrap_or("");
        let args = parts.collect::<Vec<_>>().join(" ");
        let def = match resolve_slash(head) {
            Some(d) => d,
            None => return ParsedInput::Bare(line.to_string()),
        };
        return ParsedInput::Slash {
            action: def.action.clone(),
            args,
        };
    }
    if line.starts_with(':') {
        return ParsedInput::Meta(line[1..].trim().to_string());
    }
    ParsedInput::Bare(line.to_string())
}

pub fn resolve_slash(head: &str) -> Option<&'static SlashDef> {
    SLASH_DEFS
        .iter()
        .find(|d| d.name == head || d.aliases.contains(&head))
}

/// Flat list for tab completion: name + aliases.
pub fn slash_completion_names() -> Vec<(&'static str, &'static str)> {
    let mut out = Vec::new();
    for d in SLASH_DEFS {
        out.push((d.name, d.description));
        for a in d.aliases {
            out.push((a, d.description));
        }
    }
    out
}

pub fn print_slash_menu() {
    println!("Slash commands (Tab to complete):");
    for d in SLASH_DEFS {
        let aliases = if d.aliases.is_empty() {
            String::new()
        } else {
            format!(" (alias: {})", d.aliases.join(", "))
        };
        println!("  /{:<14} {}{}", d.name, d.description, aliases);
    }
    println!("  @file            mention → Read preview");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_alias_resolves_to_stop() {
        let p = parse_line("/clean");
        assert_eq!(
            p,
            ParsedInput::Slash {
                action: SlashAction::Stop,
                args: String::new()
            }
        );
    }

    #[test]
    fn status_alias_resolves_to_session() {
        let def = resolve_slash("status").unwrap();
        assert_eq!(def.action, SlashAction::Meta("session"));
    }

    #[test]
    fn mention_parses_at_path() {
        assert_eq!(
            parse_line("@src/main.rs"),
            ParsedInput::Mention("src/main.rs".into())
        );
    }
}
